from preamble import *
from tqdm import tqdm
import Tournament
from TournamentThresholds import *
sce = Tournament.symmetric_cross_entropy
lsce = Tournament.log_symmetric_cross_entropy
isce = Tournament.ioannis_symmetric_cross_entropy
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
if float(torch.__version__.split(".")[0]+"."+torch.__version__.split(".")[1]) >= 2.4:
    amp = torch.amp
    caster = torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=device_type)
else:
    amp = torch.cuda.amp if torch.cuda.is_available() else torch.cpu.amp
    caster = torch.autocast(enabled=torch.cuda.is_available(), device_type=device_type)

# Single AMP scaler for joint backward to avoid per-model scaler interactions
scaler = amp.GradScaler(enabled=torch.cuda.is_available())



def smooth_labels(y, smoothing=0.1):
    return y * (1 - smoothing) + smoothing / y.size(-1)


class ConvergenceMonitor:
    """Track a metric (e.g., accuracy) per model, save best checkpoints and
    optionally mark a model as converged after a patience period.

    Usage:
      monitor = ConvergenceMonitor(patience=3, mode='max', save_dir='./ckpts')
      joint_eval_all(..., monitor=monitor, epoch=epoch)
    """
    def __init__(self, patience=3, mode='max', save_dir='./ckpts'):
        assert mode in ('max', 'min')
        self.patience = int(patience)
        self.mode = mode
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # Per-model state
        self._state = {}

    def _better(self, name, value):
        st = self._state.setdefault(name, {})
        if 'best' not in st:
            return True
        if self.mode == 'max':
            return value > st['best']
        else:
            return value < st['best']

    def update(self, name, value, epoch=None, model=None):
        """Update metric for model `name`. If improved, save a snapshot.

        Returns True if improvement detected (and saved), otherwise False.
        """
        st = self._state.setdefault(name, {
            'best': None,
            'best_epoch': None,
            'since_improve': 0,
            'converged': False,
        })

        improved = False
        if st['best'] is None or self._better(name, value):
            st['best'] = float(value)
            st['best_epoch'] = epoch
            st['since_improve'] = 0
            improved = True
            # save snapshot if model provided
            if model is not None:
                fn = os.path.join(self.save_dir, f"{name}_best_epoch{epoch or 'NA'}.pth")
                try:
                    torch.save(model.state_dict(), fn)
                except Exception:
                    # Do not raise: saving should not interrupt training
                    pass
                st['best_path'] = fn
        else:
            st['since_improve'] = st.get('since_improve', 0) + 1

        # mark converged if patience exceeded
        if st['since_improve'] >= self.patience:
            st['converged'] = True

        return improved


def joint_train_all(device, train_loader, models, class_count, temps = [1,1,1]):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader)
    min_logit = models['tournament'][0].tournament.min_logit
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt, sch) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target = F.one_hot(target.to(device, non_blocking=True), num_classes=class_count).float()
        tourn_min_logits = torch.ones_like(target) * min_logit
        tourn_target = torch.where(target == 0, tourn_min_logits, target)

        # zero grads for all optimizers
        for _, (m, _s, opt, sch) in models.items():
            opt.zero_grad()

        # forward under autocast
        with caster:
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            out_tourn, tourn_mid = models['tournament'][0](data, train=True)

            # loss_base = lsce(out_base, target)
            loss_base = F.cross_entropy(out_base * temps[0], target)
            # loss_mid = lsce(out_mid, target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target)
            # loss_tourn = sce(out_tourn, target)
            # loss_tourn = isce(out_tourn, tourn_target)
            # loss_tourn = F.mse_loss(out_tourn * target, target)
            loss_tourn = F.cross_entropy(out_tourn * temps[2], target) - torch.mean((tourn_mid - 0.5).abs())

        # scale the summed loss and backward once to keep AMP stable
        total_loss = loss_base + loss_mid + loss_tourn
        # scaler.scale(total_loss).backward(retain_graph=True)
        scaler.scale(total_loss).backward()

        # optional: unscale and clip gradients per-model to avoid explosion
        # (unscale requires passing the optimizer whose params' grads should be unscaled)
        for name, (m, s, opt, sch) in models.items():
            # `s` entry is ignored now (kept for compatibility in the models dict)
            try:
                scaler.unscale_(opt)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

        # step each optimizer (scaler.step handles skipped steps due to infs/nans)
        for name, (m, s, opt, sch) in models.items():
            scaler.step(opt)

        # update scaler once per iteration
        scaler.update()

        pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
        # if batch_idx % 100 == 0:
            # print()
    models["base"][-1].step()
    models["mid"][-1].step()
    models["tournament"][-1].step()

def joint_train_all_triplet(device, train_loader, models, class_count, temps = [1,1,1]):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader)
    # min_logit = models['tournament'][0].tournament.min_logit
    edge_pairs = models['tournament'][0].tournament.perms
    vectorized_triplet_loss = TournamentTripletLoss().to(device)
    vectorized_margin_loss = TournamentMarginLoss(edge_pairs).to(device)
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt, sch) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target_int = target.clone().to(device)
        target = F.one_hot(target.to(device, non_blocking=True), num_classes=class_count).float()
        # tourn_min_logits = torch.ones_like(target) * min_logit
        # tourn_target = torch.where(target == 0, tourn_min_logits, target)

        # zero grads for all optimizers
        for _, (m, _s, opt, sch) in models.items():
            opt.zero_grad()

        # forward under autocast
        with caster:
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            _, out_tourn = models['tournament'][0](data)

            # loss_base = lsce(out_base, target)
            loss_base = F.cross_entropy(out_base * temps[0], target)
            # loss_mid = lsce(out_mid, target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target)
            # loss_tourn = sce(out_tourn, target)
            # loss_tourn = isce(out_tourn, tourn_target)
            # loss_tourn = F.mse_loss(out_tourn * target, target)
            # loss_tourn = F.cross_entropy(out_tourn * temps[2], target)

            confidence_loss = -torch.mean((out_tourn - 0.5).abs())

            loss_tourn = vectorized_margin_loss(out_tourn, target_int) + confidence_loss

        # scale the summed loss and backward once to keep AMP stable
        total_loss = loss_base + loss_mid + loss_tourn
        # scaler.scale(total_loss).backward(retain_graph=True)
        scaler.scale(total_loss).backward()

        # optional: unscale and clip gradients per-model to avoid explosion
        # (unscale requires passing the optimizer whose params' grads should be unscaled)
        for name, (m, s, opt, sch) in models.items():
            # `s` entry is ignored now (kept for compatibility in the models dict)
            try:
                scaler.unscale_(opt)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

        # step each optimizer (scaler.step handles skipped steps due to infs/nans)
        for name, (m, s, opt, sch) in models.items():
            scaler.step(opt)

        # update scaler once per iteration
        scaler.update()

        pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
        # if batch_idx % 100 == 0:
            # print()
    models["base"][-1].step()
    models["mid"][-1].step()
    models["tournament"][-1].step()

def joint_eval_all(device, test_loader, models, class_count, monitor: 'ConvergenceMonitor' = None, epoch: int = None, mode = 'Val'):
    for name, (m, _s, opt,sch) in models.items():
        m.eval()
    test_loss = {k:0 for k in models.keys()}
    correct = {k:0 for k in models.keys()}
    correct['tournament_naive'] = 0
    correct['tournament_center'] = 0
    correct['tournament_bern'] = 0
    correct['tournament_seperate'] = 0
    correct['tournament_single'] = 0
    # create threshold modules on the same device as evaluation
    naive = NaiveThresholding(class_count).to(device)
    center = CenterThresholding(class_count, alpha=0.2).to(device)
    bern = BernoulliThresholding(class_count).to(device)
    seperate = SeparateConfidence(class_count).to(device)
    single = SingleConfidence(class_count).to(device)

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_oh = F.one_hot(target, num_classes=class_count).float()
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            out_tourn, out_tourn_mid = models['tournament'][0](data, train=True)
            # keep middle outputs on the same device as models/thresholds
            out_tourn_naive = naive(out_tourn_mid)
            out_tourn_center = center(out_tourn_mid)
            out_tourn_bern = bern(out_tourn_mid)
            out_tourn_seperate = seperate(out_tourn_mid)
            out_tourn_single = single(out_tourn_mid)

            test_loss['base'] += sce(out_base, target_oh, reduction='sum').item()
            test_loss['mid'] += sce(out_mid, target_oh, reduction='sum').item()
            test_loss['tournament'] += sce(out_tourn, target_oh, reduction='sum').item()


            pred_base = out_base.argmax(dim=1, keepdim=True)
            pred_mid = out_mid.argmax(dim=1, keepdim=True)
            pred_tourn = out_tourn.argmax(dim=1, keepdim=True)
            pred_tourn_naive = out_tourn_naive.argmax(dim=1, keepdim=True)
            pred_tourn_center = out_tourn_center.argmax(dim=1, keepdim=True)
            pred_tourn_bern = out_tourn_bern.argmax(dim=1, keepdim=True)
            pred_tourn_seperate = out_tourn_seperate.argmax(dim=1, keepdim=True)
            pred_tourn_single = out_tourn_single.argmax(dim=1, keepdim=True)

            correct['base'] += pred_base.eq(target.view_as(pred_base)).sum().item()
            correct['mid'] += pred_mid.eq(target.view_as(pred_mid)).sum().item()
            correct['tournament'] += pred_tourn.eq(target.view_as(pred_tourn)).sum().item()
            correct['tournament_naive'] += pred_tourn_naive.eq(target.view_as(pred_tourn_naive)).sum().item()
            correct['tournament_center'] += pred_tourn_center.eq(target.view_as(pred_tourn_center)).sum().item()
            correct['tournament_bern'] += pred_tourn_bern.eq(target.view_as(pred_tourn_bern)).sum().item()
            correct['tournament_seperate'] += pred_tourn_seperate.eq(target.view_as(pred_tourn_seperate)).sum().item()
            correct['tournament_single'] += pred_tourn_single.eq(target.view_as(pred_tourn_single)).sum().item()

    for k in models.keys():
        test_loss[k] /= len(test_loader.dataset)
        accuracy = 100. * correct[k] / len(test_loader.dataset)
        print(f"{k} {mode} set: Average loss: {test_loss[k]:.4f}, Accuracy: {correct[k]}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
        # update convergence monitor if provided
        if monitor is not None:
            try:
                monitor.update(k, float(accuracy), epoch=epoch, model=models[k][0])
            except Exception:
                pass
    naive_acc = 100. * correct['tournament_naive'] / len(test_loader.dataset)
    print(f"tournament_naive {mode} set: Accuracy: {correct['tournament_naive']}/{len(test_loader.dataset)} ({naive_acc:.2f}%)")
    center_acc = 100. * correct['tournament_center'] / len(test_loader.dataset)
    print(f"tournament_center {mode} set: Accuracy: {correct['tournament_center']}/{len(test_loader.dataset)} ({center_acc:.2f}%)")
    bern_acc = 100. * correct['tournament_bern'] / len(test_loader.dataset)
    print(f"tournament_bern {mode} set: Accuracy: {correct['tournament_bern']}/{len(test_loader.dataset)} ({bern_acc:.2f}%)")
    seperate_acc = 100. * correct['tournament_seperate'] / len(test_loader.dataset)
    print(f"tournament_seperate {mode} set: Accuracy: {correct['tournament_seperate']}/{len(test_loader.dataset)} ({seperate_acc:.2f}%)")
    single_acc = 100. * correct['tournament_single'] / len(test_loader.dataset)
    print(f"tournament_single {mode} set: Accuracy: {correct['tournament_single']}/{len(test_loader.dataset)} ({single_acc:.2f}%)")


class TournamentMarginLoss(nn.Module):
    def __init__(self, edge_pairs, margin=0.2):
        """
        edge_pairs: [E, 2] numpy array or tensor of class index pairs (i, j)
        margin: float, margin for triplet loss
        """
        super().__init__()
        edge_pairs = torch.tensor(edge_pairs, dtype=torch.long) if not torch.is_tensor(edge_pairs) else edge_pairs
        self.register_buffer('edge_pairs', edge_pairs)
        self.margin = margin

    def forward(self, batch_scores, true_classes):
        B, E = batch_scores.shape
        device = batch_scores.device
        # Use buffer directly — it's already on the correct device
        edge_pairs = self.edge_pairs
        edge_i = edge_pairs[:, 0].unsqueeze(0).expand(B, -1)  # [B, E]
        edge_j = edge_pairs[:, 1].unsqueeze(0).expand(B, -1)  # [B, E]
        true_classes_exp = true_classes.unsqueeze(1).expand(-1, E)  # [B, E]

        # Identify edges where true class is involved
        true_in_i = (edge_i == true_classes_exp)
        true_in_j = (edge_j == true_classes_exp)
        true_in_edge = true_in_i | true_in_j  # [B, E]

        involved_scores = batch_scores[true_in_edge]  # [N]
        true_wins = torch.where(true_in_i[true_in_edge], involved_scores < 0.5, involved_scores > 0.5)
        pos_scores = involved_scores[true_wins]
        neg_scores = involved_scores[~true_wins]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_scores = pos_scores.unsqueeze(1)
        neg_scores = neg_scores.unsqueeze(0)
        triplet_losses = F.relu(self.margin - (pos_scores - neg_scores))

        return triplet_losses.mean()


class TournamentTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        """
        margin: float - margin for triplet separation
        """
        super().__init__()
        self.margin = margin

    def forward(self, batch_scores, true_classes):
        """
        batch_scores: [B, E] - tournament outputs per sample
        true_classes: [B] - true class index per sample
        """
        B = batch_scores.size(0)
        device = batch_scores.device
        loss = 0.0
        count = 0

        for i in range(B):
            anchor = batch_scores[i].unsqueeze(0)  # [1, E]
            anchor_label = true_classes[i]

            # Find positives and negatives
            pos_mask = (true_classes == anchor_label) & (torch.arange(B, device=device) != i)
            neg_mask = (true_classes != anchor_label)

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            positives = batch_scores[pos_mask]  # [P, E]
            negatives = batch_scores[neg_mask]  # [N, E]

            # Compute distances
            pos_dist = F.pairwise_distance(anchor.expand_as(positives), positives)  # [P]
            neg_dist = F.pairwise_distance(anchor.expand_as(negatives), negatives)  # [N]

            # Broadcast and apply triplet loss
            pos_dist = pos_dist.unsqueeze(1)  # [P, 1]
            neg_dist = neg_dist.unsqueeze(0)  # [1, N]
            triplet_losses = F.relu(pos_dist - neg_dist + self.margin)  # [P, N]

            loss += triplet_losses.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss / count
