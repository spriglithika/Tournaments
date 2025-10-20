from preamble import *
from tqdm import tqdm
import Tournament
from TournamentThresholds import *
sce = Tournament.symmetric_cross_entropy

# Single AMP scaler for joint backward to avoid per-model scaler interactions
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())


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


def joint_train_all(device, train_loader, models, class_count):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader)
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target = F.one_hot(target.to(device, non_blocking=True), num_classes=class_count).float()
        tourn_target = smooth_labels(target, smoothing=0.4)

        # zero grads for all optimizers
        for _, (m, _s, opt) in models.items():
            opt.zero_grad()

        # forward under autocast
        with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=device.type):
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            out_tourn = models['tournament'][0](data, train=True)

            loss_base = sce(out_base, target)
            loss_mid = sce(out_mid, target)
            loss_tourn = sce(out_tourn, target)

        # scale the summed loss and backward once to keep AMP stable
        total_loss = loss_base + loss_mid + loss_tourn
        scaler.scale(total_loss).backward()

        # optional: unscale and clip gradients per-model to avoid explosion
        # (unscale requires passing the optimizer whose params' grads should be unscaled)
        for name, (m, s, opt) in models.items():
            # `s` entry is ignored now (kept for compatibility in the models dict)
            try:
                scaler.unscale_(opt)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

        # step each optimizer (scaler.step handles skipped steps due to infs/nans)
        for name, (m, s, opt) in models.items():
            scaler.step(opt)

        # update scaler once per iteration
        scaler.update()

        pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item()})
        if batch_idx % 100 == 0:
            print(out_tourn.min(), out_tourn.max())


def joint_eval_all(device, test_loader, models, class_count, monitor: 'ConvergenceMonitor' = None, epoch: int = None):
    for name, (m, _s, opt) in models.items():
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
    center = CenterThresholding(class_count, alpha=0.1).to(device)
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
            out_tourn = models['tournament'][0](data, train=True)
            # keep middle outputs on the same device as models/thresholds
            out_tourn_mid = models['tournament'][0].middle(data).detach()
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
        print(f"{k} Test set: Average loss: {test_loss[k]:.4f}, Accuracy: {correct[k]}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
        # update convergence monitor if provided
        if monitor is not None:
            try:
                monitor.update(k, float(accuracy), epoch=epoch, model=models[k][0])
            except Exception:
                pass
    naive_acc = 100. * correct['tournament_naive'] / len(test_loader.dataset)
    print(f"tournament_naive Test set: Accuracy: {correct['tournament_naive']}/{len(test_loader.dataset)} ({naive_acc:.2f}%)")
    center_acc = 100. * correct['tournament_center'] / len(test_loader.dataset)
    print(f"tournament_center Test set: Accuracy: {correct['tournament_center']}/{len(test_loader.dataset)} ({center_acc:.2f}%)")
    bern_acc = 100. * correct['tournament_bern'] / len(test_loader.dataset)
    print(f"tournament_bern Test set: Accuracy: {correct['tournament_bern']}/{len(test_loader.dataset)} ({bern_acc:.2f}%)")
    seperate_acc = 100. * correct['tournament_seperate'] / len(test_loader.dataset)
    print(f"tournament_seperate Test set: Accuracy: {correct['tournament_seperate']}/{len(test_loader.dataset)} ({seperate_acc:.2f}%)")
    single_acc = 100. * correct['tournament_single'] / len(test_loader.dataset)
    print(f"tournament_single Test set: Accuracy: {correct['tournament_single']}/{len(test_loader.dataset)} ({single_acc:.2f}%)")