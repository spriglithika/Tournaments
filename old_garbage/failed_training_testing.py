
def joint_train_all_separate(device, train_loader, models, class_count, temps = [1,1,1], lbda = 1.0):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader)
    # min_logit = models['tournament'][0].tournament.min_logit
    separated_preds = SeparateConfidence(class_count)
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt, sch) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
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
            out_tourn, tourn_mid = models['tournament'][0](data, train=True)
            sep_preds = separated_preds(tourn_mid)
            # loss_base = lsce(out_base, target)
            loss_base = F.cross_entropy(out_base * temps[0], target)
            # loss_mid = lsce(out_mid, target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target)
            # loss_tourn = sce(out_tourn, target)
            # loss_tourn = isce(out_tourn, tourn_target)
            # loss_tourn = F.mse_loss(out_tourn * target, target)
            loss_tourn = F.cross_entropy(out_tourn * temps[2], target) - lbda * torch.mean((tourn_mid - 0.5).abs())
            # loss_tourn = F.cross_entropy(sep_preds * temps[2], target) - lbda * torch.mean((tourn_mid - 0.5).abs())
            # loss_tourn = F.cross_entropy(out_tourn * temps[2], target) + F.cross_entropy(sep_preds, target) - lbda * torch.mean((tourn_mid - 0.5).abs()) # trying all three out

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

