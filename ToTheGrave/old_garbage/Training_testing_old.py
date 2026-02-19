from preamble import *
from tqdm import tqdm
import old_garbage.Tournament
from old_garbage.TournamentThresholds import *
import psutil
import time
from old_garbage.TournamentGroundTruth import edge_loss_sparse, get_gt
from copy import deepcopy
sce = old_garbage.Tournament.symmetric_cross_entropy
lsce = old_garbage.Tournament.log_symmetric_cross_entropy
isce = old_garbage.Tournament.ioannis_symmetric_cross_entropy

# Single AMP scaler for joint backward to avoid per-model scaler interactions


def smooth_labels(y, smoothing=0.1):
    return y * (1 - smoothing) + smoothing / y.size(-1)

import os
import threading
import torch

class ConvergenceMonitor:
    """Track a metric (e.g., accuracy) per model, save best checkpoints and
    optionally mark a model as converged after a patience period.
    """
    def __init__(self, patience=3, mode='max', save_dir='./ckpts'):
        assert mode in ('max', 'min')
        self.patience = int(patience)
        self.mode = mode
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._state = {}

    def _better(self, name, value):
        st = self._state.setdefault(name, {})
        if 'best' not in st:
            return True
        if self.mode == 'max':
            return value > st['best']
        else:
            return value < st['best']

    def _save_async(self, state, path):
        def _save():
            try:
                torch.save(state, path)
            except Exception as e:
                print(f"Warning: Failed to save model to {path}: {e}")
        thread = threading.Thread(target=_save)
        thread.daemon = True  # won't block program exit
        thread.start()

    def update(self, name, value, epoch=None, model=None):
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
            if model is not None:
                fn = os.path.join(self.save_dir, f"{name}_best_epoch{epoch or 'NA'}.pth")
                self._save_async(deepcopy(model.state_dict()), fn)
                st['best_path'] = fn
        else:
            st['since_improve'] = st.get('since_improve', 0) + 1

        if st['since_improve'] >= self.patience:
            st['converged'] = True

        return improved

def joint_train_all(device, train_loader, models, class_count, temps = [1,1,1], lbda = 1.0, verbose = True):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader) if verbose else train_loader
    # min_logit = models['tournament'][0].tournament.min_logit
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

            # loss_base = lsce(out_base, target)
            loss_base = F.cross_entropy(out_base * temps[0], target)
            # loss_mid = lsce(out_mid, target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target)
            # loss_tourn = sce(out_tourn, target)
            # loss_tourn = isce(out_tourn, tourn_target)
            # loss_tourn = F.mse_loss(out_tourn * target, target)

            # loss_tourn = F.cross_entropy(out_tourn * temps[2], target) - lbda * torch.mean((tourn_mid - 0.5).abs())
            loss_confidence= lbda * torch.mean((tourn_mid - 0.5).abs())
            loss_tourn = F.mse_loss(out_tourn * temps[2], target) - loss_confidence
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

        if verbose:
            # pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
            pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item() + loss_confidence.item(), 'loss_conf': loss_confidence.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
        # if batch_idx % 100 == 0:
            # print()
    models["base"][-1].step()
    models["mid"][-1].step()
    models["tournament"][-1].step()

def joint_train_all_variational(device, train_loader, models, class_count, temps = [1,1,1], lbda = [0.01, 0.001, 0.005], verbose = True):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader) if verbose else train_loader
    # min_logit = models['tournament'][0].tournament.min_logit
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt, sch) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target_int = target.to(device, non_blocking=True)
        target = F.one_hot(target_int, num_classes=class_count).float()
        # tourn_min_logits = torch.ones_like(target) * min_logit
        # tourn_target = torch.where(target == 0, tourn_min_logits, target)

        # zero grads for all optimizers
        for _, (m, _s, opt, sch) in models.items():
            opt.zero_grad()

        # forward under autocast
        with caster:
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            out_tourn, tourn_mid, tourn_sample, tourn_mahalanobis, tourn_reg_loss, tourn_kl_div = models['tournament'][0](data, train=True)

            loss_base = F.cross_entropy(out_base * temps[0], target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target)
            loss_edges = edge_loss(tourn_sample, models['tournament'][0].tournament.gt, target_int)

            # loss_tourn =  F.cross_entropy(out_tourn, target) \
            loss_tourn = loss_edges \
           + lbda[0] * tourn_mahalanobis.mean() \
           + lbda[1] * tourn_reg_loss \
           + lbda[2] * tourn_kl_div


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

        if verbose:
            # pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
            pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_mahalanobis': tourn_mahalanobis.mean().item(), 'tourn_reg_loss': tourn_reg_loss.item(), 'tourn_kl_div': tourn_kl_div.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
        # if batch_idx % 100 == 0:
            # print()
    models["base"][-1].step()
    models["mid"][-1].step()
    models["tournament"][-1].step()

def joint_train_all_ising(device, train_loader, models, class_count, temps = [1,1,1], lbda = [1.0,1.0,1.0,1.0], epoch=0, verbose = True):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader) if verbose else train_loader
    gt, _ = get_gt(class_count)
    gt = gt.to(device)
    # min_logit = models['tournament'][0].tournament.min_logit
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt, sch) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target_int = target.to(device, non_blocking=True)
        target_oh = F.one_hot(target_int, num_classes=class_count).float().to(device)
        # tourn_min_logits = torch.ones_like(target) * min_logit
        # tourn_target = torch.where(target == 0, tourn_min_logits, target)

        # zero grads for all optimizers
        for _, (m, _s, opt, sch) in models.items():
            opt.zero_grad()

        # forward under autocast
        with caster:
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            out_tourn, tourn_mid = models['tournament'][0](data, alpha = (1.5-epoch), train=True)
            # tourn_mid = models['tournament'][0](data, alpha = (1.5-epoch), train=True)
            # print(out_tourn.shape, target_oh.shape)
            # loss_base = lsce(out_base, target)
            loss_base = F.cross_entropy(out_base * temps[0], target_oh)
            # loss_mid = lsce(out_mid, target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target_oh)
            # loss_tourn = sce(out_tourn, target)
            # loss_tourn = isce(out_tourn, tourn_target)
            # loss_tourn = F.mse_loss(out_tourn * target, target)

            # loss_tourn = F.cross_entropy(out_tourn * temps[2], target) - lbda * torch.mean((tourn_mid - 0.5).abs())
            # loss_confidence = torch.mean((out_tourn - 0.5).abs())
            loss_edges, loss_entropy = edge_loss(tourn_mid, gt, target_int)
            # loss_verts = F.l1_loss(out_tourn * temps[2], target)# * target
            # pre_fixed_verts = lsce(out_tourn * temps[2], target_oh, reduction='none')
            # print(pre_fixed_verts.shape)
            # loss_verts = (pre_fixed_verts * target_oh).sum(-1).mean()
            # loss_tourn = loss_verts * lbda[0] \
            #             + loss_edges * lbda[1]
            #             + loss_entropy * lbda[2] \
            #             - loss_confidence * lbda[3]
            loss_tourn = loss_edges * lbda[1] + loss_entropy * lbda[2]
        # print(tourn_mid.mean(0))
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

        if verbose:
            # pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
            # pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_verts': loss_verts.item(), 'loss_edges': loss_edges.item(), 'loss_entropy': loss_entropy.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
            pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_edges': loss_edges.item(), 'loss_entropy': loss_entropy.item()})
        # if batch_idx % 5 == 0:
            # print(f"RAM: {psutil.virtual_memory().percent}% | CPU: {psutil.cpu_percent()}%")
    models["base"][-1].step()
    models["mid"][-1].step()
    models["tournament"][-1].step()

def train_tourn_ising(device, train_loader, model, class_count, gt_stuff, temps = [1,1,1], lbda = [1.0,1.0,1.0,1.0], epoch=0, verbose = True):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader) if verbose else train_loader
    gt, gt_idx, perms = gt_stuff
    m, _, opt, sch = model
    m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target_int = target.to(device, non_blocking=True)
        target_oh = F.one_hot(target_int, num_classes=class_count).float().to(device)
        opt.zero_grad()

        out_tourn, tourn_mid = m(data, alpha = (1.5-epoch), train=True)
        loss_edges = edge_loss_sparse(tourn_mid, (gt, gt_idx), target_int)
        total_loss = loss_edges * lbda[1]
        total_loss.backward()
        opt.zero_grad()
        opt.step()

        if verbose:
            # pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
            pbar.set_postfix({'loss_edges': loss_edges.item()})

    sch.step()

def eval_tourn(device, test_loader, model, class_count, monitor: 'ConvergenceMonitor' = None, epoch: int = None, mode = 'Val'):
    m, _s, opt,sch = model
    m.eval()
    correct = 0

    single = SingleConfidence(class_count).to(device)

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            _, out_tourn_mid = model[0](data, train=True)
            out_tourn_single = single(out_tourn_mid)

            pred_tourn_single = out_tourn_single.argmax(dim=1, keepdim=True)
            correct += pred_tourn_single.eq(target.view_as(pred_tourn_single)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"{mode} set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    if monitor is not None:
        try:
            monitor.update('Tournament', float(accuracy), epoch=epoch, model=model[0])
        except Exception:
            pass


def joint_train_all_balanced(device, train_loader, models, class_count, temps = [1,1,1], lbda = 1.0, verbose = True):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader) if verbose else train_loader
    # min_logit = models['tournament'][0].tournament.min_logit
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt, sch) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target_int = target.to(device, non_blocking=True)
        target = F.one_hot(target_int, num_classes=class_count).float()
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

            # loss_base = lsce(out_base, target)
            loss_base = F.cross_entropy(out_base * temps[0], target)
            # loss_mid = lsce(out_mid, target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target)
            # loss_tourn = sce(out_tourn, target)
            # loss_tourn = isce(out_tourn, tourn_target)
            # loss_tourn = F.mse_loss(out_tourn * target, target)

            # loss_tourn = F.cross_entropy(out_tourn * temps[2], target) - lbda * torch.mean((tourn_mid - 0.5).abs())
            # loss_confidence = torch.mean((tourn_mid - 0.5).abs())
            loss_verts = (F.mse_loss(out_tourn * temps[2], target, reduction='none') * target).sum(-1).mean() *lbda[0]
            loss_edges = edge_loss(tourn_mid, models['tournament'][0].tournament.gt, target_int) * lbda[1]
            loss_tourn = loss_verts + loss_edges  #- lbda[2] * loss_confidence
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

        if verbose:
            # pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
            pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_verts': loss_verts.item() , 'loss_edges': loss_edges.item(), 'tourn_min': out_tourn.min().item(), 'tourn_max': out_tourn.max().item()})
        # if batch_idx % 100 == 0:
            # print()
    models["base"][-1].step()
    models["mid"][-1].step()
    models["tournament"][-1].step()

def joint_train_all_mixup(device, train_loader, models, class_count, temps = [1,1,1], lbda = 1.0, verbose = True):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader) if verbose else train_loader
    # min_logit = models['tournament'][0].tournament.min_logit
    mix_up = MixUpTransform(class_count)
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt, sch) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = mix_up(data, target)
        data = data.to(device,non_blocking=True)
        target = target.to(device, non_blocking=True)
        # target = F.one_hot(target.to(device, non_blocking=True), num_classes=class_count).float()
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

            # loss_base = lsce(out_base, target)
            loss_base = F.cross_entropy(out_base * temps[0], target)
            # loss_mid = lsce(out_mid, target)
            loss_mid = F.cross_entropy(out_mid * temps[1], target)
            # loss_tourn = sce(out_tourn, target)
            # loss_tourn = isce(out_tourn, tourn_target)
            # loss_tourn = F.mse_loss(out_tourn * target, target)

            # loss_tourn = F.cross_entropy(out_tourn * temps[2], target) - lbda * torch.mean((tourn_mid - 0.5).abs())
            # loss_tourn = F.mse_loss(out_tourn * temps[2], target)* lbda[0] - lbda[1] * torch.mean((tourn_mid - 0.5).abs())
            loss_tourn = F.cross_entropy(out_tourn * temps[2], target)* lbda[0] - lbda[1] * torch.mean((tourn_mid - 0.5).abs())

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

        if verbose:
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
    center = CenterThresholding(class_count, alpha=0.05).to(device)
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
            _, out_tourn_mid = models['tournament'][0](data, train=True)
            out_tourn = torch.zeros_like(out_base)
            # keep middle outputs on the same device as models/thresholds
            out_tourn_naive = naive(out_tourn_mid)
            out_tourn_center = center(out_tourn_mid)
            out_tourn_bern = bern(out_tourn_mid)
            out_tourn_seperate = seperate(out_tourn_mid)
            out_tourn_single = single(out_tourn_mid)

            test_loss['base'] += F.cross_entropy(out_base, target_oh, reduction='sum').item()
            test_loss['mid'] += F.cross_entropy(out_mid, target_oh, reduction='sum').item()
            test_loss['tournament'] += F.cross_entropy(out_tourn, target_oh, reduction='sum').item()


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
            # correct['tournament'] += pred_tourn.eq(target.view_as(pred_tourn)).sum().item()
            correct['tournament'] += pred_tourn_single.eq(target.view_as(pred_tourn_single)).sum().item()
            correct['tournament_naive'] += pred_tourn_naive.eq(target.view_as(pred_tourn_naive)).sum().item()
            correct['tournament_center'] += pred_tourn_center.eq(target.view_as(pred_tourn_center)).sum().item()
            correct['tournament_bern'] += pred_tourn_bern.eq(target.view_as(pred_tourn_bern)).sum().item()
            correct['tournament_seperate'] += pred_tourn_seperate.eq(target.view_as(pred_tourn_seperate)).sum().item()
            correct['tournament_single'] += pred_tourn_single.eq(target.view_as(pred_tourn_single)).sum().item()
            counts = torch.zeros(6, class_count)
            # for i in range(class_count):
            #     counts[0, i] = (pred_tourn == i).sum()
            #     counts[1, i] = (pred_tourn_naive == i).sum()
            #     counts[2, i] = (pred_tourn_center == i).sum()
            #     counts[3, i] = (pred_tourn_bern == i).sum()
            #     counts[4, i] = (pred_tourn_seperate == i).sum()
            #     counts[5, i] = (pred_tourn_single == i).sum()
            # print(counts)

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

def joint_eval_all_var(device, test_loader, models, class_count, monitor: 'ConvergenceMonitor' = None, epoch: int = None, mode = 'Val'):
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
    center = CenterThresholding(class_count, alpha=0.35).to(device)
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
            out_tourn, out_tourn_mid = models['tournament'][0].inference(data)
            # keep middle outputs on the same device as models/thresholds
            out_tourn_naive = naive(out_tourn_mid)
            out_tourn_center = center(out_tourn_mid)
            out_tourn_bern = bern(out_tourn_mid)
            out_tourn_seperate = seperate(out_tourn_mid)
            out_tourn_single = single(out_tourn_mid)

            test_loss['base'] += F.cross_entropy(out_base, target_oh, reduction='sum').item()
            test_loss['mid'] += F.cross_entropy(out_mid, target_oh, reduction='sum').item()
            test_loss['tournament'] += F.cross_entropy(out_tourn, target_oh, reduction='sum').item()


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

class MixUpTransform:
    def __init__(self, num_classes, alpha=1.0):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, images, targets):
        """
        images: Tensor of shape (B, C, H, W)
        targets: Tensor of shape (B,) with class indices
        Returns:
            mixed_images: Tensor of shape (B, C, H, W)
            mixed_targets: Tensor of shape (B, num_classes) with soft labels
        """
        if self.alpha <= 0:
            return images, self._to_one_hot(targets)

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index]

        y_a = self._to_one_hot(targets)
        y_b = self._to_one_hot(targets[index])
        mixed_targets = lam * y_a + (1 - lam) * y_b

        return mixed_images, mixed_targets
    def _to_one_hot(self, targets):
        return torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()

def edge_loss(x, gt, y):
    indices = gt[y]
    mask = indices != 0
    preds_selected = torch.where(indices > 0, x, 1 - x)
    reduced_set = preds_selected[mask]
    targets_set = torch.ones_like(reduced_set)  # all should be "correct"

    probs = x[~mask]
    probs = probs.clamp(1e-3, 1 - 1e-3)
    entropy = -(probs * torch.log(probs + 1e-8) +
                (1 - probs) * torch.log(1 - probs + 1e-8))

    return lsce(reduced_set, targets_set), entropy.mean()

def entropy_regularization(x, gt, y):
    indices = gt[y]
    mask = indices == 0
    probs = x[mask]
    probs = probs.clamp(1e-3, 1 - 1e-3)
    # probs: [batch, num_classes-1], values in [0,1]
    entropy = -(probs * torch.log(probs + 1e-8) +
                (1 - probs) * torch.log(1 - probs + 1e-8))
    return entropy.mean()

