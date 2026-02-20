from preamble import *
from tqdm import tqdm
from copy import deepcopy
import os
import threading
import torch
# Single AMP scaler for joint backward to avoid per-model scaler interactions


def smooth_labels(y, smoothing=0.1):
    return y * (1 - smoothing) + smoothing / y.size(-1)



class ConvergenceMonitor:
    """Simple single-model convergence monitor.

    Usage:
        monitor = ConvergenceMonitor(patience=3, mode='max', save_dir='./ckpts')
        improved = monitor.update(value, epoch=ep, model=model)
        if monitor.converged:
            break

    This class tracks a single best scalar (`best`) according to `mode`
    ('max' or 'min'), counts how many consecutive updates have *not*
    improved (`since_improve`) and sets `converged=True` when that reaches
    `patience`. If a `model` is provided on an improving update, its
    state_dict is saved (overwriting the single best file).
    """
    def __init__(self, patience: int = 5, mode: str = 'max', save_dir: str = './ckpts', filename: str = 'best.pth'):
        assert mode in ('max', 'min')
        self.patience = int(patience)
        self.mode = mode
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = filename

        # status
        self.best = None
        self.best_epoch = None
        self.since_improve = 0
        self.converged = False
        self.best_path = None

    def _better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == 'max':
            return value > self.best
        else:
            return value < self.best

    def _save_async(self, state, path: str):
        def _save():
            tmp = path + '.tmp'
            try:
                torch.save(state, tmp)
                os.replace(tmp, path)
            except Exception as e:
                print(f"Warning: Failed to save model to {path}: {e}")
        thread = threading.Thread(target=_save)
        thread.daemon = True
        thread.start()

    def update(self, value: float, epoch: int = None, model=None) -> bool:
        """Update monitor with a new scalar `value`.

        Returns True if this value improved the best metric.
        """
        improved = False
        if self._better(value):
            self.best = float(value)
            self.best_epoch = epoch
            self.since_improve = 0
            improved = True
            if model is not None:
                fn = os.path.join(self.save_dir, self.filename)
                try:
                    self._save_async(deepcopy(model.state_dict()), fn)
                    self.best_path = fn
                except Exception:
                    # ignore save failures
                    pass
        else:
            self.since_improve += 1

        if self.since_improve >= self.patience:
            self.converged = True

        return improved

    def reset(self):
        self.best = None
        self.best_epoch = None
        self.since_improve = 0
        self.converged = False
        self.best_path = None

def train(model: nn.Module, device: str, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int, forward_kwargs: dict = {}):
    model.train()
    conf_mat = torch.zeros((model.num_classes, model.num_classes), dtype=torch.int32)
    ema_loss = None
    loss_history = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for data, target in pbar:
        data = data.to(device, non_blocking=True)
        target_int = target.to(device, non_blocking=True)
        # target_oh = F.one_hot(target_int, num_classes=model.num_classes).float().to(device)
        optimizer.zero_grad()
        loss, prediction = model(data, target_int,  train=True, **forward_kwargs)
        loss.backward()
        optimizer.step()
        for t, p in zip(target.view(-1), prediction.argmax(dim=1).view(-1)):
            conf_mat[t.long(), p.long()] += 1
        ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
        loss_history.append(loss.item())
        pbar.set_postfix({'Loss': f'{ema_loss:.4f}', 'Accuracy': f'{100. * conf_mat.trace().item() / conf_mat.sum().item():.2f}%'})
    return ema_loss, 100. * conf_mat.trace().item() / conf_mat.sum().item(), conf_mat, loss_history

def eval(model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader, forward_kwargs: dict = {}):
    model.eval()
    conf_mat = torch.zeros((model.num_classes, model.num_classes), dtype=torch.int32)
    ema_loss = None
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            loss, prediction = model(data, target, train=False, **forward_kwargs)
            for t, p in zip(target.view(-1), prediction.argmax(dim=1).view(-1)):
                conf_mat[t.long(), p.long()] += 1
            ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
            pbar.set_postfix({'Loss': f'{ema_loss:.4f}', 'Accuracy': f'{100. * conf_mat.trace().item() / conf_mat.sum().item():.2f}%'})
    return ema_loss, 100. * conf_mat.trace().item() / conf_mat.sum().item(), conf_mat

def get_confidence_predictions(model: nn.Module, test_loader: torch.utils.data.DataLoader, use_tqdm: bool = True):
    """Return (confs, preds, labels). If `use_tqdm` is False the function
    iterates silently (useful when Lightning already renders a compact bar).
    """
    all_confs = []
    all_preds = []
    all_labels = []
    all_accs = []

    model.eval()
    with torch.no_grad():
        iterator = tqdm(test_loader, desc="Evaluating for ECE") if use_tqdm else test_loader
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            _, z = model.forward(x)          # your z output
            probs = torch.softmax(z, dim=1)
            conf, pred = probs.max(dim=1)
            all_confs.append(conf.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    all_confs = torch.cat(all_confs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_confs, all_preds, all_labels

def compute_ece(confs, preds, labels, n_bins=15):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    per_bin_acc = []
    per_bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1])

        if mask.sum() == 0:
            per_bin_acc.append(float('nan'))
            per_bin_conf.append(float('nan'))
            bin_counts.append(0)
            continue

        acc = (preds[mask] == labels[mask]).float().mean()
        avg_conf = confs[mask].mean()
        frac = mask.float().mean()

        ece += frac * torch.abs(acc - avg_conf)

        per_bin_acc.append(acc.item())
        per_bin_conf.append(avg_conf.item())
        bin_counts.append(mask.sum().item())

    return ece.item(), per_bin_acc, per_bin_conf, bin_counts

def compute_ece_quantile(confs, preds, labels, n_bins=15):
    """
    Compute ECE using quantile binning.
    confs, preds, labels: 1D torch tensors of same length
    """
    assert confs.ndim == 1
    assert preds.ndim == 1
    assert labels.ndim == 1

    N = len(confs)

    # Sort by confidence
    sorted_confs, idx = torch.sort(confs)
    sorted_preds = preds[idx]
    sorted_labels = labels[idx]

    # Compute quantile boundaries (bin edges)
    # We want n_bins roughly equal-sized bins in terms of number of samples.
    bin_sizes = N // n_bins
    remainder = N % n_bins

    per_bin_acc = []
    per_bin_conf = []
    bin_counts = []

    ece = 0.0
    start = 0

    for b in range(n_bins):
        # Distribute remainder (so first "remainder" bins get one extra)
        size = bin_sizes + (1 if b < remainder else 0)

        end = start + size
        if size == 0:
            # extremely rare corner case: more bins than samples
            per_bin_acc.append(float("nan"))
            per_bin_conf.append(float("nan"))
            bin_counts.append(0)
            continue

        conf_bin = sorted_confs[start:end]
        preds_bin = sorted_preds[start:end]
        labels_bin = sorted_labels[start:end]

        avg_conf = conf_bin.mean()
        acc = (preds_bin == labels_bin).float().mean()

        # fraction of total samples
        frac = size / N

        ece += frac * torch.abs(acc - avg_conf)

        per_bin_acc.append(acc.item())
        per_bin_conf.append(avg_conf.item())
        bin_counts.append(size)

        start = end

    return ece.item(), per_bin_acc, per_bin_conf, bin_counts

def eval_ece(model: nn.Module, test_loader: torch.utils.data.DataLoader, n_bins=15, use_tqdm: bool = True):
    confs, preds, labels = get_confidence_predictions(model, test_loader, use_tqdm=use_tqdm)
    # return compute_ece(confs, preds, labels, n_bins)
    return compute_ece_quantile(confs, preds, labels, n_bins)

def ddn_extended_base_train(model: nn.Module, class_matrix: torch.Tensor, device: str, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epoch: int, forward_kwargs: dict = {}):
    model.train()
    conf_mat = torch.zeros((model.num_classes, model.num_classes), dtype=torch.int32)
    ema_loss = None
    loss_history = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for data, target in pbar:
        data = data.to(device, non_blocking=True)
        target_int = target.to(device, non_blocking=True)
        # target_oh = F.one_hot(target_int, num_classes=model.num_classes).float().to(device)
        optimizer.zero_grad()
        _, raw_prediction = model(data, None,  train=True, **forward_kwargs)
        prediction = laplace_diffuse(raw_prediction, class_matrix)
        loss = F.cross_entropy(prediction, target_int)
        loss.backward()
        optimizer.step()
        for t, p in zip(target.view(-1), prediction.argmax(dim=1).view(-1)):
            conf_mat[t.long(), p.long()] += 1
        ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
        loss_history.append(loss.item())
        pbar.set_postfix({'Loss': f'{ema_loss:.4f}', 'Accuracy': f'{100. * conf_mat.trace().item() / conf_mat.sum().item():.2f}%'})
    return ema_loss, 100. * conf_mat.trace().item() / conf_mat.sum().item(), conf_mat, loss_history

def ddn_extended_base_eval(model: nn.Module, class_matrix: torch.Tensor, device: str, test_loader: torch.utils.data.DataLoader, forward_kwargs: dict = {}):
    model.eval()
    conf_mat = torch.zeros((model.num_classes, model.num_classes), dtype=torch.int32)
    ema_loss = None
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            _, raw_prediction = model(data, None,  train=False, **forward_kwargs)
            prediction = laplace_diffuse(raw_prediction, class_matrix)
            loss = F.cross_entropy(prediction, target)
            for t, p in zip(target.view(-1), prediction.argmax(dim=1).view(-1)):
                conf_mat[t.long(), p.long()] += 1
            ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
            pbar.set_postfix({'Loss': f'{ema_loss:.4f}', 'Accuracy': f'{100. * conf_mat.trace().item() / conf_mat.sum().item():.2f}%'})
    return ema_loss, 100. * conf_mat.trace().item() / conf_mat.sum().item(), conf_mat

def laplace_diffuse(pred, C, tau=0.2):
    return pred - tau * (pred @ C.T)

def ddn_extended_eval_ece(model: nn.Module, class_matrix: torch.Tensor, test_loader: torch.utils.data.DataLoader, n_bins=15, use_tqdm: bool = True):
    confs, preds, labels = ddn_extended_get_confidence_predictions(model, class_matrix, test_loader, use_tqdm=use_tqdm)
    # return compute_ece(confs, preds, labels, n_bins)
    return compute_ece_quantile(confs, preds, labels, n_bins)

def ddn_extended_get_confidence_predictions(model: nn.Module, class_matrix: torch.Tensor, test_loader: torch.utils.data.DataLoader, use_tqdm: bool = True):
    all_confs = []
    all_preds = []
    all_labels = []
    all_accs = []

    model.eval()
    with torch.no_grad():
        iterator = tqdm(test_loader, desc="Evaluating for ECE") if use_tqdm else test_loader
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            _, z = model.forward(x)          # your z output
            z = laplace_diffuse(z, class_matrix)
            probs = torch.softmax(z, dim=1)
            conf, pred = probs.max(dim=1)
            all_confs.append(conf.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    all_confs = torch.cat(all_confs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_confs, all_preds, all_labels