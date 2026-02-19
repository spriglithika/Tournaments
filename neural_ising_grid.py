"""neural_ising_grid.py

A small, self-contained experiment harness to run extremely small MNIST trials
for the NeuralIsingTournamentFull model defined in `Models.py`.

Features:
- Build tiny datasets with N samples per class (choose which digits to include).
- Run mean-field inference with tracing of iteration-wise changes (convergence trace).
- Simple short training loop that can update both model.middle parameters and the
  adjacency matrix J when `train_adj=True` (it will add J to the optimizer).
- Simple decode from pairwise probs -> class prediction by majority votes.
- Helper routines to run parameter grids (alpha, gamma, max_iter) and plot results.

Usage (example):
    from experiments.neural_ising_grid import run_simple_demo
    run_simple_demo()

Or run from command line (python -m experiments.neural_ising_grid) which
runs a small demo and writes figures into ./experiments/outputs.

Keep this file small and easy to extend for presentations.
"""

import math
import os
from itertools import combinations
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
import csv
import json
from itertools import product

# Import local model implementation
from old_garbage.Models import NeuralIsingTournamentFull
from TournamentThresholds import SingleConfidence
from tqdm import tqdm

# -------------------------- Dataset helpers --------------------------

def make_mnist_subset(num_classes: int = 2,
                       samples_per_class: int = 10,
                       class_list: List[int] = None,
                       train: bool = True,
                       resize: int = 224,
                       device: torch.device = torch.device('cpu'),
                       dataset: str = 'mnist',
                       root: str = '.',
                       batch_size: int = None,
                       shuffle: bool = None) -> DataLoader:
    """Return a DataLoader containing up to `samples_per_class` examples for
    each class in `class_list` (or 0..num_classes-1 if not provided).

    Supports `mnist`, `cifar10`, and `cifar100` via the `dataset` argument.
    Images are converted / resized so they can be consumed by the ResNet/
    MobileNet backbones provided in `Models.py`.
    """
    if class_list is None:
        class_list = list(range(num_classes))

    dataset = dataset.lower()

    # Use ImageNet normalization (safe for pretrained backbones)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        ds = datasets.MNIST(root=os.path.join(root, 'data'), train=train, download=True, transform=transform)

    elif dataset in ('cifar10', 'cifar100'):
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        if dataset == 'cifar10':
            ds = datasets.CIFAR10(root=os.path.join(root, 'data'), train=train, download=True, transform=transform)
        else:
            ds = datasets.CIFAR100(root=os.path.join(root, 'data'), train=train, download=True, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'mnist', 'cifar10' or 'cifar100'.")

    # Robust extraction of targets (list, tensor, or attribute name differences)
    if hasattr(ds, 'targets'):
        targets = ds.targets
    elif hasattr(ds, 'labels'):
        targets = ds.labels
    else:
        # fallback: iterate dataset once (slower)
        targets = [lab for _, lab in ds]

    targets = torch.as_tensor(targets)

    selected_indices = []
    for c in class_list:
        idx = torch.where(targets == c)[0].tolist()
        selected_indices.extend(idx[:samples_per_class])

    subset = Subset(ds, selected_indices)

    # default shuffle: True for train, False for test
    if shuffle is None:
        shuffle = bool(train)

    # default batch_size: if not provided, use full subset for evaluation (no batching)
    if batch_size is None:
        batch_size = len(subset) if len(subset) > 0 else 1

    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return loader

# -------------------------- Model/inference helpers --------------------------
def edge_list_for(num_classes: int) -> List[Tuple[int, int]]:
    return list(combinations(range(num_classes), 2))

def scale_model_J(model: NeuralIsingTournamentFull, gamma: float):
    """Scale the model's adjacency J by `gamma` in-place. Works whether J is a
    Parameter or a registered buffer tensor.
    """
    if hasattr(model, 'J'):
        J = model.J
        if isinstance(J, nn.Parameter):
            with torch.no_grad():
                model.J.data = model.J.data * float(gamma)
        else:
            # buffer
            model.J = (J * float(gamma)).to(J.device)


def inference_trace(model: NeuralIsingTournamentFull,
                    x: torch.Tensor,
                    alpha: float = 1.0,
                    max_iter: int = 20,
                    tol: float = 1e-4) -> Dict[str, Any]:
    """Run the mean-field inference loop manually while recording a
    convergence trace. Returns a dict with trace, final m, final probs,
    and iterations used.

    model.middle(x) is used to compute h. We assume model.J is available.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    model = model.to(device)
    with torch.no_grad():
        # compute h (the pre-mean-field activations)
        h = model.middle(x)  # shape: (batch, num_edges)

    # We'll run the iterative updates with gradients turned off for tracing.
    J = getattr(model, 'J', None)
    if J is None:
        raise RuntimeError('Model does not have attribute J')
    J = J.to(device)

    m = torch.tanh(h)
    trace = []
    for it in range(max_iter):
        m_new = torch.tanh(h + alpha * torch.matmul(m, J))
        delta = (m_new - m).abs().mean().item()
        trace.append(delta)
        m = m_new
        if delta < tol:
            break

    probs = (m + 1.0) / 2.0
    return dict(trace=trace, final_m=m, final_probs=probs, iters=len(trace))

# -------------------------- Decoding & metrics --------------------------
def decode_votes_from_probs(probs: torch.Tensor,
                            num_classes: int,
                            edge_list: List[Tuple[int, int]]) -> torch.Tensor:
    """Given probs for edges (shape: [batch, num_edges]) where each entry is
    interpreted as probability that the first index wins over the second, build
    votes for each class and return predicted class indices (batch,).
    """
    batch = probs.shape[0]
    votes = torch.zeros((batch, num_classes), dtype=torch.int32, device=probs.device)
    for e, (i, j) in enumerate(edge_list):
        # if prob>0.5 => vote for i, else vote for j
        sel = probs[:, e] > 0.5
        votes[sel, i] += 1
        votes[~sel, j] += 1
    preds = votes.argmax(dim=1)
    return preds


def compute_accuracy_from_probs(probs: torch.Tensor, labels: torch.Tensor, num_classes: int):
    edge_list = edge_list_for(num_classes)
    preds = decode_votes_from_probs(probs, num_classes, edge_list)
    return (preds.cpu() == labels.cpu()).float().mean().item()


# -------------------------- Training routine --------------------------

def train_short(model: NeuralIsingTournamentFull,
                dataloader: DataLoader,
                num_epochs: int = 25,
                lr: float = 1e-3,
                train_adj: bool = True,
                alpha: float = 1.0,
                max_iter: int = 10,
                tol: float = 1e-4,
                device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
    """A very short, small-batch training loop that updates model.middle
    parameters and optionally the adjacency matrix J if `train_adj` is True.

    Loss is computed only on edges that involve the sample's true class.
    """
    model = model.to(device)
    model.train()

    # prepare optimizer: include model parameters and optionally J
    params = [p for p in model.middle.parameters() if p.requires_grad]
    if train_adj and hasattr(model, 'J') and isinstance(model.J, nn.Parameter):
        params = params + [model.J]

    if len(params) == 0:
        raise RuntimeError('No parameters to train. Are all model parameters frozen?')

    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCELoss(reduction='mean')

    num_classes = model.num_classes
    edge_list = edge_list_for(num_classes)
    # build edge masks for each class: mask[c] is boolean mask of edges involving c
    edge_masks = []
    edge_targets = []  # for a given class c, target value for edges involving c (1 or 0)
    for c in range(num_classes):
        mask = torch.tensor([1 if (i == c or j == c) else 0 for (i, j) in edge_list], dtype=torch.bool)
        # target is 1 when the first index equals c, 0 when the second index equals c
        t = torch.tensor([1.0 if i == c else 0.0 if j == c else 0.5 for (i, j) in edge_list], dtype=torch.float32)
        edge_masks.append(mask.to(device))
        edge_targets.append(t.to(device))

    history = {'loss': [], 'train_iters': []}

    for epoch in range(num_epochs):
        epoch_batch_iters = []
        pbar = tqdm(dataloader)
        for batch in pbar:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            iters_used = 0
            epoch_batch_iters.append(iters_used)
            _, probs = model(images, alpha=alpha, max_iter=max_iter, tol=tol)
            # probs = (m + 1.0) / 2.0

            # Build loss: only on edges involving true class for each sample
            # We'll compute loss per-sample then mean over batch
            batch_losses = []
            for bidx in range(probs.shape[0]):
                lab = labels[bidx].item()
                mask = edge_masks[lab]
                if mask.sum() == 0:
                    # no edges involve this class? shouldn't happen
                    continue
                tgt = edge_targets[lab][mask]
                pred = probs[bidx, mask]
                batch_losses.append(bce(pred, tgt))
            if len(batch_losses) == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = torch.stack(batch_losses).mean()

            loss.backward()
            opt.step()

            history['loss'].append(loss.item())
            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
        # record average iterations used during this epoch (per-batch average)
        if len(epoch_batch_iters) > 0:
            history['train_iters'].append(float(sum(epoch_batch_iters)) / len(epoch_batch_iters))
        else:
            history['train_iters'].append(0.0)
    return history


# -------------------------- Experiment runner --------------------------

def run_single_experiment(num_classes: int = 2,
                          samples_per_class: int = 8,
                          alpha: float = 1.0,
                          gamma: float = 1.0,
                          max_iter: int = 20,
                          tol: float = 1e-4,
                          train_adj: bool = False,
                          train_epochs: int = 3,
                          resize: int = 64,
                          device: torch.device = torch.device('cpu'),
                          dataset: str = 'mnist',
                          root: str = '.') -> Dict[str, Any]:
    """Run a tiny experiment: build model, optionally train shortly, then
    run inference traces on the dataset and return metrics + plots.
    """
    # build separate small train and test datasets
    train_loader = make_mnist_subset(num_classes=num_classes,
                                     samples_per_class=samples_per_class*6,
                                     train=True,
                                     resize=resize,
                                     device=device,
                                     dataset=dataset,
                                     root=root)
    test_loader = make_mnist_subset(num_classes=num_classes,
                                    samples_per_class=samples_per_class,
                                    train=False,
                                    resize=resize,
                                    device=device,
                                    dataset=dataset,
                                    root=root)

    # Evaluate on the entire test set (iterate all batches) to avoid
    # reporting results from a single random minibatch which can be misleading.
    # Collect all test images/probs for pre/post evaluation.
    all_test_images = []
    all_test_labels = []
    for tb in test_loader:
        imgs, labs = tb
        all_test_images.append(imgs)
        all_test_labels.append(labs)
    if len(all_test_images) == 0:
        raise RuntimeError('Test set is empty')
    images = torch.cat(all_test_images, dim=0)
    labels = torch.cat(all_test_labels, dim=0)
    # keep the train loader for training when requested

    # instantiate model
    model = NeuralIsingTournamentFull(num_classes, backbone='resnet18', device=device, learn_J=train_adj, freeze_backbone=False, unfreeze_last_n=2)
    # scale adjacency by gamma
    # scale_model_J(model, gamma)

    # quick pre-train accuracy
    model = model.to(device)
    model.eval()
    # compute pre-train accuracy over full test set in batches to avoid OOM
    model = model.to(device)
    model.eval()
    sc = SingleConfidence(num_classes).to(device)
    probs_list = []
    with torch.no_grad():
        # process in minibatches to avoid memory spikes
        bs = 256
        for i in range(0, images.size(0), bs):
            batch_imgs = images[i:i+bs].to(device)
            res = inference_trace(model, batch_imgs, alpha=alpha, max_iter=max_iter, tol=tol)
            probs_list.append(res['final_probs'].cpu())
    probs_all = torch.cat(probs_list, dim=0)
    acc_pre = compute_accuracy_from_probs(probs_all, labels, num_classes)
    counts_pre = sc(probs_all.to(device))
    preds_sc_pre = counts_pre.argmax(dim=1)
    acc_pre_sc = (preds_sc_pre.cpu() == labels.cpu()).float().mean().item()

    train_hist = None
    if train_epochs > 0:
        train_hist = train_short(model, train_loader, num_epochs=train_epochs, lr=1e-3, train_adj=train_adj, alpha=alpha, max_iter=max_iter, tol=tol, device=device)

    # post-train inference
    # post-train inference over whole test set
    model.eval()
    probs_list = []
    with torch.no_grad():
        bs = 256
        for i in range(0, images.size(0), bs):
            batch_imgs = images[i:i+bs].to(device)
            res = inference_trace(model, batch_imgs, alpha=alpha, max_iter=max_iter, tol=tol)
            probs_list.append(res['final_probs'].cpu())
    probs_all_post = torch.cat(probs_list, dim=0)
    acc_post = compute_accuracy_from_probs(probs_all_post, labels, num_classes)
    counts_post = sc(probs_all_post.to(device))
    preds_sc_post = counts_post.argmax(dim=1)
    acc_post_sc = (preds_sc_post.cpu() == labels.cpu()).float().mean().item()

    out = {
        'num_classes': num_classes,
        'samples_per_class': samples_per_class,
        'alpha': alpha,
        'gamma': gamma,
        'max_iter': max_iter,
        'train_adj': train_adj,
        'train_epochs': train_epochs,
        'acc_pre': acc_pre,
        'acc_pre_sc': acc_pre_sc,
        'acc_post': acc_post,
        'acc_post_sc': acc_post_sc,
        'trace_pre': res_pre['trace'],
        'trace_post': res_post['trace'],
        'J_final': (model.J.detach().cpu() if hasattr(model, 'J') else None),
        'train_history': train_hist,
    }
    return out


# -------------------------- Plotting helpers --------------------------

def plot_trace(trace: List[float], title: str = 'convergence trace', savepath: str = None):
    plt.figure()
    plt.plot(trace, marker='o')
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('mean |delta|')
    plt.title(title)
    plt.grid(True)
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)
    else:
        plt.show()


def plot_adj_matrix(J: torch.Tensor, title: str = 'adjacency', savepath: str = None):
    plt.figure(figsize=(6,6))
    sns.heatmap(J.numpy(), center=0, cmap='vlag')
    plt.title(title)
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)
    else:
        plt.show()

def plot_adj_matrix_log(J: torch.Tensor, title: str = 'adjacency', savepath: str = None):
    J_sign = torch.sign(J)
    J_log = torch.log1p(torch.abs(J)) * J_sign
    plt.figure(figsize=(6,6))
    sns.heatmap(J_log.numpy(), center=0, cmap='vlag')
    plt.title(title)
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)
    else:
        plt.show()


def save_J_heatmap(J: torch.Tensor, out_dir: str, alpha: float, gamma: float, max_iter: int, idx: int = None):
    """Save a PNG heatmap of J with a detailed filename including hyperparameters.

    Returns the relative filename written.
    """
    if J is None:
        return None
    # make safe filename portion, replace dots to keep names shell-friendly
    a_s = f"{alpha:.3f}".replace('.', 'p')
    g_s = f"{gamma:.3f}".replace('.', 'p')
    idx_s = f"{idx}" if idx is not None else '0'
    name = f"J_a{a_s}_g{g_s}_m{max_iter}_i{idx_s}.png"
    path = os.path.join(out_dir, name)
    try:
        plot_adj_matrix(J, title=f'J (a={alpha}, g={gamma}, m={max_iter})', savepath=path)
        plot_adj_matrix_log(J, title=f'log_J (a={alpha}, g={gamma}, m={max_iter})', savepath=path)
        return name
    except Exception:
        return None


# -------------------------- Demo / CLI entry --------------------------

def run_simple_demo(num_classes: int,
                    samples_per_class: int,
                    alpha: float = 1.0,
                    gamma: float = 1.0,
                    max_iter: int = 30,
                    train_adj: bool = False,
                    train_epochs: int = 3,
                    resize: int = 64,
                    out_dir: str = './experiments/outputs',
                    device: torch.device = torch.device('cpu'),
                    dataset: str = 'mnist',
                    root: str = '.'):
    """Run a single experiment and save plots. This function accepts explicit
    parameters so it can be driven from the CLI with required args for
    `num_classes` and `samples_per_class`.
    """
    os.makedirs(out_dir, exist_ok=True)
    print('Using device:', device)

    print('Running experiment with:', dict(num_classes=num_classes, samples_per_class=samples_per_class,
                                          alpha=alpha, gamma=gamma, max_iter=max_iter, train_adj=train_adj,
                                          train_epochs=train_epochs, resize=resize))
    res = run_single_experiment(num_classes=num_classes,
                                samples_per_class=samples_per_class,
                                alpha=alpha,
                                gamma=gamma,
                                max_iter=max_iter,
                                train_adj=train_adj,
                                train_epochs=train_epochs,
                                resize=resize,
                                device=device,
                                dataset=dataset,
                                root=root)

    print(f"Pre-train acc: {res['acc_pre']:.3f}, Post-train acc: {res['acc_post']:.3f}")
    plot_trace(res['trace_pre'], title='Pre-train convergence trace', savepath=os.path.join(out_dir, 'trace_pre.png'))
    # if trace_post is empty or very short, annotate accordingly inside plot_trace
    if len(res['trace_post']) == 0:
        print('Post-train trace empty (converged immediately)')
    plot_trace(res['trace_post'], title='Post-train convergence trace', savepath=os.path.join(out_dir, 'trace_post.png'))

    if res['J_final'] is not None:
        # If J is scalar (n=2) annotate the numeric value for clarity
        J = res['J_final']
        if J.numel() == 1:
            print('Final adjacency J (scalar):', float(J.item()))
        plot_adj_matrix(J, title='Final adjacency J', savepath=os.path.join(out_dir, 'J_final.png'))
        # plot_adj_matrix_log(J, title='Final adjacency J (log scale)', savepath=os.path.join(out_dir, 'J_final_log.png'))

    # save summary JSON
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"pre_acc: {res['acc_pre']}\npre_acc_sc: {res.get('acc_pre_sc', 'n/a')}\n")
        f.write(f"post_acc: {res['acc_post']}\npost_acc_sc: {res.get('post_acc_sc', 'n/a')}\n")
        if res['J_final'] is not None:
            f.write(f"J_min: {float(res['J_final'].min())}, J_max: {float(res['J_final'].max())}\n")

    print('Outputs written to', out_dir)


def run_grid_search(num_classes: int,
                    samples_per_class: int,
                    alphas: List[float],
                    gammas: List[float],
                    max_iters: List[int],
                    train_adj: bool = False,
                    train_epochs: int = 0,
                    resize: int = 64,
                    out_dir: str = './experiments/outputs/grid',
                    device: torch.device = torch.device('cpu'),
                    dataset: str = 'mnist',
                    root: str = '.'):
    """Run a grid sweep over (alpha, gamma, max_iter). Writes a CSV with
    results and, when possible, simple heatmaps.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    total = len(alphas) * len(gammas) * len(max_iters)
    i = 0
    for a, g, m in product(alphas, gammas, max_iters):
        i += 1
        print(f'Grid {i}/{total}: alpha={a}, gamma={g}, max_iter={m}')
        res = run_single_experiment(num_classes=num_classes,
                        samples_per_class=samples_per_class,
                        alpha=a,
                        gamma=g,
                        max_iter=m,
                        train_adj=train_adj,
                        train_epochs=train_epochs,
                        resize=resize,
                        device=device,
                        dataset=dataset,
                        root=root)
        # Save heatmap of final adjacency for this grid point (if available)
        J_img_name = None
        if res.get('J_final') is not None:
            J_img_name = save_J_heatmap(res['J_final'], out_dir=out_dir, alpha=a, gamma=g, max_iter=m, idx=i)

        row = dict(alpha=a, gamma=g, max_iter=m,
                   acc_pre=res['acc_pre'], acc_pre_sc=res.get('acc_pre_sc', None),
                   acc_post=res['acc_post'], acc_post_sc=res.get('acc_post_sc', None),
                   iters_pre=len(res['trace_pre']), iters_post=len(res['trace_post']),
                   J_image=J_img_name)
        if res['J_final'] is not None:
            row.update(J_min=float(res['J_final'].min()), J_max=float(res['J_final'].max()))
        rows.append(row)

    csv_path = os.path.join(out_dir, 'grid_results.csv')

    # Prepare ordered fieldnames (prefer a sensible order)
    preferred = ['alpha', 'gamma', 'max_iter', 'acc_pre', 'acc_pre_sc', 'acc_post', 'acc_post_sc',
                 'iters_pre', 'iters_post', 'J_image', 'J_min', 'J_max']
    # collect all keys present
    all_keys = list({k for r in rows for k in r.keys()})
    # start with preferred order then append any extras
    fieldnames = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]

    def fmt_val(v):
        if v is None:
            return ''
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    # write CSV with formatted (truncated) numeric values
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out_row = {k: fmt_val(r.get(k)) for k in fieldnames}
            writer.writerow(out_row)

    print('Grid results written to', csv_path)

    # Also write an aligned, human-readable table for quick inspection
    pretty_path = os.path.join(out_dir, 'grid_results_aligned.txt')
    # compute column widths
    str_rows = []
    for r in rows:
        str_rows.append([fmt_val(r.get(k)) for k in fieldnames])
    col_widths = [max(len(str(field)), max((len(row[i]) for row in str_rows), default=0)) for i, field in enumerate(fieldnames)]
    with open(pretty_path, 'w') as f:
        # header
        hdr = '  '.join(field.ljust(col_widths[i]) for i, field in enumerate(fieldnames))
        f.write(hdr + '\n')
        f.write('-' * len(hdr) + '\n')
        for row in str_rows:
            line = '  '.join(row[i].ljust(col_widths[i]) for i in range(len(fieldnames)))
            f.write(line + '\n')

    print('Aligned table written to', pretty_path)

    # If max_iters is singleton, plot heatmaps alpha x gamma for acc_post
    if len(max_iters) == 1:
        M = max_iters[0]
        import numpy as np
        vals = {(r['alpha'], r['gamma']): r['acc_post'] for r in rows}
        al = sorted(set([r['alpha'] for r in rows]))
        ga = sorted(set([r['gamma'] for r in rows]))
        mat = np.zeros((len(al), len(ga)))
        for ii, a in enumerate(al):
            for jj, g in enumerate(ga):
                mat[ii, jj] = vals.get((a, g), float('nan'))
        plt.figure(figsize=(6,5))
        sns.heatmap(mat, xticklabels=ga, yticklabels=al, annot=True, fmt='.3f', cmap='viridis')
        plt.xlabel('gamma')
        plt.ylabel('alpha')
        plt.title(f'Post-train accuracy (max_iter={M})')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'heatmap_acc_post_maxiter_{M}.png'))
        plt.close()

    return csv_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run small Neural Ising tournament experiments')
    parser.add_argument('--num_classes', type=int, help='Number of classes to include (required)')
    parser.add_argument('--samples_per_class', type=int, help='Number of samples per class (required)')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--train_adj', action='store_true', help='Allow adjacency J to be trainable')
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--resize', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='./experiments/outputs')
    parser.add_argument('--dataset', type=str, default='mnist', help="Dataset to use: 'mnist', 'cifar10', or 'cifar100'")
    parser.add_argument('--grid_alphas', type=str, default=None, help='Comma-separated alphas for grid search')
    parser.add_argument('--grid_gammas', type=str, default=None, help='Comma-separated gammas for grid search')
    parser.add_argument('--grid_max_iters', type=str, default=None, help='Comma-separated max_iter values for grid search')
    parser.add_argument('--grid_out', type=str, default='./experiments/outputs/grid')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file to run experiment or grid')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # If a config file is provided, delegate to the declarative config runner.
    if args.config is not None:
        from old_garbage.config_runner import load_and_run
        print('Running declarative config', args.config)
        load_and_run(args.config)
        raise SystemExit(0)
    else:
        run_simple_demo(num_classes=args.num_classes,
                        samples_per_class=args.samples_per_class,
                        alpha=args.alpha,
                        gamma=args.gamma,
                        max_iter=args.max_iter,
                        train_adj=args.train_adj,
                        train_epochs=args.train_epochs,
                        resize=args.resize,
                        out_dir=args.out_dir,
                        device=device,
                        dataset=args.dataset,
                        root='.')

        # optional grid search via CLI flags
        if args.grid_alphas is not None and args.grid_gammas is not None and args.grid_max_iters is not None:
            alphas = [float(x) for x in args.grid_alphas.split(',')]
            gammas = [float(x) for x in args.grid_gammas.split(',')]
            max_iters = [int(x) for x in args.grid_max_iters.split(',')]
            print('Running grid search...')
            run_grid_search(num_classes=args.num_classes,
                            samples_per_class=args.samples_per_class,
                            alphas=alphas,
                            gammas=gammas,
                            max_iters=max_iters,
                            train_adj=args.train_adj,
                            train_epochs=args.train_epochs,
                            resize=args.resize,
                            out_dir=args.grid_out,
                            device=device,
                            dataset=args.dataset,
                            root='.')
