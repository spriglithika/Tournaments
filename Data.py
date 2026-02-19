from preamble import *
from typing import List
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict


CLASS_LABELS = {
    'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'fmnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']}

def get_data_loader(num_classes: int = 2,
                       samples_per_class: int = 10,
                       class_list: List[int] = None,
                       train: bool = True,
                       resize: int = 28,
                       batch_size: int = 64,
                       dataset: str = 'mnist',
                       root: str = '.',
                       label_noise: float = 0.0,
                       imbalance=None,
                       imbalance_factor: float = 1.0) -> DataLoader:
    """Return a DataLoader containing up to `samples_per_class` examples for
    each class in `class_list` (or 0..num_classes-1 if not provided).

    Supports `mnist`, `cifar10`, `cifar100`, and `fmnist` via the `dataset` argument.
    Images are converted / resized so they can be consumed by the ResNet/
    MobileNet backbones provided in `Models.py`.
    """
    if class_list is None:
        class_list = list(range(num_classes))

    dataset = dataset.lower()

    # need mnist and cifar datasets own mean and std since we are not using pretrained model (NO IMAGENET)
    # dictionary with dataset as keys and mean and std as values
    dataset_stats = {
        'mnist': ([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]), # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        'fmnist': ([0.2860, 0.2860, 0.2860], [0.3530, 0.3530, 0.3530]), # {'T-shirt/top':0, 'Trouser':1, 'Pullover':2, 'Dress':3, 'Coat':4, 'Sandal':5, 'Shirt':6, 'Sneaker':7, 'Bag':8, 'Ankle boot':9}
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),} # {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=dataset_stats[dataset][0], std=dataset_stats[dataset][1]),
        ])
        ds = datasets.MNIST(root=os.path.join(root, 'data'), train=train, download=True, transform=transform)

    elif dataset in ('cifar10', 'cifar100'):
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_stats[dataset][0], std=dataset_stats[dataset][1]),
        ])
        if dataset == 'cifar10':
            ds = datasets.CIFAR10(root=os.path.join(root, 'data'), train=train, download=True, transform=transform)
        else:
            ds = datasets.CIFAR100(root=os.path.join(root, 'data'), train=train, download=True, transform=transform)
    elif dataset == 'fmnist':
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=dataset_stats[dataset][0], std=dataset_stats[dataset][1]),
        ])
        ds = datasets.FashionMNIST(root=os.path.join(root, 'data'), train=train, download=True, transform=transform)
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

    if label_noise > 0.0:
        # Apply label noise
        noisy_labels = add_label_noise(targets[selected_indices], num_classes, label_noise)
        subset.dataset.targets = list(subset.dataset.targets)  # ensure it's a list
        for i, idx in enumerate(selected_indices):
            subset.dataset.targets[idx] = noisy_labels[i].item()
    if imbalance is not None:
        subset = make_imbalanced(subset, imbalance)
    if imbalance_factor != 1.0:
        subset = make_longtailed_subset(subset, num_classes, imbalance_factor)

    loader = DataLoader(subset, batch_size=batch_size if train else len(subset), shuffle=train)
    return loader



def make_imbalanced(dataset, class_fractions, seed=42):
    rng = np.random.default_rng(seed)
    indices_by_class = defaultdict(list)

    for i, (_, y) in enumerate(dataset):
        indices_by_class[y].append(i)

    keep_indices = []
    for c, idxs in indices_by_class.items():
        frac = class_fractions.get(c, 1.0)
        k = int(len(idxs) * frac)
        keep_indices.extend(rng.choice(idxs, k, replace=False))

    return torch.utils.data.Subset(dataset, keep_indices)

def make_longtailed_subset(subset, num_classes, imbalance_ratio=100, seed=42):
    """
    Convert an existing Subset into a long-tailed version with a given imbalance ratio.
    - imbalance_ratio = 1   → balanced
    - imbalance_ratio = 100 → 100:1 long-tail
    Preserves dataset indexing and returns another Subset.

    Randomly permutes which class is majority/minority (reviewer-safe).
    """
    rng = np.random.default_rng(seed)

    # Step 1: Extract labels for the *subset* indices
    # We need to unwrap nested subsets correctly:
    indices = subset.indices
    base = subset.dataset
    while isinstance(base, Subset):
        base_indices = base.indices
        indices = [base_indices[i] for i in indices]
        base = base.dataset

    # Base dataset -> get labels
    if hasattr(base, 'targets'):
        all_labels = base.targets
    elif hasattr(base, 'labels'):
        all_labels = base.labels
    else:
        all_labels = [lab for _, lab in base]

    # Step 2: Group subset indices by class
    class_to_indices = {c: [] for c in range(num_classes)}
    for sub_i, base_i in enumerate(indices):
        c = int(all_labels[base_i])
        class_to_indices[c].append(sub_i)

    # Step 3: Random permutation of class order (reviewer-safe)
    perm = rng.permutation(num_classes)

    # Step 4: Compute exponential decay fractions
    ranks = np.arange(num_classes)  # 0 = largest, K-1 = smallest
    decay = imbalance_ratio ** (-ranks / (num_classes - 1))  # shape (K,)

    # Step 5: Assign counts per class using permuted ranking
    keep = []
    for orig_c, rank_c in enumerate(perm):
        frac = decay[rank_c]
        idxs = class_to_indices[orig_c]
        k = max(1, int(len(idxs) * frac))
        keep.extend(rng.choice(idxs, k, replace=False))

    # Map back to base indices
    final_indices = [subset.indices[i] for i in keep]
    return Subset(subset.dataset, final_indices)



def add_label_noise(labels, num_classes, noise_rate):
    """Add label noise.

    Behaviours:
    - If `labels` is a Tensor or list of labels: return a Tensor with noisy labels (same as before).
    - If `labels` is a `torch.utils.data.DataLoader`, `Dataset`, or `Subset`, apply noise
      to the dataset in-place (writes into `targets`/`labels` on the base dataset) and
      return the same object.

    The noise is applied by randomly selecting `noise_rate * N` entries and replacing
    each selected label with a uniformly sampled different class.
    """

    # Helper to noise a 1-D tensor of integer labels
    def _noise_tensor(lbls: torch.Tensor) -> torch.Tensor:
        noisy = lbls.clone()
        N = len(lbls)
        n_noisy = int(noise_rate * N)
        if n_noisy == 0:
            return noisy
        idx = torch.randperm(N)[:n_noisy]
        for i in idx:
            true = int(lbls[i].item())
            choices = list(range(num_classes))
            if true in choices:
                choices.remove(true)
            noisy_val = random.choice(choices)
            noisy[i] = noisy_val
        return noisy

    # If given a DataLoader, operate on its `.dataset`
    if isinstance(labels, DataLoader):
        ds = labels.dataset
        add_label_noise(ds, num_classes, noise_rate)
        return labels

    # If given a Subset or Dataset, map indices to the base dataset and write back
    if isinstance(labels, Subset):
        indices = list(labels.indices)
        base = labels.dataset
        # unwrap nested Subset
        while isinstance(base, Subset):
            inner_indices = list(base.indices)
            indices = [inner_indices[i] for i in indices]
            base = base.dataset

        base_ds = base
        # extract original labels
        if hasattr(base_ds, 'targets'):
            orig = list(base_ds.targets)
        elif hasattr(base_ds, 'labels'):
            orig = list(base_ds.labels)
        else:
            orig = [lab for _, lab in base_ds]

        sel = torch.as_tensor([orig[i] for i in indices])
        noisy = _noise_tensor(sel)

        if hasattr(base_ds, 'targets'):
            base_ds.targets = list(base_ds.targets)
            for i, idx in enumerate(indices):
                base_ds.targets[idx] = int(noisy[i].item())
        elif hasattr(base_ds, 'labels'):
            base_ds.labels = list(base_ds.labels)
            for i, idx in enumerate(indices):
                base_ds.labels[idx] = int(noisy[i].item())
        else:
            raise RuntimeError('Unable to write noisy labels back to dataset; no targets/labels attribute found.')
        return labels

    # If given a plain Dataset (not a Subset)
    if hasattr(labels, '__len__') and not torch.is_tensor(labels):
        base_ds = labels
        if hasattr(base_ds, 'targets'):
            orig = list(base_ds.targets)
            sel = torch.as_tensor(orig)
            noisy = _noise_tensor(sel)
            base_ds.targets = list(noisy.tolist())
            return base_ds
        elif hasattr(base_ds, 'labels'):
            orig = list(base_ds.labels)
            sel = torch.as_tensor(orig)
            noisy = _noise_tensor(sel)
            base_ds.labels = list(noisy.tolist())
            return base_ds
        else:
            # fall back to iterating and replacing via attribute access is unsupported
            raise RuntimeError('Unable to write noisy labels back to dataset; no targets/labels attribute found.')

    # Otherwise assume labels is a tensor/list of labels and return noisy tensor
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    return _noise_tensor(labels)
