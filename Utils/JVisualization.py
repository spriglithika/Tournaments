from preamble import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from Data import CLASS_LABELS
from Utils.AdjacencyInit import J_normed_diag_gamma, J_random_normed

def read_J(experiment: str) -> torch.Tensor:
    path = os.path.join('experiments', 'outputs', experiment, 'J.npy')
    J = np.load(path)
    return torch.tensor(J)

def get_A(num_classes):
    edge_list = list(combinations(range(num_classes), 2))
    num_edges = len(edge_list)
    A = torch.zeros((num_edges, num_classes))
    for e, (i, j) in enumerate(edge_list):
        A[e, i] = 1
        A[e, j] = -1
    return A

def compute_class_matrix(J: torch.Tensor, labels: List[int], num_classes: int, eps: float = 1e-8) -> torch.Tensor:
    """Compute the class-class interaction matrix from J and labels."""
    class_matrix = torch.zeros((num_classes, num_classes), device=J.device)
    counts = torch.zeros((num_classes, num_classes), device=J.device)

    # for i in range(len(labels)):
    #     for j in range(len(labels)):
    #         c1 = labels[i]
    #         c2 = labels[j]
    #         class_matrix[c1, c2] += J[i, j]
    #         counts[c1, c2] += 1
    A = get_A(num_classes)

    # C = class_matrix
    # C_sym = 0.5 * (C + C.T)
    # C_norm = C_sym / (counts + counts.T + eps)
    direction_matrix = A.T @ J @ A
    # print( - class_matrix)
    # Avoid division by zero
    # counts[counts == 0] = 1
    # class_matrix /= counts
    return direction_matrix

def compute_class_matrix_old(J: torch.Tensor, labels: List[int], num_classes: int, eps: float = 1e-8) -> torch.Tensor:
    """Compute the class-class interaction matrix from J and labels."""
    class_matrix = torch.zeros((num_classes, num_classes), device=J.device)
    counts = torch.zeros((num_classes, num_classes), device=J.device)

    for i in range(len(labels)):
        for j in range(len(labels)):
            c1 = labels[i]
            c2 = labels[j]
            class_matrix[c1, c2] += J[i, j]
            counts[c1, c2] += 1
    A = get_A(num_classes, J.shape[0])
    # C = class_matrix
    # C_sym = 0.5 * (C + C.T)
    # C_norm = C_sym / (counts + counts.T + eps)
    direction_matrix = class_matrix #- class_matrix.T
    # print( - class_matrix)
    # Avoid division by zero
    # counts[counts == 0] = 1
    # class_matrix /= counts
    return direction_matrix
    # return A.T @ J @ A

def plot_class_matrix(class_matrix: torch.Tensor, title: str = 'class_interaction', savepath: str = None, dataset: str = 'mnist'):
    path = os.path.join(savepath, title + '.png') if savepath else None
    plt.figure(figsize=(6,6))
    sns.heatmap(class_matrix.cpu(), center=0, cmap='vlag', annot=True, fmt=".2f", xticklabels=CLASS_LABELS.get(dataset, None), yticklabels=CLASS_LABELS.get(dataset, None))
    plt.title(title)
    # plt.xlabel('Class')
    # plt.ylabel('Class')
    if savepath:
        plt.tight_layout()
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()

def save_class_matrix(class_matrix: torch.Tensor, experiment: str, out_dir: str = os.path.join('experiments', 'outputs')):
    """Save a PNG heatmap of the class interaction matrix."""
    name = f"class_interaction_{experiment}.png"
    path = os.path.join(out_dir, name)
    try:
        plot_class_matrix(class_matrix, title=f'Class Interaction Matrix ({experiment})', savepath=path)
        np.save(os.path.join(out_dir, f"class_interaction_{experiment}.npy"), class_matrix.cpu().numpy())
        return name
    except Exception:
        return None
def make_default():
    J = J_normed_diag_gamma(num_classes=10, num_edges=45, gamma=0.5, no_diag=True)
    np.save(os.path.join('experiments', 'outputs', 'default', 'J.npy'), J)

if __name__ == "__main__":
    # experiments = ['EnergyNormJMi3', 'EnergyNormJSPBefore', 'EnergySoftmaxAfterL0p0001NormBoth']
    # out_dir = os.path.join('experiments', 'outputs')
    # os.makedirs(out_dir, exist_ok=True)

    # for exp in experiments:
    #     J = read_J(exp)
    #     # Assuming labels are stored or can be generated; here we create dummy labels for illustration
    #     num_classes = 10
    #     num_samples = J.shape[0]
    #     labels = [i % num_classes for i in range(num_samples)]  # Dummy labels
    #     class_matrix = compute_class_matrix(J, labels, num_classes)
    #     save_class_matrix(class_matrix, exp, out_dir)
    cfg_paths = [# ('FMNISTEnergyRaw', 'fmnist'),
                #  ('EnergyRaw', 'mnist'),
                #  ('CIFAREnergyRaw', 'cifar10'),
                #  ('FMNISTEnergySpectral', 'fmnist'),
                #  ('FMNISTEnergySplit', 'fmnist'),
                 ('FMNISTMF', 'fmnist'),
                 ('CIFARMF', 'cifar10'),
                #  ('FMNISTSplit', 'fmnist'),
                #  ('FMNISTSplitImbalanced', 'fmnist'),
                #  ('FMNISTDecL2', 'fmnist'),
                #  ('CIFARDec', 'cifar10'),
                 ('CIFARDecF', 'cifar10'),
                 ('CIFARDecFLongTail', 'cifar10'),
                   ]
    # cfg_paths = [('FMNISTEnergyRaw', 'fmnist'), ('FMNISTBaseModel', 'fmnist'),  ('FMNISTMidModel', 'fmnist'),
                #  ('EnergyRaw', 'mnist'), ('BaseModel', 'mnist'),  ('MidModel', 'mnist'),
                #  ('CIFAREnergyRaw', 'cifar10'), ('CIFARBaseModel', 'cifar10'), ('CIFARMidModel', 'cifar10')]
    # make_default()
    for path, dataset in cfg_paths:
        cfg_path = path
        J = read_J(cfg_path)
        num_classes = 10
        class_matrix = compute_class_matrix(J, [i % num_classes for i in range(J.shape[0])], num_classes)
        # plot_class_matrix(class_matrix, title='test_plot', savepath=None, dataset='fminst')
        plot_class_matrix(class_matrix, title='class_matrix', savepath=os.path.join('experiments', 'outputs', cfg_path), dataset=dataset)
        np.save(os.path.join('experiments', 'outputs', cfg_path, 'class_matrix.npy'), class_matrix.cpu().numpy())