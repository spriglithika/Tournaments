from preamble import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def plot_adj_matrix(J: torch.Tensor, title: str = 'adjacency', savepath: str = None):
    path = os.path.join(savepath, title + '.png') if savepath else None
    plt.figure(figsize=(6,6))
    sns.heatmap(J, center=0, cmap='vlag')
    plt.title(title)
    if savepath:
        plt.tight_layout()
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()

def plot_adj_matrix_log(J: torch.Tensor, title: str = 'log_adjacency', savepath: str = None):
    J_sign = np.sign(J)
    J_log = np.log1p(np.abs(J)) * J_sign
    path = os.path.join(savepath, title + '.png') if savepath else None
    plt.figure(figsize=(6,6))
    sns.heatmap(J_log, center=0, cmap='vlag')
    plt.title(title)
    if savepath:
        plt.tight_layout()
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()
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

class SaveModule:
    """Utility to save a module's artifacts, stats, and parameters to disk."""
    def __init__(self, module: nn.Module, out_dir: str):
        self.module = module
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save_J(self, J, mod =''):
        if J is None:
            return None
        name = f"J.npy"
        path = os.path.join(self.out_dir, name)
        np.save(path, J)
        plot_adj_matrix(J, savepath=self.out_dir)
        plot_adj_matrix_log(J, savepath=self.out_dir)
        plt.close('all')
        return name

    def save_loss_history(self, loss_history: List[float], idx: int = None):
        if loss_history is None:
            return None
        name = f"loss_history.npy"
        path = os.path.join(self.out_dir, name)
        try:
            np.save(path, np.array(loss_history))
            return name
        except Exception:
            return None

    def save_confusion_matrix(self, conf_mat: np.ndarray, mode: str = 'train', idx: int = None):
        if conf_mat is None:
            return None
        name = f"confusion_matrix_{mode}.npy"
        path = os.path.join(self.out_dir, name)
        try:
            np.save(path, conf_mat)
            return name
        except Exception:
            return None

    def save_calibration(self, confs, accs, ece, epoch: int = None):
        """Save calibration data (arrays + image) to the experiment output directory.

        Returns the filename (npz) written or None on failure.
        """
        try:
            confs_np = np.array(confs)
            accs_np = np.array(accs)
        except Exception:
            return None
        fname = f"calibration_epoch_{epoch}.npz" if epoch is not None else "calibration_data.npz"
        path = os.path.join(self.out_dir, fname)
        try:
            np.savez(path, confs=confs_np, accs=accs_np, ece=float(ece))
            # also save a plotted PNG (unique per-epoch)
            plot_fname = f"calibration_curve_epoch_{epoch}.png" if epoch is not None else 'calibration_curve.png'
            plot_calibration_curve(confs_np, accs_np, ece, savepath=self.out_dir, fname=plot_fname)
            return fname
        except Exception:
            return None

def notify(title, text):
    os.system("""
              osascript -e 'display dialog "{}" with title "{}"'
              """.format(text, title))

def plot_calibration_curve(confs, accs, ece, savepath=None, fname: str = None):
    confs = np.array(confs)
    accs = np.array(accs)
    mask = ~np.isnan(confs)

    plt.figure(figsize=(6,6))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
    plt.plot(confs[mask], accs[mask], 'o-', label=f'ECE={ece:.4f}')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.title('Calibration Curve')
    plt.legend()
    if savepath:
        if fname is None:
            fname = 'calibration_curve.png'
        path = os.path.join(savepath, fname)
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()

