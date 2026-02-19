import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def plot_loss_history(loss_history: np.ndarray, title: str = 'Loss History', savepath: str = None):
    """
    Plots the loss history over epochs.
    Args:
        loss_history (np.ndarray): Array of loss values.
        title (str): Title of the plot.
        savepath (str): Path to save the plot. If None, the plot is shown instead.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--loss_history', type=str, required=True, help='Path to the loss history file (numpy .npy format)')
    argparser.add_argument('--title', type=str, default='Loss History', help='Title of the loss history plot')
    argparser.add_argument('--savepath', type=str, default=None, help='Path to save the plot. If not provided, the plot will be shown.')
    args = argparser.parse_args()
    loss_history = np.load(args.loss_history)
    plot_loss_history(loss_history, title=args.title, savepath=args.savepath)