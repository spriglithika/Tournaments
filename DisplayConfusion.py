import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from typing import List
import os

def plot_confusion_matrix(conf_mat: np.ndarray, class_names: List[str] = None, title: str = 'Confusion Matrix', savepath: str = None):
    """
    Plots a confusion matrix.

    Args:
        conf_mat (np.ndarray): Confusion matrix to plot.
        class_names (List[str]): List of class names.
        title (str): Title of the plot.
        savepath (str): Path to save the plot. If None, the plot is shown instead.
    """
    if class_names is None:
        class_names = [str(i) for i in range(conf_mat.shape[0])]
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--path', type=str, required=True, help='Path to the confusion matrix file (numpy .npy format)')
    argparser.add_argument('--title', type=str, default='Confusion Matrix', help='Title of the confusion matrix plot')
    argparser.add_argument('--savepath', type=str, default=None, help='Path to save the plot. If not provided, the plot will be shown.')
    args = argparser.parse_args()
    conf_mat_path = os.path.join(args.path, 'confusion_matrix_test.npy')
    conf_mat = np.load(conf_mat_path)
    plot_confusion_matrix(conf_mat, title=args.title, savepath=args.savepath)
