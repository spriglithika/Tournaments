
from preamble import *
import os
import torch
from itertools import combinations

def zeroing_test():
    def other_thing(x):
        print("I ran")
        return x + 1
    a = torch.tensor(1)
    print( 1 + a * other_thing(2))
    a = torch.tensor(0)
    print( 1 + a * other_thing(2))
# zeroing_test()

def pseudo_A():
    A = torch.zeros((6, 4))
    l = list(combinations(range(4), 2))
    for e, (i, j) in enumerate(l):
        A[e, i] = 1
        A[e, j] = -1
    print( A, torch.linalg.pinv(A))
# pseudo_A()

def notify(title, text):
    os.system("""
              osascript -e 'display dialog "{}" with title "{}"'
              """.format(text, title))

# notify("Test", "Does this work?")
def check_class_labels():
    dataset = 'fmnist'
    ds = datasets.CIFAR10(root='./data', train=False, download=True)
    print( ds.classes)
    print( ds.class_to_idx)


# ---------------- model inspection helpers ----------------

def count_parameters(model: torch.nn.Module):
    """Return (total_params, trainable_params) as integers."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def model_summary(model: torch.nn.Module, top_level: bool = True):
    """Print a compact parameter summary for `model`.

    - total params
    - trainable params
    - top-level child modules with their param counts
    """
    total, trainable = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    if top_level:
        print("  Top-level modules:")
        for name, child in model.named_children():
            c_tot, c_train = count_parameters(child)
            print(f"    {name:25s}: total={c_tot:10,d}, trainable={c_train:10,d}")


# keep the sample helper call commented out so importing this module is quiet
# check_class_labels()
