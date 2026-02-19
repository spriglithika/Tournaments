
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
check_class_labels()
