import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np

def fix_random_seeds(seed=69):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False