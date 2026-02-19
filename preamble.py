import torch
import torch.nn as nn, autograd
import torch.nn.functional as F
import random
import os
import numpy as np
import math
import contextlib
import torch.distributed as dist
from torchvision import datasets, transforms, models
from itertools import combinations

# set mps cpu fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def launched_by_torchrun():
    return "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
linalg_device = 'cpu' if device.type == 'mps' else device
if launched_by_torchrun():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    device_type = device.type
if float(torch.__version__.split(".")[0]+"."+torch.__version__.split(".")[1]) >= 2.4:
    amp = torch.amp
    caster = torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=device.type)
    amp_ctx = torch.amp.autocast(enabled=False, device_type=device.type) if torch.cuda.is_available() else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
else:
    amp = torch.cuda.amp if torch.cuda.is_available() else torch.cpu.amp
    caster = torch.autocast(enabled=torch.cuda.is_available(), device_type=device.type)
    amp_ctx = torch.autocast(enabled=False, device_type=device.type) if torch.cuda.is_available() else contextlib.nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
# torch.set_default_dtype(torch.float32)

def fix_random_seeds(seed=69):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------ Weights & Biases helpers ------------------
# These helpers are intentionally lightweight and optional so importing
# `preamble` won't fail when `wandb` isn't installed. Call `init_wandb()`
# from your entrypoint (e.g. `TrainandTest.py`) when you want to enable W&B.
_wandb = None
_wandb_available = False


def init_wandb(key: str = None, silent: bool = True):
    """Attempt to import and (optionally) log in to wandb.

    Behaviour:
    - If `wandb` is not installed this is a no-op and returns None.
    - If `key` is provided it will call `wandb.login(key=key)`.
    - If `WANDB_API_KEY` is present in the environment it will be used.
    - Returns the imported `wandb` module on success, otherwise None.
    """
    global _wandb, _wandb_available
    try:
        import wandb as _wb
    except Exception:
        _wandb = None
        _wandb_available = False
        if not silent:
            print('wandb: not installed')
        return None

    api_key = key or os.environ.get('WANDB_API_KEY', None)
    try:
        if api_key:
            _wb.login(key=api_key)
        else:
            # try a passive login (will succeed if user previously logged in)
            try:
                _wb.login()
            except Exception:
                pass
        _wandb = _wb
        _wandb_available = True
        if not silent:
            print('wandb: available and initialized')
        return _wandb
    except Exception as e:
        _wandb = None
        _wandb_available = False
        if not silent:
            print(f'wandb: initialization failed ({e})')
        return None


def wandb_available() -> bool:
    return _wandb_available


def get_wandb():
    return _wandb


# Auto-initialize if an API key is present in the environment (convenience)
# This mirrors the behaviour you suggested — safe because it's wrapped
# in try/except and does nothing if wandb isn't installed.
if os.environ.get('WANDB_API_KEY'):
    try:
        init_wandb(silent=True)
    except Exception:
        pass
