import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import contextlib
import torch.distributed as dist

def launched_by_torchrun():
    return "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
if launched_by_torchrun():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    device_type = device.type
if float(torch.__version__.split(".")[0]+"."+torch.__version__.split(".")[1]) >= 2.4:
    amp = torch.amp
    caster = torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=device_type)
    amp_ctx = torch.amp.autocast(enabled=False, device_type=device_type) if torch.cuda.is_available() else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
else:
    amp = torch.cuda.amp if torch.cuda.is_available() else torch.cpu.amp
    caster = torch.autocast(enabled=torch.cuda.is_available(), device_type=device_type)
    amp_ctx = torch.autocast(enabled=False, device_type=device_type) if torch.cuda.is_available() else contextlib.nullcontext()
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False