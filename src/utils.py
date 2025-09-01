import os
import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_checkpoint(state, is_best: bool, out_dir: str, filename: str = "last.ckpt"):
    ensure_dir(out_dir)
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.ckpt")
        torch.save(state, best_path)
    return ckpt_path
