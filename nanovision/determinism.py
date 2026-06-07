"""Seeding and deterministic flags so tests and overfit runs are reproducible."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA).

    With deterministic=True, also set cuDNN to deterministic mode. This is what
    the overfit-one-batch tests rely on to be reliable under a fixed step budget.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
