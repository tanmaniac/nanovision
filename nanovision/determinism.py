"""Seeding and deterministic flags so tests and overfit runs are reproducible."""

import os
import random

import numpy as np
import torch


def default_device() -> torch.device:
    """The device heavy training/visualization demos should use: CUDA if present, else CPU.

    For viz.py and full training runs only. The graded unit tests stay on CPU deliberately -
    they must be device-agnostic (runnable without a GPU) and deterministic (CUDA relaxes
    bit-exactness, which would make gradcheck and exact-equality assertions flaky). Use this in
    viz/demo code: `dev = default_device(); model.to(dev)`.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
