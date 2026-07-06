"""A0 - fill the three holes, then run the tests.

The reference implementation lives in this assignment's `solution/primitives.py` (read it if you
get stuck). Do not import it here; implement the bodies yourself.
"""

import math

import torch
from torch import Tensor, nn


def gelu(x: Tensor) -> Tensor:
    """Exact (erf) GELU, not the tanh approximation. Input/output shape: same as x.

    See the GELU section of the README.
    """
    raise NotImplementedError("A0 Task 1: implement exact erf GELU")


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, computed from mean/var ops.

    Shapes: x is (..., dim); output is the same shape. No nn.LayerNorm.
    See the layer normalization section of the README.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("A0 Task 2: implement LayerNorm.forward")


class MLP(nn.Module):
    """Two-layer MLP: Linear -> act -> dropout -> Linear.

    Shapes: x is (..., dim); output is (..., dim).
    See the MLP section of the README.
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0, act=gelu):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("A0 Task 3: implement MLP.forward")
