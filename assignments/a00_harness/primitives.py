"""A0 - fill the three holes, then run the tests.

The reference implementation lives in this assignment's `solution/primitives.py` (read it if you
get stuck). Do not import it here; implement the bodies yourself.
"""

import math

import torch
from torch import Tensor, nn


def gelu(x: Tensor) -> Tensor:
    """Exact (erf) GELU.

    Implement: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    Input/output shape: same as x. Use torch.erf.
    """
    raise NotImplementedError("A0 Task 1: implement exact erf GELU")


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, from mean/var ops.

    Implement forward as:
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, keepdim=True, unbiased=False)
        y    = (x - mean) / sqrt(var + eps) * weight + bias
    Shapes: x is (..., dim); output is the same shape. No nn.LayerNorm.
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

    Implement forward: self.fc2(self.drop(self.act(self.fc1(x)))).
    Shapes: x is (..., dim); output is (..., dim).
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0, act=gelu):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("A0 Task 3: implement MLP.forward")
