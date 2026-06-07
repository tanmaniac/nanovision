"""Core neural-network primitives, built from low-level ops.

A0 establishes the shared library with LayerNorm, gelu, and MLP. Later
assignments append to this module: RMSNorm and SwiGLU (A1), ConvNeXtBlock (A2).
No `nn.LayerNorm` / `nn.GELU` — the point is to build the mechanism.
"""

import math

import torch
from torch import Tensor, nn


def gelu(x: Tensor) -> Tensor:
    """Exact (erf) GELU.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, from mean/var ops.

    y = (x - mean) / sqrt(var + eps) * weight + bias
    where mean and (biased) var are taken over the last `dim` axis.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


class MLP(nn.Module):
    """Two-layer MLP: Linear -> act -> dropout -> Linear."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0, act=gelu):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))
