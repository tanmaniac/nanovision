"""A1 starter - the two new primitives (RMSNorm, SwiGLU). Fill the holes.

LayerNorm, gelu, and MLP were built in A0 and are re-exported from the shared
library so the A1 transformer block can use them; you only implement the two new
LLaMA-style primitives below. The reference lives in `nanovision/primitives.py`.
"""

import torch
from torch import Tensor, nn

from nanovision.primitives import MLP, LayerNorm, gelu  # noqa: F401  (A0, not A1 tasks)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019), the LLaMA-style norm.

    No mean subtraction and no bias: rescale by the root-mean-square over the last
    axis and a learned gain.

        rms(x) = sqrt(mean(x^2, last dim) + eps)
        y      = x / rms(x) * weight

    Shapes: x is (..., dim); output is the same shape.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("A1 Task 5: implement RMSNorm.forward")


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward (Shazeer, 2020), the LLaMA-family FFN.

        SwiGLU(x) = (silu(W_gate x) * (W_up x)) @ W_down
        silu(z)   = z * sigmoid(z)

    All three linear layers are bias-free. Shapes: x is (..., dim); gate/up map to
    (..., hidden); output is (..., dim).
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("A1 Task 6: implement SwiGLU.forward (gated SiLU)")
