"""A1 - the two new primitives (RMSNorm, SwiGLU). Fill the holes.

LayerNorm, gelu, and MLP were built in A0; code that needs them imports them from
`nanovision.primitives` directly, so here you only implement the two new LLaMA-style
primitives. RMSNorm and SwiGLU are self-contained. The reference is in this
assignment's solution/primitives.py.
"""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019), the LLaMA-style norm.

    Unlike LayerNorm there is no mean subtraction and no bias: rescale by the
    root-mean-square over the last axis and a learned gain.

    Shapes: x is (..., dim); output is the same shape.
    See the RMSNorm section of the README.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("A1 Task 5: implement RMSNorm.forward")


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward (Shazeer, 2020), the LLaMA-family FFN.

    A SiLU-activated gate branch multiplies an up-projection branch, then a
    down-projection. All three linear layers are bias-free. Shapes: x is
    (..., dim); gate/up map to (..., hidden); output is (..., dim).
    See the SwiGLU section of the README.
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("A1 Task 6: implement SwiGLU.forward (gated SiLU)")
