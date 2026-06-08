"""Reference: A1 primitives (RMSNorm, SwiGLU), the LLaMA-style norm and FFN.

Answer key for the top-level `primitives.py` the student edits. The shared library
exposes these as `nanovision.primitives.{RMSNorm, SwiGLU}`. They are self-contained
(no dependency on the A0 primitives), so this file defines only the two new symbols.
"""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019), the LLaMA-style norm.

    No mean subtraction and no bias: rescale by the root-mean-square over the last
    axis and a learned gain.

        rms(x) = sqrt(mean(x^2, last dim) + eps)
        y      = x / rms(x) * weight

    Shapes: x is (..., dim); output is the same shape. eps defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward (Shazeer, 2020), the LLaMA-family FFN.

        SwiGLU(x) = (silu(W_gate x) * (W_up x)) @ W_down

    where silu(z) = z * sigmoid(z). Three linear layers, all bias-free to match
    the modern stack. `hidden` is the inner width; callers using the 8/3 rule pass
    hidden ~= 8/3 * dim (rounded) so the parameter count matches a 4x GELU MLP.

    Shapes: x is (..., dim); gate/up project to (..., hidden); output is (..., dim).
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))
