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


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019), the LLaMA-style norm.

    Unlike LayerNorm there is no mean subtraction and no bias; the input is just
    rescaled by its root-mean-square over the last axis and a learned gain:

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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al., 2022), the modernized ResNet residual unit.

    A depthwise 7x7 conv mixes spatially within each channel, then an inverted
    bottleneck (Linear dim->4*dim, gelu, Linear 4*dim->dim) mixes across channels,
    with an optional learnable layer-scale on the branch and a residual add:

        y = x + LayerScale(Linear(gelu(Linear(LayerNorm(DWConv7x7(x))))))

    The depthwise conv and the channel MLP separate spatial mixing from channel
    mixing the same way attention and the FFN do in a transformer, which is why a
    pure-conv block with this layout matches Swin/ViT at equal compute. LayerNorm
    and the two Linears run channels-last (B, H, W, dim); the conv and the residual
    run channels-first (B, dim, H, W).

    forward(x): x is (B, dim, H, W); output is (B, dim, H, W).
    """

    def __init__(self, dim: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        # Layer scale (Touvron et al., 2021): a per-channel learned gain on the
        # branch, initialized small so the block starts close to the identity.
        if layer_scale_init is not None:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, dim, H, W) -> (B, H, W, dim)
        x = self.norm(x)
        x = self.pw1(x)
        x = gelu(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, dim) -> (B, dim, H, W)
        return residual + x
