"""Reference: A2 ConvNeXtBlock (Liu et al., 2022), the modernized ResNet unit.

Answer key for the top-level `convnext.py` the student edits. The shared library
exposes this as `nanovision.primitives.ConvNeXtBlock`.

LayerNorm and gelu were built in A0. They are pulled from the A0 owner through the
student loader rather than `from nanovision.primitives import ...`, because
nanovision.primitives itself sources ConvNeXtBlock from here; importing it back
would be a cycle. The loader respects NANOVISION_IMPL, so in solution mode this
gets the A0 reference and in default mode the student's A0 code.
"""

import torch
from torch import Tensor, nn

from nanovision._student import load

_a0 = load("a00_harness", "primitives")
LayerNorm, gelu = _a0.LayerNorm, _a0.gelu


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
