"""A2 - the one new primitive (ConvNeXtBlock). Fill the hole.

You only implement the ConvNeXt block below. It needs LayerNorm and gelu (built in
A0). They are pulled from the A0 owner through the student loader rather than from
`nanovision.primitives`, because nanovision.primitives sources ConvNeXtBlock from
this file; importing it back would be a cycle. The reference is in this assignment's
solution/convnext.py.
"""

import torch
from torch import Tensor, nn

from nanovision._student import load

_a0 = load("a00_harness", "primitives")
LayerNorm, gelu = _a0.LayerNorm, _a0.gelu


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al., 2022), the modernized ResNet residual unit.

    A depthwise spatial conv and an inverted-bottleneck channel MLP on a residual
    branch, with an optional learnable layer-scale gain. The depthwise conv and the
    residual add run channels-first (B, dim, H, W); LayerNorm and the two Linears run
    channels-last (B, H, W, dim), so the block permutes between the two layouts.

    forward(x): x is (B, dim, H, W); output is (B, dim, H, W).

    See the ConvNeXt block section of the README.
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
        """x: (B, dim, H, W) -> (B, dim, H, W).

        The layer-scale gain is applied only when self.gamma is not None. See the
        ConvNeXt block section of the README.
        """
        raise NotImplementedError("A2 Task 1: implement ConvNeXtBlock.forward")
