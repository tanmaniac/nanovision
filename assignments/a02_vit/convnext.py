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

    A depthwise 7x7 conv mixes spatially within each channel, then an inverted
    bottleneck (Linear dim->4*dim, gelu, Linear 4*dim->dim) mixes across channels,
    with an optional learnable layer-scale on the branch and a residual add:

        y = x + LayerScale(Linear(gelu(Linear(LayerNorm(DWConv7x7(x))))))

    The depthwise conv and the channel MLP separate spatial mixing from channel
    mixing the same way attention and the FFN do in a transformer. LayerNorm and
    the two Linears run channels-last (B, H, W, dim); the conv and the residual run
    channels-first (B, dim, H, W).

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
        """x: (B, dim, H, W) -> (B, dim, H, W).

        Implement:
            1. residual = x
            2. x = dwconv(x)                       # depthwise 7x7, (B, dim, H, W)
            3. permute to channels-last (B, H, W, dim)
            4. x = LayerNorm(x); x = pw1(x); x = gelu(x); x = pw2(x)
            5. if self.gamma is not None: x = self.gamma * x   # layer scale
            6. permute back to (B, dim, H, W)
            7. return residual + x
        """
        raise NotImplementedError("A2 Task 1: implement ConvNeXtBlock.forward")
