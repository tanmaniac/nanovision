"""A3.5 - the tubelet embedding (TubeletEmbedding). Fill the one hole.

This is the one shared-library symbol A3.5 adds: it is exposed as
`nanovision.transformer.TubeletEmbedding`. It is the temporal analog of A2's
PatchEmbed, a non-overlapping strided 3D convolution applied per space-time tubelet.
The reference is in this assignment's solution/tubelet.py.
"""

import torch
from torch import Tensor, nn


class TubeletEmbedding(nn.Module):
    """Spatiotemporal tubelet embedding (Arnab et al., ViViT, 2021).

    The temporal analog of the image patch embedding: the provided Conv3d (kernel and
    stride both (t, p, p)) produces one token per (t x p x p) space-time tubelet, and
    the hole turns its output grid into a token sequence.

    With T' = T/t, H' = H/p, W' = W/p, the sequence has N = T' * H' * W' tokens and
    must be in temporal-outermost order: token idx = t' * (H'*W') + (h'*W' + w').
    The backbone PE, the tube mask, and the reconstruction target all assume this
    order, so it is a hard constraint, not a choice.

    forward(video): video is (B, C, T, H, W); returns (B, N, dim).

    See the tubelet embedding section of the README.
    """

    def __init__(self, in_chans: int, dim: int, patch: int, tubelet_t: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, dim,
            kernel_size=(tubelet_t, patch, patch),
            stride=(tubelet_t, patch, patch),
        )

    def forward(self, video: Tensor) -> Tensor:
        """video: (B, C, T, H, W) -> (B, N, dim), tokens in temporal-outermost order.

        See the tubelet embedding section of the README.
        """
        raise NotImplementedError("A3.5 Task 1: implement TubeletEmbedding.forward")
