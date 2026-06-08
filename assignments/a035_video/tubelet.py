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

    A Conv3d with kernel and stride both (t, p, p) over a video (B, C, T, H, W)
    produces one token per (t x p x p) space-time tubelet, then the tubelet grid is
    flattened to a token sequence:

        Conv3d(C -> dim, kernel=(t,p,p), stride=(t,p,p))   # (B, dim, T', H', W')
        flatten(2).transpose(1, 2)                          # (B, N, dim)

    with T' = T/t, H' = H/p, W' = W/p, and N = T' * H' * W'. The flatten must be
    temporal-outermost (the default flatten(2) order over (T', H', W')), so token
    idx = t' * (H'*W') + (h'*W' + w'); the backbone PE, the tube mask, and the
    reconstruction target all assume this order.

    forward(video): video is (B, C, T, H, W); returns (B, N, dim).
    """

    def __init__(self, in_chans: int, dim: int, patch: int, tubelet_t: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, dim,
            kernel_size=(tubelet_t, patch, patch),
            stride=(tubelet_t, patch, patch),
        )

    def forward(self, video: Tensor) -> Tensor:
        """video: (B, C, T, H, W) -> (B, N, dim).

        Implement:
            1. x = self.proj(video)                # (B, dim, T', H', W')
            2. x = x.flatten(2).transpose(1, 2)    # (B, N, dim), temporal-outermost
            3. return x
        """
        raise NotImplementedError("A3.5 Task 1: implement TubeletEmbedding.forward")
