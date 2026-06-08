"""Reference: A3.5 TubeletEmbedding (ViViT-style spatiotemporal patch embed).

Answer key for the top-level `tubelet.py` the student edits. The shared library
exposes this as `nanovision.transformer.TubeletEmbedding`. It is the temporal analog
of A2's PatchEmbed: a non-overlapping strided 3D convolution applies one shared linear
map to each space-time tubelet.
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

    with T' = T/t, H' = H/p, W' = W/p, and N = T' * H' * W'. The flatten is
    temporal-outermost, so token idx = t' * (H'*W') + (h'*W' + w'); the backbone's
    positional embedding, the tube mask, and the reconstruction target all assume
    this order.

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
        x = self.proj(video)              # (B, dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim), temporal-outermost
        return x
