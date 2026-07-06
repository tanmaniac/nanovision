"""Fourier positional encoding, the fix for the MLP's spectral bias.

A plain coordinate MLP fits low frequencies first and renders the scene blurry (the
spectral-bias result of Tancik et al. 2020). The encoding lifts each input coordinate to a
bank of sinusoids at geometrically spaced frequencies, so the network can represent
high-frequency detail (sharp object boundaries) early in training.

Input-range contract: the geometric frequency schedule is only well-behaved when inputs
are normalized to roughly [-1, 1] first. The caller (NeRFMLP) divides sample positions by a
scene-bound constant before encoding and feeds unit-length directions. An unnormalized
position hit with the highest-frequency band aliases badly, so this module assumes the
caller has normalized.
"""

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Map (..., D) coordinates to Fourier features with no learnable parameters.

    The encoding is applied to each of the D input coordinates. With include_input the raw
    input is concatenated first, so the output dim is D + D*2*L; without it, D*2*L.

    See the "Spectral bias and the Fourier encoding" section of the README for the encoding.

    Args:
        L: number of frequency bands.
        include_input: prepend the raw coordinates to the encoding.
    """

    def __init__(self, L: int, include_input: bool = True):
        super().__init__()
        self.L = L
        self.include_input = include_input
        # Frequency bands 2^k for k = 0..L-1, stored as a non-trainable buffer.
        bands = 2.0 ** torch.arange(L, dtype=torch.float32)
        self.register_buffer("bands", bands)

    def forward(self, x: Tensor) -> Tensor:
        """Encode x (..., D) -> (..., D + D*2*L) with include_input, else (..., D*2*L)."""
        raise NotImplementedError("implement the Fourier positional encoding")
