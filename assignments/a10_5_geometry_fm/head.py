"""The pointmap head: decoder tokens -> per-pixel 3D points and a confidence.

Each cross-attending decoder produces (B, N, dim) tokens, one per patch. PointmapHead maps
each token to a 3D point (X, Y, Z) and a confidence logit, then reshapes the N tokens back to
the (h, w) patch grid. The toy predicts one point per patch (patch resolution); real DUSt3R
predicts one point per pixel through a DPT head. The lesson is the same: regress 3D + a
learned confidence.

Confidence uses the DUSt3R parameterization C = 1 + exp(logit) (paper text after Eq. 6), so
C >= 1 always. The loss reads C; a softplus would change the trade-off, so do not substitute.
"""

import torch
from torch import Tensor, nn


class PointmapHead(nn.Module):
    """MLP head mapping decoder tokens to a pointmap and a confidence map.

    Args:
        dim: decoder token dimension.
        grid: side length of the (square) patch grid, so N = grid * grid tokens.
        hidden: hidden width of the MLP.
    """

    def __init__(self, dim: int, grid: int, hidden: int = 128):
        super().__init__()
        self.grid = grid
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),  # 3 for XYZ, 1 for the confidence logit
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Map (B, N, dim) tokens to a pointmap and a confidence map.

        N must equal grid * grid; row-major reshape recovers the (h, w) = (grid, grid) layout
        that matches the toy's patch-center ordering.

        Returns:
            pts: (B, grid, grid, 3) per-patch XYZ.
            conf: (B, grid, grid) confidence, C = 1 + exp(logit), so C >= 1.
        """
        raise NotImplementedError(
            "run the MLP, split XYZ from the confidence logit, reshape to the grid, "
            "and map the logit to C = 1 + exp(logit)"
        )
