"""The NeRF radiance-field MLP. Provided in full.

The network maps a 3D world position to a volume density sigma (independent of view) and a
view-dependent RGB color. This factorization is the original NeRF design: one density per
point, but appearance can change with the viewing direction (specular highlights, etc.).

Two details the renderer depends on:
- density passes through softplus so sigma >= 0; volume_render then takes these raw
  non-negative sigmas and does NOT re-activate them.
- color passes through sigmoid so rgb is in [0, 1].

Input normalization: sample positions are divided by `scene_bound` (roughly the scene's
half-extent) before the Fourier encoding, mapping the object into about [-1, 1]. The
encoding's 2^k frequency schedule is only valid on that range; feeding raw positions of
magnitude ~4 would alias the top band. Directions are already unit length.

The pedagogy of A9 is the positional encoding and the volume renderer, not this MLP plumbing,
so it is provided rather than holed.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from encoding import PositionalEncoding


class NeRFMLP(nn.Module):
    """Position -> density, (position, direction) -> color.

    Args:
        pos_L: frequency bands for the position encoding.
        dir_L: frequency bands for the direction encoding.
        hidden: width of the hidden layers.
        n_layers: number of hidden layers in the density trunk.
        include_input: pass raw coordinates alongside the Fourier features.
        scene_bound: positions are divided by this before encoding (maps the object to
            roughly [-1, 1] so the frequency schedule is valid).
    """

    def __init__(
        self,
        pos_L: int = 6,
        dir_L: int = 4,
        hidden: int = 128,
        n_layers: int = 4,
        include_input: bool = True,
        scene_bound: float = 4.0,
    ):
        super().__init__()
        self.scene_bound = scene_bound
        self.pos_enc = PositionalEncoding(pos_L, include_input)
        self.dir_enc = PositionalEncoding(dir_L, include_input)

        pos_dim = 3 + 3 * 2 * pos_L if include_input else 3 * 2 * pos_L
        dir_dim = 3 + 3 * 2 * dir_L if include_input else 3 * 2 * dir_L
        self.pos_dim = pos_dim
        self.skip = n_layers // 2  # re-inject the encoded position at the middle layer

        trunk = []
        in_dim = pos_dim
        for i in range(n_layers):
            if i == self.skip:
                in_dim += pos_dim
            trunk.append(nn.Linear(in_dim, hidden))
            in_dim = hidden
        self.trunk = nn.ModuleList(trunk)

        self.sigma_head = nn.Linear(hidden, 1)
        self.feat = nn.Linear(hidden, hidden)
        self.rgb_hidden = nn.Linear(hidden + dir_dim, hidden // 2)
        self.rgb_head = nn.Linear(hidden // 2, 3)

    def forward(self, positions: Tensor, directions: Tensor) -> tuple[Tensor, Tensor]:
        """Density and color for sampled points.

        Args:
            positions: (R, N, 3) world-frame sample points.
            directions: (R, 3) unit ray directions (one per ray, broadcast over samples).

        Returns:
            sigma: (R, N) non-negative density.
            rgb: (R, N, 3) color in [0, 1].
        """
        R, N, _ = positions.shape
        pos_n = positions / self.scene_bound
        enc_pos = self.pos_enc(pos_n)                       # (R, N, pos_dim)

        h = enc_pos
        for i, layer in enumerate(self.trunk):
            if i == self.skip:
                h = torch.cat([h, enc_pos], dim=-1)
            h = F.relu(layer(h))

        sigma = F.softplus(self.sigma_head(h)).squeeze(-1)  # (R, N) >= 0

        feat = self.feat(h)                                 # (R, N, hidden)
        dirs = directions[:, None, :].expand(R, N, 3)
        enc_dir = self.dir_enc(dirs)                        # (R, N, dir_dim)
        rgb = F.relu(self.rgb_hidden(torch.cat([feat, enc_dir], dim=-1)))
        rgb = torch.sigmoid(self.rgb_head(rgb))             # (R, N, 3) in [0, 1]
        return sigma, rgb
