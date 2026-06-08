"""Hyperparameters for the BEVFormer toy. Provided, not a hole.

The toy is a 4-camera ring at 32x32 imaging a few "vehicles" as colored blobs, with a centered
16x16 BEV grid (x and y in [-8, 8] m forward/lateral, 1.0 m cells). The grid is centered (not
forward-only like Lift-Splat-Shoot's) because the cameras ring the ego. Anchor heights follow the
paper: n_heights=4 from z_min=-5 m to z_max=3 m in ego z, which brackets the vehicle centroids at
z=0.75 m.

These defaults keep every test on CPU in seconds. A real BEVFormer config is a 200x200 BEV grid
over six cameras with a heavy backbone; see the README.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from nanovision.geometry import BEVGrid


@dataclass
class BEVFormerConfig:
    # Image and feature grid.
    img: int = 32              # square image side in pixels
    stride: int = 4            # backbone downsample; feature grid is img // stride per side
    n_cams: int = 4            # ring cameras

    # Model width and depth.
    dim: int = 32              # BEV query / feature channels C
    n_heads: int = 4           # attention heads (temporal + deformable)
    n_layers: int = 3          # stacked encoder layers
    ffn_hidden: int = 64       # feed-forward hidden width

    # Spatial cross-attention.
    offsets: bool = False      # learned deformable offsets (off for the overfit/temporal tests)
    n_points: int = 4          # deformable samples per reference point (offsets=True only)

    # Reference pillars.
    n_heights: int = 4         # anchor heights per BEV cell
    z_min: float = -5.0        # ego-z range of the anchor heights (meters)
    z_max: float = 3.0

    # BEV grid extent (meters), centered on the ego.
    bev_min: float = -8.0
    bev_max: float = 8.0
    bev_res: float = 1.0

    # Camera.
    focal: float | None = None  # pinhole focal in pixels; defaults to img / 2 (~90 deg FOV)
    cam_height: float = 1.5     # camera height above the ego origin (meters)

    @property
    def Hf(self) -> int:
        """Feature-grid height (= img // stride)."""
        return self.img // self.stride

    @property
    def Wf(self) -> int:
        """Feature-grid width (= img // stride)."""
        return self.img // self.stride

    @property
    def f(self) -> float:
        """Resolved focal length in pixels."""
        return float(self.focal) if self.focal is not None else self.img / 2.0

    def bev_grid(self) -> BEVGrid:
        """The centered ego BEV grid (nx along forward x, ny along lateral y)."""
        return BEVGrid(
            x_min=self.bev_min, x_max=self.bev_max,
            y_min=self.bev_min, y_max=self.bev_max,
            resolution=self.bev_res,
        )

    def K(self, dtype=torch.float32) -> Tensor:
        """Pinhole intrinsic (3, 3) consistent with the toy generator's defaults."""
        f = self.f
        c = (self.img - 1) / 2.0
        return torch.tensor(
            [[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]], dtype=dtype
        )
