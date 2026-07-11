"""Hyperparameters for the multi-modal fusion toy. Provided, not a hole.

The toy is one forward camera at 32x32 plus a LiDAR point cloud, both over an 8x16 BEV grid
(x in [0, 8] m forward, y in [-8, 8] m lateral, 1.0 m cells). Two seg classes: index 0 is
background, index 1 is vehicle. These defaults keep every test on CPU in seconds. A real fusion
stack runs a 200x200 BEV grid over six cameras and a full 32-beam sweep; see the README.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from nanovision.geometry import BEVGrid


@dataclass
class FusionConfig:
    # Image and camera.
    img: int = 32                 # square image side in pixels
    stride: int = 4               # backbone downsample (feature grid is img // stride)
    focal: float | None = None    # pinhole focal in pixels; defaults to img / 2 (~90 deg FOV)
    cam_height: float = 1.5       # camera height above the ego origin (meters)

    # Segmentation classes carried by seg_scores / PointPainting.
    n_classes: int = 2            # background (0) and vehicle (1)

    # BEV grid extent (meters). Shared by the camera and LiDAR branches (fusion is on one grid).
    bev_x_min: float = 0.0
    bev_x_max: float = 8.0
    bev_y_min: float = -8.0
    bev_y_max: float = 8.0
    bev_res: float = 1.0

    # LiDAR pillar encoder.
    lidar_hidden: int = 32        # per-point MLP hidden width
    lidar_channels: int = 16      # LiDAR BEV feature channels (C of the (C, nx, ny) map)

    # BEV fusion encoder.
    fuse_hidden: int = 32         # conv fuser hidden width
    fuse_channels: int = 16       # unified BEV feature channels

    # TransFuser attention block.
    token_dim: int = 16           # per-token model dimension
    n_heads: int = 2              # attention heads

    @property
    def f(self) -> float:
        """Resolved focal length in pixels."""
        return float(self.focal) if self.focal is not None else self.img / 2.0

    def bev_grid(self) -> BEVGrid:
        """The ego-centric BEV grid for the toy (nx along forward x, ny along lateral y)."""
        return BEVGrid(
            x_min=self.bev_x_min, x_max=self.bev_x_max,
            y_min=self.bev_y_min, y_max=self.bev_y_max,
            resolution=self.bev_res,
        )

    def K(self, dtype=torch.float32) -> Tensor:
        """Pinhole intrinsic (3, 3) consistent with the toy generator's defaults."""
        f = self.f
        c = (self.img - 1) / 2.0
        return torch.tensor(
            [[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]], dtype=dtype
        )
