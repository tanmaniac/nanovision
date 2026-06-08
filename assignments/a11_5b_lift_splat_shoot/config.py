"""Hyperparameters for the Lift-Splat-Shoot toy. Provided, not a hole.

The toy is one forward camera at 32x32 imaging a few "vehicles" as colored blobs, with an
8x16 BEV grid (x in [0, 8] m forward, y in [-8, 8] m lateral, 1.0 m cells). The depth-range /
BEV-extent match is load-bearing: a forward camera's frustum reaches only as far forward as the
deepest depth bin, so the BEV forward extent must lie within the depth range. With d_min=1,
d_max=9, d_step=1 the bins are arange(1, 9, 1) = [1..8] (EXCLUSIVE of d_max), D=8, deepest
reachable point 8 m forward - exactly the BEV x_max. Using arange(1, 10, 1) would give D=9 and
break every shape; pin the exclusive convention here.

These defaults keep every test on CPU in seconds. A real LSS config is D=41 bins from 4 to 45 m
and a 200x200 BEV grid over six cameras; see the README.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from nanovision.geometry import BEVGrid


@dataclass
class LSSConfig:
    # Image and backbone.
    img: int = 32              # square image side in pixels
    stride: int = 4            # backbone downsample; feature grid is img // stride per side
    c_backbone: int = 16       # backbone hidden channels
    c_ctx: int = 16            # context channels carried into the BEV grid

    # Depth bins (centers = arange(d_min, d_max, d_step), EXCLUSIVE of d_max).
    d_min: float = 1.0
    d_max: float = 9.0
    d_step: float = 1.0

    # BEV grid extent (meters). Forward x_max must lie within the depth range above.
    bev_x_min: float = 0.0
    bev_x_max: float = 8.0
    bev_y_min: float = -8.0
    bev_y_max: float = 8.0
    bev_res: float = 1.0

    # Camera.
    focal: float | None = None  # pinhole focal in pixels; defaults to img / 2 (~90 deg FOV)
    cam_height: float = 1.5     # camera height above the ego origin (meters)

    @property
    def D(self) -> int:
        """Number of depth bins."""
        return self.bins().numel()

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

    def bins(self, dtype=torch.float32) -> Tensor:
        """Depth-bin centers, arange(d_min, d_max, d_step) EXCLUSIVE of d_max -> (D,)."""
        return torch.arange(self.d_min, self.d_max, self.d_step, dtype=dtype)

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

    def pixel_xy(self, dtype=torch.float32) -> Tensor:
        """Image-pixel centers (u, v) for every feature cell -> (Hf, Wf, 2).

        Feature cell (i, j) corresponds to image pixel ((j + 0.5) * stride, (i + 0.5) * stride):
        i indexes feature rows (image v / height), j indexes feature columns (image u / width).
        """
        Hf, Wf, s = self.Hf, self.Wf, self.stride
        vs = (torch.arange(Hf, dtype=dtype) + 0.5) * s   # image v per feature row
        us = (torch.arange(Wf, dtype=dtype) + 0.5) * s   # image u per feature col
        gv, gu = torch.meshgrid(vs, us, indexing="ij")
        return torch.stack([gu, gv], dim=-1)             # (Hf, Wf, 2) as (u, v)
