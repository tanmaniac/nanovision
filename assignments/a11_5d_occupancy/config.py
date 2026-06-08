"""Hyperparameters for the occupancy toy. Provided, not a hole.

The grid is tiny on purpose: Z=8, Y=32, X=32 with n_classes=4 (free + 3 occupied), ~8k voxels.
A real Occ3D grid is 200x200x16 over 18 classes, ~46 MB of labels per sample, so the dense
storage here is the mechanism isolator, not the production layout (see the README on sparse
occupancy).

Sample budget: rendered depth quantizes to the sample spacing (z_far - z_near) / n_samples.
The render-supervision test asserts depth error under 0.3 m, so n_samples is set so the spacing
is well under that. With z_near=1, z_far=15 the floor is (15 - 1) / 0.15 = 93.3, so n_samples=96
gives a spacing of ~0.146 m, a factor-2 margin.

These defaults keep every test on CPU in seconds.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class OccConfig:
    # Voxel grid (Z, Y, X) and classes.
    Z: int = 8
    Y: int = 32
    X: int = 32
    n_classes: int = 4              # free=0 + 3 occupied

    # Metric grid extent (meters), ego frame: ((x0,x1),(y0,y1),(z0,z1)).
    grid_bounds: tuple = ((-4.0, 4.0), (-4.0, 4.0), (-1.0, 2.0))

    # Model widths.
    bev_channels: int = 16          # BEV feature channels C (and voxel feature channels)

    # Ray sampling bracket and budget.
    z_near: float = 1.0
    z_far: float = 15.0
    n_samples: int = 96             # >= (z_far - z_near) / 0.15 = 93.3

    # Toy scene.
    n_boxes: int = 2
    n_cams: int = 3
    img: int = 24

    def __post_init__(self):
        floor = (self.z_far - self.z_near) / 0.15
        assert self.n_samples >= floor, (
            f"n_samples={self.n_samples} below the depth-quantization floor {floor:.1f}"
        )

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.Z, self.Y, self.X)

    def voxel_centers(self, device="cpu", dtype=torch.float32) -> Tensor:
        """Ego-frame metric coordinates of every voxel center.

        Cell-center convention (align_corners=False): center i along an axis of S cells over
        [a, b] sits at a + (i + 0.5) * (b - a) / S. This matches the grid_sample normalization
        in render_occupancy_rays.

        Returns:
            [Z, Y, X, 3] (x, y, z) metric center of each voxel.
        """
        (x0, x1), (y0, y1), (z0, z1) = self.grid_bounds
        xs = x0 + (torch.arange(self.X, device=device, dtype=dtype) + 0.5) * (x1 - x0) / self.X
        ys = y0 + (torch.arange(self.Y, device=device, dtype=dtype) + 0.5) * (y1 - y0) / self.Y
        zs = z0 + (torch.arange(self.Z, device=device, dtype=dtype) + 0.5) * (z1 - z0) / self.Z
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")     # each [Z, Y, X]
        return torch.stack([gx, gy, gz], dim=-1)                   # [Z, Y, X, 3]
