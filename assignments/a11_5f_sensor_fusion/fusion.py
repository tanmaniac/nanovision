"""Multi-modal LiDAR + camera fusion: PointPainting, a pillar encoder, and BEV fusion.

This is the holed file the learner fills in. The reference lives in ``solution/fusion.py`` with
identical docstrings and provided bodies; only the hole bodies below differ. These are
assignment-LOCAL modules (no later assignment consumes a fusion primitive), so tests and
``viz.py`` import them by bare name (``from fusion import paint_points``), while the shared
primitives they build on come through ``nanovision.*``.

Three mechanisms, ordered by where the two modalities meet:
- ``paint_points``: point-level (early) fusion. Project each LiDAR point into the image and
  concatenate the per-pixel class score onto its coordinates. This is PointPainting.
- ``LidarPillarEncoder``: the LiDAR branch. Augment each point to nine features, run a per-point
  MLP, then scatter-MAX the point features into their BEV pillar. This is a PointPillars-lite
  encoder; the max pool (not a sum) keeps the pillar feature from leaking point density.
- ``BEVFuser``: feature-level fusion. Channel-concatenate the camera BEV feature map and the
  LiDAR BEV feature map on the shared grid and mix them with a small conv stack. This is the
  BEVFusion core.

Conventions match ``nanovision.geometry`` and the toy substrate: ego frame x forward, y left,
z up; camera frame OpenCV x right, y down, z forward; the extrinsic is T_cam_ego (ego -> camera).
The flat pillar index is ``ix * ny + iy``, matching the LiDAR branch reused from A11.5b.
"""

import torch
import torch.nn as nn
from torch import Tensor

from nanovision.geometry import BEVGrid, apply_transform, project_points
from nanovision.lift_splat import cumsum_pool, pillar_index


def paint_points(
    points_ego: Tensor,
    seg_scores: Tensor,
    K: Tensor,
    T_cam_ego: Tensor,
    image_hw: tuple[int, int],
) -> Tensor:
    """PointPainting: decorate each LiDAR point with the image class score at its pixel.

    For each ego-frame point: transform it to the camera frame with ``T_cam_ego``; drop it if it
    is behind the camera (camera-frame z <= 0); project it to a pixel with ``project_points``;
    drop it if the pixel falls outside ``[0, W) x [0, H)``; otherwise gather the C-dim per-pixel
    score vector at the nearest pixel and concatenate it onto the point. Dropped points get the
    zero (background) score vector. The output order is ``[x, y, z, s_1..s_C]``.

    Args:
        points_ego: (N, 3) ego-frame LiDAR points.
        seg_scores: (C, H, W) per-pixel soft class scores.
        K: (3, 3) pinhole intrinsic.
        T_cam_ego: (4, 4) extrinsic ego -> camera.
        image_hw: (H, W) image size in pixels.

    Returns:
        (N, 3 + C) painted points.
    """
    raise NotImplementedError("paint_points")


class LidarPillarEncoder(nn.Module):
    """PointPillars-lite: augment points, MLP per point, scatter-MAX into a BEV grid.

    Each point is augmented to nine features before the MLP, following PointPillars
    (arXiv:1812.05784): ``[x, y, z, xc, yc, zc, xp, yp, r]`` where ``(xc, yc, zc)`` is the point
    minus its pillar's point-mean (the cluster offset) and ``(xp, yp)`` is the offset to the
    pillar center; ``r`` is a constant 1.0 (the toy has no intensity). A per-point MLP maps the
    nine features to C channels, then a scatter-MAX over the points in each pillar gives the BEV
    feature. The max (not a sum) keeps the pillar feature from encoding how many points landed in
    it, so a LiDAR-only reader cannot recover class from point density.

    Args:
        bev_grid: the shared BEV grid.
        hidden: per-point MLP hidden width.
        out_channels: BEV feature channels C.
    """

    def __init__(self, bev_grid: BEVGrid, hidden: int, out_channels: int):
        super().__init__()
        self.bev_grid = bev_grid
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_channels)
        )

    def forward(self, points_ego: Tensor) -> Tensor:
        """Encode an ego-frame point cloud to a (C, nx, ny) BEV feature map.

        Steps:
        1. ``pillar_index`` assigns each point to a flat pillar; drop out-of-bounds points.
        2. Compute the per-pillar point-mean (``cumsum_pool`` sum / a count) and the pillar
           centers, then build the nine augmented features per point.
        3. The per-point MLP maps nine -> C.
        4. Max-pool the point features into their pillar (scatter-MAX, NOT ``cumsum_pool``);
           reshape to (C, nx, ny).

        Args:
            points_ego: (N, 3) ego-frame points.

        Returns:
            (C, nx, ny) LiDAR BEV feature map.
        """
        raise NotImplementedError("LidarPillarEncoder.forward")


class BEVFuser(nn.Module):
    """BEVFusion core: channel-concat the camera and LiDAR BEV maps, mix with a conv stack.

    The two branches are already on the same BEV grid, so fusion is a channel concat followed by
    a small conv encoder. The concatenated feature at a cell carries both LiDAR geometry (is the
    cell occupied) and camera semantics (which class), and the conv learns their conjunction.

    Args:
        cam_channels: channels of the camera BEV map.
        lidar_channels: channels of the LiDAR BEV map.
        hidden: conv hidden width.
        out_channels: unified BEV feature channels.
    """

    def __init__(self, cam_channels: int, lidar_channels: int, hidden: int, out_channels: int):
        super().__init__()
        c_in = cam_channels + lidar_channels
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, cam_bev: Tensor, lidar_bev: Tensor) -> Tensor:
        """Fuse two aligned BEV feature maps into one.

        Concatenate along the channel axis and run the conv stack.

        Args:
            cam_bev: (C_cam, nx, ny) or (B, C_cam, nx, ny) camera BEV feature.
            lidar_bev: (C_lidar, nx, ny) or (B, C_lidar, nx, ny) LiDAR BEV feature.

        Returns:
            unified BEV feature, (C_out, nx, ny) or (B, C_out, nx, ny) matching the input rank.
        """
        raise NotImplementedError("BEVFuser.forward")
