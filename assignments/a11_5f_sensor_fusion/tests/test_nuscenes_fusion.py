"""nuScenes fusion path, gated on the dataset being present.

The toy is self-contained, so this only checks that the two branches accept real nuScenes data:
LiDAR points move from the lidar sensor frame to the ego frame via lidar_to_ego, the pillar
encoder returns a BEV map on the loader's grid, and PointPainting projects the ego points into one
real camera. Skips cleanly when NUSCENES_DATAROOT is unset or the devkit is missing.
"""

import os

import pytest
import torch

from config import FusionConfig
from fusion import LidarPillarEncoder, paint_points

from nanovision.geometry import apply_transform


def _load_one_sample():
    dataroot = os.environ.get("NUSCENES_DATAROOT")
    if not dataroot or not os.path.isdir(dataroot):
        pytest.skip("NUSCENES_DATAROOT unset or missing; see the a11_5 series README")
    try:
        import nuscenes  # noqa: F401
    except ImportError:
        pytest.skip("nuscenes-devkit not installed; run: pip install -e \".[av]\"")
    from nanovision.data.nuscenes_mini import NuScenesMini

    ds = NuScenesMini(dataroot=dataroot, image_size=(400, 224))
    assert len(ds) > 0
    return ds[0]


def test_nuscenes_lidar_and_paint_shapes():
    sample = _load_one_sample()
    cfg = FusionConfig()

    # LiDAR is returned in the lidar sensor frame; move it to the ego frame.
    lidar_ego = apply_transform(sample["lidar_to_ego"], sample["lidar"])  # (N, 3)
    assert lidar_ego.shape[1] == 3

    grid = sample["bev_grid"]
    enc = LidarPillarEncoder(grid, cfg.lidar_hidden, cfg.lidar_channels)
    bev = enc(lidar_ego)
    assert bev.shape == (cfg.lidar_channels, grid.nx, grid.ny)

    # Paint the ego points into one real camera (a placeholder uniform seg map stands in for a
    # trained segmenter; only the projection/gather shape is checked here).
    rig = sample["rig"]
    cam = next(iter(sample["images"]))
    _, H, W = sample["images"][cam].shape
    C = cfg.n_classes
    seg = torch.full((C, H, W), 1.0 / C)
    painted = paint_points(lidar_ego, seg, rig.Ks[cam], rig.extrinsics[cam], (H, W))
    assert painted.shape == (lidar_ego.shape[0], 3 + C)
