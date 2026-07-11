"""Output shapes of the four fusion mechanisms.

paint_points -> (N, 3 + C); LidarPillarEncoder -> (C, nx, ny); BEVFuser -> (C_out, nx, ny) and
its batched form; TransFuserBlock preserves each token set's shape.
"""

import torch

from config import FusionConfig
from fusion import BEVFuser, LidarPillarEncoder, paint_points
from transfuser import TransFuserBlock


def test_paint_points_shape():
    cfg = FusionConfig()
    N, C = 20, cfg.n_classes
    points = torch.randn(N, 3)
    seg = torch.rand(C, cfg.img, cfg.img)
    painted = paint_points(points, seg, cfg.K(), torch.eye(4), (cfg.img, cfg.img))
    assert painted.shape == (N, 3 + C)


def test_lidar_encoder_shape():
    cfg = FusionConfig()
    grid = cfg.bev_grid()
    enc = LidarPillarEncoder(grid, cfg.lidar_hidden, cfg.lidar_channels)
    points = torch.rand(50, 3) * torch.tensor([8.0, 8.0, 1.0]) + torch.tensor([0.0, -4.0, 0.0])
    bev = enc(points)
    assert bev.shape == (cfg.lidar_channels, grid.nx, grid.ny)


def test_bev_fuser_shape():
    cfg = FusionConfig()
    grid = cfg.bev_grid()
    fuser = BEVFuser(cfg.n_classes, cfg.lidar_channels, cfg.fuse_hidden, cfg.fuse_channels)
    cam = torch.randn(cfg.n_classes, grid.nx, grid.ny)
    lidar = torch.randn(cfg.lidar_channels, grid.nx, grid.ny)
    # Unbatched (C, nx, ny) and batched (B, C, nx, ny) both round-trip.
    assert fuser(cam, lidar).shape == (cfg.fuse_channels, grid.nx, grid.ny)
    assert fuser(cam[None], lidar[None]).shape == (1, cfg.fuse_channels, grid.nx, grid.ny)


def test_transfuser_preserves_token_shapes():
    cfg = FusionConfig()
    block = TransFuserBlock(cfg.token_dim, cfg.n_heads)
    cam_tokens = torch.randn(2, 5, cfg.token_dim)
    lidar_tokens = torch.randn(2, 7, cfg.token_dim)
    cam_out, lidar_out = block(cam_tokens, lidar_tokens)
    assert cam_out.shape == cam_tokens.shape
    assert lidar_out.shape == lidar_tokens.shape
