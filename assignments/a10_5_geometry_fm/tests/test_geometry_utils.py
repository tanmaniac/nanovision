"""depth_to_pointmap / pointmap_to_depth round-trip, reprojection consistency, gradchecks.

These pin the shared pointmap utilities. The round-trip recovers the depth a pointmap was
built from; reprojection consistency checks that a cam1-frame pointmap, carried into cam2 and
projected, lands at the pixels cam2 observes (the toy's patch centers); gradcheck confirms the
ops are differentiable end to end.
"""

import torch

from toy_scene import stereo_pointmap_gt
from config import GeometryFMConfig

from nanovision.geometry import depth_to_pointmap, pointmap_to_depth, reproject_pointmap


def _patch_centers(cfg):
    grid = cfg.img_size // cfg.patch
    cen = (torch.arange(grid, dtype=torch.float64) + 0.5) * cfg.patch - 0.5
    vs, us = torch.meshgrid(cen, cen, indexing="ij")
    return torch.stack([us, vs], dim=-1)  # (grid, grid, 2)


def test_depth_pointmap_roundtrip():
    torch.manual_seed(0)
    B, H, W = 2, 16, 16
    f = 16.0
    K = torch.tensor([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]], dtype=torch.float64)
    depth = 1.0 + torch.rand(B, H, W, dtype=torch.float64)
    pts = depth_to_pointmap(depth, K)
    assert pts.shape == (B, H, W, 3)
    rt = pointmap_to_depth(pts)
    assert torch.allclose(rt, depth, atol=1e-10)


def test_depth_to_pointmap_pixel_convention():
    # A point built at depth d for pixel (i, j) must project back to (u, v) = (j, i).
    H, W = 16, 16
    f = 16.0
    K = torch.tensor([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]], dtype=torch.float64)
    depth = 2.0 * torch.ones(1, H, W, dtype=torch.float64)
    pts = depth_to_pointmap(depth, K)
    from nanovision.geometry import project_points
    px = project_points(pts, K)[0]  # (H, W, 2)
    i, j = 3, 7
    assert torch.allclose(px[i, j], torch.tensor([float(j), float(i)], dtype=torch.float64), atol=1e-9)


def test_reproject_consistency_on_toy():
    # The view-2 GT pointmap (in cam1 frame) reprojected into image 2 lands at view-2's
    # own patch centers, within a sub-pixel tolerance.
    cfg = GeometryFMConfig()
    d = stereo_pointmap_gt(cfg)
    pts2 = d["gt_pts2"].to(torch.float64).unsqueeze(0)       # (1, grid, grid, 3) cam1 frame
    T_1to2 = d["T_1to2"].to(torch.float64)
    K = d["K"].to(torch.float64)
    px = reproject_pointmap(pts2, T_1to2, K)[0]             # (grid, grid, 2)
    centers = _patch_centers(cfg)                           # (grid, grid, 2)
    valid = d["valid2"]
    err = (px - centers).norm(dim=-1)[valid]
    assert err.max() < 1e-3, f"max reprojection error {err.max().item()}"


def test_gradcheck_depth_to_pointmap():
    H, W = 4, 4
    f = 4.0
    K = torch.tensor([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]], dtype=torch.float64)
    depth = (1.0 + torch.rand(1, H, W, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda dd: depth_to_pointmap(dd, K), (depth,))


def test_gradcheck_reproject_pointmap():
    H, W = 4, 4
    f = 4.0
    K = torch.tensor([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]], dtype=torch.float64)
    pts = torch.randn(1, H, W, 3, dtype=torch.float64)
    pts[..., 2] = 2.0 + 0.1 * torch.randn(1, H, W, dtype=torch.float64)  # keep z > 0
    pts = pts.detach().requires_grad_(True)
    T = torch.eye(4, dtype=torch.float64)
    T[:3, 3] = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda pp: reproject_pointmap(pp, T, K), (pts,))
