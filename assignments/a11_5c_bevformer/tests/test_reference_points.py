"""Pillar construction and projection geometry for the BEV reference points.

The reference pillars are the anchor 3D points the BEV query later samples features at. These
checks pin the pillar shape and z-values, and confirm the projection chain agrees with the
camera-geometry primitives (a point on a camera's optical axis lands at that camera's principal
point and is marked in-frame; a camera facing the other way does not see it).
"""

import torch

from config import BEVFormerConfig

from nanovision.data import toy
from nanovision.geometry import CameraRig
from nanovision.bevformer import bev_reference_points, project_reference_points


def _rig(scene, cfg):
    K, E = scene["K"], scene["E"]
    Ks = {f"cam{i}": K for i in range(cfg.n_cams)}
    Es = {f"cam{i}": E[i] for i in range(cfg.n_cams)}
    sizes = {f"cam{i}": (cfg.img, cfg.img) for i in range(cfg.n_cams)}
    return CameraRig(Ks, Es, sizes)


def test_pillar_shape_and_heights():
    cfg = BEVFormerConfig()
    grid = cfg.bev_grid()
    ref = bev_reference_points(grid, cfg.n_heights, cfg.z_min, cfg.z_max)
    assert ref.shape == (16, 16, cfg.n_heights, 3)

    zs_expected = torch.linspace(cfg.z_min, cfg.z_max, cfg.n_heights)
    # Every pillar carries exactly those z-values.
    assert torch.allclose(ref[..., 2], zs_expected.view(1, 1, -1).expand(16, 16, -1))

    # The (x, y) of all heights in a pillar match the BEV cell center.
    centers = grid.cell_centers()                       # (nx, ny, 2)
    xy = ref[..., :2]                                    # (nx, ny, n_heights, 2)
    assert torch.allclose(xy, centers[:, :, None, :].expand_as(xy))


def test_projection_to_principal_point_and_front_back():
    cfg = BEVFormerConfig()
    scene = toy.bev_multicam_scene(n_cams=cfg.n_cams, img=cfg.img, stride=cfg.stride,
                                   focal=cfg.f, seed=0)
    rig = _rig(scene, cfg)

    # cam0 looks along ego +x (front camera). A point on its optical axis at depth 5 m is at
    # ego (5, 0, cam_height): in front of cam0, projecting to its principal point.
    depth = 5.0
    pt = torch.tensor([[depth, 0.0, cfg.cam_height]])    # (1, 3) ego
    ref = pt.reshape(1, 1, 1, 3)                         # (nx=1, ny=1, nh=1, 3)
    uv, valid = project_reference_points(ref, rig, (cfg.img, cfg.img))

    # cam0 sees it at the principal point: normalized grid coord ~ (0, 0) (align_corners=False
    # maps the image center (cx+0.5, cy+0.5) = (img/2, img/2) to gx=gy=0 for cx=cy=(img-1)/2).
    assert valid[0, 0, 0, 0].item() is True
    assert torch.allclose(uv[0, 0, 0, 0], torch.zeros(2), atol=1e-4)

    # cam2 faces ego -x (back camera, n_cams=4): the front point is behind it, not in-frame.
    assert valid[2, 0, 0, 0].item() is False


def test_projection_round_trip():
    cfg = BEVFormerConfig()
    scene = toy.bev_multicam_scene(n_cams=cfg.n_cams, img=cfg.img, stride=cfg.stride,
                                   focal=cfg.f, seed=1)
    rig = _rig(scene, cfg)
    grid = cfg.bev_grid()
    ref = bev_reference_points(grid, cfg.n_heights, cfg.z_min, cfg.z_max)
    uv, valid = project_reference_points(ref, rig, (cfg.img, cfg.img))

    # For an in-frame point, undo the align_corners=False normalization and compare to a direct
    # world_to_pixel call. uv[..., 0] = 2*(u + 0.5)/W - 1  =>  u = (uv0 + 1)/2 * W - 0.5.
    W = H = cfg.img
    pts = ref.reshape(-1, 3)
    for ci, name in enumerate(rig.names):
        px, val = rig.world_to_pixel(name, pts)
        v_in = val.reshape(grid.nx, grid.ny, cfg.n_heights)
        u_back = (uv[ci, ..., 0] + 1.0) / 2.0 * W - 0.5
        v_back = (uv[ci, ..., 1] + 1.0) / 2.0 * H - 0.5
        uv_back = torch.stack([u_back, v_back], dim=-1).reshape(-1, 2)
        m = val
        assert torch.allclose(uv_back[m], px[m], atol=1e-4)
        # And the projection masks agree.
        assert torch.equal(v_in, valid[ci])
