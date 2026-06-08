"""Spatial cross-attention: shape, the geometric hit-mask, the no-hit guard, and gradients.

The hit-mask test constructs a single-view cell rather than hardcoding an index (the toy's FOV
geometry decides which cells are single-view). The no-hit / divide-by-zero path is unit-tested
directly with a synthetic all-False mask, because a 360-degree ring leaves few geometric no-hit
cells. gradcheck confirms gradients flow from the loss back to the image features through
grid_sample.
"""

import torch

from config import BEVFormerConfig

from nanovision.data import toy
from nanovision.geometry import CameraRig
from nanovision.bevformer import (
    bev_reference_points,
    project_reference_points,
    SpatialCrossAttention,
)


def _rig(scene, cfg):
    K, E = scene["K"], scene["E"]
    Ks = {f"cam{i}": K for i in range(cfg.n_cams)}
    Es = {f"cam{i}": E[i] for i in range(cfg.n_cams)}
    sizes = {f"cam{i}": (cfg.img, cfg.img) for i in range(cfg.n_cams)}
    return CameraRig(Ks, Es, sizes)


def _setup(cfg, seed=0):
    scene = toy.bev_multicam_scene(n_cams=cfg.n_cams, img=cfg.img, stride=cfg.stride,
                                   focal=cfg.f, seed=seed)
    rig = _rig(scene, cfg)
    grid = cfg.bev_grid()
    ref = bev_reference_points(grid, cfg.n_heights, cfg.z_min, cfg.z_max)
    uv, valid = project_reference_points(ref, rig, (cfg.img, cfg.img))
    return grid, uv, valid


def test_shape():
    cfg = BEVFormerConfig()
    grid, uv, valid = _setup(cfg)
    nx, ny, C = grid.nx, grid.ny, cfg.dim
    sca = SpatialCrossAttention(C, cfg.n_heads, offsets=False)
    query = torch.randn(nx, ny, C)
    feats = torch.randn(cfg.n_cams, C, cfg.Hf, cfg.Wf)
    out = sca(query, feats, uv, valid)
    assert out.shape == (nx, ny, C)


def test_single_view_cell_pools_the_hit_camera():
    cfg = BEVFormerConfig()
    grid, uv, valid = _setup(cfg)
    nx, ny, C = grid.nx, grid.ny, cfg.dim

    # Each camera's feature map is a distinct constant. With a single-view cell, the pooled value
    # (before projections) must equal that one camera's constant.
    consts = torch.arange(1, cfg.n_cams + 1, dtype=torch.float32)  # 1, 2, 3, 4
    feats = torch.zeros(cfg.n_cams, C, cfg.Hf, cfg.Wf)
    for c in range(cfg.n_cams):
        feats[c] = consts[c]

    # Identity value + output projection so the pooled constant survives, no residual bias.
    sca = SpatialCrossAttention(C, cfg.n_heads, offsets=False)
    with torch.no_grad():
        sca.value_proj.weight.copy_(torch.eye(C)); sca.value_proj.bias.zero_()
        sca.out_proj.weight.copy_(torch.eye(C)); sca.out_proj.bias.zero_()
    query = torch.zeros(nx, ny, C)  # residual add of 0, so out == pooled constant

    hit = valid.any(-1).sum(0)                       # (nx, ny): cameras seeing each cell
    single = (hit == 1).nonzero()
    assert single.numel() > 0, "no single-view cell at this focal"

    # Pick a single-view cell whose valid reference points sit INTERIOR to the image (away from
    # the +-1 border by a margin). On a constant feature map, bilinear sampling returns the
    # constant exactly only when the sample is not interpolating against the zero-padding at the
    # exact image edge; corner cells project to gx/gy = +-1 and would read half the constant.
    chosen = None
    for cand in single.tolist():
        i, j = cand
        c = valid[:, i, j].any(-1).nonzero()[0].item()
        vh = valid[c, i, j]                           # which heights are in-frame for that cam
        coords = uv[c, i, j][vh]                      # (n_valid, 2) the contributing samples
        if coords.abs().max() < 0.95:
            chosen = (i, j, c)
            break
    assert chosen is not None, "no interior single-view cell at this focal"
    i, j, hit_cam = chosen

    out = sca(query, feats, uv, valid)
    assert torch.allclose(out[i, j], consts[hit_cam] * torch.ones(C), atol=1e-4)


def test_no_hit_cell_keeps_query():
    cfg = BEVFormerConfig()
    grid, uv, valid = _setup(cfg)
    nx, ny, C = grid.nx, grid.ny, cfg.dim
    sca = SpatialCrossAttention(C, cfg.n_heads, offsets=False)
    query = torch.randn(nx, ny, C)
    feats = torch.randn(cfg.n_cams, C, cfg.Hf, cfg.Wf)

    # Force cell (0, 0) to be seen by no camera at any height; the output query there is unchanged.
    valid_mod = valid.clone()
    valid_mod[:, 0, 0, :] = False
    out = sca(query, feats, uv, valid_mod)
    assert torch.allclose(out[0, 0], query[0, 0], atol=1e-6)


def test_gradcheck_features():
    cfg = BEVFormerConfig()
    grid, uv, valid = _setup(cfg)
    nx, ny, C = grid.nx, grid.ny, cfg.dim
    sca = SpatialCrossAttention(C, cfg.n_heads, offsets=False).double()
    query = torch.zeros(nx, ny, C, dtype=torch.float64)
    feats = torch.randn(cfg.n_cams, C, cfg.Hf, cfg.Wf, dtype=torch.float64,
                        requires_grad=True)
    uv64 = uv.double()

    def f(x):
        return sca(query, x, uv64, valid).sum()

    assert torch.autograd.gradcheck(f, (feats,), eps=1e-6, atol=1e-4, rtol=1e-3)
