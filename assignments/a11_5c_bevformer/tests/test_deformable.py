"""The deformable spatial-cross-attention path: it reduces to bilinear SCA at zero offsets.

With a zero-initialized offset head every deformable sample lands on the reference point and the
softmax weights sum to 1, so the weighted sum is the bilinear sample. Sharing the value and output
projections with the simplified path isolates the offsets from a projection mismatch. A backward
pass confirms the offset head receives gradient (the learned offsets are trainable).
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


def _setup(cfg, seed=0):
    scene = toy.bev_multicam_scene(n_cams=cfg.n_cams, img=cfg.img, stride=cfg.stride,
                                   focal=cfg.f, seed=seed)
    K, E = scene["K"], scene["E"]
    Ks = {f"cam{i}": K for i in range(cfg.n_cams)}
    Es = {f"cam{i}": E[i] for i in range(cfg.n_cams)}
    sizes = {f"cam{i}": (cfg.img, cfg.img) for i in range(cfg.n_cams)}
    rig = CameraRig(Ks, Es, sizes)
    grid = cfg.bev_grid()
    ref = bev_reference_points(grid, cfg.n_heights, cfg.z_min, cfg.z_max)
    uv, valid = project_reference_points(ref, rig, (cfg.img, cfg.img))
    return grid, uv, valid


def test_zero_offset_equals_bilinear():
    torch.manual_seed(0)
    cfg = BEVFormerConfig()
    grid, uv, valid = _setup(cfg)
    nx, ny, C = grid.nx, grid.ny, cfg.dim
    query = torch.randn(nx, ny, C)
    feats = torch.randn(cfg.n_cams, C, cfg.Hf, cfg.Wf)

    simple = SpatialCrossAttention(C, cfg.n_heads, offsets=False)
    deform = SpatialCrossAttention(C, cfg.n_heads, offsets=True, n_points=cfg.n_points)
    # Share the value + output projections so only the (zero) offsets differ.
    deform.value_proj.load_state_dict(simple.value_proj.state_dict())
    deform.out_proj.load_state_dict(simple.out_proj.state_dict())

    o_simple = simple(query, feats, uv, valid)
    o_deform = deform(query, feats, uv, valid)
    assert torch.allclose(o_simple, o_deform, atol=1e-5)


def test_offset_head_receives_gradient():
    torch.manual_seed(0)
    cfg = BEVFormerConfig()
    grid, uv, valid = _setup(cfg)
    nx, ny, C = grid.nx, grid.ny, cfg.dim
    query = torch.randn(nx, ny, C)
    feats = torch.randn(cfg.n_cams, C, cfg.Hf, cfg.Wf)

    deform = SpatialCrossAttention(C, cfg.n_heads, offsets=True, n_points=cfg.n_points)
    out = deform(query, feats, uv, valid)
    out.sum().backward()
    g = deform.offset_head.weight.grad
    assert g is not None and g.abs().sum().item() > 0.0
