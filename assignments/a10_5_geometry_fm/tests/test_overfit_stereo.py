"""Overfit GeometryFM on a small set of stereo pairs and measure the cross-attention effect.

The training set is several view pairs drawn from the ring, so the relative pose between the
two views VARIES across the batch. That variation is what forces cross-attention to matter: a
single fixed pair lets the network bake one relative pose into its weights and ignore the
other view, which is the centered-sphere degeneracy the toy is built to avoid. With varied
poses, placing view 2's points in view 1's frame needs information from view 1's tokens, so
disabling cross-attention raises the loss floor.

Two assertions:
  1. With cross-attention on, the normalized pointmap error falls below a measured threshold.
  2. Disabling cross-attention (zeroing the cross memory) raises the pointmap-error floor.

Thresholds are set from measurement (see the README's validation section); this is a bounded
training run on CPU, ~2500 Adam steps on 8 tiny pairs.
"""

import torch
from dataclasses import replace

from config import GeometryFMConfig
from model import GeometryFM
from loss import pointmap_loss, normalize_scale
from toy_scene import stereo_pointmap_gt


_PAIRS = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 5), (2, 6), (3, 6), (3, 7)]


def _build_batch(cfg):
    batch = [stereo_pointmap_gt(replace(cfg, view1=a, view2=b)) for a, b in _PAIRS]
    i1 = torch.stack([d["img1"] for d in batch])
    i2 = torch.stack([d["img2"] for d in batch])
    g1 = torch.stack([d["gt_pts1"] for d in batch])
    g2 = torch.stack([d["gt_pts2"] for d in batch])
    v1 = torch.stack([d["valid1"] for d in batch])
    v2 = torch.stack([d["valid2"] for d in batch])
    return i1, i2, g1, g2, v1, v2


def _normalized_pointmap_error(model, batch, use_cross):
    i1, i2, g1, g2, v1, v2 = batch
    with torch.no_grad():
        p1, c1, p2, c2 = model(i1, i2, use_cross=use_cross)
        pred = torch.stack([p1, p2], dim=1)
        gt = torch.stack([g1, g2], dim=1)
        val = torch.stack([v1, v2], dim=1)
        z = normalize_scale(pred, val)
        zb = normalize_scale(gt, val)
        ell = (pred / z.view(-1, 1, 1, 1, 1) - gt / zb.view(-1, 1, 1, 1, 1)).norm(dim=-1)
        m = val.float()
        return float((ell * m).sum() / m.sum())


def _train(cfg, batch, use_cross, steps, seed=0):
    torch.manual_seed(seed)
    model = GeometryFM(cfg)
    i1, i2, g1, g2, v1, v2 = batch
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(steps):
        opt.zero_grad()
        p1, c1, p2, c2 = model(i1, i2, use_cross=use_cross)
        loss = pointmap_loss(p1, p2, g1, g2, c1, c2, v1, v2, cfg.alpha)
        loss.backward()
        opt.step()
    return model


def test_overfit_and_cross_attention_helps():
    cfg = GeometryFMConfig()
    batch = _build_batch(cfg)
    steps = 2500

    model_cross = _train(cfg, batch, use_cross=True, steps=steps, seed=0)
    err_cross = _normalized_pointmap_error(model_cross, batch, use_cross=True)

    model_nocross = _train(cfg, batch, use_cross=False, steps=steps, seed=0)
    err_nocross = _normalized_pointmap_error(model_nocross, batch, use_cross=False)

    # 1. With cross-attention the model fits the pointmaps well. Threshold from measurement.
    assert err_cross < 0.05, f"pointmap error with cross-attention {err_cross}"

    # 2. Disabling cross-attention raises the floor: each view can no longer read the other,
    #    so placing view 2 in cam1 frame under a varying relative pose is harder. Threshold
    #    from measurement (a clear margin, not a hairline).
    assert err_nocross > 1.5 * err_cross, (
        f"cross-attention did not lower the floor: with={err_cross} without={err_nocross}"
    )
