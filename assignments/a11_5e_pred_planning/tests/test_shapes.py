"""Output shapes of roi_align_bev and the multimodal trajectory head."""

import torch

from config import PredConfig
from predict import MultimodalTrajectoryHead, roi_align_bev

from nanovision.data import toy


def test_roi_align_shape():
    cfg = PredConfig()
    scene = toy.pred_toy_scene(channels=cfg.in_ch, horizon=cfg.horizon, seed=0)
    tokens = roi_align_bev(scene["bev_feat"], scene["centers"], cfg.roi_size, cfg.radius)
    N = scene["centers"].shape[0]
    assert tokens.shape == (N, cfg.roi_size ** 2, cfg.in_ch)


def test_head_shape():
    torch.manual_seed(0)
    cfg = PredConfig()
    scene = toy.pred_toy_scene(channels=cfg.in_ch, horizon=cfg.horizon, seed=0)
    head = MultimodalTrajectoryHead(
        cfg.in_ch, dim=cfg.dim, n_modes=cfg.n_modes, horizon=cfg.horizon,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads, roi_size=cfg.roi_size, radius=cfg.radius,
    )
    trajs, scores = head(scene["bev_feat"], scene["centers"])
    N = scene["centers"].shape[0]
    assert trajs.shape == (N, cfg.n_modes, cfg.horizon, 2)
    assert scores.shape == (N, cfg.n_modes)
