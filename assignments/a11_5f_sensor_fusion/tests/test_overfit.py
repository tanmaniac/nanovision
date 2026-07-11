"""Overfit the fused BEV segmentation on one scene.

This shows the fusion pipeline composes end to end and is differentiable: the LiDAR pillar
encoder, the BEV fuser, and a seg head together route the vehicle cells to the right BEV pillars
on a single scene. It is a composition/optimization check, not the fusion-beats-single claim
(which needs held-out scenes; see test_fusion_beats_single).
"""

import torch
import torch.nn as nn

from config import FusionConfig
from fusion import BEVFuser, LidarPillarEncoder

from nanovision.data import toy


def _iou(pred_bool, gt_bool):
    inter = (pred_bool & gt_bool).sum().float()
    union = (pred_bool | gt_bool).sum().float()
    return (inter / union.clamp(min=1)).item()


def test_overfit_fused_bev_segmentation():
    torch.manual_seed(0)
    cfg = FusionConfig()
    grid = cfg.bev_grid()
    scene = toy.bev_fusion_scene(seed=0)
    cam_bev = scene["cam_bev"]
    lidar = scene["lidar"]
    gt = scene["bev_gt"][None, None]

    enc = LidarPillarEncoder(grid, cfg.lidar_hidden, cfg.lidar_channels)
    fuser = BEVFuser(cfg.n_classes, cfg.lidar_channels, cfg.fuse_hidden, cfg.fuse_channels)
    head = nn.Conv2d(cfg.fuse_channels, 1, kernel_size=1)
    params = list(enc.parameters()) + list(fuser.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    first = None
    for _ in range(600):
        logit = head(fuser(cam_bev, enc(lidar))[None])
        loss = loss_fn(logit, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()

    with torch.no_grad():
        pred = head(fuser(cam_bev, enc(lidar))[None]) > 0.0
    iou = _iou(pred, gt > 0.5)

    assert final < 0.05, f"final BCE {final} (start {first})"
    assert iou > 0.8, f"fused BEV IoU {iou} (final BCE {final})"
