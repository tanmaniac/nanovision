"""Overfit the assembled BEVFormer encoder + segmentation head on one multi-cam toy frame.

This shows the query-pull pipeline composes end to end and is differentiable: a small backbone,
the projected reference pillars, the spatial cross-attention, and the BEV head together route each
vehicle's image blob to the correct BEV cell and segment it. With one fixed scene the network
overfits, so this checks the mechanism composes, not generalization. Simplified spatial
cross-attention, no temporal.
"""

import torch

from config import BEVFormerConfig

from nanovision.data import toy
from nanovision.geometry import CameraRig
from nanovision.bevformer import BEVFormerSeg


def _iou(pred_bool, gt_bool):
    inter = (pred_bool & gt_bool).sum().float()
    union = (pred_bool | gt_bool).sum().float()
    return (inter / union.clamp(min=1)).item()


def test_overfit_bev_segmentation():
    torch.manual_seed(0)
    cfg = BEVFormerConfig()
    scene = toy.bev_multicam_scene(n_cams=cfg.n_cams, n_vehicles=4, img=cfg.img,
                                   stride=cfg.stride, focal=cfg.f, seed=0)
    K, E = scene["K"], scene["E"]
    Ks = {f"cam{i}": K for i in range(cfg.n_cams)}
    Es = {f"cam{i}": E[i] for i in range(cfg.n_cams)}
    sizes = {f"cam{i}": (cfg.img, cfg.img) for i in range(cfg.n_cams)}
    rig = CameraRig(Ks, Es, sizes)

    imgs = scene["images"][0]                            # (n_cam, 3, 32, 32)
    gt = scene["bev_gt"][0]                              # (nx, ny)

    model = BEVFormerSeg(cfg)
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, cfg.dim, 3, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(cfg.dim, cfg.dim, 3, stride=2, padding=1), torch.nn.ReLU(),
    )
    params = list(model.parameters()) + list(backbone.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    first = None
    for _ in range(1500):
        feats = backbone(imgs)                           # (n_cam, C, Hf, Wf)
        logit = model(feats, rig)                        # (1, nx, ny)
        loss = loss_fn(logit[0], gt)
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()

    with torch.no_grad():
        pred = model(backbone(imgs), rig)[0] > 0.0       # threshold the logit at 0
    iou = _iou(pred, gt > 0.5)

    assert final < 0.05, f"final BCE {final} (start {first})"
    assert iou > 0.6, f"BEV IoU {iou} (final BCE {final})"
