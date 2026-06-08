"""Mechanism D: overfit the full Lift-Splat-Shoot model on one toy BEV scene.

This shows the LSS pipeline composes end to end and is differentiable: the backbone, the
outer-product lift, the frustum geometry, the sort+cumsum splat, and the BEV head together can
route a vehicle's image blob to the correct BEV pillar and segment it.

Honest limitation. With one camera and one fixed scene, no two objects share a pixel at
different depths, so depth is identifiable but only trivially - the network memorizes one depth
distribution per pixel and never has to RESOLVE depth ambiguity. The frustum geometry is still
exercised (each depth bin along a ray maps to a DISTINCT pillar, and ix increases monotonically
with depth, so a wrong depth lands the feature in a wrong pillar and the BCE penalizes it), so
this is a real test that the mechanism composes, NOT evidence that implicit depth is learned in
any non-trivial sense. The README states the same.
"""

import torch

from config import LSSConfig

from nanovision.data import toy
from nanovision.lift_splat import LiftSplatShoot


def _iou(pred_bool, gt_bool):
    inter = (pred_bool & gt_bool).sum().float()
    union = (pred_bool | gt_bool).sum().float()
    return (inter / union.clamp(min=1)).item()


def test_overfit_bev_segmentation():
    torch.manual_seed(0)
    cfg = LSSConfig()
    scene = toy.bev_toy_scene(
        n_vehicles=3, img=cfg.img, stride=cfg.stride,
        d_min=cfg.d_min, d_max=cfg.d_max, d_step=cfg.d_step,
        focal=cfg.f, cam_height=cfg.cam_height, seed=0,
    )
    image = scene["image"]                       # (1, 3, 32, 32)
    K, E = scene["K"], scene["E"]
    gt = scene["bev_gt"][None, None]             # (1, 1, nx, ny)

    model = LiftSplatShoot(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    first = None
    for _ in range(1500):
        logit = model(image, K, E)               # (1, 1, nx, ny)
        loss = loss_fn(logit, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()

    with torch.no_grad():
        pred = model(image, K, E) > 0.0          # threshold the LOGIT at 0 (prob 0.5)
    iou = _iou(pred, gt > 0.5)

    assert final < 0.05, f"final BCE {final} (start {first})"
    assert iou > 0.6, f"BEV IoU {iou} (final BCE {final})"
