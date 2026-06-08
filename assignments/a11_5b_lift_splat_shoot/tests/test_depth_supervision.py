"""BEVDepth: explicit depth supervision with a cross-entropy on the GT depth bin.

The label at a vehicle pixel is the nearest depth bin center, argmin_k |z_cam - bins()[k]|,
computed with the same bins() tensor the model uses (the toy generator already does this). The
label is only valid when z_cam lies in [d_min, d_max], which the toy guarantees by clamping
vehicle depth to the reachable range. Overfitting the depth logits with bevdepth_depth_loss
drives the per-cell argmax to the labeled bin.
"""

import torch

from config import LSSConfig

from nanovision.data import toy
from nanovision.lift_splat import DepthLift, bevdepth_depth_loss


def test_no_label_is_zero_loss():
    cfg = LSSConfig()
    logits = torch.randn(1, cfg.D, cfg.Hf, cfg.Wf)
    labels = torch.zeros(1, cfg.Hf, cfg.Wf, dtype=torch.long)
    mask = torch.zeros(1, cfg.Hf, cfg.Wf, dtype=torch.bool)
    loss = bevdepth_depth_loss(logits, labels, mask)
    assert loss.item() == 0.0


def test_single_cell_gradient_points_at_bin():
    # With one labeled cell, the gradient lowers the loss by raising the target-bin logit.
    cfg = LSSConfig()
    logits = torch.zeros(1, cfg.D, cfg.Hf, cfg.Wf, requires_grad=True)
    labels = torch.zeros(1, cfg.Hf, cfg.Wf, dtype=torch.long)
    mask = torch.zeros(1, cfg.Hf, cfg.Wf, dtype=torch.bool)
    labels[0, 0, 0] = 3
    mask[0, 0, 0] = True
    loss = bevdepth_depth_loss(logits, labels, mask)
    loss.backward()
    g = logits.grad[0, :, 0, 0]
    # The target bin's logit should be pushed UP (negative grad), others down.
    assert g[3] < 0
    assert (g[torch.arange(cfg.D) != 3] > 0).all()


def test_overfit_depth_head():
    torch.manual_seed(0)
    cfg = LSSConfig()
    scene = toy.bev_toy_scene(
        n_vehicles=3, img=cfg.img, stride=cfg.stride,
        d_min=cfg.d_min, d_max=cfg.d_max, d_step=cfg.d_step,
        focal=cfg.f, cam_height=cfg.cam_height, seed=0,
    )
    image = scene["image"]
    labels = scene["depth_bin_labels"][None]     # (1, Hf, Wf)
    mask = scene["depth_mask"][None]             # (1, Hf, Wf)

    # A tiny backbone + the real DepthLift; overfit the depth logits path.
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, cfg.c_backbone, 3, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(cfg.c_backbone, cfg.c_backbone, 3, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(cfg.c_backbone, cfg.c_backbone, 3, padding=1), torch.nn.ReLU(),
    )
    lift = DepthLift(c_in=cfg.c_backbone, D=cfg.D, C_ctx=cfg.c_ctx)
    params = list(backbone.parameters()) + list(lift.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)

    for _ in range(800):
        feat = backbone(image)
        depth_logits, _ = lift(feat)             # (1, D, Hf, Wf)
        loss = bevdepth_depth_loss(depth_logits, labels, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = loss.item()

    with torch.no_grad():
        feat = backbone(image)
        depth_logits, _ = lift(feat)
        pred = depth_logits.argmax(dim=1)        # (1, Hf, Wf)
    acc = (pred[mask] == labels[mask]).float().mean().item()

    assert final < 0.05, f"final depth CE {final}"
    assert acc > 0.95, f"argmax accuracy {acc} (final CE {final})"
