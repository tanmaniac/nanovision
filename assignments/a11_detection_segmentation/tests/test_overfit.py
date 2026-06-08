"""Overfit the 4-image toy detection set: the DETR mechanism runs end to end.

This proves the matcher + loss + model overfit a tiny fixed set. It proves nothing about
convergence speed or accuracy at real scale (see the README's toy-scope disclaimer). The
total loss falls from ~4.8 to a floor near 0.1-0.25, dominated by the GIoU residual that the
4-pixel patch stride leaves on box localization; the matched predicted box centers land on
the ground-truth centers to well under 0.06 of the image size.

Measured at seed 0, 500 Adam steps (lr 1e-3), CPU, ~15s: first ~4.8, final ~0.11, matched
center error (max coordinate) ~0.008. Thresholds are set comfortably above those floors so
the test does not thrash: final < 0.5 and < 0.1*first, center error < 0.06.
"""

import torch

from config import DETRConfig
from loss import detr_loss
from matcher import HungarianMatcher
from model import DETR
from nanovision.data import toy
from nanovision.determinism import set_seed


def test_overfit_toy_detection():
    set_seed(0)
    cfg = DETRConfig()
    model = DETR(cfg)
    img, boxes, labels, mask = toy.detection_batch(
        batch=cfg.batch, num_classes=cfg.num_classes, max_objects=cfg.max_objects, seed=0
    )
    gt_labels = [labels[b][mask[b]] for b in range(cfg.batch)]
    gt_boxes = [boxes[b][mask[b]] for b in range(cfg.batch)]

    matcher = HungarianMatcher(cfg.cost_class, cfg.cost_l1, cfg.cost_giou)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    first = None
    for _ in range(500):
        logits, pred_boxes = model(img)
        indices = matcher.forward(logits, pred_boxes, gt_labels, gt_boxes)
        total, _ = detr_loss(
            logits, pred_boxes, gt_labels, gt_boxes, indices, cfg.num_classes,
            eos_coef=cfg.eos_coef, weight_class=cfg.loss_class,
            weight_l1=cfg.loss_l1, weight_giou=cfg.loss_giou,
        )
        if first is None:
            first = total.item()
        opt.zero_grad()
        total.backward()
        opt.step()

    final = total.item()
    assert final < 0.5, f"overfit loss floored at {final:.3f}, expected < 0.5"
    assert final < 0.1 * first, f"loss only fell from {first:.3f} to {final:.3f}"

    # Matched predicted box centers sit on the ground-truth centers.
    model.eval()
    with torch.no_grad():
        logits, pred_boxes = model(img)
        indices = matcher.forward(logits, pred_boxes, gt_labels, gt_boxes)
        max_center_err = 0.0
        for b, (row, col) in enumerate(indices):
            for r, c in zip(row, col):
                err = (pred_boxes[b, r, :2] - gt_boxes[b][c, :2]).abs().max().item()
                max_center_err = max(max_center_err, err)
    assert max_center_err < 0.06, f"matched center error {max_center_err:.4f} >= 0.06"
