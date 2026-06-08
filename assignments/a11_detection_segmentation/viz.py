"""Overfit the toy detection set and draw the result. Provided, not graded.

Trains the DETR model on the 4-image colored-squares toy, then for each image draws the
ground-truth boxes and the matched predicted boxes side by side and annotates which query
slot was matched to which ground-truth object. Writes figures to out/.

Run from the repo root with the solution filled in (the ViT backbone and the A11 holes must
be implemented):
    NANOVISION_IMPL=solution python -m assignments.a11_detection_segmentation.viz

Uses the GPU when present (default_device); tensors that feed matplotlib are moved to CPU.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import DETRConfig  # noqa: E402
from loss import detr_loss  # noqa: E402
from matcher import HungarianMatcher  # noqa: E402
from model import DETR  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device, set_seed  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _draw_box(ax, box_cxcywh, size, color, label=None, ls="-"):
    cx, cy, w, h = [v * size for v in box_cxcywh]
    rect = patches.Rectangle(
        (cx - w / 2, cy - h / 2), w, h, linewidth=2, edgecolor=color,
        facecolor="none", linestyle=ls,
    )
    ax.add_patch(rect)
    if label is not None:
        ax.text(cx - w / 2, cy - h / 2 - 1, label, color=color, fontsize=8)


def main():
    set_seed(0)
    dev = default_device()
    cfg = DETRConfig()
    model = DETR(cfg).to(dev)

    img, boxes, labels, mask = toy.detection_batch(
        batch=cfg.batch, num_classes=cfg.num_classes, max_objects=cfg.max_objects, seed=0
    )
    img, boxes, labels, mask = img.to(dev), boxes.to(dev), labels.to(dev), mask.to(dev)
    gt_labels = [labels[b][mask[b]] for b in range(cfg.batch)]
    gt_boxes = [boxes[b][mask[b]] for b in range(cfg.batch)]

    matcher = HungarianMatcher(cfg.cost_class, cfg.cost_l1, cfg.cost_giou)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(600):
        logits, pred_boxes = model(img)
        indices = matcher.forward(logits, pred_boxes, gt_labels, gt_boxes)
        total, comps = detr_loss(
            logits, pred_boxes, gt_labels, gt_boxes, indices, cfg.num_classes,
            eos_coef=cfg.eos_coef, weight_class=cfg.loss_class,
            weight_l1=cfg.loss_l1, weight_giou=cfg.loss_giou,
        )
        opt.zero_grad()
        total.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:4d}  loss {total.item():.4f}  "
                  f"cls {comps['class'].item():.4f} l1 {comps['l1'].item():.4f} "
                  f"giou {comps['giou'].item():.4f}")

    model.eval()
    with torch.no_grad():
        logits, pred_boxes = model(img)
        indices = matcher.forward(logits, pred_boxes, gt_labels, gt_boxes)

    palette = ["red", "lime", "blue", "yellow", "magenta", "cyan"]
    fig, axes = plt.subplots(1, cfg.batch, figsize=(3 * cfg.batch, 3.2))
    img_cpu = img.cpu()
    pred_cpu = pred_boxes.cpu()
    for b in range(cfg.batch):
        ax = axes[b]
        ax.imshow(img_cpu[b].permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        row, col = indices[b]
        for r, c in zip(row.tolist(), col.tolist()):
            cls = int(gt_labels[b][c])
            gtb = gt_boxes[b][c].cpu().tolist()
            prb = pred_cpu[b, r].tolist()
            _draw_box(ax, gtb, cfg.img_size, "white", label=f"gt c{cls}", ls="--")
            _draw_box(ax, prb, cfg.img_size, palette[cls % len(palette)],
                      label=f"q{r}", ls="-")
        ax.set_title(f"image {b}", fontsize=9)
    fig.suptitle("dashed white = ground truth, solid color = matched query prediction")
    fig.tight_layout()
    out = _OUT / "detection_overfit.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
