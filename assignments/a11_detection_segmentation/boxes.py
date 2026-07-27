"""Box format conversions and generalized IoU. Assignment-local.

Boxes are normalized to [0, 1] image coordinates. Two formats appear: cxcywh (center x,
center y, width, height), which the model predicts and the toy returns, and xyxy (top-left
and bottom-right corners), which the area computations want. The cxcywh -> xyxy conversion
assumes positive width and height, true for sigmoid box outputs and positive-extent ground
truth.

Generalized IoU (Rezatofighi et al., 2019, https://arxiv.org/abs/1902.09630) extends IoU
to a signed score in (-1, 1] that stays informative when the boxes do not overlap: plain
IoU is 0 for any disjoint pair regardless of how far apart they are, so its gradient is flat
and useless for a regression that starts far from the target. GIoU subtracts the fraction of
the smallest enclosing box not covered by the union, which keeps decreasing as the boxes
move apart, so the matching cost and the box loss both use it.
"""

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(b: Tensor) -> Tensor:
    """(cx, cy, w, h) -> (x0, y0, x1, y1), normalized [0, 1]. Last dim is 4."""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(b: Tensor) -> Tensor:
    """(x0, y0, x1, y1) -> (cx, cy, w, h), normalized [0, 1]. Last dim is 4.

    The inverse of box_cxcywh_to_xyxy.
    """
    raise NotImplementedError("convert xyxy corners back to cxcywh")


def generalized_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise GIoU between two sets of cxcywh boxes.

    boxes1 is (N1, 4), boxes2 is (N2, 4), both cxcywh in [0, 1]. Returns (N1, N2) with
    GIoU[i, j] in (-1, 1]. See the generalized IoU section of the README for the definition.

    Correctness contracts: clamp the intersection width/height to >= 0 so disjoint boxes give
    zero intersection, not a spurious negative one; add eps to BOTH denominators (the union and
    the enclosing-box area) so zero-area degenerate boxes do not divide by zero. The result is
    differentiable, so the same function is reused in the box loss.
    """
    raise NotImplementedError("implement pairwise generalized IoU")
