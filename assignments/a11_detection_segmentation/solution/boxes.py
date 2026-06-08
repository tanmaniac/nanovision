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
    """(x0, y0, x1, y1) -> (cx, cy, w, h), normalized [0, 1]. Last dim is 4."""
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0], dim=-1)


def generalized_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise GIoU between two sets of cxcywh boxes.

    boxes1 is (N1, 4), boxes2 is (N2, 4), both cxcywh in [0, 1]. Returns (N1, N2) with
    GIoU[i, j] in (-1, 1]. GIoU = IoU - |C \\ (A u B)| / |C|, where C is the smallest box
    enclosing A and B. The intersection width/height is clamped to >= 0 so disjoint boxes
    contribute zero intersection area, not a spurious negative one, and eps is added to both
    denominators (the IoU union and |C|) so zero-area degenerate boxes do not divide by zero.
    Differentiable, so the same function is reused in the box loss.
    """
    b1 = box_cxcywh_to_xyxy(boxes1)        # (N1, 4)
    b2 = box_cxcywh_to_xyxy(boxes2)        # (N2, 4)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])   # (N1,)
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])   # (N2,)

    # Intersection box (broadcast to (N1, N2, 2) corners).
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])        # top-left
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])        # bottom-right
    wh = (rb - lt).clamp(min=0)                             # clamp: no negative overlap
    inter = wh[..., 0] * wh[..., 1]                         # (N1, N2)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + eps)

    # Smallest enclosing box C.
    lt_c = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb_c = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]                    # (N1, N2)

    return iou - (area_c - union) / (area_c + eps)
