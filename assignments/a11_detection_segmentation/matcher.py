"""Hungarian bipartite matcher (the COST side of DETR). Assignment-local.

The matcher decides which predicted query slot is compared to which ground-truth object. It
builds a per-image cost matrix and solves the one-to-one assignment that minimizes total
cost with the Hungarian algorithm (scipy.optimize.linear_sum_assignment). This is the COST,
not the loss: it is a fixed, non-differentiable function computed in numpy. The returned
indices are constants in the backward pass - no gradient flows through the matcher. The loss
(loss.py) is the differentiable objective computed AFTER these indices are fixed.

Cost for query i against ground-truth object j:
    C[i, j] = cost_class * (-softmax(logits_i)[class_j])
            + cost_l1    * ||box_i - box_j||_1          (cxcywh)
            + cost_giou  * (-GIoU(box_i, box_j))
with the DETR weights class 1, L1 5, GIoU 2. The class term uses the raw predicted
probability of the true class (higher probability lowers the cost); the loss instead uses
cross-entropy. L1 and GIoU are the same function in both the cost and the loss.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from boxes import generalized_iou


class HungarianMatcher:
    """One-to-one bipartite matching between queries and ground-truth objects."""

    def __init__(self, cost_class: float = 1.0, cost_l1: float = 5.0, cost_giou: float = 2.0):
        self.cost_class = cost_class
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,   # (B, N, C+1)
        pred_boxes: Tensor,    # (B, N, 4) cxcywh
        gt_labels: list[Tensor],   # B tensors of shape (M_b,)
        gt_boxes: list[Tensor],    # B tensors of shape (M_b, 4) cxcywh
    ) -> list[tuple[Tensor, Tensor]]:
        """Return per-image (row_idx, col_idx): query rows matched to gt columns.

        row_idx/col_idx are length min(N, M_b) = M_b here (N >= M). The indices carry no
        gradient (this whole call runs under no_grad and goes through numpy).

        For each image b:
          1. If M_b == 0, append two empty long tensors and continue.
          2. prob = softmax(pred_logits[b], dim=-1), shape (N, C+1).
          3. cost_class = -prob[:, gt_labels[b]], shape (N, M): negative probability of each
             gt's true class, per query.
          4. cost_l1 = torch.cdist(pred_boxes[b], gt_boxes[b], p=1), shape (N, M).
          5. cost_giou = -generalized_iou(pred_boxes[b], gt_boxes[b]), shape (N, M).
          6. C = cost_class*self.cost_class + cost_l1*self.cost_l1 + cost_giou*self.cost_giou.
          7. assert torch.isfinite(C).all() before handing C to scipy.
          8. row, col = linear_sum_assignment(C.cpu().numpy()); return them as long tensors.
        """
        raise NotImplementedError("build the cost matrix and run the Hungarian assignment")
