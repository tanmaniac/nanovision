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
        """
        B = pred_logits.shape[0]
        out: list[tuple[Tensor, Tensor]] = []
        for b in range(B):
            labels_b = gt_labels[b]
            boxes_b = gt_boxes[b]
            M = labels_b.shape[0]
            if M == 0:
                empty = torch.zeros(0, dtype=torch.long)
                out.append((empty, empty))
                continue

            prob = pred_logits[b].softmax(-1)          # (N, C+1)
            # Class cost: negative probability of each gt label, gathered over queries.
            cost_class = -prob[:, labels_b]            # (N, M)
            # L1 cost over cxcywh box coordinates.
            cost_l1 = torch.cdist(pred_boxes[b], boxes_b, p=1)   # (N, M)
            # GIoU cost (negative, since higher GIoU is better).
            cost_giou = -generalized_iou(pred_boxes[b], boxes_b)  # (N, M)

            C = self.cost_class * cost_class + self.cost_l1 * cost_l1 + self.cost_giou * cost_giou
            assert torch.isfinite(C).all(), "cost matrix has non-finite entries"

            row, col = linear_sum_assignment(C.cpu().numpy())
            out.append((
                torch.as_tensor(row, dtype=torch.long),
                torch.as_tensor(col, dtype=torch.long),
            ))
        return out
