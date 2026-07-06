"""Hungarian bipartite matcher (the COST side of DETR). Assignment-local.

The matcher decides which predicted query slot is compared to which ground-truth object. It
builds a per-image cost matrix and solves the one-to-one assignment that minimizes total
cost with the Hungarian algorithm. This is the COST, not the loss: it is a fixed,
non-differentiable function computed in numpy. The returned indices are constants in the
backward pass - no gradient flows through the matcher. The loss (loss.py) is the
differentiable objective computed AFTER these indices are fixed.

The per-pair cost sums a class term, an L1 box term, and a GIoU term with fixed weights; see
the bipartite-matching section of the README for the cost. The class term uses the raw
predicted probability of the true class (higher probability lowers the cost) while the loss
uses cross-entropy; L1 and GIoU are the same function in both the cost and the loss.
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

        Build the per-image (N, M) cost matrix from the three weighted terms (see the
        bipartite-matching section of the README) and solve the one-to-one assignment with the
        Hungarian algorithm. An image with no ground-truth objects (M_b == 0) returns two empty
        long tensors.

        Contract: assert the cost matrix is finite before handing it to the solver (a NaN or
        inf silently corrupts the assignment).
        """
        raise NotImplementedError("build the cost matrix and run the Hungarian assignment")
