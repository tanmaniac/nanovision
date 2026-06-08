"""The differentiable DETR training loss (the LOSS side). Assignment-local.

Computed AFTER the matcher fixes the assignment indices. The indices are treated as
constants here; no gradient flows back through the matching. Three terms, summed with the
DETR weights class 1, L1 5, GIoU 2:

  (a) Classification: cross-entropy over ALL N queries. A matched query's target is its gt
      class; an unmatched query's target is the no-object class (index num_classes). The
      no-object class is downweighted by eos_coef = 0.1 through a length-(C+1) CE weight
      vector, weight[num_classes] = eos_coef and 1.0 for the real classes. This is the
      class imbalance fix, not a post-hoc scaling of unmatched queries: most of the N=10
      queries match nothing on a 1-2 object image, so without the downweight the no-object
      term dominates and the model collapses to predicting no-object everywhere.
  (b) Box L1 on the matched (query, gt) pairs, cxcywh.
  (c) GIoU loss 1 - GIoU on the matched pairs.

The class term differs from the matching cost: the cost uses -p[c] (raw probability), the
loss uses cross-entropy -log p[c]. Same information, different function. L1 and GIoU are the
same function used in both.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from boxes import generalized_iou


def detr_loss(
    pred_logits: Tensor,   # (B, N, C+1)
    pred_boxes: Tensor,    # (B, N, 4) cxcywh
    gt_labels: list[Tensor],   # B tensors (M_b,)
    gt_boxes: list[Tensor],    # B tensors (M_b, 4)
    indices: list[tuple[Tensor, Tensor]],   # per-image (row_idx, col_idx) from the matcher
    num_classes: int,
    *,
    eos_coef: float = 0.1,
    weight_class: float = 1.0,
    weight_l1: float = 5.0,
    weight_giou: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return (total_loss, components). total_loss is a scalar with requires_grad.

    Steps:
      1. target_classes: a (B, N) long tensor filled with num_classes (the no-object class).
         For each image b with non-empty indices (row, col), set
         target_classes[b, row] = gt_labels[b][col].
      2. ce_weight: a length-(C+1) vector of ones (matching pred_logits.dtype) with
         ce_weight[num_classes] = eos_coef.
         loss_class = F.cross_entropy(pred_logits.reshape(B*N, C+1),
                                      target_classes.reshape(B*N), weight=ce_weight).
      3. Gather the matched predicted and gt boxes across images (pred_boxes[b, row] vs
         gt_boxes[b][col]); concatenate to mp, mg of shape (sum_M, 4).
         loss_l1 = F.l1_loss(mp, mg). loss_giou = (1 - generalized_iou(mp, mg).diagonal())
         .mean() (the matched pairs are the diagonal of the pairwise GIoU). If there are no
         matched pairs, use a zero that still depends on pred_boxes (e.g. pred_boxes.sum()*0)
         so the graph stays connected.
      4. total = weight_class*loss_class + weight_l1*loss_l1 + weight_giou*loss_giou.
      5. Return total and {"class": loss_class, "l1": loss_l1, "giou": loss_giou}.
    """
    raise NotImplementedError("implement the DETR set-prediction loss")
