"""The differentiable DETR training loss (the LOSS side). Assignment-local.

Computed AFTER the matcher fixes the assignment indices. The indices are treated as
constants here; no gradient flows back through the matching. The loss sums three terms (a
classification cross-entropy over all queries, a box L1, and a GIoU loss) with fixed weights.
See the set-prediction loss section of the README for the terms and weights.

The no-object class is downweighted (eos_coef) to counter class imbalance: on a 1-2 object
image most of the N queries match nothing, so without the downweight the no-object term
dominates and the model collapses to predicting no-object everywhere. This is a class-imbalance
fix, not a post-hoc scaling of unmatched queries.

The class term differs from the matching cost: the cost uses raw probability, the loss uses
cross-entropy. Same information, different function. L1 and GIoU are the same function used in
both the cost and the loss.
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

    Classification is a cross-entropy over all N queries with the no-object class downweighted
    by eos_coef; box L1 and the GIoU loss are computed on the matched (query, gt) pairs only.
    The three terms are summed with weight_class / weight_l1 / weight_giou. See the
    set-prediction loss section of the README for the exact terms.

    components is the dict {"class", "l1", "giou"} of the three unweighted terms.

    Contract: when an image has no matched pairs, the box terms must still be a zero that
    depends on pred_boxes so the autograd graph stays connected.
    """
    raise NotImplementedError("implement the DETR set-prediction loss")
