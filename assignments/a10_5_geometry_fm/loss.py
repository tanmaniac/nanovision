"""The DUSt3R confidence-weighted, scale-normalized regression loss.

DUSt3R regresses both pointmaps in camera 1's frame and supervises them with a single loss
that does two things at once: it removes the global scale ambiguity (a monocular pair only
fixes geometry up to one overall scale) and it learns a per-pixel confidence so the network
can down-weight pixels it cannot predict (sky, occlusion, reflective surfaces) while paying a
penalty for doing so.

Two correctness points the build must respect:

1. ONE shared scale over the UNION of both pointmaps' valid points (Eq. 5), not a separate
   scale per map. A per-map scale would independently rescale view 1 and view 2 and destroy
   the relative scale between them, which is exactly the information the shared cam1-frame
   representation carries. normalize_scale therefore returns one scalar per batch element.
2. Confidence is C = 1 + exp(logit) >= 1 (the head already applies this); the loss reads C.

The confidence term lets the network lower C on pixels it cannot fit, so confidence is
learned from the data rather than driven to an extreme. See the "The confidence-weighted,
scale-normalized loss" section of the README for the loss.
"""

import torch
from torch import Tensor


def normalize_scale(pts: Tensor, valid: Tensor, eps: float = 1e-8) -> Tensor:
    """One shared scale over the union of both pointmaps' valid points (DUSt3R Eq. 5).

    Both pointmaps stack into pts so they share one scalar per batch element; do not compute
    a scale per map. See the "The confidence-weighted, scale-normalized loss" section of the
    README for the scale.

    Args:
        pts: (B, 2, h, w, 3) the two pointmaps stacked on dim 1 (index 0 = map 1, 1 = map 2).
        valid: (B, 2, h, w) bool mask of valid pixels for each map.
        eps: floor on the denominator and the returned scale.

    Returns:
        z: (B,) the single positive scale per batch element.
    """
    raise NotImplementedError(
        "compute the per-batch mean norm of valid points over BOTH stacked pointmaps"
    )


def pointmap_loss(
    pred_pts1: Tensor,
    pred_pts2: Tensor,
    gt_pts1: Tensor,
    gt_pts2: Tensor,
    pred_conf1: Tensor,
    pred_conf2: Tensor,
    valid1: Tensor,
    valid2: Tensor,
    alpha: float = 0.2,
) -> Tensor:
    """Confidence-weighted, scale-normalized pointmap regression loss over both views.

    Scale-normalize the two predicted maps by a single joint z and the two GT maps by a
    single joint zbar, both from normalize_scale (one shared scale per side, not per map).
    The loss weights each valid pixel's residual by its confidence and averages over the
    valid pixels of BOTH views; alpha = 0.2 (paper). See the "The confidence-weighted,
    scale-normalized loss" section of the README for the residual and the loss.

    Args:
        pred_pts1, pred_pts2: (B, h, w, 3) predicted pointmaps, both in cam1 frame.
        gt_pts1, gt_pts2: (B, h, w, 3) GT pointmaps, both in cam1 frame.
        pred_conf1, pred_conf2: (B, h, w) confidence C >= 1 from the head.
        valid1, valid2: (B, h, w) bool masks.
        alpha: weight of the -log(C) confidence regularizer.

    Returns:
        scalar loss.
    """
    raise NotImplementedError(
        "joint-scale-normalize pred and gt, compute per-pixel L2, weight by C, subtract "
        "alpha*log(C), and average over valid pixels of both views"
    )
