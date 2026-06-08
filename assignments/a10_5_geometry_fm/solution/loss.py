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

The confidence term -alpha * log(C) lets the network lower C on pixels it cannot fit. With a
fixed residual ell, the per-pixel cost C*ell - alpha*log(C) is minimized at C = alpha/ell, a
finite confidence, so confidence is learned, not driven to either extreme.
"""

import torch
from torch import Tensor


def normalize_scale(pts: Tensor, valid: Tensor, eps: float = 1e-8) -> Tensor:
    """One shared scale over the union of both pointmaps' valid points (DUSt3R Eq. 5).

    The scale for batch element b is the mean L2 norm of every valid 3D point across BOTH
    pointmaps:

        z_b = ( sum over valid points in map 1 of ||X_i|| + sum over valid points in map 2
                of ||X_i|| ) / ( |valid in map 1| + |valid in map 2| ).

    Both pointmaps stack into pts so they share one scalar; do not compute a scale per map.

    Args:
        pts: (B, 2, h, w, 3) the two pointmaps stacked on dim 1 (index 0 = map 1, 1 = map 2).
        valid: (B, 2, h, w) bool mask of valid pixels for each map.
        eps: floor on the denominator and the returned scale.

    Returns:
        z: (B,) the single positive scale per batch element.
    """
    norms = pts.norm(dim=-1)                          # (B, 2, h, w)
    m = valid.to(norms.dtype)                         # (B, 2, h, w)
    num = (norms * m).flatten(1).sum(dim=1)           # (B,)
    den = m.flatten(1).sum(dim=1).clamp(min=eps)      # (B,)
    return (num / den).clamp(min=eps)                 # (B,)


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

    Scale-normalize the two predicted maps by a single z (their joint scale) and the two GT
    maps by a single zbar (the GT joint scale), both from normalize_scale. Per valid pixel i
    of view v, the residual is the L2 distance between the scaled prediction and scaled GT:

        ell_i^v = || pred_pts_i^v / z  -  gt_pts_i^v / zbar ||.

    The total loss sums the confidence-weighted residual over both views and averages over
    the valid pixels:

        L = mean over valid pixels of ( C_i^v * ell_i^v - alpha * log(C_i^v) ),

    with the mean taken over the count of valid pixels across BOTH views. alpha = 0.2 (paper).

    Args:
        pred_pts1, pred_pts2: (B, h, w, 3) predicted pointmaps, both in cam1 frame.
        gt_pts1, gt_pts2: (B, h, w, 3) GT pointmaps, both in cam1 frame.
        pred_conf1, pred_conf2: (B, h, w) confidence C >= 1 from the head.
        valid1, valid2: (B, h, w) bool masks.
        alpha: weight of the -log(C) confidence regularizer.

    Returns:
        scalar loss.
    """
    pred = torch.stack([pred_pts1, pred_pts2], dim=1)   # (B, 2, h, w, 3)
    gt = torch.stack([gt_pts1, gt_pts2], dim=1)         # (B, 2, h, w, 3)
    conf = torch.stack([pred_conf1, pred_conf2], dim=1)  # (B, 2, h, w)
    valid = torch.stack([valid1, valid2], dim=1)         # (B, 2, h, w)

    z = normalize_scale(pred, valid)                     # (B,)
    zbar = normalize_scale(gt, valid)                    # (B,)
    zb = z.view(-1, 1, 1, 1, 1)
    zbb = zbar.view(-1, 1, 1, 1, 1)

    diff = pred / zb - gt / zbb                          # (B, 2, h, w, 3)
    ell = diff.norm(dim=-1)                              # (B, 2, h, w)

    m = valid.to(ell.dtype)
    per_pixel = conf * ell - alpha * torch.log(conf)     # (B, 2, h, w)
    total = (per_pixel * m).sum()
    count = m.sum().clamp(min=1.0)
    return total / count
