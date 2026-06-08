"""Adaptive density control, read-and-understand. The course implements opacity pruning only.

Real 3D Gaussian splatting interleaves the differentiable photometric fit with a
non-differentiable heuristic that adds and removes Gaussians, called adaptive density
control (ADC). Every few hundred steps it:

  - clones under-reconstructed Gaussians (small Gaussians in regions with large positional
    gradient: the scene needs more of them there) and nudges the copies along the gradient;
  - splits over-reconstructed Gaussians (large Gaussians with large gradient) into two
    smaller ones sampled from the parent's distribution, shrinking their scale;
  - prunes Gaussians whose opacity has fallen below a threshold, or whose screen-space
    footprint has grown too large;
  - periodically resets all opacities toward zero, so Gaussians that are not earning their
    place fade out and get pruned, while useful ones recover their opacity through the fit.

ADC is what lets the representation start from a sparse point cloud and grow to millions of
Gaussians that tile the scene at the right density. It is a discrete outer loop around the
differentiable inner fit, not part of the gradient path, and it is out of the graded scope
here. This file implements only the prune step, which is enough to see the idea.

Assignment-local. Import bare.
"""

import torch

from gaussian import GaussianModel


def prune_low_opacity(model: GaussianModel, threshold: float) -> GaussianModel:
    """Return a new GaussianModel keeping only Gaussians with opacity above `threshold`.

    The simplified stand-in for the prune step of adaptive density control. Non-
    differentiable (it changes the parameter count); call it between optimization phases,
    not inside the gradient path.

    Args:
        model: the current GaussianModel.
        threshold: keep Gaussians with sigmoid(opacity_logit) > threshold.

    Returns:
        a new GaussianModel with the surviving Gaussians.
    """
    with torch.no_grad():
        keep = model.opacities > threshold
    return GaussianModel(
        model.means[keep].detach().clone(),
        model.log_scales[keep].detach().clone(),
        model.quats[keep].detach().clone(),
        model.opacity_logits[keep].detach().clone(),
        model.color_logits[keep].detach().clone(),
    )
