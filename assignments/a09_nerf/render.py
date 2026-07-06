"""The discretized volume renderer: emission-absorption quadrature with alpha compositing.

This is the single most important equation in the assignment: the continuous volume
rendering integral, discretized onto N ray samples and evaluated by front-to-back alpha
compositing. The same compositing returns in Gaussian splatting (A10) and in occupancy /
neural-SDF rendering (A11.5d).

See the "The volume rendering integral" section of the README for the integral and its
discrete quadrature.

Shared OWNED file. Import its symbols through nanovision.volume, never by bare name.
"""

import torch
from torch import Tensor


def volume_render(
    sigmas: Tensor,
    colors: Tensor,
    deltas: Tensor,
    *,
    white_background: bool = False,
) -> tuple[Tensor, Tensor]:
    """Composite per-sample densities and colors into a pixel color.

    Contract: sigmas are ALREADY non-negative. The MLP applies the softplus/ReLU that
    keeps density >= 0, so this function must NOT re-activate them (double-applying the
    activation changes the alpha values).

    Args:
        sigmas: (R, N) non-negative volume densities per ray sample.
        colors: (R, N, 3) per-sample RGB in [0, 1].
        deltas: (R, N) segment lengths between consecutive samples.
        white_background: composite the leftover transmittance onto white.

    Returns:
        color: (R, 3) rendered pixel color.
        weights: (R, N) per-sample compositing weights, used by the ablation and
            (optionally) hierarchical fine sampling.

    See the "The volume rendering integral" section of the README.
    """
    raise NotImplementedError("implement the discretized volume renderer")
