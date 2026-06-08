"""The discretized volume renderer: emission-absorption quadrature with alpha compositing.

This is the single most important equation in the assignment. The continuous volume
rendering integral
    C(r) = integral T(t) sigma(r(t)) c(r(t), d) dt,   T(t) = exp(-integral_0^t sigma ds)
discretizes, on N samples with segment lengths delta_i, to
    alpha_i = 1 - exp(-sigma_i delta_i)
    T_i = prod_{j<i} (1 - alpha_j)         (exclusive cumulative product, T_0 = 1)
    w_i = T_i alpha_i,   C = sum_i w_i c_i
T(t) = exp(-integral sigma ds) is Beer-Lambert's law for transmittance through an
absorbing medium; alpha_i = 1 - exp(-sigma_i delta_i) is the exact opacity of one segment
of constant density. This same front-to-back alpha compositing returns in Gaussian
splatting (A10) and in occupancy / neural-SDF rendering (A11.5d).

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
        weights: (R, N) per-sample compositing weights w_i = T_i alpha_i, used by the
            ablation and (optionally) hierarchical fine sampling.
    """
    raise NotImplementedError("implement the discretized volume renderer")
