"""The diffusion core: forward noising, the three prediction parameterizations, the
score connection, and the training loss.

All tensors are image batches (B, C, H, W); abar_t is a per-sample signal level broadcast
to (B, 1, 1, 1) by schedule.gather.
"""

import torch
from torch import Tensor

from schedule import gather


def q_sample(x0: Tensor, t: Tensor, eps: Tensor, alphas_bar: Tensor) -> Tensor:
    """Closed-form forward process q(x_t | x_0) (Ho et al. 2020):

        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

    t is (B,) integer timesteps; eps is standard normal noise the same shape as x0.
    """
    abar_t = gather(alphas_bar, t)
    return abar_t.sqrt() * x0 + (1.0 - abar_t).sqrt() * eps


def v_target(x0: Tensor, eps: Tensor, abar_t: Tensor) -> Tensor:
    """The v-prediction target (Salimans & Ho 2022): v = sqrt(abar)*eps - sqrt(1-abar)*x0.

    This is the rotation of (x0, eps) by the same angle that maps (x0, eps) to (x_t, v).
    It is well-conditioned at both noise extremes, unlike eps (singular at t=T) and x0
    (singular at t=0). abar_t is (B, 1, 1, 1).
    """
    return abar_t.sqrt() * eps - (1.0 - abar_t).sqrt() * x0


def to_x0_eps(pred: Tensor, x_t: Tensor, abar_t: Tensor, kind: str) -> tuple[Tensor, Tensor]:
    """Convert a model output of the given kind into (x0_hat, eps_hat).

    The three parameterizations are algebraically equivalent given
    x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps. With a = sqrt(abar), b = sqrt(1-abar):
      eps: x0_hat = (x_t - b*eps)/a
      x0 : eps_hat = (x_t - a*x0)/b
      v  : x0_hat = a*x_t - b*v ,  eps_hat = b*x_t + a*v  (the inverse rotation)
    Samplers call this so they work with any parameterization.
    """
    a = abar_t.sqrt()
    b = (1.0 - abar_t).sqrt()
    if kind == "eps":
        eps_hat = pred
        x0_hat = (x_t - b * eps_hat) / a
    elif kind == "x0":
        x0_hat = pred
        eps_hat = (x_t - a * x0_hat) / b
    elif kind == "v":
        x0_hat = a * x_t - b * pred
        eps_hat = b * x_t + a * pred
    else:
        raise ValueError(f"unknown parameterization {kind!r}")
    return x0_hat, eps_hat


def score_from_eps(eps: Tensor, abar_t: Tensor) -> Tensor:
    """The score grad_{x_t} log p_t(x_t) = -eps / sqrt(1 - alpha_bar_t) (Tweedie's formula).

    A network trained to predict eps is, up to this scaling, a score estimator. This is
    why eps-prediction is denoising score matching (Vincent 2011).
    """
    return -eps / (1.0 - abar_t).sqrt()


def diffusion_loss(model, x0: Tensor, alphas_bar: Tensor, *, kind: str = "v",
                   num_classes: int | None = None, cfg_drop_prob: float = 0.1,
                   min_snr_gamma: float | None = None, labels: Tensor | None = None,
                   generator: torch.Generator | None = None) -> Tensor:
    """The DDPM training objective with a parameterization-aware Min-SNR weighting.

    Sample t ~ U[0, T), eps ~ N(0, I), build x_t, optionally drop the label to the null
    index for classifier-free guidance, predict the target by `kind`, and take MSE. If
    min_snr_gamma is set, weight each sample by the Min-SNR weight in that loss's native
    space (the weights differ by parameterization; a single min(SNR,g)/SNR is wrong for v).
    """
    B = x0.shape[0]
    T = alphas_bar.shape[0]
    device = x0.device
    t = torch.randint(0, T, (B,), device=device, generator=generator)
    eps = torch.randn(x0.shape, device=device, generator=generator)
    abar_t = gather(alphas_bar, t)
    x_t = q_sample(x0, t, eps, alphas_bar)

    if labels is not None and num_classes is not None and cfg_drop_prob > 0:
        drop = torch.rand(B, device=device, generator=generator) < cfg_drop_prob
        labels = labels.clone()
        labels[drop] = num_classes                 # the null index is the extra row

    pred = model(x_t, t, labels)
    if kind == "eps":
        target = eps
    elif kind == "x0":
        target = x0
    elif kind == "v":
        target = v_target(x0, eps, abar_t)
    else:
        raise ValueError(f"unknown parameterization {kind!r}")

    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)   # per-sample (B,)
    if min_snr_gamma is not None:
        snr = (abar_t / (1.0 - abar_t)).flatten()         # (B,)
        capped = snr.clamp(max=min_snr_gamma)
        if kind == "eps":
            w = capped / snr
        elif kind == "x0":
            w = capped
        else:                                             # v
            w = capped / (snr + 1.0)
        mse = w * mse
    return mse.mean()
