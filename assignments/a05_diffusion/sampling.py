"""Samplers: DDPM ancestral (stochastic) and DDIM (deterministic, sub-sampled), plus the
classifier-free guidance combine.

Both samplers are parameterization-agnostic: they get (x0_hat, eps_hat) from `_predict`
and apply their update rule. `_predict` (provided) runs the network, does the
classifier-free guidance two-pass when guidance != 1, and optionally clamps the predicted
x0 to [-1, 1] (the x0 estimate is unreliable at high t and can blow the trajectory up).
After clamping it recomputes eps_hat from the clamped x0 so the pair stays consistent with
x_t. Index convention: timesteps run high to low; alphas_bar[-1] := 1 (the clean image) is
the implicit previous level at the final step, so that step adds no noise and returns x0.
"""

import torch
from torch import Tensor

from diffusion import to_x0_eps


def classifier_free_guidance(eps_cond: Tensor, eps_uncond: Tensor, w: float) -> Tensor:
    """Extrapolate the conditional score away from the unconditional one (Ho & Salimans
    2022): eps_guided = eps_uncond + w * (eps_cond - eps_uncond).

    w = 1 is the plain conditional (no guidance boost); w > 1 sharpens at the cost of
    diversity; w = 0 is unconditional. (The diffusers `guidance_scale` equals w + 1.)
    """
    raise NotImplementedError("implement the classifier-free guidance combine")


def _predict(model, x: Tensor, t: Tensor, abar_t: Tensor, kind: str, labels, guidance: float,
             clip_x0: bool) -> tuple[Tensor, Tensor]:
    """Run the model and return (x0_hat, eps_hat), with CFG and optional x0 clamping."""
    if guidance != 1.0 and labels is not None:
        null = torch.full_like(labels, model.null_index)
        x0_c, eps_c = to_x0_eps(model(x, t, labels), x, abar_t, kind)
        x0_u, eps_u = to_x0_eps(model(x, t, null), x, abar_t, kind)
        eps_hat = classifier_free_guidance(eps_c, eps_u, guidance)
        x0_hat = (x - (1.0 - abar_t).sqrt() * eps_hat) / abar_t.sqrt()
    else:
        x0_hat, eps_hat = to_x0_eps(model(x, t, labels), x, abar_t, kind)
    if clip_x0:
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        eps_hat = (x - abar_t.sqrt() * x0_hat) / (1.0 - abar_t).sqrt()
    return x0_hat, eps_hat


def ddpm_sample(model, shape, alphas_bar: Tensor, *, kind: str = "v",
                variance: str = "beta_tilde", clip_x0: bool = True, labels=None,
                guidance: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    """Ancestral DDPM sampler (Ho et al. 2020), t from T-1 down to 0.

    Start from x ~ N(0, I) of `shape`. At each t, get x0_hat from _predict, form the
    posterior mean (eq 6-7):
      mu = sqrt(abar_prev)*beta_t/(1-abar_t) * x0_hat
         + sqrt(alpha_t)*(1-abar_prev)/(1-abar_t) * x_t
    with alpha_t = abar_t/abar_prev, beta_t = 1 - alpha_t, and abar_prev being abar[t-1]
    (abar_{-1} := 1). Add noise at every step except t=0: x = mu + sqrt(var)*z where var is
    beta_tilde_t = (1-abar_prev)/(1-abar_t)*beta_t for variance="beta_tilde" (default) or
    beta_t for variance="beta". Return the final x.
    """
    raise NotImplementedError("implement the DDPM ancestral sampler")


def ddim_sample(model, shape, alphas_bar: Tensor, timesteps, *, kind: str = "v",
                eta: float = 0.0, clip_x0: bool = True, labels=None, guidance: float = 1.0,
                generator: torch.Generator | None = None) -> Tensor:
    """DDIM sampler (Song et al. 2020). `timesteps` is a decreasing subset of indices.

    Start from x ~ N(0, I). At each step with current index t and next index prev (use
    abar_prev := 1 when prev is past the end), get (x0_hat, eps_hat) from _predict and step
      sigma  = eta * sqrt((1-abar_prev)/(1-abar_t)) * sqrt(1 - abar_t/abar_prev)
      x_prev = sqrt(abar_prev)*x0_hat + sqrt(1 - abar_prev - sigma^2)*eps_hat + sigma*z
    eta = 0 is deterministic (the probability-flow ODE); eta = 1 produces variance
    beta_tilde_t. Add the sigma*z noise only when eta > 0 and a real prev step exists.
    Return the final x.
    """
    raise NotImplementedError("implement the DDIM sampler")
