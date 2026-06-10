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
    diversity; w = 0 is unconditional. (The diffusers `guidance_scale` is this same w, with guidance_scale=1 the plain conditional.)
    """
    return eps_uncond + w * (eps_cond - eps_uncond)


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

    Posterior mean (eq 6-7):
      mu = sqrt(abar_prev)*beta_t/(1-abar_t) * x0_hat
         + sqrt(alpha_t)*(1-abar_prev)/(1-abar_t) * x_t
    Variance: the true posterior variance is beta_tilde_t = (1-abar_prev)/(1-abar_t)*beta_t
    ("beta_tilde", default); beta_t is the other fixed choice Ho et al. report as
    comparable. Noise is added at every step except t=0.
    """
    device = alphas_bar.device
    T = alphas_bar.shape[0]
    abar = alphas_bar
    abar_prev = torch.cat([abar.new_ones(1), abar[:-1]])   # abar_prev[t] = abar[t-1], [0]=1
    x = torch.randn(shape, device=device, generator=generator)
    for ti in reversed(range(T)):
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        abar_t, abar_pm = abar[ti], abar_prev[ti]
        beta_t = 1.0 - abar_t / abar_pm
        x0_hat, _ = _predict(model, x, t, abar_t, kind, labels, guidance, clip_x0)
        mean = (abar_pm.sqrt() * beta_t / (1.0 - abar_t)) * x0_hat \
            + ((abar_t / abar_pm).sqrt() * (1.0 - abar_pm) / (1.0 - abar_t)) * x
        if ti > 0:
            var = (1.0 - abar_pm) / (1.0 - abar_t) * beta_t if variance == "beta_tilde" else beta_t
            noise = torch.randn(shape, device=device, generator=generator)
            x = mean + var.sqrt() * noise
        else:
            x = mean
    return x


def ddim_sample(model, shape, alphas_bar: Tensor, timesteps, *, kind: str = "v",
                eta: float = 0.0, clip_x0: bool = True, labels=None, guidance: float = 1.0,
                generator: torch.Generator | None = None) -> Tensor:
    """DDIM sampler (Song et al. 2020). `timesteps` is a decreasing subset of indices.

      x_prev = sqrt(abar_prev)*x0_hat + sqrt(1 - abar_prev - sigma^2)*eps_hat + sigma*z
      sigma  = eta * sqrt((1-abar_prev)/(1-abar_t)) * sqrt(1 - abar_t/abar_prev)

    eta = 0 is deterministic (the probability-flow ODE); eta = 1 produces variance
    beta_tilde_t and matches the beta_tilde DDPM sampler on the full consecutive grid.
    """
    device = alphas_bar.device
    abar = alphas_bar
    ts = list(timesteps)
    x = torch.randn(shape, device=device, generator=generator)
    for i, ti in enumerate(ts):
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        abar_t = abar[ti]
        prev = ts[i + 1] if i + 1 < len(ts) else -1
        abar_prev = abar[prev] if prev >= 0 else abar.new_tensor(1.0)   # abar_{-1} := 1
        x0_hat, eps_hat = _predict(model, x, t, abar_t, kind, labels, guidance, clip_x0)
        sigma = eta * ((1.0 - abar_prev) / (1.0 - abar_t)).sqrt() * (1.0 - abar_t / abar_prev).sqrt()
        x = abar_prev.sqrt() * x0_hat + (1.0 - abar_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps_hat
        if eta > 0 and prev >= 0:
            noise = torch.randn(shape, device=device, generator=generator)
            x = x + sigma * noise
    return x
