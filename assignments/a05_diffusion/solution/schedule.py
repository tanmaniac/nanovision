"""Forward noising schedules: linear (Ho et al. 2020) and cosine (Nichol & Dhariwal 2021).

A schedule is the sequence of cumulative signal levels alpha_bar_t. Index convention:
arrays have length T, indices 0..T-1. alphas_bar[0] is the least noised level (one forward
step from a clean image), alphas_bar[T-1] ~ 0 is pure noise. betas[t] = 1 - alpha_t where
alpha_t = alphas_bar[t] / alphas_bar[t-1], with alphas_bar[-1] := 1 (the clean image) as
the implicit base for betas[0].
"""

import math

import torch
from torch import Tensor


def linear_alpha_bar(T: int, beta_start: float = 1e-4,
                     beta_end: float = 2e-2) -> tuple[Tensor, Tensor]:
    """Linear beta schedule (Ho et al. 2020), beta ramps linearly over T steps.

    Returns (betas (T,), alphas_bar (T,)). These constants are calibrated for T=1000; at
    much smaller T the chain barely noises (alphas_bar[-1] stays well above 0), so do not
    assume alphas_bar[-1] ~ 0 here unless T is ~1000.
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)
    return betas, alphas_bar


def cosine_alpha_bar(T: int, s: float = 0.008) -> tuple[Tensor, Tensor]:
    """Cosine schedule (Nichol & Dhariwal 2021, eq 17).

        f(t) = cos^2( (t/T + s) / (1 + s) * pi/2 ),  alpha_bar_t = f(t) / f(0)

    The small offset s keeps beta_t from getting too tiny near t=0. betas are clipped to
    <= 0.999 to avoid singular values near t=T. Defined directly on alpha_bar via t/T, so
    it is self-normalizing in T (alphas_bar[-1] ~ 0 at any T).

    Returns (betas (T,), alphas_bar (T,)).
    """
    steps = torch.arange(T + 1, dtype=torch.float64)          # 0..T, T+1 grid points
    f = torch.cos((steps / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    abar_full = f / f[0]                                       # abar_full[0] = 1
    betas = 1.0 - abar_full[1:] / abar_full[:-1]              # betas[t] = 1 - abar_t/abar_{t-1}
    betas = betas.clamp(max=0.999)
    alphas_bar = abar_full[1:]                                # abar_1..abar_T -> length T
    return betas.float(), alphas_bar.float()


def gather(a: Tensor, t: Tensor) -> Tensor:
    """Index a (T,) schedule tensor by an integer time batch t (B,) and reshape to
    (B, 1, 1, 1) so it broadcasts against an image batch (B, C, H, W)."""
    return a.to(t.device)[t].view(-1, 1, 1, 1)
