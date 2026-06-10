"""A DDPM action head as the diffusion-vs-flow contrast (secondary, not the build target).

This is the course's diffusion epsilon-prediction objective re-conditioned on c, generating the
same H-step action chunk as a denoising chain instead of a flow ODE. It exists so the README and
viz can compare DDPM (a noise schedule and many reverse steps) against flow matching (plain
velocity regression and a few Euler steps) on one task. The flow head is the headline mechanism;
this is the baseline it is measured against.

RDT-1B (Liu et al. 2024) is a diffusion TRANSFORMER for bimanual manipulation; it is named in the
README as a reading pointer, not built here, and is not the same thing as this plain DDPM denoiser.

Convention matches the course's diffusion assignment: a_t = sqrt(abar_t) a_chunk + sqrt(1-abar_t)
eps, the network predicts eps, and the loss is MSE on eps. Shapes: a_chunk, a_t, eps are (B, H, 2);
the integer timestep is (B,); c is (B, cond_in); alphas_bar is (T,).
"""

import math

import torch
from torch import Tensor, nn

from flow import sinusoidal_embedding


def make_schedule(T: int) -> Tensor:
    """Linear-beta cumulative signal schedule alphas_bar (T,). Provided."""
    betas = torch.linspace(1e-4, 2e-2, T)
    return torch.cumprod(1.0 - betas, dim=0)


class DDPMHead(nn.Module):
    """An MLP eps_theta(a_t, t, c) -> (B, H, 2). __init__ and forward provided."""

    def __init__(self, cfg, cond_in: int, T: int):
        super().__init__()
        self.H = cfg.chunk
        self.act_dim = cfg.act_dim
        self.time_dim = cfg.time_dim
        self.T = T
        self.cond_proj = nn.Linear(cond_in, cfg.cond_dim)
        in_dim = self.H * self.act_dim + cfg.time_dim + cfg.cond_dim
        w = cfg.hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, w), nn.SiLU(),
            nn.Linear(w, w), nn.SiLU(),
            nn.Linear(w, self.H * self.act_dim),
        )

    def forward(self, a_t: Tensor, t: Tensor, c: Tensor) -> Tensor:
        B = a_t.shape[0]
        temb = sinusoidal_embedding(t.float() / self.T, self.time_dim)
        cemb = self.cond_proj(c)
        x = torch.cat([a_t.reshape(B, -1), temb, cemb], dim=-1)
        return self.net(x).reshape(B, self.H, self.act_dim)


def ddpm_loss(head: DDPMHead, a_chunk: Tensor, c: Tensor, alphas_bar: Tensor,
              generator: torch.Generator | None = None) -> Tensor:
    """The DDPM epsilon-prediction loss."""
    B = a_chunk.shape[0]
    T = alphas_bar.shape[0]
    t = torch.randint(0, T, (B,), device=a_chunk.device, generator=generator)
    eps = torch.randn(a_chunk.shape, device=a_chunk.device, dtype=a_chunk.dtype, generator=generator)
    abar_t = alphas_bar.to(a_chunk.device)[t].reshape(B, 1, 1)
    a_t = abar_t.sqrt() * a_chunk + (1.0 - abar_t).sqrt() * eps
    eps_hat = head(a_t, t, c)
    return ((eps_hat - eps) ** 2).mean()


def ddpm_sample(head: DDPMHead, c: Tensor, H: int, alphas_bar: Tensor,
                generator: torch.Generator | None = None) -> Tensor:
    """The ancestral DDPM reverse chain (Ho et al. 2020)."""
    B = c.shape[0]
    device = c.device
    abar = alphas_bar.to(device)
    T = abar.shape[0]
    abar_prev = torch.cat([abar.new_ones(1), abar[:-1]])
    a = torch.randn(B, H, head.act_dim, device=device, dtype=c.dtype, generator=generator)
    for ti in reversed(range(T)):
        t = torch.full((B,), ti, device=device, dtype=torch.long)
        abar_t = abar[ti]
        abar_pm = abar_prev[ti]
        alpha_t = abar_t / abar_pm
        beta_t = 1.0 - alpha_t
        eps_hat = head(a, t, c)
        x0_hat = (a - (1.0 - abar_t).sqrt() * eps_hat) / abar_t.sqrt()
        mean = (abar_pm.sqrt() * beta_t / (1.0 - abar_t)) * x0_hat \
            + (alpha_t.sqrt() * (1.0 - abar_pm) / (1.0 - abar_t)) * a
        if ti > 0:
            var = (1.0 - abar_pm) / (1.0 - abar_t) * beta_t
            noise = torch.randn(a.shape, device=device, dtype=c.dtype, generator=generator)
            a = mean + var.sqrt() * noise
        else:
            a = mean
    return a
