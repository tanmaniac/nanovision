"""A DDPM action head as the diffusion-vs-flow contrast (secondary, not the build target).

This is the course's diffusion epsilon-prediction objective re-conditioned on c, generating the
same H-step action chunk as a denoising chain instead of a flow ODE. It exists so the README and
viz can compare DDPM (a noise schedule and many reverse steps) against flow matching (plain
velocity regression and a few Euler steps) on one task. The flow head is the headline mechanism;
this is the baseline it is measured against.

RDT-1B (Liu et al. 2024) is a diffusion TRANSFORMER for bimanual manipulation; it is named in the
README as a reading pointer, not built here, and is not the same thing as this plain DDPM denoiser.

Convention matches the course's diffusion assignment: the network predicts eps (not x0) and the loss
is MSE on eps. Shapes: a_chunk, a_t, eps are (B, H, 2); the integer timestep is (B,); c is
(B, cond_in); alphas_bar is (T,). The math is in the DDPM contrast section of the README.
"""

import math

import torch
from torch import Tensor, nn

from flow import sinusoidal_embedding


def make_schedule(T: int) -> Tensor:
    """Linear-beta cumulative signal schedule alphas_bar (T,). Provided.

    betas ramp linearly 1e-4 -> 2e-2; alphas_bar = cumprod(1 - beta). At small T the chain does
    not fully noise (alphas_bar[-1] stays above 0), which is fine for this 2D toy.
    """
    betas = torch.linspace(1e-4, 2e-2, T)
    return torch.cumprod(1.0 - betas, dim=0)


class DDPMHead(nn.Module):
    """An MLP eps_theta(a_t, t, c) -> (B, H, 2). __init__ and forward provided.

    Same wiring as the flow head: flatten the noisy chunk, embed the integer timestep sinusoidally
    (normalized to [0, 1] by T), project the conditioning, concatenate, MLP, reshape.
    """

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
    """The DDPM epsilon-prediction loss. HOLE.

    Noise a_chunk to a random timestep with the forward process, then regress the network's
    prediction onto the sampled noise eps (predict eps, not x0) by MSE. a_chunk is (B, H, 2), c is
    (B, cond_in), alphas_bar is (T,).

    See the DDPM contrast section of the README.
    """
    raise NotImplementedError("implement the DDPM epsilon-prediction loss")


def ddpm_sample(head: DDPMHead, c: Tensor, H: int, alphas_bar: Tensor,
                generator: torch.Generator | None = None) -> Tensor:
    """The ancestral DDPM reverse chain. HOLE.

    Start from a_T ~ N(0, I) of shape (B, H, 2) and run the reverse chain from t=T-1 down to 0,
    recovering the posterior mean at each step and adding Gaussian noise except at t=0. Return the
    final a (B, H, 2). This is the same ancestral sampler as the course's diffusion topic.

    See the DDPM contrast section of the README.
    """
    raise NotImplementedError("implement the ancestral DDPM sampler")
