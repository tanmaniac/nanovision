"""The conditional flow-matching (CFM) action head. This is the build target.

The action head is the generative model from the diffusion and flow-matching topics, re-conditioned
on robot state instead of a class label. It learns a velocity field that transports a Gaussian
sample to the demonstrated action chunk along a straight path, integrated with a few Euler steps at
inference (pi0, Black et al. 2024).

Convention (matching the course's flow-matching assignment): t=0 is noise z0 ~ N(0, I), t=1 is the
data action chunk a_chunk. The straight path is z_t = (1-t) z0 + t a_chunk, so the conditional
velocity dz_t/dt = a_chunk - z0 is CONSTANT in t. The network regresses onto that constant target.

Shapes: a_chunk, z0, z_t are (B, H, 2); t is (B, 1, 1) broadcast over (H, 2); c is (B, cond_dim).
"""

import math

import torch
from torch import Tensor, nn


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a scalar t (B,) into (B, dim). Provided.

    Same construction as the transformer's positional encoding and the diffusion time embedding.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def cfm_target(a_chunk: Tensor, z0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """The straight-path interpolant and the velocity target.

      z_t = (1 - t) * z0 + t * a_chunk
      v   = a_chunk - z0                 (the target, CONSTANT in t)

    a_chunk, z0 are (B, H, 2); t is (B, 1, 1) and broadcasts over (H, 2). Return (z_t, v).
    """
    z_t = (1.0 - t) * z0 + t * a_chunk
    v = a_chunk - z0
    return z_t, v


class FlowHead(nn.Module):
    """An MLP velocity field v_theta(z_t, t, c) -> (B, H, 2). __init__ and forward provided."""

    def __init__(self, cfg, cond_in: int):
        super().__init__()
        self.H = cfg.chunk
        self.act_dim = cfg.act_dim
        self.time_dim = cfg.time_dim
        self.cond_proj = nn.Linear(cond_in, cfg.cond_dim)
        in_dim = self.H * self.act_dim + cfg.time_dim + cfg.cond_dim
        w = cfg.hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, w), nn.SiLU(),
            nn.Linear(w, w), nn.SiLU(),
            nn.Linear(w, self.H * self.act_dim),
        )

    def forward(self, z_t: Tensor, t: Tensor, c: Tensor) -> Tensor:
        B = z_t.shape[0]
        temb = sinusoidal_embedding(t.reshape(B), self.time_dim)
        cemb = self.cond_proj(c)
        x = torch.cat([z_t.reshape(B, -1), temb, cemb], dim=-1)
        return self.net(x).reshape(B, self.H, self.act_dim)


def flow_loss(head: FlowHead, a_chunk: Tensor, c: Tensor,
              generator: torch.Generator | None = None) -> Tensor:
    """The conditional flow-matching loss: MSE between predicted and target velocity."""
    z0 = torch.randn(a_chunk.shape, device=a_chunk.device, dtype=a_chunk.dtype, generator=generator)
    t = torch.rand(a_chunk.shape[0], 1, 1, device=a_chunk.device, dtype=a_chunk.dtype,
                   generator=generator)
    z_t, v = cfm_target(a_chunk, z0, t)
    v_hat = head(z_t, t, c)
    return ((v_hat - v) ** 2).mean()


def flow_sample(head: FlowHead, c: Tensor, H: int, n_steps: int,
                generator: torch.Generator | None = None) -> Tensor:
    """Euler-integrate the ODE from t=0 (noise) to t=1 (action)."""
    B = c.shape[0]
    z = torch.randn(B, H, head.act_dim, device=c.device, dtype=c.dtype, generator=generator)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B, 1, 1), k * dt, device=c.device, dtype=c.dtype)
        z = z + dt * head(z, t, c)
    return z
