"""The conditional flow-matching (CFM) action head. This is the build target.

The action head is the generative model from the diffusion and flow-matching topics, re-conditioned
on robot state instead of a class label. It learns a velocity field that transports a Gaussian
sample to the demonstrated action chunk along a straight path, integrated with a few Euler steps at
inference (pi0, Black et al. 2024).

Convention (matching the course's flow-matching assignment): t=0 is noise z0 ~ N(0, I), t=1 is the
data action chunk a_chunk. The network regresses onto the conditional velocity of the straight path,
which is CONSTANT in t.

Shapes: a_chunk, z0, z_t are (B, H, 2); t is (B, 1, 1) broadcast over (H, 2); c is (B, cond_dim).
The math is in the conditional flow matching section of the README.
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
    """The straight-path interpolant and the velocity target. HOLE.

    Build z_t on the straight line from the noise z0 (at t=0) to the action chunk a_chunk (at t=1),
    and return the constant conditional velocity target.

    a_chunk, z0 are (B, H, 2); t is (B, 1, 1) and broadcasts over (H, 2). Return (z_t, v), both
    (B, H, 2). The target v does NOT depend on t: a velocity field that has a t-dependent target is
    wrong (a test asserts t-independence).

    See the conditional flow matching section of the README.
    """
    raise NotImplementedError("implement the CFM interpolant z_t and the constant velocity target v")


class FlowHead(nn.Module):
    """An MLP velocity field v_theta(z_t, t, c) -> (B, H, 2). __init__ and forward provided.

    The chunk-state z_t is flattened to (B, H*2), the timestep is embedded sinusoidally, and the
    conditioning c is projected. The three are concatenated and passed through an MLP that outputs
    the per-step velocity, reshaped to (B, H, 2).
    """

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
    """The conditional flow-matching loss. HOLE.

    Plain velocity regression: over fresh z0 ~ N(0, I) and t ~ Uniform(0, 1), the mean squared error
    between the network's predicted velocity and the cfm_target velocity. The objective is
    unweighted: do not weight by t.

    See the conditional flow matching section of the README.
    """
    raise NotImplementedError("implement the conditional flow-matching loss")


def flow_sample(head: FlowHead, c: Tensor, H: int, n_steps: int,
                generator: torch.Generator | None = None) -> Tensor:
    """Euler-integrate the ODE from t=0 (noise) to t=1 (action). HOLE.

    Start from z ~ N(0, I) of shape (B, H, 2) and take n_steps forward-Euler steps of the learned
    velocity field, integrating from t=0 to t=1. Return the final z as the predicted action chunk
    (B, H, 2). No external ODE library; this is a short loop. Integrating in the wrong direction
    (starting at t=1 instead of t=0) or with the wrong step size is the likely bug.

    See the conditional flow matching section of the README.
    """
    raise NotImplementedError("implement Euler ODE sampling for the flow head")
