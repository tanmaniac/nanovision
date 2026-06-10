"""Hyperparameters for the VLA action-head toy. Provided, not a hole.

The build target is a conditional flow-matching (CFM) action head in the pi0 line (Black et al.,
2024). pi0 pairs a ~3B-parameter vision-language backbone with a ~300M-parameter action expert
that generates 7-DoF action chunks by flow matching. Here everything shrinks to a 2D point-mass
reacher with a small MLP velocity field, so the mechanism (a velocity field that transports a
Gaussian sample to the demonstrated action chunk along a straight path, integrated in ~10 Euler
steps) is visible and every CPU test runs in seconds. The numbers here demonstrate the mechanism;
they do not predict pi0-scale manipulation behavior.
"""

from dataclasses import dataclass


@dataclass
class VLAConfig:
    act_dim: int = 2            # action is a 2D velocity (vx, vy)
    chunk: int = 4             # default chunk length H; the ablation sweeps {1, 4, 16}
    cond_dim: int = 64          # width of the conditioning embedding fed to the heads
    hidden: int = 128           # MLP hidden width
    time_dim: int = 64          # sinusoidal time-embedding width for the flow/ddpm heads

    n_flow_steps: int = 10      # Euler steps for the flow-matching ODE sampler at inference
    ddpm_T: int = 50            # diffusion steps for the DDPM contrast head

    v_max: float = 0.05         # action clip (matches env.V_MAX); small so a chunk of 16 fits an episode
    eps: float = 0.05           # success radius (matches env.EPS)
    n_goals: int = 4

    repr: str = "onehot"        # goal encoding: "onehot" or "coord"


@dataclass
class GradcheckConfig:
    """Smaller widths for the float64 gradcheck so it stays fast."""

    act_dim: int = 2
    chunk: int = 2
    cond_dim: int = 8
    hidden: int = 16
    time_dim: int = 8
    n_flow_steps: int = 4
    ddpm_T: int = 8
    v_max: float = 0.1
    eps: float = 0.05
    n_goals: int = 4
    repr: str = "onehot"
