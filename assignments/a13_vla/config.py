"""Hyperparameters for the VLA action-head capstone. Provided, not a hole.

The build target is a conditional flow-matching (CFM) action head in the pi0 line (Black et al.,
2024). pi0 pairs a ~3B-parameter vision-language backbone with a ~300M-parameter action expert that
generates 7-DoF action chunks by flow matching. Here everything shrinks to a 2-link reacher
controlled from a 64x64 camera image: a small CNN encodes the frame into the conditioning vector and
a small MLP velocity field generates the 2D torque chunk by flow matching. The mechanism (a velocity
field that transports a Gaussian sample to the demonstrated action chunk along a straight path,
integrated in ~10 Euler steps) is visible, and every CPU mechanism test runs in seconds without the
robot library.

The point-mass multimodal side-demo (mode-averaging of a regressor vs a generative head) reuses the
goal/v_max fields; the reacher path uses the act_dim / obs / embed fields.
"""

from dataclasses import dataclass


@dataclass
class VLAConfig:
    act_dim: int = 2            # reacher action is a 2D joint torque (shoulder, wrist) in [-1, 1]
    chunk: int = 4             # default chunk length H; the ablation sweeps {1, 4, 16}
    cond_dim: int = 64          # width the heads project the conditioning to inside the MLP
    hidden: int = 256           # MLP hidden width
    time_dim: int = 64          # sinusoidal time-embedding width for the flow/ddpm heads

    n_flow_steps: int = 10      # Euler steps for the flow-matching ODE sampler at inference
    ddpm_T: int = 50            # diffusion steps for the DDPM contrast head

    # Image observation and the CNN encoder. The encoder output width embed_dim is the heads'
    # cond_in: the action heads read this vector and are agnostic to where it came from.
    obs_ch: int = 3
    obs_size: int = 64
    embed_dim: int = 128

    # Reacher episode budget for demo collection and rollout.
    max_steps: int = 80         # cap before declaring an episode a miss

    # Point-mass side-demo fields (the retained multimodal regression-vs-generative lesson).
    v_max: float = 0.05         # point-mass action clip for the 2D side-demo
    eps: float = 0.05           # point-mass success radius for the 2D side-demo
    n_goals: int = 4
    repr: str = "onehot"        # point-mass goal encoding: "onehot" or "coord"


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
    obs_ch: int = 3
    obs_size: int = 64
    embed_dim: int = 8
    max_steps: int = 80
    v_max: float = 0.1
    eps: float = 0.05
    n_goals: int = 4
    repr: str = "onehot"
