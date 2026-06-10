"""Hyperparameters for the RSSM / DreamerV3 cartpole-from-pixels build. Provided, not a hole.

These are the DreamerV3-aligned values measured to train a continuous-control agent on
dm_control cartpole-balance from 64x64 pixels: the categorical latent is 32 categoricals x
32 classes = 1024 discrete dimensions, the deterministic GRU state is 512-wide, and the encoder
embedding is 1024-wide. The action is a single continuous force in [-1, 1] fed to the RSSM as a
(B, 1) float; the actor is a Tanh-Normal policy trained by dynamics backprop, not REINFORCE.

The two-hot bins live in SYMLOG space: bins = linspace(-20, 20, 255). The target is pushed through
symlog before encoding, and decode applies symexp once after taking the bin expectation, matching
the canonical DreamerV3 implementation (DiscDist with transfwd=symlog, transbwd=symexp,
buckets=linspace(-20, 20, 255)). The round-trip stays exact because a clean two-hot label has its
expectation exactly at symlog(y), and symexp(symlog(y)) = y. Symlog-space bins also keep the decode
bounded: with value-space bins the outer buckets would sit at symexp(20) ~ 5e8 and any residual
softmax tail mass there would dominate the expectation, blowing up imagined rewards and returns. The
bins are built lazily in nets.py from these fields via nets.value_bins(cfg).

kl_dyn_scale and kl_rep_scale are DreamerV3's beta_dyn / beta_rep (0.5 / 0.1), NOT DreamerV2's
0.8 / 0.2 KL balance. The 5:1 ratio moves the prior toward the posterior faster than the reverse.

The imagination horizon is 8 (DreamerV3 uses 15-16 at scale; 8 keeps the untrained-critic bootstrap
from inflating early imagined returns on this short build). gamma is 0.997, lambda 0.95.
"""

from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    # Observation and encoder.
    obs_size: int = 64              # 64x64 RGB render from dm_control
    obs_ch: int = 3
    embed_dim: int = 1024           # flattened encoder output width

    # RSSM latent.
    h_dim: int = 512                # deterministic GRU state width
    n_cat: int = 32                 # number of categorical heads
    n_cls: int = 32                 # classes per head -> 1024 latent dims
    action_dim: int = 1             # continuous 1-D force fed to the RSSM as a (B, 1) float

    # KL objective (DreamerV3 eq. 2-3).
    free_bits: float = 1.0          # free-bits floor in nats, applied to the SUMMED-over-heads KL
    kl_dyn_scale: float = 0.5       # beta_dyn: weight on the dynamics loss (sg(q) || p)
    kl_rep_scale: float = 0.1       # beta_rep: weight on the representation loss (q || sg(p))

    # Categorical sampling.
    unimix: float = 0.01            # uniform mixture weight blended into every categorical

    # Two-hot bins (symlog space).
    n_bins: int = 255
    bin_lo: float = -20.0
    bin_hi: float = 20.0

    # Actor-critic in imagination.
    horizon: int = 8                # imagination rollout length
    gamma: float = 0.997            # discount
    lam: float = 0.95               # lambda-return mixing
    ent_coef: float = 1e-4          # actor entropy bonus weight
    ret_ema_decay: float = 0.99     # EMA decay for the 5-95 percentile return spread


@dataclass
class GradcheckConfig:
    """A shrunk config so float64 gradcheck stays fast and deterministic."""

    obs_size: int = 16
    obs_ch: int = 3
    embed_dim: int = 16
    h_dim: int = 8
    n_cat: int = 4
    n_cls: int = 4
    action_dim: int = 1
    free_bits: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1
    unimix: float = 0.01
    n_bins: int = 31
    bin_lo: float = -20.0
    bin_hi: float = 20.0
    horizon: int = 4
    gamma: float = 0.997
    lam: float = 0.95
    ent_coef: float = 1e-4
    ret_ema_decay: float = 0.99
