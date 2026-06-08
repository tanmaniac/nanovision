"""Hyperparameters for the latent-diffusion DiT toy.

Two stages: a KL-regularized VAE compresses a 16x16x1 image into a 4x4x4 continuous
latent (downsample factor f = 4 per spatial axis), and a small diffusion transformer
(DiT) predicts a flow-matching velocity in that latent space, conditioned on timestep and
class label. beta is the KL weight; it is small (1e-4) so reconstruction dominates and the
latent keeps spatial structure rather than collapsing toward pure noise.
"""

from dataclasses import dataclass


@dataclass
class DiTConfig:
    # image / VAE
    image_size: int = 16
    channels: int = 1
    num_classes: int = 3
    latent_dim: int = 4         # C, the latent channel count
    f: int = 4                  # per-axis spatial downsample (16 -> 4)
    beta: float = 1e-4          # KL weight in the VAE loss

    # DiT
    patch_size: int = 1         # p; with a 4x4 latent this gives N = 16 tokens
    d_model: int = 64
    n_heads: int = 2
    n_blocks: int = 4
    mlp_ratio: int = 4
    time_dim: int = 64          # sinusoidal timestep-embedding width

    # sampling
    n_steps: int = 50           # Euler steps for the velocity ODE
