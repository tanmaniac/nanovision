"""Hyperparameters for the tiny VQ tokenizer.

A 16x16 image is encoded to a 4x4 grid of D-dimensional vectors, each snapped to one of K
codebook entries, and decoded back. A small autoregressive transformer then models the 16
discrete tokens. Everything is tiny: the point of A6.5 is vector quantization with the
straight-through estimator and the discrete autoregressive prior, not a competitive
tokenizer.
"""

from dataclasses import dataclass


@dataclass
class VQConfig:
    img_size: int = 16
    channels: int = 1
    hidden: int = 64          # encoder/decoder conv width
    code_dim: int = 16        # D, codebook vector dimension
    num_codes: int = 32       # K, codebook size
    beta: float = 0.25        # commitment-loss weight (van den Oord et al. 2017)
    grid: int = 4             # latent grid side (4x4 -> 16 tokens)

    # Autoregressive prior over the 16 tokens.
    prior_dim: int = 64
    prior_depth: int = 3
    prior_heads: int = 4
