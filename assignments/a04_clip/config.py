"""Hyperparameters for the tiny CLIP/SigLIP dual encoder.

One small config object. The towers are deliberately tiny: a 16x16 image with patch 4
(16 patch tokens) and a short token caption. The point of A4 is the contrastive loss
and the zero-shot procedure, not a competitive encoder, so the encoders are small
stand-ins built on the shared transformer.

Temperature/bias initialization follows the papers: logit_scale is stored in log space
and initialized to log(1/0.07) (CLIP); the SigLIP paper instead initializes its temp to
log(10), close enough that the toy is indifferent. bias is initialized to -10 (the
SigLIP value), which keeps the many negative pairs near zero loss at the start.
"""

import math
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    # Image tower.
    img_size: int = 16
    patch: int = 4            # 4x4 grid -> 16 patch tokens
    in_chans: int = 3
    img_dim: int = 64
    img_depth: int = 3
    img_heads: int = 4

    # Text tower.
    vocab_size: int = 32      # ids: 0 = pad, 1..n_classes = class tokens,
                              # next = attribute tokens, vocab_size-1 = EOS (largest id)
    max_len: int = 8          # caption length (padded)
    txt_dim: int = 64
    txt_depth: int = 3
    txt_heads: int = 4

    # Shared embedding space.
    embed_dim: int = 64
    mlp_ratio: float = 4.0

    # Contrastive parameters.
    init_logit_scale: float = math.log(1.0 / 0.07)  # CLIP temperature init (log space)
    init_bias: float = -10.0                         # SigLIP bias init
    logit_scale_max: float = math.log(100.0)         # clamp the log param (1/tau <= 100)

    # Toy data.
    n_classes: int = 4        # latent classes; < overfit_batch so classes repeat
    n_attrs: int = 4          # nuisance attribute per pair

    # Overfit-one-batch / training.
    overfit_batch: int = 8    # synthetic (image, caption) pairs
    lr: float = 1e-3          # Adam learning rate
    steps: int = 400          # overfit steps
    seed: int = 0
