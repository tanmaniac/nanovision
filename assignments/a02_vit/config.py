"""Hyperparameters for the tiny ViT used by the tests, viz, and CIFAR run.

One small config object so the magic numbers live in one place. The defaults are
the CIFAR-10-scale tiny ViT from the research scope: 32x32 images, patch 4 (so
N = 64 patch tokens), dim 64, depth 2, 4 heads.
"""

from dataclasses import dataclass


@dataclass
class ViTConfig:
    img_size: int = 32        # input image side length in pixels
    patch: int = 4            # patch side; N = (img_size / patch) ** 2 = 64
    in_chans: int = 3         # RGB input channels
    dim: int = 64             # model / token dimension
    depth: int = 2            # number of transformer encoder blocks
    n_heads: int = 4          # attention heads (dim must divide by this)
    mlp_ratio: float = 4.0    # FFN expansion (classic ViT GELU MLP, 4x)
    n_registers: int = 4      # learnable register tokens (Darcet et al., 2024)
    num_classes: int = 10     # CIFAR-10 / synthetic class count
    pool: str = "cls"         # image representation: "cls" or "mean"

    # Overfit-one-batch settings (the wiring test and viz curve).
    overfit_lr: float = 3e-3  # Adam learning rate
    overfit_steps: int = 500  # optimization steps on the single batch
    overfit_batch: int = 8    # number of synthetic images
    seed: int = 0             # determinism seed for the overfit batch
