"""Hyperparameters for the tiny MAE and DINO models used by tests, viz, and runs.

One small config object so the magic numbers live in one place. The defaults are
the CIFAR-10-scale tiny setup from the research scope: 32x32 images, patch 4 (so
N = 64 patch tokens). MAE uses an asymmetric encoder (dim 64, depth 4) and a
lighter decoder (dim 32, depth 2) with 75% masking. DINO uses the same tiny ViT
backbone with a projection head to K prototypes, an EMA teacher, and centering +
sharpening temperatures.
"""

from dataclasses import dataclass


@dataclass
class SSLConfig:
    # Image / patch grid (shared by MAE and DINO).
    img_size: int = 32        # input image side length in pixels
    patch: int = 4            # patch side; N = (img_size / patch) ** 2 = 64
    in_chans: int = 3         # RGB input channels

    # MAE encoder (large) and decoder (light): the asymmetric design.
    enc_dim: int = 64         # encoder token dimension
    enc_depth: int = 4        # encoder transformer blocks
    enc_heads: int = 4        # encoder attention heads
    dec_dim: int = 48         # decoder token dimension (smaller than encoder)
    dec_depth: int = 2        # decoder transformer blocks (shallower than encoder)
    dec_heads: int = 4        # decoder attention heads
    mask_ratio: float = 0.75  # fraction of patches masked (He et al., 2022)

    # DINO backbone (the student/teacher ViT) and projection head.
    dino_dim: int = 64        # backbone token dimension
    dino_depth: int = 4       # backbone transformer blocks
    dino_heads: int = 4       # backbone attention heads
    out_dim: int = 128        # K prototypes in the projection head
    head_hidden: int = 256    # projection-head hidden width
    teacher_temp: float = 0.04  # teacher softmax temperature (sharpening)
    student_temp: float = 0.1   # student softmax temperature
    ema_momentum: float = 0.996      # teacher EMA momentum lambda (teaching default)
    center_momentum: float = 0.9     # centering buffer EMA momentum
    # The collapse test needs a fast-tracking teacher so the teacher follows the
    # student into the degenerate state within a few steps. The overfit test instead
    # freezes the teacher (a captured fixed target), so it does not use this.
    collapse_momentum: float = 0.9   # fast-tracking teacher for the collapse test

    # Multi-crop: two global crops + several smaller local crops.
    n_global: int = 2         # number of global crops
    n_local: int = 4          # number of local crops
    global_size: int = 32     # global crop side length (full image)
    local_size: int = 16      # local crop side length

    # Overfit-one-batch and collapse-test settings (gating signals).
    overfit_batch: int = 8    # number of synthetic images in the fixed batch
    mae_lr: float = 2e-3      # MAE Adam learning rate
    mae_steps: int = 800      # MAE overfit steps
    dino_lr: float = 1e-3     # DINO Adam learning rate
    dino_steps: int = 150     # DINO overfit / collapse steps
    seed: int = 0             # determinism seed
