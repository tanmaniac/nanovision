"""Hyperparameters for the tiny video MAE used by tests, viz, and runs.

One small config object so the magic numbers live in one place. The defaults are a
6-frame 16x16 toy clip with spatial patch 4 and temporal tubelet 2, so the tubelet
grid is T'=3 by 4x4 = 48 tokens. The encoder is the asymmetric MAE design (enc dim 64
depth 4, a lighter decoder dim 48 depth 2).

Masking: tube masking keeps whole spatial columns across all temporal steps, so the
ratio is quantized to 1 - k/16 (16 = the 4x4 spatial grid). VideoMAE (He/Tong et al.,
2022) uses 90-95%; we use 0.875 (keep 2 of 16 spatial columns -> 6 of 48 tubelets),
high enough to be in the paper's spirit while a depth-4 toy still overfits one clip.
The README states the 90-95% standard explicitly.
"""

from dataclasses import dataclass


@dataclass
class VideoSSLConfig:
    # Clip / tubelet grid.
    img_size: int = 16        # spatial side length in pixels
    patch: int = 4            # spatial patch side; spatial grid = (img/patch)^2 = 16
    tubelet_t: int = 2        # temporal tubelet depth
    n_frames: int = 6         # T; temporal tokens T' = n_frames / tubelet_t = 3
    in_chans: int = 3         # RGB

    # Asymmetric MAE: large encoder, light decoder.
    enc_dim: int = 64         # encoder token dimension
    enc_depth: int = 4        # encoder transformer blocks
    enc_heads: int = 4        # encoder attention heads
    dec_dim: int = 48         # decoder token dimension (smaller than encoder)
    dec_depth: int = 2        # decoder transformer blocks (shallower than encoder)
    dec_heads: int = 4        # decoder attention heads
    mlp_ratio: float = 4.0

    # Tube masking. keep_spatial = round((1 - mask_ratio) * spatial_grid).
    mask_ratio: float = 0.875  # keep 2 of 16 spatial columns; VideoMAE uses 0.9-0.95

    # Overfit-one-batch settings (the gating signal).
    overfit_batch: int = 8    # synthetic clips in the fixed batch
    n_blobs: int = 3          # independently moving blobs per clip
    mae_lr: float = 2e-3      # Adam learning rate
    mae_steps: int = 800      # overfit steps
    seed: int = 0             # determinism seed
