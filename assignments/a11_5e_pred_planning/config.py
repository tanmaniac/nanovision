"""Hyperparameters for the motion-prediction toy. Provided, not a hole.

Dims are tiny so every CPU test runs in seconds: dim=64, n_modes=6, horizon=12, n_heads=4,
n_layers=2, roi_size=3 over an 8-channel BEV grid. A production motion decoder (MTR-style) uses
~64 intention queries, six attention layers, and a wider model; this is the mechanism isolator,
not the production layout (see the README).

roi_align normalization convention. roi_align_bev samples bev_feat (C, nx, ny) with
F.grid_sample. The grid is fed as (1, C, H, W) with H = nx, W = ny, and grid_sample reads the
last grid dimension as (x = width, y = height). So the grid's last dim must be
(normalized ny-coord, normalized nx-coord) - a SWAP relative to centers' (x_cell, y_cell) order.
Each axis is normalized with the align_corners=False cell-center rule
g = 2 * (cell + 0.5) / S - 1, S = nx or ny, with padding_mode="border" so edge agents sample the
boundary feature, not spurious zeros. This matches the grid_sample convention used in
A11.5c/A11.5d. RoI offsets are built in cell units (linspace(-radius, +radius, roi_size)), so the
window is radius cells regardless of grid resolution.
"""

from dataclasses import dataclass


@dataclass
class PredConfig:
    # BEV / RoI.
    in_ch: int = 8                  # BEV feature channels C (matches pred_toy_scene default)
    roi_size: int = 3               # RoI grid side; roi_size**2 tokens per agent
    radius: float = 1.0             # RoI half-extent in cell units

    # Decoder.
    dim: int = 64
    n_modes: int = 6                # K trajectory hypotheses
    horizon: int = 12               # future length T
    n_heads: int = 4
    n_layers: int = 2

    # Annealing recipe for the soft -> hard winner-take-all schedule.
    tau0: float = 3.0               # initial temperature
    anneal_frac: float = 0.6        # fraction of steps over which tau decays linearly to 0
    cls_weight: float = 1.0
