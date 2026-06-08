"""Hyperparameters for the NeRF toy scene. Provided.

The toy is one colored solid sphere imaged from a ring of cameras at 16x16 resolution. The
defaults keep every test on CPU in seconds: 32 samples per ray, a 128-wide MLP, 6 position
bands. The full Lego scene (800x800, 20k+ steps, ~22 dB PSNR) is out of scope; see the README.
"""

from dataclasses import dataclass


@dataclass
class NeRFConfig:
    H: int = 16
    W: int = 16
    n_views: int = 6
    n_samples: int = 32         # stratified samples per ray

    pos_L: int = 6              # position encoding bands (tied to the spectral-bias ablation)
    dir_L: int = 4              # direction encoding bands
    hidden: int = 128
    n_layers: int = 4
    include_input: bool = True

    radius: float = 1.0         # sphere radius (world units)
    sphere_sigma: float = 8.0   # constant interior density of the toy sphere
    cam_dist: float = 4.0       # camera distance from the origin
    scene_bound: float = 4.0    # positions divided by this before encoding (~[-1, 1])

    white_background: bool = True
    lr: float = 5e-4
