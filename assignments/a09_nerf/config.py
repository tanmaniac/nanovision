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

    # High-frequency surface albedo for the spectral-bias ablation. A smooth sphere has no
    # high-frequency content, so encoding cannot help there; these stripes are the signal the
    # encoding resolves and a raw-coordinate MLP blurs. focal_mult zooms the textured sphere
    # to fill the frame so the texture, not the background, drives the PSNR.
    texture_freq: float = 12.0
    focal_mult: float = 2.0

    # The ablation render (viz) needs a denser, slightly larger capture than the CPU test
    # above: separating "encoding resolves the texture" from "encoding overfits a few views"
    # requires enough views to constrain the high-frequency field. The graded test stays at
    # the small fast capture (n_views, H) and only overfits training rays, so it is unaffected.
    abl_n_views: int = 40
    abl_H: int = 48
    abl_steps: int = 3500

    white_background: bool = True
    lr: float = 5e-4
