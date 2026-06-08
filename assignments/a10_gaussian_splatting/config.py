"""Hyperparameters for the 3D Gaussian splatting toy.

A10 fits a cloud of N 3D Gaussians to the posed colored-sphere images from the NeRF
assignment (nanovision.data.toy.nerf_synthetic_scene) by gradient descent through the
differentiable splat rasterizer. Tests render at 16x16 for speed; the viz fit uses 32x32.
Separate learning rates per parameter group follow the original 3D Gaussian splatting,
where positions move slowly and opacity/color move fast.
"""

from dataclasses import dataclass


@dataclass
class SplatConfig:
    n_gaussians: int = 200      # cloud size for the viz fit
    n_views: int = 6            # cameras on the ring (matches nerf_synthetic_scene)
    H: int = 32                 # viz render size
    W: int = 32
    sh_degree: int = 0          # view-independent color (sigmoid RGB); higher SH not built

    dilation: float = 0.3       # px^2 added to the 2D-covariance diagonal (EWA filter)

    # per-group learning rates (positions slow, appearance fast)
    lr_means: float = 1e-2
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 5e-2
    lr_color: float = 1e-2

    n_steps: int = 1500         # Adam steps for the viz multi-view fit

    init_spread: float = 1.0    # half-width of the init cube, centered on the world origin
    init_scale: float = 0.2     # initial isotropic Gaussian std
