"""The differentiable splat rasterizer: depth-sort, evaluate, alpha-composite.

Each projected 2D Gaussian contributes a per-pixel weight that falls off with the squared
Mahalanobis distance from its center. The Gaussians are sorted front to back by depth and
composited with the SAME exclusive-transmittance alpha compositing as the ray-marched
volume renderer (A9's volume_render):

    C(x) = sum_i c_i alpha_i prod_{j<i}(1 - alpha_j) + T_final * bg

with alpha_i = opacity_i * exp(-0.5 * d^T Sigma_2D_i^-1 d), d = x - mu_i, clamped to 0.99.
The only difference from the ray-marched renderer is the source of alpha: there it comes
from 1 - exp(-sigma*delta) along a ray; here it comes from a depth-sorted projected 2D
Gaussian. The compositing math is identical.

The depth sort is non-differentiable (argsort carries no gradient), but that is fine: it
only reorders the per-Gaussian tensors. Gradients flow through the gathered VALUES (means,
covariances, colors, opacities), so autograd is correct as long as `depths` itself is not
treated as a differentiable input to the final loss.

Assignment-local. Import bare.
"""

import torch
from torch import Tensor


def splat_render(means2d: Tensor, cov2d: Tensor, colors: Tensor, opacities: Tensor,
                 depths: Tensor, H: int, W: int, *, bg: float = 0.0) -> Tensor:
    """Rasterize projected Gaussians into an (H, W, 3) image by front-to-back compositing.

    Steps:
      1. Sort the Gaussians by depth, nearest first, and gather every per-Gaussian tensor
         by that order (the sort is non-differentiable; the gathered values are).
      2. Invert each 2x2 covariance once, in closed form (the 0.3 dilation guards the
         determinant), and broadcast the conic over all pixels. Do not invert per pixel.
      3. For each pixel x and Gaussian i, alpha_i = opacity_i * exp(-0.5 * d^T C_i d) with
         d = x - mu_i and C_i = Sigma_2D_i^-1, clamped to <= 0.99.
      4. Composite: C = sum_i c_i alpha_i T_i + T_final * bg, with the exclusive
         transmittance T_i = prod_{j<i}(1 - alpha_j), T_0 = 1.

    Vectorized over pixels: all N Gaussians are evaluated against the H*W pixel grid with
    broadcasting, no Python tile loops.

    Args:
        means2d: (N, 2) pixel centers.
        cov2d: (N, 2, 2) screen-space covariances.
        colors: (N, 3) RGB in [0, 1].
        opacities: (N,) in [0, 1].
        depths: (N,) camera-space z, the sort key (front-to-back = ascending z).
        H, W: image size.
        bg: background gray level applied to the leftover transmittance.

    Returns:
        (H, W, 3) rendered image in [0, 1].
    """
    raise NotImplementedError("sort by depth, evaluate the 2D Gaussians, alpha-composite")
