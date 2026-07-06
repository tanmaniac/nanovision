"""The differentiable splat rasterizer: depth-sort, evaluate, alpha-composite.

Each projected 2D Gaussian contributes a per-pixel weight that falls off with the squared
Mahalanobis distance from its center. The Gaussians are sorted front to back by depth and
composited with the SAME exclusive-transmittance alpha compositing as the ray-marched
volume renderer (A9's volume_render). The only difference is the source of alpha: there it
comes from the density along a ray; here it comes from a depth-sorted projected 2D
Gaussian. The compositing math is identical.

See the "The rasterizer" section of the README for the per-pixel alpha and the compositing.

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

    The 2x2 covariances are inverted once in closed form (the dilation guards the
    determinant) and broadcast over the pixel grid, not inverted per pixel. See the "The
    rasterizer" section of the README for the depth sort, the per-pixel alpha, and the
    compositing.

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
