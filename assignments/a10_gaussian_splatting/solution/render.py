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
    device, dtype = means2d.device, means2d.dtype
    # 1. Sort front to back (ascending camera-space z) and gather the values. argsort
    # carries no gradient; the gathered values do.
    order = torch.argsort(depths)
    means2d = means2d[order]
    cov2d = cov2d[order]
    colors = colors[order]
    opacities = opacities[order]

    # 2. Closed-form 2x2 inverse (the conic), once per Gaussian. The 0.3 dilation keeps the
    # determinant bounded away from zero.
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 0]
    d = cov2d[:, 1, 1]
    det = a * d - b * c                                   # (N,)
    inv = torch.stack([d, -b, -c, a], dim=-1).reshape(-1, 2, 2) / det[:, None, None]

    # 3. Pixel grid (u, v) = (x, y). means2d is (u, v) from project_points.
    vs, us = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                            torch.arange(W, dtype=dtype, device=device), indexing="ij")
    px = torch.stack([us, vs], dim=-1).reshape(-1, 2)    # (P, 2), P = H*W

    dx = px[None, :, :] - means2d[:, None, :]            # (N, P, 2)
    # Mahalanobis quadratic form d^T C d, broadcast the conic over pixels.
    cdx = torch.einsum("nij,npj->npi", inv, dx)          # (N, P, 2)
    power = (dx * cdx).sum(-1)                            # (N, P)
    g = torch.exp(-0.5 * power)                           # (N, P)
    alpha = (opacities[:, None] * g).clamp(max=0.99)     # (N, P)

    # 4. Front-to-back compositing with exclusive transmittance T_i = prod_{j<i}(1-alpha_j).
    one_minus = 1.0 - alpha                              # (N, P)
    ones = torch.ones(1, alpha.shape[1], dtype=dtype, device=device)
    T = torch.cumprod(torch.cat([ones, one_minus[:-1]], dim=0), dim=0)  # (N, P)
    weights = alpha * T                                  # (N, P)
    color = torch.einsum("np,nc->pc", weights, colors)  # (P, 3)
    T_final = (T[-1] * one_minus[-1]) if alpha.shape[0] > 0 else torch.ones_like(color[:, 0])
    color = color + T_final[:, None] * bg               # leftover transmittance onto bg
    return color.reshape(H, W, 3)
