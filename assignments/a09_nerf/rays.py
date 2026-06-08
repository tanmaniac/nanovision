"""Ray generation and stratified depth sampling for volume rendering.

A pixel back-projects to a ray in the camera frame (the pinhole model), which rotates into
world coordinates by the camera-to-world matrix. The ray origin is the camera center. Depth
samples are stratified: split [near, far] into uniform bins and (optionally) jitter one
sample inside each bin so training sees a continuous range of depths rather than a fixed grid.

Convention: OpenCV +z forward (matching nanovision.geometry), not the original NeRF's OpenGL
-z. rays_d is normalized to unit length, so a depth value z is Euclidean distance along the
ray and near/far are distances, not z-depths. The same unit rays_d feeds the MLP color branch.

Shared OWNED file. Import its symbols through nanovision.volume, never by bare name.
"""

import torch
from torch import Tensor

from nanovision.geometry import unproject


def stratified_sample_rays(
    H: int,
    W: int,
    K: Tensor,
    c2w: Tensor,
    near: float,
    far: float,
    n_samples: int,
    *,
    perturb: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build one ray per pixel and stratified depth samples along each ray.

    Args:
        H, W: image height and width.
        K: (3, 3) pinhole intrinsic.
        c2w: (4, 4) camera-to-world transform (OpenCV +z forward).
        near, far: depth bounds along the ray (distances, since rays_d is unit length).
        n_samples: number of stratified samples per ray.
        perturb: jitter one sample uniformly within each [near, far] bin.
        generator: optional RNG for the jitter.

    Returns:
        rays_o: (H*W, 3) ray origins (the camera center, repeated).
        rays_d: (H*W, 3) unit-length world-frame ray directions.
        z_vals: (H*W, n_samples) sample distances stratified in [near, far].
    """
    raise NotImplementedError("implement stratified ray + depth sampling")


def sample_along_rays(rays_o: Tensor, rays_d: Tensor, z_vals: Tensor) -> Tensor:
    """3D sample points along each ray: o + z * d.

    Args:
        rays_o: (R, 3) origins.
        rays_d: (R, 3) unit directions.
        z_vals: (R, N) sample distances.

    Returns:
        (R, N, 3) world-frame points.
    """
    raise NotImplementedError("implement point sampling along rays")


def deltas_from_z(z_vals: Tensor) -> Tensor:
    """Segment lengths between consecutive samples, with a large final delta.

    Args:
        z_vals: (R, N) sample distances along the ray.

    Returns:
        (R, N) deltas. The last delta is set to 1e10 so the final sample absorbs the
        remaining transmittance (the original NeRF choice). Because rays_d is unit length,
        no ||d|| scale factor is needed.
    """
    raise NotImplementedError("implement consecutive-difference deltas")
