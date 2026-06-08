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
    device = c2w.device
    vs, us = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    px = torch.stack([us.reshape(-1), vs.reshape(-1)], dim=-1)        # (H*W, 2)
    # Camera-frame ray directions: lift each pixel to depth 1 (OpenCV +z forward).
    dirs_cam = unproject(px, torch.ones(px.shape[0], device=device), K)  # (H*W, 3)

    R_mat = c2w[:3, :3]
    rays_d = dirs_cam @ R_mat.T                                       # rotate to world
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)               # unit length
    rays_o = c2w[:3, 3].expand_as(rays_d)                            # camera center

    # Stratified depth sampling: uniform bins in [near, far], one jittered sample per bin.
    t = torch.linspace(0.0, 1.0, n_samples + 1, device=device)
    edges = near + (far - near) * t                                  # (n_samples+1,)
    lower = edges[:-1]
    upper = edges[1:]
    if perturb:
        u = torch.rand(rays_d.shape[0], n_samples, device=device, generator=generator)
        z_vals = lower + (upper - lower) * u                         # (R, n_samples)
    else:
        mids = 0.5 * (lower + upper)
        z_vals = mids.expand(rays_d.shape[0], n_samples).clone()
    return rays_o, rays_d, z_vals


def sample_along_rays(rays_o: Tensor, rays_d: Tensor, z_vals: Tensor) -> Tensor:
    """3D sample points along each ray: o + z * d.

    Args:
        rays_o: (R, 3) origins.
        rays_d: (R, 3) unit directions.
        z_vals: (R, N) sample distances.

    Returns:
        (R, N, 3) world-frame points.
    """
    return rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]


def deltas_from_z(z_vals: Tensor) -> Tensor:
    """Segment lengths between consecutive samples, with a large final delta.

    Args:
        z_vals: (R, N) sample distances along the ray.

    Returns:
        (R, N) deltas. The last delta is set to 1e10 so the final sample absorbs the
        remaining transmittance (the original NeRF choice). Because rays_d is unit length,
        no ||d|| scale factor is needed.
    """
    diffs = z_vals[..., 1:] - z_vals[..., :-1]                       # (R, N-1)
    last = torch.full_like(z_vals[..., :1], 1e10)
    return torch.cat([diffs, last], dim=-1)
