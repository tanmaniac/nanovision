"""Closed-form GT pointmaps for the two-view sphere toy. Provided.

This is the supervision the GeometryFM model overfits. There is no network here: given the
two camera poses, the intrinsic K, and a chosen sphere (center + radius), it computes the
exact 3D surface point each patch-center ray hits, for both views, expressed in CAMERA 1's
frame. A patch whose center ray misses the sphere is invalid (masked out of the loss).

Why off-center the sphere and widen the baseline. A single sphere centered at the world
origin viewed from a symmetric ring gives two near-identical views, so placing view 2's
points in view 1's frame needs almost no information from view 2 and the cross-attention is
idle. Translating the sphere off the origin and choosing two ring cameras far apart makes
each view see a different surface portion, so the cross-view attention has work to do. The
cross-attention ablation in the tests/viz measures whether this actually lowers the loss
floor.

Both pointmaps live in cam1's frame:
  X^{1,1} : view-1 pixels as 3D points in cam1 frame (the points are already in cam1).
  X^{2,1} : view-2 pixels as 3D points in cam1 frame, obtained cam2 -> world -> cam1 with
            X^{2,1} = inv(T1) @ T2 @ X^{2,2}, T_i = camera-i-to-world (the toy's c2w poses).

The pixel grid uses the patch centers of a (grid x grid) tiling of the image, matching the
PointmapHead's row-major (h, w) reshape and the toy's K (principal point at ((W-1)/2,
(H-1)/2)).
"""

import torch
from torch import Tensor

from nanovision.geometry import apply_transform, invert_transform
from nanovision.data import toy


def _patch_center_pixels(img_size: int, patch: int, dtype, device) -> Tensor:
    """(grid, grid, 2) pixel (u, v) at the center of each patch, row-major over (row, col)."""
    grid = img_size // patch
    centers = (torch.arange(grid, dtype=dtype, device=device) + 0.5) * patch - 0.5  # per axis
    vs, us = torch.meshgrid(centers, centers, indexing="ij")  # vs = row, us = col
    return torch.stack([us, vs], dim=-1)  # (grid, grid, 2) -> (u, v)


def _ray_sphere_front(o: Tensor, d: Tensor, center: Tensor, radius: float):
    """Front (near) ray-sphere intersection.

    o: (..., 3) ray origins, d: (..., 3) unit ray directions, center: (3,). Returns the 3D
    entry point (..., 3) and a bool hit mask (...). For a missed ray the point is o (masked).
    """
    oc = o - center
    b = 2.0 * (d * oc).sum(-1)
    c = (oc * oc).sum(-1) - radius * radius
    disc = b * b - 4.0 * c
    hit = disc > 0.0
    sqrt_disc = torch.sqrt(disc.clamp(min=0.0))
    t0 = (-b - sqrt_disc) / 2.0          # near root
    hit = hit & (t0 > 0.0)               # in front of the camera
    point = o + t0[..., None] * d
    return point, hit


def stereo_pointmap_gt(cfg) -> dict:
    """Closed-form GT pointmaps for the two chosen views, both in cam1 frame.

    Args:
        cfg: GeometryFMConfig (uses img_size, patch, radius, cam_dist, sphere_center,
            view1, view2, n_ring).

    Returns:
        dict with:
          img1, img2: (3, H, W) the two toy images (channel-first, for the ViT).
          gt_pts1, gt_pts2: (grid, grid, 3) GT pointmaps in cam1 frame.
          valid1, valid2: (grid, grid) bool masks (ray hits the sphere).
          K: (3, 3) intrinsic.
          T1, T2: (4, 4) camera-to-world poses of the two views.
          T_1to2: (4, 4) relative pose cam1 -> cam2 (for the reprojection check).
    """
    images, poses, K, _, _ = toy.nerf_synthetic_scene(
        n_views=cfg.n_ring, H=cfg.img_size, W=cfg.img_size,
        radius=cfg.radius, cam_dist=cfg.cam_dist,
    )
    dtype = K.dtype
    device = K.device
    center = torch.tensor(cfg.sphere_center, dtype=dtype, device=device)

    T1 = poses[cfg.view1]                 # cam1 -> world
    T2 = poses[cfg.view2]                 # cam2 -> world
    img1 = images[cfg.view1].permute(2, 0, 1).contiguous()  # (3, H, W)
    img2 = images[cfg.view2].permute(2, 0, 1).contiguous()

    px = _patch_center_pixels(cfg.img_size, cfg.patch, dtype, device)  # (grid, grid, 2)
    u, v = px[..., 0], px[..., 1]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Camera-frame ray directions (OpenCV +z forward), unit length.
    dirs_cam = torch.stack(
        [(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1
    )  # (grid, grid, 3)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    def view_points(T):
        R = T[:3, :3]
        o = T[:3, 3].expand_as(dirs_cam)              # camera center in world
        d_world = dirs_cam @ R.T                       # rotate dirs to world
        d_world = d_world / d_world.norm(dim=-1, keepdim=True)
        pt_world, hit = _ray_sphere_front(o, d_world, center, cfg.radius)
        return pt_world, hit

    pw1, hit1 = view_points(T1)
    pw2, hit2 = view_points(T2)

    # Express both in cam1 frame: world -> cam1 is inv(T1).
    T_world_to_1 = invert_transform(T1)
    gt_pts1 = apply_transform(T_world_to_1, pw1)       # X^{1,1}
    gt_pts2 = apply_transform(T_world_to_1, pw2)       # X^{2,1}

    # Relative pose cam1 -> cam2: world->cam2 composed with cam1->world.
    T_1to2 = invert_transform(T2) @ T1

    return {
        "img1": img1, "img2": img2,
        "gt_pts1": gt_pts1, "gt_pts2": gt_pts2,
        "valid1": hit1, "valid2": hit2,
        "K": K, "T1": T1, "T2": T2, "T_1to2": T_1to2,
    }
