"""Pinhole projection and the SE(3) transform toolkit (A9).

This is the canonical, readable implementation the learner studies. These base
camera-geometry primitives - the pinhole model and its inverse, and the four rigid-
transform operations - are reused throughout the 3D and autonomous-driving parts of
the course through the `nanovision.geometry` shim. NeRF builds them first because ray
generation needs back-projection (unproject).

Conventions
-----------
Camera frame is OpenCV-style: +x right, +y down, +z forward (into the scene). The
intrinsic matrix K maps a camera-frame point (X, Y, Z) with Z > 0 to a pixel via
u = fx*X/Z + cx, v = fy*Y/Z + cy.

SE(3) transforms are 4x4 homogeneous matrices that map column-vector points on the
left: ``p' = T @ p_homogeneous``. ``apply_transform`` handles the homogeneous
bookkeeping for (N, 3) point arrays. A transform ``T_b_a`` reads "a-to-b": it takes
points expressed in frame a and returns them in frame b.

Everything is float-tensor and autograd-compatible (CPU is enough for the tests).

This file is owned by A9 and must NOT import nanovision.geometry (that would be a
self-cycle through the shim). It depends only on torch.
"""

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Pinhole projection
# ---------------------------------------------------------------------------


def project_points(pts_cam: Tensor, K: Tensor) -> Tensor:
    """Project camera-frame points to pixels with the pinhole model.

    Args:
        pts_cam: (..., 3) points in the camera frame (OpenCV axes, +z forward);
            leading batch dimensions pass through.
        K: (3, 3) intrinsic matrix.

    Returns:
        (N, 2) pixel coordinates (u, v). Points with z <= 0 are behind the
        camera; their pixels are returned but are not meaningful (the caller
        filters on depth).

    Formula:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
    """
    x = pts_cam[..., 0]
    y = pts_cam[..., 1]
    z = pts_cam[..., 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return torch.stack([u, v], dim=-1)


def unproject(px: Tensor, depth: Tensor, K: Tensor) -> Tensor:
    """Back-project pixels at a given depth to camera-frame points.

    Args:
        px: (..., 2) pixel coordinates (u, v); leading batch dimensions pass through.
        depth: broadcastable to px's leading dimensions, or scalar. Along +z (meters).
        K: (3, 3) intrinsic matrix.

    Returns:
        (N, 3) points in the camera frame.

    Formula (the inverse of project_points):
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d
    """
    u = px[..., 0]
    v = px[..., 1]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    d = depth if torch.is_tensor(depth) else torch.as_tensor(depth)
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d * torch.ones_like(u)
    return torch.stack([x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# SE(3) primitives
# ---------------------------------------------------------------------------


def make_transform(R: Tensor, t: Tensor) -> Tensor:
    """Assemble a 4x4 SE(3) matrix from rotation R and translation t.

    Args:
        R: (3, 3) rotation matrix.
        t: (3,) translation.

    Returns:
        (4, 4) homogeneous transform
            [[R, t],
             [0, 1]].
    """
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(T: Tensor, pts: Tensor) -> Tensor:
    """Apply a 4x4 SE(3) transform to a batch of 3-D points.

    Args:
        T: (4, 4) transform.
        pts: (..., 3) points; leading batch dimensions pass through.

    Returns:
        (N, 3) transformed points, computed as (R @ p) + t via homogeneous
        coordinates: append a 1, multiply by T, drop the homogeneous row.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return pts @ R.T + t


def invert_transform(T: Tensor) -> Tensor:
    """Invert a 4x4 SE(3) transform using its structure (no general inverse).

    For T = [[R, t], [0, 1]] the inverse is [[R^T, -R^T t], [0, 1]].
    """
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.T
    Tinv = torch.eye(4, dtype=T.dtype, device=T.device)
    Tinv[:3, :3] = Rt
    Tinv[:3, 3] = -Rt @ t
    return Tinv


def compose_transforms(*Ts: Tensor) -> Tensor:
    """Compose a sequence of 4x4 transforms left-to-right.

    compose_transforms(A, B, C) returns A @ B @ C, so applying the result to a
    point is the same as applying C, then B, then A.
    """
    if len(Ts) == 0:
        raise ValueError("compose_transforms needs at least one transform")
    out = Ts[0]
    for T in Ts[1:]:
        out = out @ T
    return out
