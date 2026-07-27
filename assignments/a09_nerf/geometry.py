"""Pinhole projection and the SE(3) transform toolkit (A9).

These are the base camera-geometry primitives for the whole 3D and AV part of the
course: the pinhole model and its inverse, and the four rigid-transform operations.
NeRF builds them first because ray generation needs back-projection (unproject); the
later 3D and autonomous-driving assignments import them through the `nanovision.geometry`
shim, never bare.

The reference implementation lives in this assignment's `solution/geometry.py` (read it
if you get stuck). Do not import it here; implement the bodies yourself.

Conventions
-----------
Camera frame is OpenCV-style: +x right, +y down, +z forward (into the scene). The
intrinsic matrix K maps a camera-frame point (X, Y, Z) with Z > 0 to a pixel by the
standard pinhole model (see the README).

SE(3) transforms are 4x4 homogeneous matrices applied on the left: p' = T @ p_homogeneous.
A transform T_b_a reads "a-to-b": it takes points in frame a and returns them in frame b.
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
        pts_cam: (..., 3) points in the camera frame (OpenCV axes, +z forward).
            Usually (N, 3), but any leading batch dimensions must pass through: the
            geometry foundation models assignment calls this with (B, H, W, 3).
        K: (3, 3) intrinsic matrix.

    Returns:
        (..., 2) pixel coordinates (u, v), matching pts_cam's leading dimensions.

    See the "Pinhole projection" section of the README.
    """
    raise NotImplementedError("implement pinhole project_points")


def unproject(px: Tensor, depth: Tensor, K: Tensor) -> Tensor:
    """Back-project pixels at a given depth to camera-frame points.

    Args:
        px: (..., 2) pixel coordinates (u, v). Usually (N, 2), but any leading batch
            dimensions must pass through: the geometry foundation models assignment
            calls this with (B, H, W, 2).
        depth: broadcastable to px's leading dimensions, or scalar. Depth along
            +z (meters).
        K: (3, 3) intrinsic matrix.

    Returns:
        (..., 3) points in the camera frame, matching px's leading dimensions. This is
        the inverse of project_points.

    See the "Pinhole projection" section of the README.
    """
    raise NotImplementedError("implement unproject")


# ---------------------------------------------------------------------------
# SE(3) primitives
# ---------------------------------------------------------------------------


def make_transform(R: Tensor, t: Tensor) -> Tensor:
    """Assemble a 4x4 SE(3) matrix from rotation R and translation t.

    Args:
        R: (3, 3) rotation matrix.
        t: (3,) translation.

    Returns:
        (4, 4) homogeneous SE(3) transform.

    See the Rigid transforms and SE(3) section of the README.
    """
    raise NotImplementedError("implement make_transform")


def apply_transform(T: Tensor, pts: Tensor) -> Tensor:
    """Apply a 4x4 SE(3) transform to a batch of 3-D points.

    Args:
        T: (4, 4) transform.
        pts: (..., 3) points. Usually (N, 3), but any leading batch dimensions must
            pass through: the geometry foundation models assignment calls this with
            (B, H, W, 3).

    Returns:
        (..., 3) transformed points, matching pts' leading dimensions.

    See the Rigid transforms and SE(3) section of the README.
    """
    raise NotImplementedError("implement apply_transform")


def invert_transform(T: Tensor) -> Tensor:
    """Invert a 4x4 SE(3) transform using its structure, not a general matrix inverse.

    See the Rigid transforms and SE(3) section of the README.
    """
    raise NotImplementedError("implement invert_transform")


def compose_transforms(*Ts: Tensor) -> Tensor:
    """Compose a sequence of 4x4 transforms left-to-right.

    Applying compose_transforms(A, B, C) to a point is the same as applying C, then B,
    then A. See the Rigid transforms and SE(3) section of the README.
    """
    raise NotImplementedError("implement compose_transforms")
