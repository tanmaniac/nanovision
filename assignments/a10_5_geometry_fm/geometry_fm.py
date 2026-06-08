"""Pointmap and depth utilities for DUSt3R-style geometry foundation models (A10.5).

A pointmap is a per-pixel map of 3D coordinates: for an image of shape (H, W), the
pointmap is (H, W, 3) holding the camera-frame XYZ of the surface each pixel sees. These
three functions convert between depth, pointmaps, and pixels, and they are reused by the
overfit model and the cross-view consistency check.

This file is owned by A10.5 and re-exported through `nanovision.geometry`. Import the camera
primitives it builds on from there too, never bare:

    from nanovision.geometry import unproject, project_points, apply_transform

The pixel-center convention matches the toy scene (nerf_synthetic_scene): the principal
point sits at ((W-1)/2, (H-1)/2) and pixel (i, j) has center coordinate (u, v) = (j, i).
Get this wrong and the depth round-trip is off by half a pixel.
"""

import torch
from torch import Tensor

from nanovision.geometry import unproject, project_points, apply_transform


def depth_to_pointmap(depth: Tensor, K: Tensor) -> Tensor:
    """Back-project a depth map to a per-pixel pointmap in the camera frame.

    Args:
        depth: (B, H, W) depth along +z (OpenCV camera, +z forward).
        K: (3, 3) pinhole intrinsic.

    Returns:
        (B, H, W, 3) camera-frame points. Pixel (i, j) maps to the point
        unproject((u, v)=(j, i), depth[b, i, j], K), i.e. the pixel grid uses
        u = column index, v = row index, with the principal point at ((W-1)/2, (H-1)/2)
        carried inside K. Reuses `unproject`.
    """
    raise NotImplementedError("back-project depth to a per-pixel camera-frame pointmap")


def pointmap_to_depth(pts: Tensor) -> Tensor:
    """Read the depth (z-channel) off a camera-frame pointmap.

    Args:
        pts: (B, H, W, 3) camera-frame points.

    Returns:
        (B, H, W) depth = the z component. The exact inverse of depth_to_pointmap's
        depth (the round-trip anchor).
    """
    raise NotImplementedError("return the z-component of the pointmap as depth")


def reproject_pointmap(pts_cam1: Tensor, T_1to2: Tensor, K: Tensor) -> Tensor:
    """Reproject a cam1-frame pointmap into image-2 pixel coordinates.

    Transform the points from camera 1 into camera 2 with the relative pose T_1to2
    (a 4x4 that takes a cam1 point to cam2), then project with K. This is the cross-view
    consistency check: image-1 pixels carried as 3D points and reprojected into image 2
    should land where image 2 observes the same surface.

    Args:
        pts_cam1: (B, H, W, 3) points in camera 1's frame.
        T_1to2: (4, 4) relative pose, cam1 -> cam2.
        K: (3, 3) intrinsic.

    Returns:
        (B, H, W, 2) pixel coordinates (u, v) in image 2.
    """
    raise NotImplementedError("transform cam1 points to cam2 and project to image-2 pixels")
