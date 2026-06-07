"""Synthetic cameras and a synthetic 4-camera rig for the A11.5a tests.

No dataset is required. The rig places four cameras at the origin looking out
along +x, +y, -x, -y in the world (ego) frame, each with OpenCV camera axes
(+z forward). This is enough to test world_to_pixel visibility and IPM.
"""

import math

import torch

from geometry import BEVGrid, CameraRig, make_transform


def make_K(fx=400.0, fy=400.0, w=400, h=224) -> torch.Tensor:
    """A simple pinhole intrinsic with the principal point at the image center."""
    return torch.tensor(
        [[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def _rot_world_to_cam_facing(yaw: float) -> torch.Tensor:
    """Rotation taking world points to a camera that looks along world direction
    given by `yaw` (radians, 0 = +x, pi/2 = +y), with OpenCV camera axes:
    cam +z = view direction, cam +x = right, cam +y = down (world -z).
    """
    fwd = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
    down = torch.tensor([0.0, 0.0, -1.0])  # world -z is camera down
    right = torch.cross(down, fwd, dim=0)
    right = right / right.norm()
    # Columns of R_cam_world^T are the camera axes in world coords:
    # rows of R_cam_world are [right; down; fwd].
    R = torch.stack([right, down, fwd], dim=0)
    return R


def make_synthetic_rig(w=400, h=224, fx=400.0) -> CameraRig:
    """Four cameras at the origin facing +x, +y, -x, -y (ego frame)."""
    names = ["front", "left", "back", "right"]
    yaws = [0.0, math.pi / 2, math.pi, -math.pi / 2]
    Ks, extr, sizes = {}, {}, {}
    for name, yaw in zip(names, yaws):
        R = _rot_world_to_cam_facing(yaw)
        # camera at world origin -> t = -R @ 0 = 0
        extr[name] = make_transform(R, torch.zeros(3))
        Ks[name] = make_K(fx=fx, fy=fx, w=w, h=h)
        sizes[name] = (w, h)
    return CameraRig(Ks, extr, sizes)


def make_ground_camera(height=1.5, pitch_deg=-15.0, w=400, h=224, fx=300.0):
    """A single forward-looking camera at `height` above the ground, pitched down.

    Returns (K, extrinsic_world_to_cam). World/ego frame: x forward, y left,
    z up. The camera is at (0, 0, height) looking forward and down by pitch_deg.
    """
    K = make_K(fx=fx, fy=fx, w=w, h=h)
    pitch = math.radians(pitch_deg)
    # Forward-looking base (look along +x): cam axes in world.
    # right = -y_world (camera +x is to the right when facing +x), down tilts.
    # Build R_cam_world from camera basis vectors expressed in world.
    # Looking direction (cam +z): forward and slightly down.
    fwd = torch.tensor([math.cos(pitch), 0.0, math.sin(pitch)])
    right = torch.tensor([0.0, -1.0, 0.0])  # camera +x = world -y
    down = torch.cross(fwd, right, dim=0)
    down = down / down.norm()
    R = torch.stack([right, down, fwd], dim=0)  # rows = cam axes in world
    cam_pos = torch.tensor([0.0, 0.0, float(height)])
    t = -R @ cam_pos
    extr = make_transform(R, t)
    return K, extr


def checkerboard(w=400, h=224, sq=20, c=3) -> torch.Tensor:
    """A checkerboard image (c, h, w) in [0, 1]."""
    ys = torch.arange(h).view(h, 1)
    xs = torch.arange(w).view(1, w)
    board = ((xs // sq + ys // sq) % 2).float()
    return board.unsqueeze(0).repeat(c, 1, 1)
