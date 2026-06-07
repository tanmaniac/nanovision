"""A11.5a starter - fill the holes, then run the tests.

The reference implementation lives in `nanovision/geometry.py` (read it if you
get stuck). Do not import it here; implement the bodies yourself.

Conventions
-----------
Camera frame is OpenCV-style: +x right, +y down, +z forward (into the scene),
matching the nuScenes camera convention. SE(3) transforms are 4x4 homogeneous
matrices applied on the left: p' = T @ p_homogeneous. A transform T_b_a reads
"a-to-b": takes points in frame a, returns them in frame b. Everything is
float-tensor and autograd-compatible.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Task 1: pinhole projection
# ---------------------------------------------------------------------------


def project_points(pts_cam: Tensor, K: Tensor) -> Tensor:
    """Project camera-frame points to pixels with the pinhole model.

    Args:
        pts_cam: (N, 3) points in the camera frame (OpenCV axes, +z forward).
        K: (3, 3) intrinsic matrix.

    Returns:
        (N, 2) pixel coordinates (u, v).

    Formula:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
    """
    raise NotImplementedError("A11.5a Task 1: implement pinhole project_points")


def unproject(px: Tensor, depth: Tensor, K: Tensor) -> Tensor:
    """Back-project pixels at a given depth to camera-frame points.

    Args:
        px: (N, 2) pixel coordinates (u, v).
        depth: (N,) or scalar depth along +z (meters).
        K: (3, 3) intrinsic matrix.

    Returns:
        (N, 3) points in the camera frame.

    Formula (the inverse of project_points):
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d
    """
    raise NotImplementedError("A11.5a Task 1: implement unproject")


# ---------------------------------------------------------------------------
# Task 2: SE(3) primitives
# ---------------------------------------------------------------------------


def make_transform(R: Tensor, t: Tensor) -> Tensor:
    """Assemble a 4x4 SE(3) matrix from rotation R and translation t.

    Args:
        R: (3, 3) rotation matrix.
        t: (3,) translation.

    Returns:
        (4, 4) homogeneous transform [[R, t], [0, 1]].
    """
    raise NotImplementedError("A11.5a Task 2: implement make_transform")


def apply_transform(T: Tensor, pts: Tensor) -> Tensor:
    """Apply a 4x4 SE(3) transform to a batch of 3-D points.

    Args:
        T: (4, 4) transform.
        pts: (N, 3) points.

    Returns:
        (N, 3) transformed points, (R @ p) + t via homogeneous coordinates.
    """
    raise NotImplementedError("A11.5a Task 2: implement apply_transform")


def invert_transform(T: Tensor) -> Tensor:
    """Invert a 4x4 SE(3) transform using its structure (no general inverse).

    For T = [[R, t], [0, 1]] the inverse is [[R^T, -R^T t], [0, 1]].
    """
    raise NotImplementedError("A11.5a Task 2: implement invert_transform")


def compose_transforms(*Ts: Tensor) -> Tensor:
    """Compose a sequence of 4x4 transforms left-to-right.

    compose_transforms(A, B, C) returns A @ B @ C, so applying the result to a
    point is the same as applying C, then B, then A.
    """
    raise NotImplementedError("A11.5a Task 2: implement compose_transforms")


# ---------------------------------------------------------------------------
# BEV grid contract (provided - read it, no hole)
# ---------------------------------------------------------------------------


@dataclass
class BEVGrid:
    """Ego-centric bird's-eye-view grid, the shared contract for the AV module.

    Centered on the ego vehicle in the ego frame. x is forward, y is left
    (nuScenes ego frame, +z up). Defaults: [-50, 50] m on both axes at 0.5 m
    resolution, a 200x200 grid. LSS / BEVFormer / occupancy reuse this.
    """

    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    resolution: float = 0.5

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    def cell_centers(self, dtype=torch.float32) -> Tensor:
        """(nx, ny, 2) ego-frame (x, y) coordinates of every cell center."""
        xs = self.x_min + (torch.arange(self.nx, dtype=dtype) + 0.5) * self.resolution
        ys = self.y_min + (torch.arange(self.ny, dtype=dtype) + 0.5) * self.resolution
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack([gx, gy], dim=-1)


# ---------------------------------------------------------------------------
# Task 3: multi-camera rig
# ---------------------------------------------------------------------------


class CameraRig:
    """A set of cameras, each with intrinsics K and an extrinsic transform.

    Extrinsics are stored as T_cam_world (world-to-camera) per camera: the 4x4
    SE(3) that takes a point in the world/ego frame and expresses it in that
    camera's frame. For a vehicle, "world" is the ego frame.

    Args:
        Ks: dict name -> (3, 3) intrinsic.
        extrinsics: dict name -> (4, 4) world-to-camera transform.
        image_sizes: dict name -> (width, height); used by world_to_pixel for
            in-frame visibility.
    """

    def __init__(self, Ks: dict, extrinsics: dict, image_sizes: dict | None = None):
        if set(Ks) != set(extrinsics):
            raise ValueError("Ks and extrinsics must have the same camera names")
        self.names = list(Ks.keys())
        self.Ks = Ks
        self.extrinsics = extrinsics  # T_cam_world per camera
        self.image_sizes = image_sizes or {}

    def world_to_cam(self, name: str, pts_world: Tensor) -> Tensor:
        """Transform world/ego points into camera `name`'s frame. (N,3)->(N,3).

        Use apply_transform with this camera's extrinsic (already world-to-cam).
        """
        raise NotImplementedError("A11.5a Task 3: implement world_to_cam")

    def cam_to_world(self, name: str, pts_cam: Tensor) -> Tensor:
        """Transform camera-frame points back into the world/ego frame.

        Invert the world-to-camera extrinsic, then apply_transform.
        """
        raise NotImplementedError("A11.5a Task 3: implement cam_to_world")

    def world_to_pixel(self, name: str, pts_world: Tensor):
        """Project world/ego points into camera `name`.

        Returns:
            px: (N, 2) pixel coordinates.
            valid: (N,) bool mask of points in front of the camera (z > 0) and,
                when image_sizes is known, inside the image bounds.

        Steps: world_to_cam, then project_points; valid is cam z > 0 AND (if
        image_sizes[name] is set) 0 <= u < w and 0 <= v < h.
        """
        raise NotImplementedError("A11.5a Task 3: implement world_to_pixel")


# ---------------------------------------------------------------------------
# Task 4: inverse perspective mapping (flat-ground BEV)
# ---------------------------------------------------------------------------
#
# Flat-ground assumption and where it breaks: ipm_to_bev maps every BEV cell to
# its ground point at z = ground_z, projects that single point into the camera,
# and samples the image there. Exact for things on the ground (lane markings).
# Wrong for anything elevated: a point at height h projects to the same pixel as
# a ground point farther from the camera, so elevated objects get painted into a
# BEV cell beyond their true footprint, smeared toward the camera. The fix is
# real depth (LSS, A11.5b) or explicit 3-D queries (BEVFormer, A11.5c).


def ipm_to_bev(
    images: dict,
    rig: CameraRig,
    bev_grid: BEVGrid,
    ground_z: float = 0.0,
) -> Tensor:
    """Warp camera images onto the ego ground plane to form a naive BEV image.

    For each BEV cell center (x, y) at height ground_z in the ego frame, project
    that 3-D ground point into each camera and bilinearly sample the image.
    Cells are filled by whichever camera sees them; later cameras overwrite
    earlier ones where they overlap.

    Args:
        images: dict name -> (C, H, W) float image tensor in [0, 1].
        rig: the CameraRig (extrinsics are ego-to-camera).
        bev_grid: the BEVGrid contract.
        ground_z: ground-plane height in the ego frame (meters).

    Returns:
        (C, nx, ny) BEV image. Cells with no camera coverage are zero.

    Steps:
        1. Get cell centers (nx, ny, 2); build ego ground points (M, 3) with
           z = ground_z.
        2. For each camera: rig.world_to_pixel the ground points; normalize
           pixels to grid_sample's [-1, 1] range
           (gx = 2u/(W-1) - 1, gy = 2v/(H-1) - 1); reshape to (1, nx, ny, 2);
           call F.grid_sample(img[None], grid, mode="bilinear",
           padding_mode="zeros", align_corners=True).
        3. Write the sampled colors into bev only where valid (last camera wins).
    """
    raise NotImplementedError("A11.5a Task 4: implement ipm_to_bev")
