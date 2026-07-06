"""A11.5a - fill the holes, then run the tests.

The reference implementation lives in this assignment's `solution/geometry.py` (read it if you
get stuck). Do not import it here; implement the bodies yourself.

The pinhole model (project_points/unproject) and the four SE(3) primitives
(make_transform/apply_transform/invert_transform/compose_transforms) are built in the NeRF
assignment (A9) and imported here through the `nanovision.geometry` shim. This assignment owns
the multi-camera rig and the flat-ground IPM warp on top of them.

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

from nanovision.geometry import (
    apply_transform,
    invert_transform,
    project_points,
)

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
# Task 1: multi-camera rig
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
        """Transform world/ego points into camera `name`'s frame. (N,3)->(N,3)."""
        raise NotImplementedError("A11.5a Task 1: implement world_to_cam")

    def cam_to_world(self, name: str, pts_cam: Tensor) -> Tensor:
        """Transform camera-frame points back into the world/ego frame. (N,3)->(N,3)."""
        raise NotImplementedError("A11.5a Task 1: implement cam_to_world")

    def world_to_pixel(self, name: str, pts_world: Tensor):
        """Project world/ego points into camera `name`.

        Returns:
            px: (N, 2) pixel coordinates.
            valid: (N,) bool mask of points in front of the camera (z > 0) and,
                when image_sizes[name] is known, inside the image bounds (0 <= u < w,
                0 <= v < h).

        See the four-step lidar-to-camera-chain section of the README.
        """
        raise NotImplementedError("A11.5a Task 1: implement world_to_pixel")


# ---------------------------------------------------------------------------
# Task 2: inverse perspective mapping (flat-ground BEV)
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

    See the flat-ground-IPM section of the README for the per-cell data flow (ground point,
    projection, sample). Sampling conventions this file pins (not spelled out in the README):
    pixels are mapped to grid_sample's [-1, 1] extent with the align_corners=True map
    gx = 2u/(W-1) - 1, gy = 2v/(H-1) - 1; sampling is bilinear with zero padding; where cameras
    overlap the last camera wins.
    """
    raise NotImplementedError("A11.5a Task 2: implement ipm_to_bev")
