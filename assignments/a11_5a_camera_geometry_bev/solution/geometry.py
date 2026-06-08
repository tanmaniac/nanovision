"""Camera geometry for the autonomous-driving module (A11.5a).

This is the canonical, readable implementation the learner studies. The
assignment's ``solution/geometry.py`` re-exports from here; ``starter/geometry.py``
is the holed copy the learner fills in.

Conventions
-----------
Camera frame is OpenCV-style: +x right, +y down, +z forward (into the scene).
This matches the nuScenes camera convention, so the intrinsic matrix K maps a
camera-frame point (X, Y, Z) with Z > 0 to a pixel via u = fx*X/Z + cx,
v = fy*Y/Z + cy.

SE(3) transforms are 4x4 homogeneous matrices that map column-vector points
on the left: ``p' = T @ p_homogeneous``. ``apply_transform`` handles the
homogeneous bookkeeping for (N, 3) point arrays. A transform ``T_b_a`` reads
"a-to-b": it takes points expressed in frame a and returns them in frame b.

Everything is float-tensor and autograd-compatible (CPU is enough for the tests).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Pinhole projection
# ---------------------------------------------------------------------------


def project_points(pts_cam: Tensor, K: Tensor) -> Tensor:
    """Project camera-frame points to pixels with the pinhole model.

    Args:
        pts_cam: (N, 3) points in the camera frame (OpenCV axes, +z forward).
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
        pts: (N, 3) points.

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


# ---------------------------------------------------------------------------
# BEV grid contract (shared by the whole AV module)
# ---------------------------------------------------------------------------


@dataclass
class BEVGrid:
    """Ego-centric bird's-eye-view grid, defined once for the whole module.

    The grid is centered on the ego vehicle in the ego frame. x is forward,
    y is left (matching the nuScenes ego frame, which is right-handed with
    +z up). A cell stores the ground-plane footprint at z = 0 in ego frame.

    Defaults follow the common nuScenes BEV setup: [-50, 50] m in both axes at
    0.5 m resolution, a 200x200 grid. LSS, BEVFormer, and occupancy in later
    assignments reuse this contract (occupancy adds a z axis).

    Attributes:
        x_min, x_max: forward extent in meters (ego x).
        y_min, y_max: lateral extent in meters (ego y, +left).
        resolution: meters per cell (square cells).
    """

    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    resolution: float = 0.5

    @property
    def nx(self) -> int:
        """Number of cells along x (forward)."""
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def ny(self) -> int:
        """Number of cells along y (lateral)."""
        return int(round((self.y_max - self.y_min) / self.resolution))

    def cell_centers(self, dtype=torch.float32) -> Tensor:
        """Ego-frame (x, y) coordinates of every cell center.

        Returns:
            (nx, ny, 2) tensor. Index [i, j] holds the (x, y) center of the
            cell in row i (x) and column j (y).
        """
        xs = self.x_min + (torch.arange(self.nx, dtype=dtype) + 0.5) * self.resolution
        ys = self.y_min + (torch.arange(self.ny, dtype=dtype) + 0.5) * self.resolution
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack([gx, gy], dim=-1)


# ---------------------------------------------------------------------------
# Multi-camera rig
# ---------------------------------------------------------------------------


class CameraRig:
    """A set of cameras, each with its intrinsics K and an extrinsic transform.

    Extrinsics are stored as ``T_cam_world`` (world-to-camera) per camera: the
    4x4 SE(3) that takes a point in the world/ego frame and expresses it in that
    camera's frame. For a multi-camera vehicle, "world" is the ego frame, so the
    rig describes the 6-camera nuScenes setup directly.

    Args:
        Ks: dict name -> (3, 3) intrinsic.
        extrinsics: dict name -> (4, 4) world-to-camera transform.
        image_sizes: dict name -> (width, height) in pixels, used by
            world_to_pixel to report in-frame visibility.
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
        return apply_transform(self.extrinsics[name], pts_world)

    def cam_to_world(self, name: str, pts_cam: Tensor) -> Tensor:
        """Transform camera-frame points back into the world/ego frame."""
        return apply_transform(invert_transform(self.extrinsics[name]), pts_cam)

    def world_to_pixel(self, name: str, pts_world: Tensor):
        """Project world/ego points into camera `name`.

        Returns:
            px: (N, 2) pixel coordinates.
            valid: (N,) bool mask of points that are in front of the camera
                (z > 0) and, when image_sizes is known, inside the image
                bounds.
        """
        pts_cam = self.world_to_cam(name, pts_world)
        px = project_points(pts_cam, self.Ks[name])
        valid = pts_cam[..., 2] > 0
        if name in self.image_sizes:
            w, h = self.image_sizes[name]
            u, v = px[..., 0], px[..., 1]
            valid = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return px, valid


# ---------------------------------------------------------------------------
# Inverse perspective mapping (flat-ground BEV)
# ---------------------------------------------------------------------------
#
# Flat-ground assumption and where it breaks: ipm_to_bev maps every BEV cell to
# its ground point at z = ground_z, projects that single point into the camera,
# and samples the image there. This is exact for things that actually lie on the
# ground (lane markings, road texture). It is wrong for anything elevated: a
# point at height h projects to the same pixel as a ground point that is farther
# from the camera, so an object above the ground gets painted into a BEV cell
# beyond its true footprint, smeared toward the camera. Tall objects (cars,
# trucks, pedestrians) stretch outward. The fix is real depth (LSS, A11.5b) or
# explicit 3-D queries (BEVFormer, A11.5c); the flat-ground homography cannot
# recover height from a single view.


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
    """
    centers = bev_grid.cell_centers()  # (nx, ny, 2), ego x/y
    nx, ny, _ = centers.shape
    # Ground points in ego frame: (nx*ny, 3) with z = ground_z.
    flat = centers.reshape(-1, 2)
    zc = torch.full((flat.shape[0], 1), float(ground_z), dtype=flat.dtype)
    pts_ego = torch.cat([flat, zc], dim=-1)  # (M, 3)

    any_c = next(iter(images.values()))
    C = any_c.shape[0]
    bev = torch.zeros(C, nx, ny, dtype=any_c.dtype)
    filled = torch.zeros(nx, ny, dtype=torch.bool)

    for name in rig.names:
        if name not in images:
            continue
        img = images[name]  # (C, H, W)
        _, H, W = img.shape
        px, valid = rig.world_to_pixel(name, pts_ego)  # (M,2), (M,)
        u, v = px[..., 0], px[..., 1]
        # Normalize to grid_sample's [-1, 1] convention.
        gx = 2.0 * u / (W - 1) - 1.0
        gy = 2.0 * v / (H - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(1, nx, ny, 2)
        sampled = F.grid_sample(
            img.unsqueeze(0), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )[0]  # (C, nx, ny)
        valid_grid = valid.reshape(nx, ny)
        write = valid_grid  # overwrite policy: last camera wins
        bev[:, write] = sampled[:, write]
        filled = filled | write

    return bev
