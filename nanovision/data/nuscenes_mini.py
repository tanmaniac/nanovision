"""nuScenes-mini loader and calibration utilities (A11.5a).

This is provided boilerplate - the learner does not implement the devkit
plumbing. The taught mechanism (SE(3) primitives, projection, IPM) lives in
``nanovision.geometry`` and is tested on synthetic cameras with no dataset.

The module is importable even when ``nuscenes-devkit`` / ``pyquaternion`` are
not installed: the devkit import happens lazily inside the loader, never at
module top. A clear, actionable error is raised when the dataset path is unset
or missing.

Dataset step zero (see the assignment README): create a nuScenes account,
accept the license, download ``v1.0-mini`` (~4 GB), install the ``av`` extra
(``pip install -e ".[av]"``), and set ``NUSCENES_DATAROOT`` to the directory
that contains the ``v1.0-mini`` folder.

Conventions
-----------
nuScenes stores rotations as pyquaternion scalar-first quaternions (w, x, y, z)
in both ``calibrated_sensor`` and ``ego_pose``. Camera images are pre-undistorted,
so K is an exact pinhole intrinsic with no distortion terms. The ego frame is
right-handed (x forward, y left, z up); camera frames are OpenCV (x right,
y down, z forward). The BEV grid is the ego-centric ``BEVGrid`` from
``nanovision.geometry``.
"""

import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from nanovision.geometry import BEVGrid, CameraRig, invert_transform, make_transform

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def _missing_dataroot_message(dataroot) -> str:
    return (
        "nuScenes-mini dataset not found.\n"
        f"  NUSCENES_DATAROOT resolved to: {dataroot!r}\n"
        "Step zero:\n"
        "  1. Create a nuScenes account and accept the license at "
        "https://www.nuscenes.org/nuscenes#download\n"
        "  2. Download the v1.0-mini split (~4 GB) and extract it.\n"
        "  3. pip install -e \".[av]\"   (installs nuscenes-devkit + pyquaternion)\n"
        "  4. export NUSCENES_DATAROOT=/path/to/dir  "
        "(the dir that contains the v1.0-mini folder)\n"
    )


def quaternion_wxyz_to_matrix(q) -> Tensor:
    """Rotation matrix from a scalar-first (w, x, y, z) quaternion.

    nuScenes / pyquaternion store quaternions scalar-first. This builds the 3x3
    rotation directly so the loader does not depend on pyquaternion for the math
    (pyquaternion is only needed to parse the devkit records).

    Args:
        q: length-4 array-like (w, x, y, z).

    Returns:
        (3, 3) rotation matrix tensor.
    """
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return torch.eye(3)
    s = 2.0 / n
    R = torch.tensor(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )
    return R


def transform_from_record(translation, rotation_wxyz) -> Tensor:
    """4x4 sensor/ego-to-parent transform from a nuScenes calibration record.

    ``calibrated_sensor`` gives sensor-to-ego; ``ego_pose`` gives ego-to-global.
    Both store translation (meters) and a scalar-first quaternion. The result
    maps a point in the child frame to the parent frame:
        p_parent = R * p_child + t.
    """
    R = quaternion_wxyz_to_matrix(rotation_wxyz)
    t = torch.tensor([float(v) for v in translation], dtype=torch.float32)
    return make_transform(R, t)


class NuScenesMini(Dataset):
    """nuScenes v1.0-mini wrapper exposing per-sample geometry and images.

    Per sample (a keyframe), ``__getitem__`` returns a dict with:
        - ``images``: dict camera -> (3, H, W) float image in [0, 1], downsampled.
        - ``rig``: a ``CameraRig`` with per-camera K (scaled to the downsampled
          size) and ego-to-camera extrinsics.
        - ``lidar``: (N, 3) lidar points in the lidar sensor frame.
        - ``lidar_to_ego``, ``ego_pose_lidar``, ``ego_pose_cam`` (per camera):
          the SE(3) transforms for the four-step lidar-to-camera chain.
        - ``bev_grid``: the shared ``BEVGrid`` contract.

    Args:
        dataroot: dataset root (the dir containing v1.0-mini). Falls back to the
            NUSCENES_DATAROOT env var.
        version: dataset version string.
        image_size: (width, height) to downsample images to (default 400x224).

    Raises:
        FileNotFoundError: if the dataroot is unset or does not exist.
        ImportError: with an actionable message if the devkit is not installed.
    """

    def __init__(
        self,
        dataroot: str | None = None,
        version: str = "v1.0-mini",
        image_size: tuple[int, int] = (400, 224),
        bev_grid: BEVGrid | None = None,
    ):
        dataroot = dataroot or os.environ.get("NUSCENES_DATAROOT")
        if not dataroot or not os.path.isdir(dataroot):
            raise FileNotFoundError(_missing_dataroot_message(dataroot))

        try:
            from nuscenes.nuscenes import NuScenes  # lazy: not needed to import
        except ImportError as e:
            raise ImportError(
                "nuscenes-devkit is not installed. Run: pip install -e \".[av]\""
            ) from e

        self.dataroot = dataroot
        self.version = version
        self.image_size = image_size
        self.bev_grid = bev_grid or BEVGrid()
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.sample_tokens = [s["token"] for s in self.nusc.sample]

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _load_image(self, filename: str) -> Tensor:
        from PIL import Image

        path = os.path.join(self.dataroot, filename)
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        w, h = self.image_size
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        scale = (w / w0, h / h0)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), scale

    def _scale_intrinsic(self, K: Tensor, scale) -> Tensor:
        sx, sy = scale
        K = K.clone()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K

    def __getitem__(self, idx: int) -> dict:
        sample = self.nusc.get("sample", self.sample_tokens[idx])

        # Lidar: points, sensor-to-ego extrinsic, ego pose at lidar time.
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = self.nusc.get(
            "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
        )
        lidar_ego = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        lidar_to_ego = transform_from_record(
            lidar_cs["translation"], lidar_cs["rotation"]
        )
        ego_pose_lidar = transform_from_record(
            lidar_ego["translation"], lidar_ego["rotation"]
        )
        lidar_path = os.path.join(self.dataroot, lidar_sd["filename"])
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        lidar = torch.from_numpy(pts).contiguous()

        images, Ks, extrinsics, image_sizes, ego_pose_cam = {}, {}, {}, {}, {}
        for cam in CAMERAS:
            if cam not in sample["data"]:
                continue
            cam_sd = self.nusc.get("sample_data", sample["data"][cam])
            cam_cs = self.nusc.get(
                "calibrated_sensor", cam_sd["calibrated_sensor_token"]
            )
            cam_ego = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

            img, scale = self._load_image(cam_sd["filename"])
            images[cam] = img
            image_sizes[cam] = self.image_size

            K = torch.tensor(
                np.array(cam_cs["camera_intrinsic"]), dtype=torch.float32
            )
            Ks[cam] = self._scale_intrinsic(K, scale)

            # Extrinsic stored by the rig is ego-to-camera (world_to_cam).
            cam_to_ego = transform_from_record(
                cam_cs["translation"], cam_cs["rotation"]
            )
            extrinsics[cam] = invert_transform(cam_to_ego)
            ego_pose_cam[cam] = transform_from_record(
                cam_ego["translation"], cam_ego["rotation"]
            )

        rig = CameraRig(Ks, extrinsics, image_sizes)
        return {
            "images": images,
            "rig": rig,
            "lidar": lidar,
            "lidar_to_ego": lidar_to_ego,
            "ego_pose_lidar": ego_pose_lidar,
            "ego_pose_cam": ego_pose_cam,
            "bev_grid": self.bev_grid,
        }
