"""A11.5a viz: always produces PNGs in out/, headless.

If NUSCENES_DATAROOT is set and the data + devkit are present, this renders the
6-camera lidar overlay (naive vs temporal-correct) and the stitched BEV from a
real sample. Otherwise it falls back to a fully synthetic scene: a cube projected
into a synthetic rig, and a ground checkerboard warped into the BEV grid.

Run with: make viz A=a115a_camera_geometry_bev  (uses the reference solution).
"""

import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from nanovision.determinism import set_seed  # noqa: E402
from nanovision.geometry import (  # noqa: E402
    BEVGrid,
    CameraRig,
    ipm_to_bev,
    make_transform,
)

OUT = Path(__file__).parent / "out"


def _synthetic_rig(w=400, h=224, fx=320.0):
    """Six cameras at the origin, roughly the nuScenes layout, facing outward."""
    names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
    yaws = [0.0, 55.0, -55.0, 180.0, 125.0, -125.0]
    Ks, extr, sizes = {}, {}, {}
    for name, yaw_deg in zip(names, yaws):
        yaw = math.radians(yaw_deg)
        fwd = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
        down = torch.tensor([0.0, 0.0, -1.0])
        right = torch.cross(down, fwd, dim=0)
        right = right / right.norm()
        R = torch.stack([right, down, fwd], dim=0)
        extr[name] = make_transform(R, torch.zeros(3))
        Ks[name] = torch.tensor(
            [[fx, 0.0, w / 2], [0.0, fx, h / 2], [0.0, 0.0, 1.0]]
        )
        sizes[name] = (w, h)
    return CameraRig(Ks, extr, sizes)


def _cube(center, size=1.0):
    base = torch.tensor(center, dtype=torch.float32)
    s = size / 2
    offs = torch.tensor(
        [[dx, dy, dz] for dx in (-s, s) for dy in (-s, s) for dz in (-s, s)]
    )
    return base + offs


def _checkerboard(w=400, h=224, sq=24):
    ys = torch.arange(h).view(h, 1)
    xs = torch.arange(w).view(1, w)
    board = ((xs // sq + ys // sq) % 2).float()
    return board.unsqueeze(0).repeat(3, 1, 1)


def synthetic_scene():
    set_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    rig = _synthetic_rig()

    # 1. Project a cube into the rig and draw which cameras see it.
    cube = _cube([8.0, 1.0, 0.0], size=2.0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, name in zip(axes.flat, rig.names):
        px, valid = rig.world_to_pixel(name, cube)
        w, h = rig.image_sizes[name]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f"CAM_{name}  ({int(valid.sum())}/8 corners)")
        if valid.any():
            ax.scatter(px[valid, 0], px[valid, 1], c="tab:red", s=40)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Synthetic cube projected into the 6-camera rig")
    fig.tight_layout()
    p1 = OUT / "synthetic_cube_projection.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # 2. Warp a ground checkerboard from a tilted camera into the BEV grid.
    from math import radians

    pitch = radians(-15.0)
    fwd = torch.tensor([math.cos(pitch), 0.0, math.sin(pitch)])
    right = torch.tensor([0.0, -1.0, 0.0])
    down = torch.cross(fwd, right, dim=0)
    down = down / down.norm()
    R = torch.stack([right, down, fwd], dim=0)
    cam_pos = torch.tensor([0.0, 0.0, 1.5])
    extr = make_transform(R, -R @ cam_pos)
    K = torch.tensor([[300.0, 0.0, 200.0], [0.0, 300.0, 112.0], [0.0, 0.0, 1.0]])
    grnd_rig = CameraRig({"cam": K}, {"cam": extr}, {"cam": (400, 224)})
    img = _checkerboard()
    grid = BEVGrid(x_min=0.0, x_max=40.0, y_min=-20.0, y_max=20.0, resolution=0.25)
    bev = ipm_to_bev({"cam": img}, grnd_rig, grid, ground_z=0.0)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 5))
    a0.imshow(img.permute(1, 2, 0).numpy())
    a0.set_title("camera image (ground checkerboard)")
    a0.axis("off")
    a1.imshow(bev.permute(1, 2, 0).numpy(), origin="lower")
    a1.set_title("flat-ground IPM into BEV (ego x up, y right)")
    a1.axis("off")
    fig.tight_layout()
    p2 = OUT / "synthetic_ipm_bev.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"synthetic fallback wrote {p1} and {p2}")


def real_scene(dataroot):
    from nanovision.data.nuscenes_mini import NuScenesMini
    from nanovision.geometry import (
        apply_transform,
        compose_transforms,
        invert_transform,
        project_points,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    ds = NuScenesMini(dataroot=dataroot, image_size=(400, 224))
    s = ds[0]
    lidar = s["lidar"]
    lidar_h = torch.cat([lidar, torch.ones(lidar.shape[0], 1)], dim=-1)[:, :3]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, name in zip(axes.flat, s["rig"].names):
        img = s["images"][name]
        ax.imshow(img.permute(1, 2, 0).numpy())
        cam_extr = s["rig"].extrinsics[name]
        K = s["rig"].Ks[name]
        for tag, ego_cam, color in [
            ("naive", s["ego_pose_lidar"], "tab:blue"),
            ("correct", s["ego_pose_cam"][name], "tab:red"),
        ]:
            T = compose_transforms(
                cam_extr,
                invert_transform(ego_cam),
                s["ego_pose_lidar"],
                s["lidar_to_ego"],
            )
            pc = apply_transform(T, lidar_h)
            front = pc[:, 2] > 1.0
            px = project_points(pc[front], K)
            w, h = s["rig"].image_sizes[name]
            inb = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
            ax.scatter(px[inb, 0], px[inb, 1], s=1, c=color, alpha=0.4, label=tag)
        ax.set_title(name)
        ax.axis("off")
        ax.legend(markerscale=4, loc="upper right")
    fig.suptitle("lidar overlay: naive (blue) vs temporal-correct (red)")
    fig.tight_layout()
    p1 = OUT / "nuscenes_lidar_overlay.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    bev = ipm_to_bev(s["images"], s["rig"], s["bev_grid"], ground_z=0.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(bev.permute(1, 2, 0).numpy(), origin="lower")
    ax.set_title("stitched flat-ground BEV (6 cameras)")
    ax.axis("off")
    fig.tight_layout()
    p2 = OUT / "nuscenes_bev.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"real-scene wrote {p1} and {p2}")


def main():
    dataroot = os.environ.get("NUSCENES_DATAROOT")
    have_data = bool(dataroot) and os.path.isdir(dataroot or "")
    have_devkit = True
    try:
        import nuscenes  # noqa: F401
    except ImportError:
        have_devkit = False
    if have_data and have_devkit:
        try:
            real_scene(dataroot)
            return
        except Exception as e:  # noqa: BLE001 - fall back, never fail viz
            print(f"real scene failed ({e}); falling back to synthetic", file=sys.stderr)
    synthetic_scene()


if __name__ == "__main__":
    main()
