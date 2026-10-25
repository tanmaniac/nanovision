"""Render the README figures for the camera-geometry notes.

Both figures use our road camera, the single camera of the IPM test: 1.5 m above the ego origin,
looking along ego +x and pitched 15 degrees down, focal length 300 px, a 400x224 image (the size
the nuScenes loader downsamples camera images to).

road_camera_side.png. Side view (ego x, z) of the camera, a pole standing on the ground 8 m ahead,
and the rays through points on the pole at heights 0, 0.5, 1.0, 1.4 and 1.7 m. Each ray is
continued to the ground plane, where flat-ground IPM writes the pixel: 8, 12, 24 and 120 m; the
1.7 m ray points upward and never meets the ground.

ipm_streak.png. A ray-cast camera image of a ground checkerboard (1 m squares) with a box standing
on it (a car-sized block, x in [8, 12.5] m, y in [-1, 1] m, 1.4 m tall), and the flat-ground IPM
of that image computed with the reference ipm_to_bev. The checkerboard comes out undistorted; the
box becomes a streak running away from the camera, starting at its true footprint (outlined).

Run from this directory:

    python make_ipm_figures.py
"""

import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NANOVISION_IMPL", "solution")
import torch  # noqa: E402

from nanovision.geometry import BEVGrid, CameraRig, ipm_to_bev, make_transform  # noqa: E402

H_CAM, PITCH, F, W, H = 1.5, math.radians(-15.0), 300.0, 400, 224
CX, CY = W / 2, H / 2

# Rows of R_cam_ego are the camera axes (right, down, forward) written in the ego frame.
fwd = np.array([math.cos(PITCH), 0.0, math.sin(PITCH)])
right = np.array([0.0, -1.0, 0.0])
down = np.cross(fwd, right)
R = np.stack([right, down, fwd])
C = np.array([0.0, 0.0, H_CAM])  # optical center in the ego frame

BOX_LO = np.array([8.0, -1.0, 0.0])
BOX_HI = np.array([12.5, 1.0, 1.4])


def side_view():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axhline(0, color="0.3", lw=1)
    ax.plot([8, 8], [0, 1.7], color="tab:red", lw=3, solid_capstyle="butt")
    ax.plot([0], [H_CAM], "ko", ms=6)
    ax.annotate("camera, 1.5 m", (0, H_CAM), (0.4, 2.15), fontsize=9)
    ax.plot([0, 32], [H_CAM, H_CAM], ls=":", color="0.5", lw=1)
    ax.text(26, H_CAM + 0.06, "camera height (horizon ray)", fontsize=8, color="0.4")
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    for h, col in zip([0.0, 0.5, 1.0, 1.4, 1.7], colors):
        if h < H_CAM:
            d = 8 * H_CAM / (H_CAM - h)
            xe = min(d, 32)
            ze = H_CAM + (h - H_CAM) * xe / 8
            ax.plot([0, xe], [H_CAM, ze], color=col, lw=1.2)
            if d <= 32:
                ax.plot([d], [0], "x", color=col, ms=9, mew=2)
                ax.text(d, -0.32, f"{d:.0f} m", ha="center", fontsize=9, color=col)
            else:
                ax.annotate(f"h = {h} m lands at {d:.0f} m", (32, ze), (24.5, 0.45),
                            fontsize=8, color=col, arrowprops=dict(arrowstyle="->", color=col))
        else:
            ze = H_CAM + (h - H_CAM) * 32 / 8
            ax.plot([0, 32], [H_CAM, ze], color=col, lw=1.2)
            ax.text(20, H_CAM + (h - H_CAM) * 20 / 8 + 0.12, "h = 1.7 m: ray rises, no ground point",
                    fontsize=8, color=col)
        ax.plot([8], [h], "o", color=col, ms=5)
        ax.text(8.25, h + 0.02 if h > 0 else 0.17, f"h = {h}", fontsize=8, color=col, va="center")
    ax.set_xlim(-0.5, 32)
    ax.set_ylim(-0.55, 2.6)
    ax.set_xlabel("ego x (m, forward)")
    ax.set_ylabel("ego z (m, up)")
    ax.set_title("Flat-ground IPM writes a pole at 8 m into ground points at 8, 12, 24, 120 m")
    fig.tight_layout()
    fig.savefig("road_camera_side.png", dpi=130)
    plt.close(fig)


def render():
    """Ray-cast the checkerboard-plus-box scene into the road camera."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([(us - CX) / F, (vs - CY) / F, np.ones_like(us, float)], -1)
    d = d_cam @ R  # camera direction to ego direction: R^T d, written for row vectors
    img = np.empty((H, W, 3))
    img[:] = (0.75, 0.85, 0.95)  # sky
    # ground hit
    with np.errstate(divide="ignore", invalid="ignore"):
        t_g = np.where(d[..., 2] < 0, -C[2] / d[..., 2], np.inf)
        # box hit (slab test)
        t1 = (BOX_LO - C) / d
        t2 = (BOX_HI - C) / d
    t_near = np.nanmax(np.minimum(t1, t2), -1)
    t_far = np.nanmin(np.maximum(t1, t2), -1)
    hit_box = (t_near <= t_far) & (t_near > 0) & (t_near < t_g)
    ground = np.isfinite(t_g) & ~hit_box
    with np.errstate(invalid="ignore"):
        p = C + t_g[..., None] * d
        chk = (np.floor(p[..., 0]) + np.floor(p[..., 1])) % 2
    shade = np.where(chk == 0, 0.35, 0.7)
    img[ground] = np.repeat(shade[ground][:, None], 3, 1)
    pb = C + t_near[..., None] * d
    top = np.isclose(pb[..., 2], BOX_HI[2], atol=1e-6)
    img[hit_box] = (0.85, 0.15, 0.1)
    img[hit_box & top] = (1.0, 0.45, 0.35)
    return img


def streak():
    img = render()
    K = torch.tensor([[F, 0, CX], [0, F, CY], [0, 0, 1.0]], dtype=torch.float32)
    extr = make_transform(torch.tensor(R, dtype=torch.float32),
                          torch.tensor(-R @ C, dtype=torch.float32))
    rig = CameraRig({"cam": K}, {"cam": extr}, {"cam": (W, H)})
    grid = BEVGrid(x_min=0.0, x_max=40.0, y_min=-10.0, y_max=10.0, resolution=0.1)
    timg = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    bev = ipm_to_bev({"cam": timg}, rig, grid).permute(1, 2, 0).numpy()

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    a0.imshow(img)
    a0.set_title("camera image (ground checkerboard, 1.4 m box at 8 m)")
    a0.axis("off")
    # show BEV with ego x up and ego y to the left, as seen from above
    a1.imshow(bev[::-1, ::-1], extent=[10, -10, 0, 40], aspect="auto")
    a1.add_patch(plt.Rectangle((1, 8), -2, 4.5, fill=False, ec="yellow", lw=2))
    a1.set_xlim(10, -10)
    a1.set_xlabel("ego y (m, positive to the left)")
    a1.set_ylabel("ego x (m, forward)")
    a1.set_title("flat-ground IPM (yellow: true footprint)")
    fig.tight_layout()
    fig.savefig("ipm_streak.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    side_view()
    streak()
    print("wrote road_camera_side.png and ipm_streak.png")
