"""Figures for the README: our vehicle pillar, and pillar coverage of the toy rig.

Run from this directory: python make_pillar_figures.py. It uses the toy scene generator and the
camera rig from the nanovision package, and writes the pinhole projection out in numpy so it does
not depend on the assignment's holes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nanovision.data import toy  # noqa: E402

IMG, F, C0, H_CAM = 32, 16.0, 15.5, 1.5
ZS = np.linspace(-5.0, 3.0, 4)          # anchor heights of every pillar
CELL = (7, 11)                          # our vehicle pillar: BEV row 7, column 11
CENTER = (-0.5, 3.5)                    # its ego (x, y) center


def cam_axes(k, n=4):
    a = 2 * np.pi * k / n
    ca, sa = np.cos(a), np.sin(a)
    return np.array([sa, -ca, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([ca, sa, 0.0])


def project(k, p):
    xc, yc, zc = cam_axes(k)
    d = np.asarray(p, float) - np.array([0.0, 0.0, H_CAM])
    X, Y, Z = d @ xc, d @ yc, d @ zc
    if Z <= 0:
        return None
    return F * X / Z + C0, F * Y / Z + C0


def in_frame(uv):
    return uv is not None and 0 <= uv[0] < IMG and 0 <= uv[1] < IMG


scene = toy.bev_multicam_scene(n_cams=4, n_vehicles=4, img=IMG, stride=4, focal=F, seed=0)
img_left = scene["images"][0, 1].permute(1, 2, 0).numpy()
veh = scene["vehicles"][0].numpy()

# Figure 1: our vehicle pillar, side view in the left camera and the left camera image.
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
ymax = 7.0
for s in (-1, 1):  # vertical field of view of the left camera (v from 0 to 32 px)
    slope = ((32 if s > 0 else 0) - 0.5 - C0 + 0.5) / F
    ax0.plot([0, ymax], [H_CAM, H_CAM - slope * ymax], color="0.6", lw=1, ls="--")
ax0.add_patch(plt.Rectangle((veh[0, 1] - 1.0, 0), 2.0, 1.5, color="orange", alpha=0.5, lw=0))
ax0.plot(0, H_CAM, "ks", ms=8)
ax0.text(0.15, H_CAM + 0.3, "left camera", fontsize=9)
for z in ZS:
    uv = project(1, (CENTER[0], CENTER[1], z))
    ok = in_frame(uv)
    col = "tab:blue" if ok else "tab:red"
    ax0.plot([0, CENTER[1]], [H_CAM, z], color=col, lw=0.8, alpha=0.6)
    ax0.plot(CENTER[1], z, "o", color=col, ms=8, mfc=col if ok else "white")
    ax0.text(CENTER[1] + 0.25, z, f"z = {z:+.2f} m, v = {uv[1]:.1f}", fontsize=8, va="center")
ax0.axhline(0, color="k", lw=0.6)
ax0.axvspan(3.0, 4.0, color="0.85", zorder=0)
ax0.set_xlim(-0.3, ymax)
ax0.set_ylim(-5.6, 3.6)
ax0.set_xlabel("ego y (m), the left camera's viewing direction")
ax0.set_ylabel("ego z (m)")
ax0.set_title("side view: the pillar over BEV cell (7, 11)")

ax1.imshow(img_left, extent=(-0.5, IMG - 0.5, IMG - 0.5, -0.5))
for e in np.arange(0, IMG + 1, 4) - 0.5:  # the 8x8 feature cells
    ax1.axvline(e, color="w", lw=0.4, alpha=0.6)
    ax1.axhline(e, color="w", lw=0.4, alpha=0.6)
for z in ZS:
    uv = project(1, (CENTER[0], CENTER[1], z))
    if in_frame(uv):
        ax1.plot(uv[0], uv[1], "o", color="tab:blue", mec="k", ms=8)
        ax1.text(uv[0] + 1, uv[1], f"z = {z:+.2f}", color="w", fontsize=8, va="center")
ax1.set_xlim(-0.5, IMG - 0.5)
ax1.set_ylim(IMG - 0.5, -0.5)
ax1.set_title("left camera image, 8x8 feature cells")
ax1.set_xlabel("u (px)")
ax1.set_ylabel("v (px)")
fig.tight_layout()
fig.savefig("vehicle_pillar.png", dpi=130)

# Figure 2: for each BEV cell, how many of its 4 pillar heights land in the camera that sees it.
n = 16
count = np.zeros((n, n), int)
which = -np.ones((n, n), int)
for i in range(n):
    for j in range(n):
        x, y = -8 + i + 0.5, -8 + j + 0.5
        for k in range(4):
            c = sum(in_frame(project(k, (x, y, z))) for z in ZS)
            if c > count[i, j]:
                count[i, j], which[i, j] = c, k
fig, ax = plt.subplots(figsize=(5.6, 5))
cmap = plt.get_cmap("viridis", 5)
im = ax.imshow(count, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5, extent=(-8, 8, -8, 8))
for gi, gj in scene["bev_gt"][0].nonzero().tolist():
    ax.add_patch(plt.Rectangle((-8 + gj, -8 + gi), 1, 1, fill=False, ec="orange", lw=2.5))
ax.add_patch(plt.Rectangle((-8 + CELL[1], -8 + CELL[0]), 1, 1, fill=False, ec="red", lw=1.5,
                           ls="--"))
for t in (-8, 8):
    ax.plot([0, t], [0, t], color="w", lw=0.8, ls=":")
    ax.plot([0, t], [0, -t], color="w", lw=0.8, ls=":")
for k, name in enumerate(["front", "left", "back", "right"]):
    a = 2 * np.pi * k / 4
    ax.text(5.5 * np.sin(a), 5.5 * np.cos(a), name, color="w", ha="center", va="center",
            fontsize=10, fontweight="bold")
ax.set_xlabel("ego y (m), lateral, left is +")
ax.set_ylabel("ego x (m), forward")
ax.invert_xaxis()
cb = fig.colorbar(im, ax=ax, ticks=range(5))
cb.set_label("pillar heights in frame (of 4)")
ax.set_title("pillar coverage of the toy rig")
fig.tight_layout()
fig.savefig("pillar_coverage.png", dpi=130)
pairs = int((count > 0).sum())
print("cells seen:", pairs, "all 4 heights:", int((count == 4).sum()),
      "partial:", pairs - int((count == 4).sum()), "unseen:", int((count == 0).sum()))
