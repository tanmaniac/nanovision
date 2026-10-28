"""Render the README figures for the sensor-fusion notes from one toy scene (seed 200).

fusion_evidence.png. The 8x16 BEV grid of the held-out scene that viz.py also shows, four ways:
the camera's vehicle score (cam_bev channel 1, smeared over whole forward columns), the number of
LiDAR points per cell, the cells where both say "something vehicle-like" (camera vehicle score
times "a blob point landed here"), and the ground-truth vehicle cells. The two columns of our
two-column example (lateral cells 5 and 10) are outlined.

painted_points.png. The camera image with every in-image LiDAR point projected onto it, colored
by the vehicle score PointPainting would append to it.

Uses only the toy generator and a few lines of pinhole projection, not the assignment code.
Run from this directory:

    python make_fusion_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from nanovision.data import toy

sc = toy.bev_fusion_scene(seed=200)
nx, ny = sc["bev_gt"].shape
L = sc["lidar"]
ix = torch.floor(L[:, 0]).long()
iy = torch.floor(L[:, 1] + 8.0).long()
k = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
count = torch.zeros(nx, ny)
count.index_put_((ix[k], iy[k]), torch.ones(int(k.sum())), accumulate=True)
blob = torch.zeros(nx, ny, dtype=torch.bool)
kb = k.clone()
kb[240:] = False  # the first 6 * 40 points are the blobs, the rest is ground
blob[ix[kb], iy[kb]] = True
cam = sc["cam_bev"][1]
both = cam * blob.float()
gt = sc["bev_gt"]

panels = [
    (cam, "camera vehicle score", "Blues", 1),
    (count, "LiDAR points per cell", "Greys", None),
    (both, "camera AND points", "Purples", 1),
    (gt, "ground truth (vehicles)", "Oranges", 1),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, (m, title, cmap, vmax) in zip(axes, panels):
    im = ax.imshow(m.numpy(), origin="lower", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("lateral cell iy")
    ax.set_ylabel("forward cell ix")
    for col in (5, 10):
        ax.add_patch(Rectangle((col - 0.5, -0.5), 1, nx, fill=False, ec="red", lw=1.5))
    for (cx, cy) in sc["vehicle_cells"].tolist():
        ax.plot(cy, cx, marker="s", ms=11, mfc="none", mec="tab:orange", mew=1.5)
    for (cx, cy) in sc["clutter_cells"].tolist():
        ax.plot(cy, cx, marker="o", ms=11, mfc="none", mec="tab:green", mew=1.5)
    if title.startswith("LiDAR"):
        fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("fusion_evidence.png", dpi=110)
plt.close(fig)

# Painted points over the image.
img = sc["image"][0].permute(1, 2, 0).numpy()
H, W = img.shape[:2]
E, K = sc["E"], sc["K"]
pc = L @ E[:3, :3].T + E[:3, 3]  # ego -> camera frame
px = (pc @ K.T)[:, :2] / pc[:, 2:3]  # pinhole projection to (u, v)
u = px[:, 0].round().long()
v = px[:, 1].round().long()
ok = (pc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
score = sc["seg_scores"][1, v.clamp(0, H - 1), u.clamp(0, W - 1)]
fig, (a0, a1) = plt.subplots(1, 2, figsize=(9, 4.2))
a0.imshow(img, interpolation="nearest")
a0.set_title("toy camera image")
a1.imshow(img * 0.35 + 0.65, interpolation="nearest")
s = a1.scatter(px[ok, 0], px[ok, 1], c=score[ok], cmap="viridis", vmin=0, vmax=1, s=10)
a1.set_title(f"{int(ok.sum())} of {L.shape[0]} LiDAR points, colored by\npainted vehicle score")
fig.colorbar(s, ax=a1, fraction=0.046)
for a in (a0, a1):
    a.set_xlim(-0.5, W - 0.5)
    a.set_ylim(H - 0.5, -0.5)
plt.tight_layout()
plt.savefig("painted_points.png", dpi=110)
print("in-image points", int(ok.sum()), "of", L.shape[0])
