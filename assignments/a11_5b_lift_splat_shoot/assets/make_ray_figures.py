"""Render the README figures for the Lift-Splat-Shoot notes.

Both figures use the toy scene's geometry: one forward camera 1.5 m above the ego origin, a
32x32 image with focal length 16 px, a stride-4 feature grid (8x8 cells), depth bins at
1, 2, ..., 8 m, and the 8x16 BEV grid (x in [0, 8) m forward, y in [-8, 8) m lateral, 1 m cells).
The pinhole back-projection is written out in numpy here, so the figures do not use the
assignment code.

one_ray_side.png. Side view (ego x, z) of the ray through feature cell (4, 3), the cell that sees
the toy vehicle about 4 m ahead. Dots mark the eight depth-bin points on the ray, sized by an
example depth distribution; the cross marks where flat-ground inverse perspective mapping (IPM)
would put the same pixel.

frustum_top.png. Top view (ego x, y) of all 8 x 8 x 8 = 512 frustum points over the BEV grid,
colored by depth bin, with the ray of cell (4, 3) highlighted, the toy vehicles' ground-truth
cells shaded, and the 8 m bin (which falls exactly on x = 8, outside the half-open grid) hollow.
The right panel counts how many frustum points land in each pillar.

Run from this directory:

    python make_ray_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

img, stride, f, h = 32, 4, 16.0, 1.5
c = (img - 1) / 2.0
Hf = Wf = img // stride
bins = np.arange(1.0, 9.0, 1.0)
x_min, x_max, y_min, y_max, res = 0.0, 8.0, -8.0, 8.0, 1.0
nx, ny = 8, 16


def ego_point(i, j, d):
    u, v = (j + 0.5) * stride, (i + 0.5) * stride
    X, Y = (u - c) * d / f, (v - c) * d / f       # camera frame, OpenCV
    return np.array([d, -X, h - Y])               # ego: x = Z, y = -X, z = h - Y


# Side view of one ray.
i0, j0 = 4, 3
pts = np.array([ego_point(i0, j0, d) for d in bins])
logits = -0.8 * (bins - 4.0) ** 2
alpha = np.exp(logits) / np.exp(logits).sum()

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.axhline(0, color="k", lw=1)
ax.add_patch(plt.Rectangle((3.1, 0), 1.8, 1.5, color="tab:orange", alpha=0.35, lw=0))
ax.text(4.0, 1.58, "vehicle", ha="center", fontsize=9)
slope = (pts[1, 2] - pts[0, 2]) / (pts[1, 0] - pts[0, 0])
x_ground = -h / slope
xs = np.linspace(0, x_ground, 50)
ax.plot(xs, h + slope * xs, color="gray", ls="--", lw=1)
ax.plot([0], [h], marker="s", color="k", ms=8)
ax.text(0.1, h + 0.12, "camera", fontsize=9)
for k, (p, a) in enumerate(zip(pts, alpha)):
    face = "tab:blue" if p[0] < x_max else "none"
    ax.scatter(p[0], p[2], s=40 + 900 * a, facecolor=face, edgecolor="tab:blue", zorder=3)
    ax.text(p[0], p[2] + 0.22, f"{bins[k]:.0f} m\n{a:.2f}", ha="center", fontsize=7)
ax.plot([x_ground], [0], marker="x", color="tab:red", ms=10, mew=2)
ax.text(x_ground, -0.35, f"IPM puts it here\n(x = {x_ground:.1f} m)", ha="center",
        fontsize=8, color="tab:red")
for b in range(nx + 1):
    ax.plot([b, b], [-0.08, 0.08], color="k", lw=1)
ax.text(0.5, -0.35, "BEV rows ix = 0..7 (1 m each)", fontsize=8)
ax.set_xlim(-0.3, x_ground + 0.8)
ax.set_ylim(-0.7, 2.2)
ax.set_xlabel("ego x, forward (m)")
ax.set_ylabel("ego z, up (m)")
ax.set_title("One ray of the toy camera: a point at each depth bin, weighted by the depth "
             "distribution", fontsize=10)
fig.tight_layout()
fig.savefig("one_ray_side.png", dpi=130)

# Top view of the whole frustum.
P = np.array([[[ego_point(i, j, d) for j in range(Wf)] for i in range(Hf)] for d in bins])
gt_cells = [(0, 6), (1, 6), (3, 8), (4, 8), (6, 12), (7, 12)]   # toy scene, seed 0
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
for ixc, iyc in gt_cells:
    ax.add_patch(plt.Rectangle((y_min + iyc * res, x_min + ixc * res), res, res,
                               color="tab:orange", alpha=0.35, lw=0))
for b in range(nx + 1):
    ax.plot([y_min, y_max], [b, b], color="0.85", lw=0.6, zorder=0)
for b in range(ny + 1):
    ax.plot([y_min + b, y_min + b], [x_min, x_max], color="0.85", lw=0.6, zorder=0)
cmap = plt.get_cmap("viridis")
for k, d in enumerate(bins):
    xy = P[k].reshape(-1, 3)
    col = cmap(k / (len(bins) - 1))
    inside = xy[:, 0] < x_max
    ax.scatter(xy[:, 1], xy[:, 0], s=10, facecolor=col if inside.all() else "none",
               edgecolor=col, label=f"{d:.0f} m")
ray = P[:, i0, j0]
ax.plot(ray[:, 1], ray[:, 0], color="tab:red", lw=1.5, zorder=4)
inside = ray[:, 0] < x_max
ax.scatter(ray[inside, 1], ray[inside, 0], s=25, color="tab:red", zorder=5)
ax.scatter(ray[~inside, 1], ray[~inside, 0], s=25, facecolor="white", edgecolor="tab:red",
           zorder=5)
ax.text(ray[-1, 1] + 0.3, ray[-1, 0] + 0.15, "ray of cell (4, 3)", color="tab:red", fontsize=8)
ax.set_xlim(y_max + 0.5, y_min - 0.5)        # ego +y (left) drawn on the left
ax.set_ylim(x_min - 0.3, x_max + 0.6)
ax.set_xlabel("ego y (m), left of the car is on the left")
ax.set_ylabel("ego x, forward (m)")
ax.set_title("Frustum points over the 8x16 BEV grid", fontsize=10)
ax.legend(fontsize=7, ncol=2, loc="lower left", title="depth bin", title_fontsize=7)

counts = np.zeros((nx, ny), dtype=int)
for p in P.reshape(-1, 3):
    ix, iy = int(np.floor((p[0] - x_min) / res)), int(np.floor((p[1] - y_min) / res))
    if 0 <= ix < nx and 0 <= iy < ny:
        counts[ix, iy] += 1
im = ax2.imshow(counts, origin="lower", cmap="Blues",
                extent=(y_min, y_max, x_min, x_max))
ax2.set_xlim(y_max, y_min)
for ixr in range(nx):
    for iyr in range(ny):
        if counts[ixr, iyr]:
            ax2.text(y_min + iyr + 0.5, ixr + 0.5, str(counts[ixr, iyr]), ha="center",
                     va="center", fontsize=6)
ax2.set_xlabel("ego y (m)")
ax2.set_ylabel("ego x, forward (m)")
ax2.set_title("Frustum points per pillar", fontsize=10)
fig.colorbar(im, ax=ax2, fraction=0.03)
fig.tight_layout()
fig.savefig("frustum_top.png", dpi=130)
print("points per row ix:", counts.sum(axis=1).tolist(), "dropped:", 512 - counts.sum())
print("IPM ground hit of the cell (4, 3) ray: x =", round(x_ground, 3))
