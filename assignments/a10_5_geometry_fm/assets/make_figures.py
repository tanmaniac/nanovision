"""Render the README figures for the DUSt3R notes.

sphere_pair.png. Our sphere pair (views 0 and 3 of the toy ring, the config defaults) seen
from above in camera 1's frame: the two cameras, the sphere, and the ground-truth pointmaps
X^{1,1} and X^{2,1}, then the depth (z) channel of each pointmap as a 4x4 patch grid. The
geometry comes from toy_scene.stereo_pointmap_gt, the supervision the model is trained on.

confidence_cost.png. The per-pixel cost C*ell - alpha*log(C), minimized over C, as a function
of the residual ell, for C = 1 + exp(logit) (so C >= 1) and for softplus (C > 0), with
alpha = 0.2.

Run from this directory:

    python make_figures.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_here = Path(__file__).resolve().parent
_assign = _here.parent
_repo = _assign.parents[1]
os.environ.setdefault("NANOVISION_IMPL", "solution")
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_assign))

import torch  # noqa: E402
from config import GeometryFMConfig  # noqa: E402
from toy_scene import stereo_pointmap_gt  # noqa: E402

cfg = GeometryFMConfig()
d = stereo_pointmap_gt(cfg)
g1 = d["gt_pts1"].numpy()
g2 = d["gt_pts2"].numpy()
v1 = d["valid1"].numpy()
v2 = d["valid2"].numpy()
T1 = d["T1"].numpy()
T12 = d["T_1to2"].numpy()
T21 = np.linalg.inv(T12)  # cam2 -> cam1

center_w = np.array(list(cfg.sphere_center) + [1.0])
center_c1 = (np.linalg.inv(T1) @ center_w)[:3]

fig = plt.figure(figsize=(13, 5.2))
ax = fig.add_subplot(1, 3, 1)
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(center_c1[0] + cfg.radius * np.cos(th), center_c1[2] + cfg.radius * np.sin(th),
        color="0.6", lw=1, label="sphere (outline at camera height)")


def draw_cam(T, color, name):
    o = T[:3, 3]
    fwd = T[:3, 2]
    right = T[:3, 0]
    half = 8.0 / 16.0  # half image width / focal
    for s in (-1, 1):
        e = o + 1.2 * (fwd + s * half * right)
        ax.plot([o[0], e[0]], [o[2], e[2]], color=color, lw=1)
    ax.plot(*o[[0, 2]], marker="s", color=color, ms=7)
    ax.annotate(name, o[[0, 2]], textcoords="offset points", xytext=(6, -12), color=color)


draw_cam(np.eye(4), "tab:blue", "camera 1")
draw_cam(T21, "tab:orange", "camera 2")
for pts, val, color, lab, cam in ((g1, v1, "tab:blue", r"$X^{1,1}$", np.zeros(3)),
                                  (g2, v2, "tab:orange", r"$X^{2,1}$", T21[:3, 3])):
    P = pts[val]
    for p in P:
        ax.plot([cam[0], p[0]], [cam[2], p[2]], color=color, lw=0.3, alpha=0.4)
    ax.scatter(P[:, 0], P[:, 2], color=color, s=18, zorder=3, label=lab + " (valid patches)")
ax.set_xlabel("x in camera 1's frame")
ax.set_ylabel("z in camera 1's frame (camera 1 looks along +z)")
ax.set_aspect("equal")
ax.set_title("Both pointmaps in camera 1's frame (top view)")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(-0.8, 8.2)

for k, (pts, val, name) in enumerate(((g1, v1, r"$X^{1,1}$: image 1's patches"),
                                      (g2, v2, r"$X^{2,1}$: image 2's patches"))):
    a = fig.add_subplot(1, 3, k + 2)
    z = np.where(val, pts[..., 2], np.nan)
    a.imshow(z, cmap="viridis", vmin=1.5, vmax=5.6)
    for i in range(4):
        for j in range(4):
            txt = f"{z[i, j]:.2f}" if val[i, j] else "miss"
            a.text(j, i, txt, ha="center", va="center", fontsize=9,
                   color="white" if val[i, j] else "black")
    a.set_xticks(range(4))
    a.set_yticks(range(4))
    a.set_xlabel("patch column j")
    a.set_ylabel("patch row i")
    a.set_title(name + ",\nz in camera 1's frame")
fig.tight_layout()
fig.savefig(_here / "sphere_pair.png", dpi=110)
plt.close(fig)

# Confidence cost.
alpha = 0.2
ell = np.linspace(0.003, 1.5, 600)
interior = alpha * (1 + np.log(ell / alpha))
floor_cost = np.where(ell < alpha, interior, ell)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(ell, ell, color="0.6", ls=":", label=r"plain residual $\ell$ ($C = 1$ fixed)")
ax.plot(ell, floor_cost, color="tab:blue", lw=2, label=r"$C = 1 + e^{\mathrm{logit}}$ ($C \geq 1$)")
ax.plot(ell, interior, color="tab:red", lw=1.5, ls="--", label=r"softplus ($C > 0$)")
ax.axvline(alpha, color="0.8", lw=1)
ax.text(alpha + 0.02, -0.5, r"$\ell = \alpha$", color="0.4")
ax.set_xlabel(r"residual $\ell$ of one pixel")
ax.set_ylabel(r"cost at the best confidence, $\min_C\,(C\ell - \alpha\log C)$")
ax.set_title(r"Best achievable per-pixel cost, $\alpha = 0.2$")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(_here / "confidence_cost.png", dpi=110)
plt.close(fig)
print("wrote sphere_pair.png and confidence_cost.png")
