"""Render the README figures for the VQ tokenizer notes.

tokens.png. Four 16x16 shape images, the 4x4 grid of code indices a trained VQ-VAE assigns
to each, and the decoder's reconstruction from those 16 indices. The VQ-VAE is the reference
solution trained on CPU with the settings viz.py uses (16 images, 2000 Adam steps).

patch_codebook.png. Our two-pixel patch example: the four kinds of two-pixel patch in the
shape images, drawn with area proportional to how often they occur, and the Voronoi cells of
a three-code codebook in which one code is dead.

ste.png. One-dimensional quantizer with codes at -1 and +1: the forward value (a step) and
the identity map the straight-through estimator uses in the backward pass.

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

_HERE = Path(__file__).resolve().parent
_ASSIGN = _HERE.parent
_REPO = _ASSIGN.parents[1]
os.environ["NANOVISION_IMPL"] = "solution"
for p in (_REPO, _ASSIGN, _ASSIGN / "solution"):
    sys.path.insert(0, str(p))

# ---------------------------------------------------------------- patch_codebook.png
# Frequencies of horizontally adjacent pixel pairs, measured on 2000 images from
# nanovision.data.toy.diffusion_image_batch (seed 0).
freq = {(-1, -1): 0.711, (-1, 1): 0.035, (1, -1): 0.032, (1, 1): 0.222}
names = {(-1, -1): "background", (-1, 1): "left edge", (1, -1): "right edge", (1, 1): "interior"}
E = np.array([[-0.9, -0.9], [0.8, 0.8], [3.0, 3.0]])

g = np.linspace(-2.0, 3.6, 500)
X, Y = np.meshgrid(g, g)
pts = np.stack([X.ravel(), Y.ravel()], 1)
cell = ((pts[:, None, :] - E[None]) ** 2).sum(-1).argmin(1).reshape(X.shape)

fig, ax = plt.subplots(figsize=(5.2, 5.2))
colors = ["#c6dbef", "#fdd0a2", "#e5e5e5"]
ax.contourf(X, Y, cell, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors)
ax.contour(X, Y, cell, levels=[0.5, 1.5], colors="gray", linewidths=0.8)
for z, p in freq.items():
    ax.scatter(*z, s=3000 * p, color="black", alpha=0.75, zorder=3)
    ax.annotate(f"{names[z]}\n{100 * p:.1f}%", z, textcoords="offset points",
                xytext=(0, -30 if z[1] < 0 else 18), ha="center", fontsize=9)
for k, e in enumerate(E):
    ax.scatter(*e, marker="*", s=260, color="tab:red", zorder=4)
    label = f"$E_{k}$" + (" (dead)" if k == 2 else "")
    ax.annotate(label, e, textcoords="offset points", xytext=(-60 if k == 2 else 8, -18 if k == 2 else 6), color="tab:red", fontsize=11)
ax.set_xlabel("left pixel")
ax.set_ylabel("right pixel")
ax.set_title("Two-pixel patches and the Voronoi cells of a 3-code codebook", fontsize=10)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(_HERE / "patch_codebook.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- ste.png
z = np.linspace(-2, 2, 801)
zq = np.where(z < 0, -1.0, 1.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.plot(z[z < 0], zq[z < 0], color="tab:blue", lw=2)
a1.plot(z[z >= 0], zq[z >= 0], color="tab:blue", lw=2, label="forward: $z_q$ (nearest code)")
a1.plot(z, z, color="tab:orange", ls="--", label="what the backward pass assumes: $z_e$ (STE)")
a1.scatter([0.3], [1.0], color="black", zorder=3)
a1.annotate("$z_e = 0.3 \\mapsto z_q = +1$", (0.3, 1.0), textcoords="offset points",
            xytext=(6, -18), fontsize=9)
a1.set_xlabel("$z_e$")
a1.set_ylabel("quantizer output")
a1.legend(fontsize=8, loc="upper left")
a1.set_title("What each pass uses", fontsize=10)
d_true = np.zeros_like(z)
a2.plot(z, d_true, color="tab:blue", lw=2, label="true derivative (0 off the boundary)")
a2.plot(z, np.ones_like(z), color="tab:orange", ls="--", label="STE derivative (1)")
a2.axvline(0, color="gray", lw=0.6, ls=":")
a2.set_ylim(-0.4, 1.4)
a2.set_xlabel("$z_e$")
a2.set_ylabel("$\\partial z_q / \\partial z_e$")
a2.legend(fontsize=8, loc="center right")
a2.set_title("Derivative seen by the encoder", fontsize=10)
fig.tight_layout()
fig.savefig(_HERE / "ste.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- tokens.png
import torch  # noqa: E402

from config import VQConfig  # noqa: E402
from vqvae import VQVAE, vq_vae_loss  # noqa: E402

from nanovision.data import toy  # noqa: E402

torch.manual_seed(0)
cfg = VQConfig()
x, _ = toy.diffusion_image_batch(16, num_classes=3, size=cfg.img_size,
                                 channels=cfg.channels, seed=0)
vae = VQVAE(cfg)
opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
for _ in range(2000):
    opt.zero_grad()
    x_hat, idx, vq = vae(x)
    vq_vae_loss(x, x_hat, vq).backward()
    opt.step()
with torch.no_grad():
    x_hat, idx, _ = vae(x)

show = [0, 1, 2, 3]
fig, axes = plt.subplots(3, len(show), figsize=(2.3 * len(show), 7))
for j, i in enumerate(show):
    axes[0, j].imshow(x[i, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
    grid = idx[i].numpy()
    axes[1, j].imshow(grid, cmap="tab20", vmin=0, vmax=31)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            axes[1, j].text(c, r, str(grid[r, c]), ha="center", va="center", fontsize=10)
    axes[2, j].imshow(x_hat[i, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
    for a in axes[:, j]:
        a.set_xticks([])
        a.set_yticks([])
axes[0, 0].set_ylabel("image (256 pixels)")
axes[1, 0].set_ylabel("tokens (16 codes)")
axes[2, 0].set_ylabel("decoded from tokens")
fig.tight_layout()
fig.savefig(_HERE / "tokens.png", dpi=110)
plt.close(fig)
print("wrote tokens.png, patch_codebook.png, ste.png")
