"""Render the README figures for the vision transformer notes.

patch_tokens.png. Left: our four-patch image (a 4x4 grayscale image cut into four 2x2
patches), with pixel values, patch indices, and the two-channel tokens of the worked
example. Middle: a 32x32 picture cut into the 8x8 grid of 4x4 patches the assignment's
ViT uses, with the row-major token index g*r + c of a few patches. Right: the same 64
patches in a random order. Without a positional embedding the encoder receives the same
set of tokens from the middle and right pictures.

pos_interp.png. Left: sample positions of an 8x8 grid and a 12x12 grid in normalized
image coordinates under the align_corners=False convention (each sample at the center of
its cell). Middle and right: one channel of an 8x8 positional-embedding table and its
bicubic resize to 12x12 with torch.nn.functional.interpolate.

Run from this directory:

    python make_vit_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- patch_tokens.png
small = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], float)
k1 = np.ones((2, 2))
k2 = np.array([[1, 1], [-1, -1]])
tokens = []
for r in range(2):
    for c in range(2):
        p = small[2 * r:2 * r + 2, 2 * c:2 * c + 2]
        tokens.append(((p * k1).sum(), (p * k2).sum()))

# A 32x32 picture of a house: roof triangle, body, door, window.
H = 32
house = np.zeros((H, H))
yy, xx = np.mgrid[0:H, 0:H]
roof = (yy >= 3) & (yy < 14) & (np.abs(xx - 15.5) <= (yy - 3) * 1.2 + 0.5)
body = (yy >= 14) & (yy < 29) & (xx >= 6) & (xx < 26)
house[roof] = 0.55
house[body] = 0.85
house[(yy >= 20) & (yy < 29) & (xx >= 13) & (xx < 19)] = 0.25   # door
house[(yy >= 16) & (yy < 20) & (xx >= 8) & (xx < 12)] = 0.25    # window
house[(yy >= 29)] = 0.4                                         # ground

p, g = 4, 8
rng = np.random.default_rng(0)
perm = rng.permutation(g * g)
patches = house.reshape(g, p, g, p).transpose(0, 2, 1, 3).reshape(g * g, p, p)
shuffled = patches[perm].reshape(g, g, p, p).transpose(0, 2, 1, 3).reshape(H, H)

fig, axs = plt.subplots(1, 3, figsize=(13, 5.2))
ax = axs[0]
ax.imshow(small, cmap="gray", vmin=-0.3, vmax=1.3)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{int(small[i, j])}", ha="center", va="center",
                color="black" if small[i, j] > 0.5 else "white", fontsize=12)
for k in (1.5,):
    ax.axhline(k, color="tab:red", lw=2)
    ax.axvline(k, color="tab:red", lw=2)
for idx, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax.text(2 * c + 0.5, 2 * r + 0.5, f"patch {idx}", color="tab:red", fontsize=10,
            ha="center", va="center", bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=1))
ax.set_xlabel("tokens: " + ", ".join(f"({t[0]:g}, {t[1]:g})" for t in tokens), fontsize=10)
ax.set_title("our four-patch image (2x2 patches)")
ax.set_xticks([])
ax.set_yticks([])

ax = axs[1]
ax.imshow(house, cmap="gray", vmin=0, vmax=1)
for k in range(1, g):
    ax.axhline(k * p - 0.5, color="tab:red", lw=0.6)
    ax.axvline(k * p - 0.5, color="tab:red", lw=0.6)
for r, c in [(0, 0), (0, 1), (0, 7), (1, 0), (3, 4), (7, 7)]:
    ax.text(c * p + 1.5, r * p + 1.5, f"{g * r + c}", ha="center", va="center",
            color="yellow", fontsize=8, bbox=dict(facecolor="black", alpha=0.7, lw=0, pad=1))
ax.set_title("32x32 picture, 8x8 grid of 4x4 patches\n(token index = 8 r + c)")
ax.set_xticks([])
ax.set_yticks([])

ax = axs[2]
ax.imshow(shuffled, cmap="gray", vmin=0, vmax=1)
for k in range(1, g):
    ax.axhline(k * p - 0.5, color="tab:red", lw=0.6)
    ax.axvline(k * p - 0.5, color="tab:red", lw=0.6)
ax.set_title("the same 64 patches, shuffled")
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig("patch_tokens.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- pos_interp.png
old, new = 8, 12
r = np.arange(old)
# A smooth example channel: varies slowly along rows and columns.
chan = np.cos(2 * np.pi * (r[:, None] + 0.5) / 8.0) + 0.6 * np.sin(np.pi * (r[None, :] + 0.5) / 8.0)
t = torch.tensor(chan, dtype=torch.float32)[None, None]
up = F.interpolate(t, size=(new, new), mode="bicubic", align_corners=False)[0, 0].numpy()

fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
ax = axs[0]
co = (np.arange(old) + 0.5) / old
cn = (np.arange(new) + 0.5) / new
X, Y = np.meshgrid(co, co)
ax.scatter(X, Y, s=40, color="tab:blue", label="8x8 grid samples")
X, Y = np.meshgrid(cn, cn)
ax.scatter(X, Y, s=12, marker="x", color="tab:orange", label="12x12 grid samples")
for k in np.linspace(0, 1, old + 1):
    ax.axhline(k, color="tab:blue", lw=0.4, alpha=0.5)
    ax.axvline(k, color="tab:blue", lw=0.4, alpha=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)
ax.set_aspect("equal")
ax.set_title("sample i at (i + 0.5) / g (align_corners=False)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)

vmin, vmax = min(chan.min(), up.min()), max(chan.max(), up.max())
ax = axs[1]
im = ax.imshow(chan, cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("one channel of the table, 8x8")
ax.set_xticks([])
ax.set_yticks([])
ax = axs[2]
ax.imshow(up, cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("bicubic resize to 12x12")
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(im, ax=axs[1:], fraction=0.03)
fig.savefig("pos_interp.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("tokens of the four-patch image:", tokens)
