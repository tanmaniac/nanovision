"""Render the README figures for the latent DiT notes.

two_image_example.png. Our two-image example: two training pictures (a disk at the left and a
disk at the right, drawn with the same rule as the toy shape data) each encoded to a 1D Gaussian
q(z | A) = N(-1, s^2), q(z | B) = N(+1, s^2). With a decoder that can output the best answer at
every z, squared-error training makes the decoder output the posterior mean of the image given z,
x_hat(z) = w_A(z) x_A + w_B(z) x_B with w_B(z) = sigmoid(2 z / s^2). Rows: s = 0.5, s = 1, and the
collapsed case where both images encode to N(0, 1).

patchify.png. Row-major token order on the 4x4 latent grid for patch size p = 2 and p = 1.

Run from this directory:

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIZE = 16


def disk(cx, cy, r=3.0):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    img = -np.ones((SIZE, SIZE))
    img[(xs - cx) ** 2 + (ys - cy) ** 2 <= r**2] = 1.0
    return img


x_a = disk(4.0, 8.0)
x_b = disk(12.0, 8.0)


def gauss(z, m, s):
    return np.exp(-0.5 * ((z - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


rows = [
    ("$\\sigma = 0.5$", -1.0, 1.0, 0.5),
    ("$\\sigma = 1$", -1.0, 1.0, 1.0),
    ("collapsed, both $\\mathcal{N}(0, 1)$", 0.0, 0.0, 1.0),
]
z_show = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
zz = np.linspace(-3.5, 3.5, 400)

fig = plt.figure(figsize=(12, 6.6))
gs = fig.add_gridspec(len(rows), 1 + len(z_show), width_ratios=[3.2] + [1] * len(z_show),
                      wspace=0.15, hspace=0.7)
for r, (name, ma, mb, s) in enumerate(rows):
    ax = fig.add_subplot(gs[r, 0])
    ax.plot(zz, gauss(zz, ma, s), color="tab:orange", label="$q(z \\mid A)$, left disk")
    ax.plot(zz, gauss(zz, mb, s), color="tab:blue", ls="--" if ma == mb else "-",
            label="$q(z \\mid B)$, right disk")
    ax.plot(zz, gauss(zz, 0, 1), color="gray", ls=":", label="prior $\\mathcal{N}(0, 1)$")
    for zv in z_show:
        ax.axvline(zv, color="k", lw=0.4, alpha=0.4)
    ax.set_title(name, fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel("latent $z$")
    if r == 0:
        ax.legend(fontsize=7, loc="upper right")
    for k, zv in enumerate(z_show):
        pa, pb = gauss(zv, ma, s), gauss(zv, mb, s)
        wb = pb / (pa + pb)
        img = (1 - wb) * x_a + wb * x_b
        axi = fig.add_subplot(gs[r, 1 + k])
        axi.imshow(img, cmap="gray", vmin=-1, vmax=1)
        axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(f"z = {zv:g}\n$w_B$ = {wb:.2f}", fontsize=8)
fig.suptitle("Our two-image example: what the best decoder outputs at each latent value",
             fontsize=11)
fig.savefig("two_image_example.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# Patchify token order.
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
colors = plt.cm.Pastel1(np.arange(4))
for ax, p in zip(axes, (2, 1)):
    nw = 4 // p
    for i in range(4):
        for j in range(4):
            n = (i // p) * nw + (j // p)
            fc = colors[n] if p == 2 else colors[(i + j) % 2 + 2]
            ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1, facecolor=fc, edgecolor="gray"))
            a, b = i % p, j % p
            if p == 2:
                ax.text(j + 0.5, 3 - i + 0.5, f"token {n}\n(a,b)=({a},{b})",
                        ha="center", va="center", fontsize=7)
            else:
                ax.text(j + 0.5, 3 - i + 0.5, f"token {n}", ha="center", va="center", fontsize=8)
    for k in range(0, 5, p):
        ax.axhline(k, color="k", lw=2); ax.axvline(k, color="k", lw=2)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect("equal")
    ax.set_xticks(np.arange(4) + 0.5, [f"col {j}" for j in range(4)], fontsize=8)
    ax.set_yticks(np.arange(4) + 0.5, [f"row {i}" for i in range(3, -1, -1)], fontsize=8)
    ax.set_title(f"p = {p}: {(4 // p) ** 2} tokens of width {p * p * 4}", fontsize=10)
fig.suptitle("Patchify on the 4x4 latent grid (each cell holds 4 channel values)", fontsize=10)
fig.tight_layout()
fig.savefig("patchify.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("wrote two_image_example.png, patchify.png")
