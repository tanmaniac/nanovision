"""Generate the lecture-note figures for the Gaussian splatting README.

Run from this directory: `python make_figures.py`. Writes toy_views.png, two_gaussians.png,
and ewa_check.png next to this script. The toy views come from the course's scene generator;
the two-Gaussian example and the EWA check are computed here with numpy from the formulas in
the notes and do not use the assignment code. In the EWA check, samples with camera-space depth
z <= 1 are dropped before projecting (this affects only the large off-axis Gaussian), so the
divide by depth stays away from zero and the scatter stays in view.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[2]))  # repo root, for nanovision.data.toy

from nanovision.data import toy  # noqa: E402

# The toy camera: view 0 of the 16x16 scene, at world (4, 0, 0) looking at the origin.
F, CX, CY, H, W = 16.0, 7.5, 7.5, 16, 16
# World-to-camera rotation for that camera: camera x = world z, camera y = world y,
# camera z = 4 - world x.
W2C_R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
W2C_T = np.array([0.0, 0.0, 4.0])
DILATION = 0.3

# Our two-Gaussian example (world frame).
GAUSS = [
    dict(name="near (red)", mean=np.array([0.8, 0.0, -0.25]), scales=np.array([0.2, 0.2, 0.2]),
         color=np.array([1.0, 0.0, 0.0]), opacity=0.8),
    dict(name="far (blue)", mean=np.array([0.0, 0.0, 0.25]), scales=np.array([0.1, 0.1, 0.5]),
         color=np.array([0.0, 0.0, 1.0]), opacity=0.9),
]


def jacobian(m):
    x, y, z = m
    return np.array([[F / z, 0.0, -F * x / z**2], [0.0, F / z, -F * y / z**2]])


def project(g, dilation=DILATION):
    mc = W2C_R @ g["mean"] + W2C_T
    cov3 = np.diag(g["scales"] ** 2)  # identity rotation in the world frame
    J = jacobian(mc)
    cov2 = J @ W2C_R @ cov3 @ W2C_R.T @ J.T + dilation * np.eye(2)
    mu2 = np.array([F * mc[0] / mc[2] + CX, F * mc[1] / mc[2] + CY])
    return mu2, cov2, mc[2]


def render(gs, bg=1.0):
    proj = [project(g) for g in gs]
    order = np.argsort([p[2] for p in proj])
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    px = np.stack([us, vs], -1).astype(float)
    C = np.zeros((H, W, 3))
    T = np.ones((H, W))
    alphas, weights = [], []
    for i in order:
        mu, cov, _ = proj[i]
        d = px - mu
        inv = np.linalg.inv(cov)
        power = np.einsum("hwi,ij,hwj->hw", d, inv, d)
        a = np.minimum(gs[i]["opacity"] * np.exp(-0.5 * power), 0.99)
        alphas.append(a)
        weights.append(a * T)
        C += (a * T)[..., None] * gs[i]["color"]
        T = T * (1 - a)
    C += T[..., None] * bg
    return C, proj, order, alphas, weights, T


def ellipse(ax, mu, cov, k=1.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang,
                         fill=False, **kw))


def fig_toy_views():
    images, _, _, _, _ = toy.nerf_synthetic_scene(n_views=6, H=32, W=32)
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].numpy(), interpolation="nearest")
        ax.set_title("held out" if i == 5 else f"view {i}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(_here / "toy_views.png", dpi=110)
    plt.close(fig)


def fig_two_gaussians():
    C, proj, order, alphas, weights, Tfin = render(GAUSS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # (a) top-down view of the world x-z plane.
    ax = axes[0]
    ax.plot([4], [0], "ks")
    ax.annotate("camera", (4, 0), textcoords="offset points", xytext=(-10, 8))
    half = np.arctan((W / 2) / F)
    for s in (-1, 1):
        ax.plot([4, 4 - 5.5], [0, s * 5.5 * np.tan(half)], "k:", lw=0.8)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls="--", color="0.6"))
    ax.text(-0.95, -1.25, "toy sphere", color="0.5", fontsize=9)
    for g, col in zip(GAUSS, ["red", "blue"]):
        sx, sz = g["scales"][0], g["scales"][2]
        ax.add_patch(Ellipse((g["mean"][0], g["mean"][2]), 2 * sx, 2 * sz, color=col, alpha=0.5))
    ax.set_xlim(-1.6, 4.5); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xlabel("world x"); ax.set_ylabel("world z")
    ax.set_title("(a) seen from above, one-sigma ellipses")

    # (b) the 16x16 render with the projected one-sigma ellipses.
    ax = axes[1]
    ax.imshow(np.clip(C, 0, 1), interpolation="nearest")
    for (mu, cov, _), col in zip(proj, ["darkred", "navy"]):
        ellipse(ax, mu, cov, k=1.0, color=col, lw=1.5)
        ellipse(ax, mu, cov - DILATION * np.eye(2), k=1.0, color=col, lw=1.0, ls=":")
    ax.axhline(7, color="k", lw=0.6, ls="--")
    ax.set_title("(b) render; one-sigma footprints\n(dotted: before the dilation)")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")

    # (c) row v = 7: per-Gaussian alpha, compositing weights, leftover transmittance.
    ax = axes[2]
    u = np.arange(W)
    names = [GAUSS[i]["name"] for i in order]
    cols = ["red", "blue"]
    for k, i in enumerate(order):
        ax.plot(u, alphas[k][7], "--", color=cols[i], label=rf"$\alpha$ {names[k]}")
        ax.plot(u, weights[k][7], "-", color=cols[i], label=rf"weight $T\alpha$ {names[k]}")
    ax.plot(u, Tfin[7], "k-", label=r"$T_{\mathrm{final}}$ (background)")
    ax.set_xlabel("u (px), row v = 7"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(c) compositing along row v = 7")
    plt.tight_layout()
    fig.savefig(_here / "two_gaussians.png", dpi=110)
    plt.close(fig)
    return C


def fig_ewa_check():
    rng = np.random.default_rng(0)
    cases = [
        ("our far Gaussian (small)", np.array([0.25, 0.0, 4.0]), np.diag([0.5, 0.1, 0.1]) ** 2),
        ("large Gaussian off the axis", np.array([1.2, 0.0, 3.0]), np.diag([0.5, 0.5, 1.0]) ** 2),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (title, m, cov) in zip(axes, cases):
        pts = rng.multivariate_normal(m, cov, size=4000)
        pts = pts[pts[:, 2] > 1.0]
        uv = np.stack([F * pts[:, 0] / pts[:, 2] + CX, F * pts[:, 1] / pts[:, 2] + CY], -1)
        ax.scatter(uv[:, 0], uv[:, 1], s=1, alpha=0.25, color="0.4")
        J = jacobian(m)
        mu2 = np.array([F * m[0] / m[2] + CX, F * m[1] / m[2] + CY])
        cov2 = J @ cov @ J.T
        for k in (1, 2):
            ellipse(ax, mu2, cov2, k=k, color="C1", lw=1.5)
        med = np.median(uv, 0)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill=False, ls="--", color="0.5"))
        ax.plot(*mu2, "x", color="C1", ms=9, label="EWA mean and 1, 2 sigma")
        ax.plot(*med, "+", color="k", ms=9, label="median of exact projections")
        ax.set_xlim(-2, 34); ax.set_ylim(22, -7); ax.set_aspect("equal")
        ax.set_title(title); ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
        ax.legend(fontsize=8, loc="lower right")
        print(title, "EWA cov", np.round(cov2, 3).tolist(), "sample cov",
              np.round(np.cov(uv.T), 3).tolist())
    plt.tight_layout()
    fig.savefig(_here / "ewa_check.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    fig_toy_views()
    C = fig_two_gaussians()
    print("pixel (u=7, v=7) color", np.round(C[7, 7], 4))
    fig_ewa_check()
