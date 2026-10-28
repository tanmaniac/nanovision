"""Figures for the a14_3 lecture notes: our sideways-step example and its four
essential-matrix candidates. Pure NumPy + matplotlib; run from this directory:

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Our sideways-step example: camera 1 at the origin, (R, t) = (I, (1, 0, 0)) = T_2_1, so
# camera 2's center is -R^T t = (-1, 0, 0). Same intrinsics as the assignment's scene.
F_PX, CX, CY = 500.0, 320.0, 240.0
T_VEC = np.array([1.0, 0.0, 0.0])
POINTS = np.array([[0.0, 0.0, 4.0],      # the point the notes work by hand
                   [1.0, -0.6, 5.0],
                   [-1.2, 0.8, 6.0]])
COLORS = ["#d62728", "#1f77b4", "#2ca02c"]


def project(R, t, X):
    Xc = R @ X + t
    return np.array([F_PX * Xc[0] / Xc[2] + CX, F_PX * Xc[1] / Xc[2] + CY])


def draw_camera(ax, center, z_dir, label, color="k", size=0.5, dx=0, dy=-14):
    """Top view (x right, z up the page): a wedge opening along the optical axis."""
    zx, zz = z_dir[0], z_dir[2]
    side = np.array([zz, -zx])
    tip = np.array([center[0], center[2]])
    fwd = np.array([zx, zz])
    a = tip + size * (fwd + 0.6 * side)
    b = tip + size * (fwd - 0.6 * side)
    ax.fill([tip[0], a[0], b[0]], [tip[1], a[1], b[1]], color=color, alpha=0.8)
    ax.annotate(label, tip, textcoords="offset points", xytext=(dx, dy), ha="center",
                fontsize=9, color=color)


def sideways_figure():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.0),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    c1, c2 = np.zeros(3), -T_VEC
    # The doubled scene (every point and the baseline scaled by 2) in gray.
    for X in POINTS:
        for c in (2 * c1, 2 * c2):
            ax.plot([c[0], 2 * X[0]], [c[2], 2 * X[2]], color="0.8", lw=0.8, ls="--")
        ax.plot(2 * X[0], 2 * X[2], "o", color="0.75", ms=5)
    draw_camera(ax, 2 * c2, np.array([0, 0, 1.0]), "cam 2, doubled", color="0.6",
                dx=0, dy=-28)
    for X, col in zip(POINTS, COLORS):
        for c in (c1, c2):
            ax.plot([c[0], X[0]], [c[2], X[2]], color=col, lw=1.2)
        ax.plot(X[0], X[2], "o", color=col, ms=6)
    draw_camera(ax, c1, np.array([0, 0, 1.0]), "cam 1", dx=22)
    draw_camera(ax, c2, np.array([0, 0, 1.0]), "cam 2", dx=-10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z, depth (m)")
    ax.set_title("Top view: rays from both cameras\n(gray: scene and baseline doubled)")
    ax.set_aspect("equal")
    ax.set_xlim(-4.0, 3.0)
    ax.set_ylim(-1.6, 12.5)

    for X, col in zip(POINTS, COLORS):
        u1 = project(np.eye(3), np.zeros(3), X)
        u2 = project(np.eye(3), T_VEC, X)
        bx.axhline(u1[1], color=col, lw=1.0, ls="--")
        bx.plot(*u1, "s", mfc="none", mec=col, ms=8)
        bx.plot(*u2, "o", color=col, ms=6)
        print(f"X={X}: u1={u1}, u2={u2}")
    bx.set_xlim(0, 640)
    bx.set_ylim(480, 0)
    bx.set_aspect("equal")
    bx.set_xlabel("u (px)")
    bx.set_ylabel("v (px)")
    bx.set_title("Image 2: epipolar lines (dashed), matches (dots),\n"
                 "image-1 positions (squares)")
    fig.tight_layout()
    fig.savefig("sideways_step.png", dpi=130)
    plt.close(fig)


def cheirality_figure():
    R_flip = np.diag([1.0, -1.0, -1.0])  # a half turn about the baseline
    cands = [(np.eye(3), T_VEC, "(I, +t)"), (np.eye(3), -T_VEC, "(I, -t)"),
             (R_flip, T_VEC, "(R_flip, +t)"), (R_flip, -T_VEC, "(R_flip, -t)")]
    x1, x2 = np.array([0.0, 0.0]), np.array([0.25, 0.0])
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, (R, t, name) in zip(axes, cands):
        P2 = np.hstack([R, t[:, None]])
        A = np.stack([x1[0] * P1[2] - P1[0], x1[1] * P1[2] - P1[1],
                      x2[0] * P2[2] - P2[0], x2[1] * P2[2] - P2[1]])
        Xh = np.linalg.svd(A)[2][-1]
        X = Xh[:3] / Xh[3]
        z1, z2 = X[2], (R @ X + t)[2]
        c2 = -R.T @ t
        ok = z1 > 0 and z2 > 0
        for c in (np.zeros(3), c2):
            ax.plot([c[0], X[0]], [c[2], X[2]], color="0.4", lw=1.0)
        ax.plot(X[0], X[2], "o", color="#2ca02c" if ok else "#d62728", ms=8)
        side = 1 if c2[0] < 0 else -1
        draw_camera(ax, np.zeros(3), np.array([0, 0, 1.0]), "cam 1", dx=20 * side, dy=-16)
        draw_camera(ax, c2, R.T @ np.array([0, 0, 1.0]), "cam 2", color="#1f77b4",
                    dx=-20 * side, dy=12)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_title(f"{name}\ndepth in cam 1 = {z1:+.0f}, in cam 2 = {z2:+.0f}", fontsize=10)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-5.5, 5.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        print(f"{name}: X={X}, z1={z1:+.3f}, z2={z2:+.3f}, center2={c2}")
    axes[0].set_ylabel("z (m)")
    fig.tight_layout()
    fig.savefig("cheirality.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sideways_figure()
    cheirality_figure()
