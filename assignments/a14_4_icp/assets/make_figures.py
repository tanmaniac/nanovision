"""Figure for the a14_4 lecture notes: our corner example, a 2D scan of a room corner moved
by a pure translation, and one step of point-to-point and of point-to-plane ICP from it.
Pure NumPy + matplotlib; run from this directory:

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Our corner example. Target: three points on the wall x = 0 and three on the wall y = 0,
# with their unit normals. Source: the same six points moved by g = (0.2, 0.7), so the true
# alignment (source -> target) is the translation -g = (-0.2, -0.7).
TARGET = np.array([[0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]], float)
NORMALS = np.array([[1, 0]] * 3 + [[0, 1]] * 3, float)
G = np.array([0.2, 0.7])
SOURCE = TARGET + G

GRAY, RED, BLUE = "#777777", "#d62728", "#1f77b4"


def nearest(P):
    d = np.linalg.norm(P[:, None] - TARGET[None], axis=2)
    return d.argmin(1)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def point_to_point_step(P):
    """Closed-form 2D Procrustes on the nearest-neighbor pairs (with the det correction)."""
    Q = TARGET[nearest(P)]
    pb, qb = P.mean(0), Q.mean(0)
    H = (P - pb).T @ (Q - qb)
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ np.diag([1.0, np.linalg.det(V @ U.T)]) @ U.T
    return P @ R.T + (qb - R @ pb)


def point_to_plane_step(P):
    """One Gauss-Newton step on the point-to-plane cost over (rho_x, rho_y, theta), applied
    as a left update with the SE(2) exponential."""
    idx = nearest(P)
    Q, N = TARGET[idx], NORMALS[idx]
    a = np.column_stack([N[:, 0], N[:, 1], P[:, 0] * N[:, 1] - P[:, 1] * N[:, 0]])
    b = -np.sum(N * (P - Q), axis=1)
    rho_x, rho_y, th = np.linalg.solve(a.T @ a, a.T @ b)
    V = np.eye(2) if abs(th) < 1e-12 else np.array(
        [[np.sin(th), -(1 - np.cos(th))], [1 - np.cos(th), np.sin(th)]]) / th
    return P @ rot(th).T + V @ np.array([rho_x, rho_y])


def draw_walls(ax):
    ax.plot([0, 0], [0, 3.6], color=GRAY, lw=6, alpha=0.18, solid_capstyle="butt")
    ax.plot([0, 3.6], [0, 0], color=GRAY, lw=6, alpha=0.18, solid_capstyle="butt")
    ax.scatter(*TARGET.T, s=45, color=GRAY, zorder=3, label="target points")
    ax.set_aspect("equal")
    ax.set_xlim(-0.9, 3.9)
    ax.set_ylim(-0.6, 4.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))

    # Left: the nearest-neighbor pairs and their gaps split into normal and tangential parts.
    ax = axes[0]
    draw_walls(ax)
    idx = nearest(SOURCE)
    for p, i in zip(SOURCE, idx):
        q, n = TARGET[i], NORMALS[i]
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
        normal_part = np.dot(q - p, n) * n
        foot = p + normal_part
        ax.plot(*np.stack([p, foot]).T, color=BLUE, lw=2.2, zorder=2)
        ax.plot(*np.stack([foot, q]).T, color=RED, lw=2.2, ls=":", zorder=2)
    ax.scatter(*SOURCE.T, s=45, color="k", marker="s", zorder=4, label="source points")
    ax.plot([], [], color="k", lw=1, label="nearest-neighbor pair")
    ax.plot([], [], color=BLUE, lw=2.2, label="gap along the target normal")
    ax.plot([], [], color=RED, lw=2.2, ls=":", label="gap along the wall")
    ax.set_title("Start: source = target moved by (0.2, 0.7)")
    ax.legend(loc="upper right", fontsize=8)

    # Right: one step of each method from the same pairs.
    ax = axes[1]
    draw_walls(ax)
    P1 = point_to_point_step(SOURCE)
    L1 = point_to_plane_step(SOURCE)
    ax.scatter(*SOURCE.T, s=30, facecolor="none", edgecolor="k", marker="s", zorder=3,
               label="source at start")
    ax.scatter(*P1.T, s=45, color=RED, zorder=4, label="after one point-to-point step")
    ax.scatter(*L1.T, s=110, facecolor="none", edgecolor=BLUE, lw=2, zorder=5,
               label="after one point-to-plane step")
    ax.set_title("One step of each method")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("corner_example.png", dpi=130)

    # Numbers quoted in the notes and the caption.
    print("pairs (target index per source point):", idx)
    P = SOURCE.copy()
    for k in range(1, 4):
        P = point_to_point_step(P)
        err = np.abs(P - TARGET).max()
        v = P[2] - P[0]  # direction of the wall x = 0 in the moved source
        ang = np.degrees(np.arctan2(-v[0], v[1]))
        print(f"point-to-point iter {k}: max point error {err:.4f} m, rotation {ang:.2f} deg")
    print("point-to-plane after one step, max point error:", np.abs(L1 - TARGET).max())


if __name__ == "__main__":
    main()
