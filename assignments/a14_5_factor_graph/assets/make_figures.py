"""Figures for the a14_5 lecture notes. Run from this directory:

    python make_figures.py

corridor.png  our corridor loop (four poses on a line, one loop closure), solved in closed form
              with NumPy, and the 4x4 system matrix H of its normal equations.
drift_graph.png  the 24-pose drift graph from sim.py before and after Gauss-Newton, the
              per-pose position error, and the block-sparsity pattern of H. This one runs the
              reference C++ optimizer (it builds solution/ through _impl.py on first use).
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("NANOVISION_IMPL", "solution")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GRAY, RED, BLUE, GREEN = "#777777", "#d62728", "#1f77b4", "#2ca02c"

# ---- Our corridor loop -----------------------------------------------------------------
# Truth 0, 1, 2, 0. Odometry reads +1.1, +1.1, -2.0 (information 100 each); the loop closure
# (pose 3 -> pose 0) reads 0 (information 300). Residual of edge (i, j): (x_j - x_i) - z.
EDGES = [(0, 1, 1.1, 100.0), (1, 2, 1.1, 100.0), (2, 3, -2.0, 100.0), (3, 0, 0.0, 300.0)]
TRUTH = np.array([0.0, 1.0, 2.0, 0.0])
INIT = np.array([0.0, 1.1, 2.2, 0.2])  # dead reckoning: the odometry chained from x0 = 0


def corridor_system(x):
    H, g = np.zeros((4, 4)), np.zeros(4)
    for i, j, z, w in EDGES:
        r = (x[j] - x[i]) - z
        J = np.zeros(4)
        J[i], J[j] = -1.0, 1.0
        H += w * np.outer(J, J)
        g += w * J * r
    return H, g


def corridor_figure():
    H, g = corridor_system(INIT)
    delta = np.linalg.solve(H[1:, 1:], -g[1:])  # anchor x0
    opt = INIT + np.r_[0.0, delta]

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(10.5, 3.8),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    k = np.arange(4)
    ax.plot(k, TRUTH, "-", color=GRAY, lw=4, alpha=0.5, label="truth")
    ax.plot(k, INIT, "o--", color=RED, label="dead reckoning (initial estimate)")
    ax.plot(k, opt, "s-", color=BLUE, label="after one Gauss-Newton step")
    for kk in k:
        ax.annotate(f"{INIT[kk]:.2f}", (kk, INIT[kk]), textcoords="offset points",
                    xytext=(-30, 6), color=RED, fontsize=8)
        ax.annotate(f"{opt[kk]:.2f}", (kk, opt[kk]), textcoords="offset points",
                    xytext=(8, -12), color=BLUE, fontsize=8)
    ax.annotate("", xy=(0, -0.25), xytext=(3, -0.25),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(1.5, -0.62, "loop closure: x0 - x3 measured as 0", color=GREEN, ha="center",
            fontsize=9)
    ax.set_xticks(k, [f"x{kk}" for kk in k])
    ax.set_ylabel("position along the corridor (m)")
    ax.set_ylim(-0.8, 2.6)
    ax.set_title("our corridor loop")
    ax.legend(fontsize=8, loc="upper left")

    axh.imshow(np.where(H != 0, 0.25, 0.0), cmap="Greys", vmin=0, vmax=1)
    for a in range(4):
        for b in range(4):
            loop = {a, b} == {0, 3}
            axh.text(b, a, f"{H[a, b]:.0f}", ha="center", va="center",
                     color="#127812" if loop else "black", fontsize=10,
                     fontweight="bold" if loop else "normal")
    axh.set_xticks(range(4), [f"x{kk}" for kk in range(4)])
    axh.set_yticks(range(4), [f"x{kk}" for kk in range(4)])
    axh.set_title("H: tridiagonal band from odometry,\ncorner entries from the loop closure",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig("corridor.png", dpi=130)
    plt.close(fig)
    print("corridor: init", INIT, "opt", opt)


# ---- The 24-pose drift graph -------------------------------------------------------------
def drift_figure():
    import _impl as M
    import sim
    from _helpers import pose_distance, trajectory_positions

    pg = sim.pose_graph_drift(seed=0)  # the same graph viz.py shows
    ei, ej, meas, info = sim.unpack_edges(pg["edges"])
    opt_T, cost = M.optimize_pose_graph(pg["init"], ei, ej, meas, info, 20)
    gt, init, opt = (trajectory_positions(p) for p in (pg["gt"], pg["init"], opt_T))
    n = len(gt)

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(14, 4.4),
                                     gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})
    close = lambda xy: np.vstack([xy, xy[:1]])
    a0.plot(*close(gt).T, "-", color=GRAY, lw=4, alpha=0.5, label="ground truth")
    a0.plot(*init.T, "o--", color=RED, ms=3, label="dead reckoning")
    a0.plot(*close(opt).T, "s-", color=BLUE, ms=3, label="optimized")
    a0.plot(*np.vstack([init[-1], init[0]]).T, "-", color=GREEN, lw=2,
            label="loop-closure edge (initial)")
    a0.set_aspect("equal")
    a0.set_xlabel("x (m)")
    a0.set_ylabel("y (m)")
    a0.set_title("drift graph, seed 0")
    a0.legend(fontsize=7, loc="center")

    e_init = [pose_distance(a, b) for a, b in zip(pg["init"], pg["gt"])]
    e_opt = [pose_distance(a, b) for a, b in zip(opt_T, pg["gt"])]
    a1.plot(e_init, "o--", color=RED, ms=3, label="dead reckoning")
    a1.plot(e_opt, "s-", color=BLUE, ms=3, label="optimized")
    a1.set_xlabel("pose index")
    a1.set_ylabel("position error (m)")
    a1.set_title("error along the trajectory")
    a1.legend(fontsize=8)

    img = np.zeros((n, n))
    for i in range(n):
        img[i, i] = 1
    for i, j in zip(ei, ej):
        img[i, j] = img[j, i] = 1
    img[n - 1, 0] = img[0, n - 1] = 2
    a2.imshow(img, cmap=matplotlib.colors.ListedColormap(["white", "black", GREEN]))
    a2.set_title("nonzero 6x6 blocks of H\n(green: the loop-closure edge)", fontsize=10)
    a2.set_xlabel("pose j")
    a2.set_ylabel("pose i")
    fig.tight_layout()
    fig.savefig("drift_graph.png", dpi=130)
    plt.close(fig)
    print("drift: end error before %.3f after %.3f, final cost %.3f"
          % (e_init[-1], e_opt[-1], cost))
    print("drift: max error before %.3f after %.3f" % (max(e_init), max(e_opt)))


if __name__ == "__main__":
    corridor_figure()
    drift_figure()
