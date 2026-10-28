"""Render the README figures for the EKF-SLAM assignment.

corridor.png. Our corridor example: a robot on a line, two landmarks, measurements of the
signed offset z = l - x. The robot maps l0 from x0 = 0, drives out (dead-reckoned to 10, truly
at 10.2), maps l1, drives back (dead-reckoned to 1, truly at 1.3), and re-sees l0. Rows: the
belief just before re-seeing l0, after the update with the full joint covariance, and after the
same update in a filter that never kept the robot-landmark cross-covariance.

loop_run.png. EKF-SLAM on the assignment's simulated loop (sim.py and config.py, seed 0, 300
steps, known association, the same driver logic as runner.py). The filter is reimplemented
here in NumPy so the figure does not need the C++ build. Two snapshots: the step where the mean
landmark error peaks, and the last step. The script also prints the run statistics the README
quotes: landmark-landmark cross-block and diagonal-block Frobenius norms at the end, the robot
NEES averaged over six equal segments, how many steps exceed the 95% chi-square bound, and the
peak and final mean landmark error.

Run from this directory:

    python make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _helpers import covariance_ellipse, robot_nees, wrap_angle as wrap  # noqa: E402
from config import DT, GATE, INIT_P_DIAG, LANDMARKS, Q_DIAG, R_DIAG  # noqa: E402
from sim import simulate  # noqa: E402


# ---- EKF-SLAM in NumPy (same equations as solution/ekf_slam.cpp) ---------------------------
def predict(mu, P, u, dt, Q):
    th = mu[2]
    F = np.eye(3)
    F[0, 2] = -u[0] * dt * np.sin(th)
    F[1, 2] = u[0] * dt * np.cos(th)
    mu = mu.copy()
    mu[:3] = [mu[0] + u[0] * dt * np.cos(th), mu[1] + u[0] * dt * np.sin(th), wrap(th + u[1] * dt)]
    P = P.copy()
    P[:3, :3] = F @ P[:3, :3] @ F.T + Q
    P[:3, 3:] = F @ P[:3, 3:]
    P[3:, :3] = P[:3, 3:].T
    return mu, P


def add_landmark(mu, P, z, R):
    r, phi = z
    b = mu[2] + phi
    c, s = np.cos(b), np.sin(b)
    Gr = np.array([[1, 0, -r * s], [0, 1, r * c]])
    Gz = np.array([[c, -r * s], [s, r * c]])
    m = mu.size
    mu2 = np.concatenate([mu, [mu[0] + r * c, mu[1] + r * s]])
    P2 = np.zeros((m + 2, m + 2))
    P2[:m, :m] = P
    P2[m:, :m] = Gr @ P[:3, :]
    P2[:m, m:] = P2[m:, :m].T
    P2[m:, m:] = Gr @ P[:3, :3] @ Gr.T + Gz @ R @ Gz.T
    return mu2, P2


def update(mu, P, z, j, R):
    lm = mu[3 + 2 * j:5 + 2 * j]
    dx, dy = lm - mu[:2]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    zhat = np.array([r, wrap(np.arctan2(dy, dx) - mu[2])])
    H = np.zeros((2, mu.size))
    H[:, :3] = [[-dx / r, -dy / r, 0], [dy / q, -dx / q, -1]]
    H[:, 3 + 2 * j:5 + 2 * j] = [[dx / r, dy / r], [-dy / q, dx / q]]
    y = z - zhat
    y[1] = wrap(y[1])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    mu = mu + K @ y
    mu[2] = wrap(mu[2])
    IKH = np.eye(mu.size) - K @ H
    return mu, IKH @ P @ IKH.T + K @ R @ K.T


def run(sim):
    Q, R = np.diag(Q_DIAG), np.diag(R_DIAG)
    mu, P = sim["poses"][0].copy(), np.diag(INIT_P_DIAG)
    ids = {}

    def fold(mu, P, reading):
        lid, z = reading
        if lid not in ids:
            mu, P = add_landmark(mu, P, z, R)
            ids[lid] = (mu.size - 3) // 2 - 1
        else:
            mu, P = update(mu, P, z, ids[lid], R)
        return mu, P

    for rd in sim["observations"][0]:
        mu, P = fold(mu, P, rd)
    hist = []
    for k, u in enumerate(sim["controls"]):
        mu, P = predict(mu, P, u, DT, Q)
        for rd in sim["observations"][k + 1]:
            mu, P = fold(mu, P, rd)
        hist.append((mu.copy(), P.copy(), dict(ids)))
    return hist


# ---- Our corridor example ------------------------------------------------------------------
def corridor():
    r, q = 0.01, 0.04
    # state [x, l0]; map l0 from x0 = 0 (known exactly): z = 3
    mu = np.array([0.0, 3.0])
    P = np.array([[0.0, 0.0], [0.0, r]])
    # drive out, command +10 (truth +10.2)
    mu[0] += 10.0
    P[0, 0] += q
    # map l1 from x1: z = 13 - 10.2 = 2.8; l1 = x + z, G_r = 1, G_z = 1
    mu = np.append(mu, mu[0] + 2.8)
    P = np.pad(P, ((0, 1), (0, 1)))
    P[2, :2] = P[0, :2]
    P[:2, 2] = P[0, :2]
    P[2, 2] = P[0, 0] + r
    # drive back, command -9 (truth -8.9)
    mu[0] -= 9.0
    P[0, 0] += q
    before = (mu.copy(), P.copy())
    z = 3.0 - 1.3  # re-see l0 from the true x2 = 1.3
    H = np.array([[-1.0, 1.0, 0.0]])

    def upd(mu, P):
        y = z - (mu[1] - mu[0])
        S = (H @ P @ H.T)[0, 0] + r
        K = (P @ H.T)[:, 0] / S
        P2 = P - np.outer(K, K) * S
        return mu + K * y, P2, K, S, y

    full = upd(*before)
    Pbd = np.diag(np.diag(before[1]))
    blockdiag = upd(before[0], Pbd)
    return before, full, blockdiag


def fig_corridor(before, full, bd):
    truth = {"x": 1.3, "l0": 3.0, "l1": 13.0}
    rows = [("before re-seeing $\\ell_0$", before[0], before[1]),
            ("after, full joint covariance", full[0], full[1]),
            ("after, cross-covariance dropped", bd[0], bd[1])]
    colors = {"x": "#d62728", "l0": "#1f77b4", "l1": "#2ca02c"}
    names = {"x": "robot $x$", "l0": "$\\ell_0$", "l1": "$\\ell_1$"}
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 4.6), sharex=True)
    for ax, (title, mu, P) in zip(axes, rows):
        for i, key in enumerate(["x", "l0", "l1"]):
            sd = np.sqrt(P[i, i])
            ax.errorbar(mu[i], 0, xerr=sd, fmt="o", color=colors[key], capsize=5, ms=6,
                        label=names[key])
            ax.plot(truth[key], 0, marker="|", color="k", ms=18, mew=1.5)
            ax.annotate(f"{mu[i]:.2f} $\\pm$ {sd:.3f}", (mu[i], 0), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=colors[key])
        ax.set_yticks([])
        ax.set_ylim(-1, 1.4)
        ax.set_title(title, fontsize=10, loc="left")
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper center", ncol=3, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.55, 1.35))
    axes[-1].set_xlabel("position along the corridor (m); black ticks are the true positions")
    axes[-1].set_xlim(-0.5, 14)
    fig.tight_layout()
    fig.savefig("corridor.png", dpi=130)
    plt.close(fig)


def fig_loop(sim, hist, k_peak):
    poses = sim["poses"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, k in zip(axes, [k_peak, len(hist) - 1]):
        mu, P, ids = hist[k]
        ax.plot(poses[: k + 2, 0], poses[: k + 2, 1], color="0.6", lw=1.2, label="true path")
        est = np.array([h[0][:2] for h in hist[: k + 1]])
        ax.plot(est[:, 0], est[:, 1], color="#d62728", lw=1.2, label="estimated path")
        ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], marker="x", color="k", s=40,
                   label="true landmarks", zorder=3)
        for lid, j in ids.items():
            m = mu[3 + 2 * j:5 + 2 * j]
            e = covariance_ellipse(m, P[3 + 2 * j:5 + 2 * j, 3 + 2 * j:5 + 2 * j], n_sigma=3.0)
            ax.plot(e[:, 0], e[:, 1], color="#1f77b4", lw=1)
            ax.plot(*m, "o", color="#1f77b4", ms=3)
        e = covariance_ellipse(mu[:2], P[:2, :2], n_sigma=3.0)
        ax.plot(e[:, 0], e[:, 1], color="#d62728", lw=1)
        ax.plot(*poses[0, :2], "s", color="k", ms=5, label="start")
        err = np.mean([np.linalg.norm(mu[3 + 2 * j:5 + 2 * j] - LANDMARKS[lid])
                       for lid, j in ids.items()])
        ax.set_title(f"step {k + 1}: mean landmark error {err:.2f} m", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig("loop_run.png", dpi=120)
    plt.close(fig)


def main():
    before, full, bd = corridor()
    print("corridor before mu", before[0], "\nP\n", before[1])
    print("full: K", full[2], "S", full[3], "y", full[4], "mu", full[0], "var", np.diag(full[1]))
    print("blockdiag: K", bd[2], "mu", bd[0], "var", np.diag(bd[1]))
    fig_corridor(before, full, bd)

    sim = simulate(n_steps=300, seed=0, process_noise=True, meas_noise=True)
    hist = run(sim)
    poses = sim["poses"]
    nees = np.array([robot_nees(mu, P, poses[k + 1]) for k, (mu, P, _) in enumerate(hist)])
    bound = chi2.ppf(0.95, df=3)
    print(f"GATE {GATE:.4f}  chi2_3 95% {bound:.4f}  chi2_2 99% {chi2.ppf(0.99, 2):.4f}")
    print("NEES segment means", [round(float(s.mean()), 2) for s in np.array_split(nees, 6)])
    print("NEES steps above bound", int((nees > bound).sum()), "of", len(nees))
    lm_err = np.array([np.mean([np.linalg.norm(mu[3 + 2 * j:5 + 2 * j] - LANDMARKS[lid])
                                for lid, j in ids.items()]) for mu, P, ids in hist])
    k_peak = int(lm_err.argmax())
    print(f"landmark error peak {lm_err.max():.3f} at step {k_peak + 1}, final {lm_err[-1]:.3f}")
    mu, P, _ = hist[-1]
    N = (mu.size - 3) // 2
    blk = lambda a, b: np.linalg.norm(P[3 + 2 * a:5 + 2 * a, 3 + 2 * b:5 + 2 * b])
    off = [blk(a, b) for a in range(N) for b in range(a + 1, N)]
    dia = [blk(a, a) for a in range(N)]
    print(f"N={N} cross blocks {min(off):.3f}..{max(off):.3f}  diag {min(dia):.3f}..{max(dia):.3f}")
    fig_loop(sim, hist, k_peak)


if __name__ == "__main__":
    main()
