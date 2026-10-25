"""Render the README figure for our four-corner example.

A point mass sits at the center (0.5, 0.5) of the unit square. The demonstrations head for one of
four goals at the corners (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85), with equal
probability, at speed 0.05 per step, and the conditioning does not say which goal. So the
demonstrated action at the center is one of four vectors of length 0.05 along the diagonals.

fourcorner.png. Left panel: the four demonstrated actions and the least-squares regressor's output,
their mean, which is the zero vector. Right panel: the same four actions rescaled by 0.05 / sqrt(2)
so that they sit at (+-1, +-1), Gaussian noise samples at flow time t = 0, and the trajectories of the
exact marginal flow-matching velocity from those samples. Because the four modes are the product of
two independent +-1 coordinates, the marginal velocity factorizes per coordinate:
u_t(x) = (tanh(t x / (1-t)^2) - x) / (1-t). The trajectories are integrated with fine Euler steps.

Run from this directory:

    python make_fourcorner_figure.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

goals = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]])
center = np.array([0.5, 0.5])
d = goals - center
modes = d / np.linalg.norm(d, axis=1, keepdims=True) * 0.05      # four actions, length 0.05
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.6))

for m, col in zip(modes, colors):
    ax_l.arrow(0, 0, m[0], m[1], color=col, width=0.0008, head_width=0.004,
               length_includes_head=True)
ax_l.scatter([0], [0], marker="X", s=140, color="tab:red", zorder=5,
             label="least-squares regressor (the mean)")
ax_l.set_xlim(-0.05, 0.05)
ax_l.set_ylim(-0.05, 0.05)
ax_l.set_aspect("equal")
ax_l.axhline(0, color="k", lw=0.3)
ax_l.axvline(0, color="k", lw=0.3)
ax_l.set_xlabel("action x-component")
ax_l.set_ylabel("action y-component")
ax_l.set_title("Demonstrated actions at the center (arrows)\nand their mean")
ax_l.legend(loc="lower center", fontsize=8)


def u(x, t):
    return (np.tanh(t * x / (1 - t) ** 2) - x) / (1 - t)


rng = np.random.default_rng(0)
z0 = rng.standard_normal((40, 2))
steps = 4000
t_end = 0.999
dt = t_end / steps
traj = [z0.copy()]
z = z0.copy()
for k in range(steps):
    z = z + dt * u(z, k * dt)
    traj.append(z.copy())
traj = np.stack(traj)                                              # (steps+1, n, 2)
end = np.sign(traj[-1])
unit = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
for i in range(z0.shape[0]):
    j = int(np.argmin(np.abs(unit - end[i]).sum(1)))
    ax_r.plot(traj[:, i, 0], traj[:, i, 1], color=colors[j], lw=0.8, alpha=0.8)
ax_r.scatter(z0[:, 0], z0[:, 1], s=10, color="0.3", zorder=4, label="noise samples (t = 0)")
ax_r.scatter(unit[:, 0], unit[:, 1], s=60, color=colors, edgecolor="k", zorder=5,
             label="demonstrated actions (t = 1)")
ax_r.scatter([0], [0], marker="X", s=140, color="tab:red", zorder=5, label="the mean")
ax_r.set_aspect("equal")
ax_r.set_xlim(-2.6, 2.6)
ax_r.set_ylim(-2.6, 2.6)
ax_r.set_xlabel("action x-component / 0.035")
ax_r.set_ylabel("action y-component / 0.035")
ax_r.set_title("Flow trajectories of the exact marginal velocity\n(rescaled units)")
ax_r.legend(loc="upper left", fontsize=7)

fig.tight_layout()
fig.savefig("fourcorner.png", dpi=110)
print("wrote fourcorner.png")
