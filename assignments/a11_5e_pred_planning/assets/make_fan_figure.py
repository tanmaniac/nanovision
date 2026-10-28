"""Render the README figure of our three-intention fan.

One agent drives at 1.5 m/s for 6 s (12 steps of 0.5 s) under a constant-turn-rate model, with
turn rate +1/6, 0, or -1/6 rad/s for left, straight, and right, the same rollout as
`pred_toy_scene` in nanovision/data/toy.py. The rollout is written out in numpy here, so the
figure does not use the assignment code.

fan.png. Left panel: the three intentions and their conditional mean, the trajectory a
single-mode squared-error regressor converges to. Right panel: the two-mode configuration hard
winner-take-all can get stuck in, one mode on the right turn and one mode on the mean of the left
turn and straight.

Plot axes: forward distance is vertical, lateral distance horizontal with the agent's left drawn
on the left.

Run from this directory:

    python make_fan_figure.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

v, w, dt, T = 1.5, 1.0 / 6.0, 0.5, 12
t = np.arange(0, T + 1) * dt          # include the start point for drawing
left = np.stack([v / w * np.sin(w * t), v / w * (1 - np.cos(w * t))], 1)
straight = np.stack([v * t, 0 * t], 1)
right = left * [1, -1]
mean3 = (left + straight + right) / 3
mean_ls = (left + straight) / 2

cols = {"left": "tab:blue", "straight": "tab:green", "right": "tab:purple"}


def draw(ax, path, **kw):
    ax.plot(-path[:, 1], path[:, 0], **kw)        # horizontal = -y so that left is drawn left


fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
for ax in axes:
    for name, p in (("left", left), ("straight", straight), ("right", right)):
        draw(ax, p, color=cols[name], lw=2, alpha=0.35, label=f"{name} (true future)")
        ax.plot(-p[-1, 1], p[-1, 0], "o", color=cols[name], alpha=0.5)
    ax.plot(0, 0, "ks", ms=7)
    ax.set_xlabel("lateral (m), agent's left drawn on the left")
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.5, 10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("forward (m)")

draw(axes[0], mean3, color="tab:red", lw=2.5, label="conditional mean (one mode)")
axes[0].plot(-mean3[-1, 1], mean3[-1, 0], "*", color="tab:red", ms=14)
axes[0].annotate("(8.05, 0.00)\n0.95 m short of straight,\n4.16 m from each turn",
                 xy=(0, mean3[-1, 0]), xytext=(1.0, 5.2), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
axes[0].set_title("One mode: the average of three intentions")
axes[0].legend(loc="lower left", fontsize=8)

draw(axes[1], right, color="tab:orange", lw=2.5, ls="--", label="mode A (on the right turn)")
draw(axes[1], mean_ls, color="tab:red", lw=2.5, label="mode B (mean of left and straight)")
axes[1].plot(-mean_ls[-1, 1], mean_ls[-1, 0], "*", color="tab:red", ms=14)
axes[1].set_title("Two live modes: one of them averages two intentions")
axes[1].legend(loc="lower left", fontsize=8)

fig.tight_layout()
fig.savefig("fan.png", dpi=110)
print("wrote fan.png")
