"""Render the README figures for the ±1 example.

1D noise x0 ~ N(0, 1), data x1 in {-1, +1} with equal probability, linear path.

pm1_trajectories.png. Left panel: the straight conditional paths x_t = (1-t) x0 + t x1 for random
(x0, x1) pairs. Right panel: the trajectories of the exact marginal field
u_t(x) = (tanh(t x / (1-t)^2) - x) / (1-t) (Exercise 1) from the same x0, integrated with fine
Euler steps.

pm1_path.png. The probability path p_t = 0.5 N(-t, (1-t)^2) + 0.5 N(+t, (1-t)^2) at several t.

Run from this directory:

    python make_pm1_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
n = 24
x0 = np.sort(rng.standard_normal(n))
x1 = rng.choice([-1.0, 1.0], size=n)

fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

t = np.linspace(0.0, 1.0, 2)
for a, b in zip(x0, x1):
    ax_c.plot(t, (1 - t) * a + t * b, color="tab:blue" if b > 0 else "tab:orange", lw=1)
ax_c.set_title("Conditional paths: straight, they cross")
ax_c.set_xlabel("t")
ax_c.set_ylabel("x")

steps = 4000
t_end = 0.999
dt = t_end / steps
traj = np.empty((steps + 1, n))
traj[0] = x0
x = x0.copy()
for k in range(steps):
    tk = k * dt
    u = (np.tanh(tk * x / (1 - tk) ** 2) - x) / (1 - tk)
    x = x + u * dt
    traj[k + 1] = x
ts = np.linspace(0.0, t_end, steps + 1)
for i in range(n):
    ax_m.plot(ts, traj[:, i], color="tab:blue" if traj[-1, i] > 0 else "tab:orange", lw=1)
ax_m.set_title("Marginal-field trajectories: curved, no crossings")
ax_m.set_xlabel("t")

for ax in (ax_c, ax_m):
    ax.axhline(1, color="gray", lw=0.5, ls=":")
    ax.axhline(-1, color="gray", lw=0.5, ls=":")

fig.tight_layout()
fig.savefig("pm1_trajectories.png", dpi=120)
print("endpoints min/max |x_1| - 1:", np.abs(np.abs(traj[-1]) - 1).max())

# The +-1 example's probability path: p_t = 0.5 N(-t, (1-t)^2) + 0.5 N(+t, (1-t)^2).
fig2, ax = plt.subplots(figsize=(7, 3.2))
xs = np.linspace(-2.5, 2.5, 1001)
for tt, shade in zip((0.0, 0.3, 0.5, 0.7, 0.85), (0.15, 0.35, 0.55, 0.75, 0.95)):
    sd = 1 - tt
    p = 0.5 * (np.exp(-((xs - tt) ** 2) / (2 * sd**2)) + np.exp(-((xs + tt) ** 2) / (2 * sd**2)))
    p /= sd * np.sqrt(2 * np.pi)
    ax.plot(xs, p, color=plt.cm.viridis(shade), lw=1.5, label=f"t = {tt}")
for x1 in (-1, 1):
    ax.axvline(x1, color="gray", lw=0.8, ls=":")
ax.set_xlabel("x")
ax.set_ylabel("density $p_t(x)$")
ax.set_title("The ±1 example: noise N(0, 1) at t = 0, two data points at ±1 at t = 1")
ax.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("pm1_path.png", dpi=120)
print("sign preserved:", bool(np.all(np.sign(traj[-1]) == np.sign(x0))))
