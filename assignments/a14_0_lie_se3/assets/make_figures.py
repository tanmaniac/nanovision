"""Render the README figures for the Lie-group notes. Pure NumPy, closed-form exp.

arc_example.png. Our quarter-turn arc example, seen from above: the twist
xi = [rho; theta] = [1, 0, 0, 0, 0, pi/2] (drive 1 m forward along body x while turning
90 degrees left, in unit time). Left panel: the constant-twist path T(s) = exp(s xi) with
the body frame drawn at five values of s. Right panel: the entrywise blend
R(s) = (1-s) I + s R1, t(s) = s t1, whose axes shrink and whose position cuts the chord.

blend_metrics.png. det R(s) and the orthonormality error ||R^T R - I||_F along both paths for
the pose pair the Rerun visualization uses (identity to se3_exp([2, 1, 0.5, 0.4, -1.6, 2.2])).

Run from this directory:

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def so3_left_jacobian(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-8:
        return np.eye(3) + 0.5 * K + K @ K / 6
    return np.eye(3) + (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K


def se3_exp(xi):
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[3:])
    T[:3, 3] = so3_left_jacobian(xi[3:]) @ xi[:3]
    return T


# Figure 1: our quarter-turn arc example.
xi = np.array([1.0, 0, 0, 0, 0, np.pi / 2])
T1 = se3_exp(xi)
R1, t1 = T1[:3, :3], T1[:3, 3]
print("arc end translation", t1[:2], "(2/pi =", 2 / np.pi, ")")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
ss = np.linspace(0, 1, 101)
arc = np.array([se3_exp(s * xi)[:2, 3] for s in ss])
L = 0.25
for ax, title in zip(axes, ["Constant twist, exp(s xi): stays a rigid frame",
                            "Entrywise blend: axes shrink, path cuts the chord"]):
    ax.plot(arc[:, 0], arc[:, 1], color="0.6", lw=1, ls=":")
    ax.plot([0, t1[0]], [0, t1[1]], color="0.6", lw=1, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].annotate("", xy=(1.0, -0.12), xytext=(0, -0.12),
                 arrowprops=dict(arrowstyle="->", color="tab:purple"))
axes[0].text(0.5, -0.2, r"$\rho = (1, 0, 0)$, the linear velocity", color="tab:purple",
             ha="center", va="top", fontsize=9)
for s in [0, 0.25, 0.5, 0.75, 1.0]:
    T = se3_exp(s * xi)
    Rn = (1 - s) * np.eye(3) + s * R1
    tn = s * t1
    for ax, R, t in [(axes[0], T[:3, :3], T[:3, 3]), (axes[1], Rn, tn)]:
        ax.arrow(t[0], t[1], L * R[0, 0], L * R[1, 0], color="tab:red", width=0.006,
                 head_width=0.04, length_includes_head=True)
        ax.arrow(t[0], t[1], L * R[0, 1], L * R[1, 1], color="tab:green", width=0.006,
                 head_width=0.04, length_includes_head=True)
    axes[1].text(tn[0] + 0.05, tn[1] - 0.09, f"det {np.linalg.det(Rn):.2f}", fontsize=8)
axes[0].text(t1[0] + 0.05, t1[1] + 0.05, "t = (0.64, 0.64)", fontsize=9)
axes[0].set_xlim(-0.2, 1.1)
axes[0].set_ylim(-0.35, 1.0)
fig.tight_layout()
fig.savefig("arc_example.png", dpi=130)
plt.close(fig)

# Figure 2: determinant and orthonormality error along the visualization's pose pair.
Tv = se3_exp(np.array([2.0, 1.0, 0.5, 0.4, -1.6, 2.2]))
Rv = Tv[:3, :3]
theta = np.array([0.4, -1.6, 2.2])
print("viz rotation angle", np.linalg.norm(theta), np.degrees(np.linalg.norm(theta)))
ss = np.linspace(0, 1, 61)
dm, dn, om, on = [], [], [], []
for s in ss:
    Rm = so3_exp(s * theta)  # the rotation block of T0 boxplus s (T1 boxminus T0), T0 = I
    Rn = (1 - s) * np.eye(3) + s * Rv
    dm.append(np.linalg.det(Rm))
    dn.append(np.linalg.det(Rn))
    om.append(np.linalg.norm(Rm.T @ Rm - np.eye(3)))
    on.append(np.linalg.norm(Rn.T @ Rn - np.eye(3)))
print("naive min det", min(dn), "max orth err", max(on), "manifold max |det-1|",
      max(abs(np.array(dm) - 1)), "max orth", max(om))
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.3))
a.plot(ss, dm, label="manifold (constant twist)")
a.plot(ss, dn, label="entrywise blend")
a.set_title("det R(s)", fontsize=10)
b.plot(ss, om, label="manifold (constant twist)")
b.plot(ss, on, label="entrywise blend")
b.set_title(r"orthonormality error $\|R^\top R - I\|_F$", fontsize=10)
for ax in (a, b):
    ax.set_xlabel("s")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("blend_metrics.png", dpi=130)
plt.close(fig)
