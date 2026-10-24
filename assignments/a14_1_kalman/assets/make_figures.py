"""Render the README figures for the Kalman-filter assignment.

first_reading.png. Our first-reading example: the demo robot at its start pose (origin, facing
+x) with the prior P0 = diag(0.2, 0.2, 0.1), one range-bearing reading of the landmark at (6, 2)
with the demo's noise (0.15 m, 3 degrees). Left panel: the geometry and the unobservable motion
(orbit the landmark at constant range while turning by the same angle). Right panel: the prior and
posterior 1-sigma position ellipses after one EKF update.

unscented_exp.png. Our e^x example: x ~ N(0.3, 0.4^2) pushed through f(x) = e^x. The tangent line
used by linearization, the three sigma points (alpha = 0.5, beta = 2, kappa = 0) and their images,
and the true, linearized, and unscented means of f(x).

demo_run.png. The EKF on the assignment's simulated run (same simulator, seed, and noise as
config.py and _helpers.simulate_unicycle), with the 1-sigma position ellipse every 10 steps. The
EKF and a UKF (the same wiring as _helpers.ukf_predict / ukf_update) are reimplemented here in
NumPy, so the figure does not need the C++ build. The script prints the run statistics quoted in
the README: the across/along ellipse ratio, the heading standard deviation range, the mean
position error of each filter, and the largest EKF-UKF position gap.

Run from this directory:

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LANDMARK = np.array([6.0, 2.0])
SIG_R, SIG_PHI = 0.15, np.deg2rad(3.0)
R = np.diag([SIG_R**2, SIG_PHI**2])
Q = np.diag(np.array([0.02, 0.02, 0.01]) ** 2)
P0 = np.diag([0.2, 0.2, 0.1])
DT, V, OMEGA, N_STEPS, SEED = 0.1, 1.0, 0.25, 120, 0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def h(x, l=LANDMARK):
    dx, dy = l[0] - x[0], l[1] - x[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])


def H_jac(x, l=LANDMARK):
    dx, dy = l[0] - x[0], l[1] - x[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0]])


def f(x, u, dt):
    return np.array([x[0] + u[0] * dt * np.cos(x[2]), x[1] + u[0] * dt * np.sin(x[2]),
                     wrap(x[2] + u[1] * dt)])


def F_jac(x, u, dt):
    F = np.eye(3)
    F[0, 2] = -u[0] * dt * np.sin(x[2])
    F[1, 2] = u[0] * dt * np.cos(x[2])
    return F


def ekf_update(mu, P, z):
    H = H_jac(mu)
    y = z - h(mu)
    y[1] = wrap(y[1])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    mu = mu + K @ y
    mu[2] = wrap(mu[2])
    A = np.eye(3) - K @ H
    return mu, A @ P @ A.T + K @ R @ K.T


def ellipse(mean, cov, n=100):
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, n)
    pts = vecs @ (np.sqrt(np.clip(vals, 0, None))[:, None] * np.stack([np.cos(t), np.sin(t)]))
    return pts.T + mean


# ---- first_reading.png -------------------------------------------------------
x0 = np.zeros(3)
mu1, P1 = ekf_update(x0.copy(), P0, h(x0))
r0 = np.hypot(*LANDMARK)
d = LANDMARK / r0
p = np.array([-d[1], d[0]])

fig, (ax_g, ax_z) = plt.subplots(1, 2, figsize=(11, 4.8))
th = np.linspace(-0.5, 0.5, 200) + np.arctan2(-LANDMARK[1], -LANDMARK[0])
ax_g.plot(LANDMARK[0] + r0 * np.cos(th), LANDMARK[1] + r0 * np.sin(th), ":", color="gray",
          lw=1, label="constant range")
ax_g.plot([0, LANDMARK[0]], [0, LANDMARK[1]], color="tab:green", lw=1, label="ray to landmark")
for a in (-0.35, 0.0, 0.35):
    # pose obtained by orbiting the landmark by angle a and turning by the same angle
    c, s = np.cos(a), np.sin(a)
    pos = LANDMARK + np.array([[c, -s], [s, c]]) @ (-LANDMARK)
    head = a
    z = h(np.array([pos[0], pos[1], head]))
    ax_g.annotate("", xy=pos + 0.9 * np.array([np.cos(head), np.sin(head)]), xytext=pos,
                  arrowprops=dict(arrowstyle="->", color="black" if a == 0 else "tab:purple"))
    ax_g.plot(*pos, "o", color="black" if a == 0 else "tab:purple", ms=4)
    ax_g.plot([pos[0], LANDMARK[0]], [pos[1], LANDMARK[1]], color="tab:purple", lw=0.5,
              alpha=0.0 if a == 0 else 0.6)
    print(f"orbit {a:+.2f}: range {z[0]:.4f}, bearing {np.degrees(z[1]):.4f} deg")
ax_g.plot(*LANDMARK, "*", color="goldenrod", ms=16, label="landmark (6, 2)")
ax_g.set_title("Poses that give the same range and bearing")
ax_g.set_aspect("equal")
ax_g.set_xlabel("x (m)")
ax_g.set_ylabel("y (m)")
ax_g.legend(loc="lower right", fontsize=8)

pe = ellipse(np.zeros(2), P0[:2, :2])
po = ellipse(mu1[:2], P1[:2, :2])
ax_z.plot(pe[:, 0], pe[:, 1], color="gray", label="prior, 1-sigma")
ax_z.plot(po[:, 0], po[:, 1], color="tab:red", label="after one reading, 1-sigma")
ax_z.plot([0, 0.8 * d[0]], [0, 0.8 * d[1]], color="tab:green", lw=1, label="toward landmark")
ax_z.annotate("", xy=0.6 * p, xytext=-0.6 * p,
              arrowprops=dict(arrowstyle="<->", color="tab:purple"))
ax_z.text(*(0.62 * p), " across the ray", color="tab:purple", fontsize=8)
ax_z.plot(0, 0, "ko", ms=4)
ax_z.set_aspect("equal")
ax_z.set_xlim(-0.75, 0.85)
ax_z.set_ylim(-0.65, 0.65)
ax_z.set_title("Position uncertainty before and after the reading")
ax_z.set_xlabel("x (m)")
ax_z.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("first_reading.png", dpi=130)
print("first reading: along", np.sqrt(d @ P1[:2, :2] @ d), "across", np.sqrt(p @ P1[:2, :2] @ p),
      "heading deg", np.degrees(np.sqrt(P1[2, 2])))

# ---- unscented_exp.png -------------------------------------------------------
mu, sig = 0.3, 0.4
alpha, beta, kappa, n = 0.5, 2.0, 0.0, 1
lam = alpha**2 * (n + kappa) - n
pts = np.array([mu, mu + np.sqrt(n + lam) * sig, mu - np.sqrt(n + lam) * sig])
Wm = np.array([lam / (n + lam), 0.5 / (n + lam), 0.5 / (n + lam)])
ut_mean = Wm @ np.exp(pts)
true_mean = np.exp(mu + sig**2 / 2)
lin_mean = np.exp(mu)
print("e^x: true", true_mean, "linearized", lin_mean, "unscented", ut_mean)

fig, ax = plt.subplots(figsize=(7, 4.8))
xs = np.linspace(-0.9, 1.5, 300)
ax.plot(xs, np.exp(xs), color="black", lw=1.5, label="f(x) = exp(x)")
ax.plot(xs, np.exp(mu) * (1 + (xs - mu)), "--", color="tab:red", lw=1,
        label="tangent at the mean (linearization)")
dens = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
ax.fill_between(xs, 0, 0.25 * dens, color="tab:blue", alpha=0.15, label="input N(0.3, 0.4^2), scaled")
ys = np.linspace(0.3, 4.5, 400)
ldens = np.exp(-0.5 * ((np.log(ys) - mu) / sig) ** 2) / (ys * sig * np.sqrt(2 * np.pi))
ax.fill_betweenx(ys, -0.9, -0.9 + 0.25 * ldens, color="tab:orange", alpha=0.25,
                 label="density of f(x), scaled")
for k, xp in enumerate(pts):
    ax.plot([xp, xp], [0, np.exp(xp)], ":", color="tab:blue", lw=1)
    ax.plot([-0.9, xp], [np.exp(xp), np.exp(xp)], ":", color="tab:blue", lw=1)
    ax.plot(xp, 0, "o", color="tab:blue", ms=6, label="sigma points" if k == 0 else None)
ax.axhline(lin_mean, color="tab:red", lw=1)
ax.axhline(ut_mean, color="tab:blue", lw=1.2)
ax.axhline(true_mean, color="tab:orange", lw=1.5, ls=(0, (4, 3)), zorder=5)
ax.text(-0.62, true_mean + 0.04, "true mean 1.462 (dashed)", color="tab:orange", fontsize=8,
        va="bottom")
ax.text(0.78, ut_mean + 0.04, "unscented mean 1.458 (solid)", color="tab:blue", fontsize=8,
        va="bottom")
ax.text(0.78, lin_mean - 0.03, "linearized mean 1.350", color="tab:red", fontsize=8, va="top")
ax.set_xlim(-0.9, 1.5)
ax.set_ylim(0, 4.5)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Pushing a Gaussian through exp: one tangent vs three points")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("unscented_exp.png", dpi=130)

# ---- demo_run.png --------------------------------------------------------------
rng = np.random.default_rng(SEED)
x = np.zeros(3)
gt, meas = [x.copy()], []
for _ in range(N_STEPS):
    x = np.array([x[0] + V * DT * np.cos(x[2]), x[1] + V * DT * np.sin(x[2]),
                  wrap(x[2] + OMEGA * DT)])
    x = x + rng.normal(0.0, np.sqrt(np.diag(Q)))
    x[2] = wrap(x[2])
    gt.append(x.copy())
    z = h(x) + rng.normal(0.0, [SIG_R, SIG_PHI])
    meas.append(z)
gt = np.array(gt)

u = np.array([V, OMEGA])


def sigma_points(m, C, alpha=0.5, beta=2.0, kappa=0.0):
    k = m.size
    lam = alpha**2 * (k + kappa) - k
    L = np.linalg.cholesky((k + lam) * C)
    pts = np.vstack([m, m + L.T, m - L.T])
    Wm = np.full(2 * k + 1, 0.5 / (k + lam))
    Wm[0] = lam / (k + lam)
    Wc = Wm.copy()
    Wc[0] += 1 - alpha**2 + beta
    return pts, Wm, Wc


def unscented(pts, Wm, Wc, noise):
    m = Wm @ pts
    d = pts - m
    return m, (Wc[:, None] * d).T @ d + noise


def ukf_step(m, C, z):
    pts, Wm, Wc = sigma_points(m, C)
    m, C = unscented(np.array([f(s, u, DT) for s in pts]), Wm, Wc, Q)
    pts, Wm, Wc = sigma_points(m, C)
    zs = np.array([h(s) for s in pts])
    zm, S = unscented(zs, Wm, Wc, R)
    Pxz = ((Wc[:, None] * (pts - m)).T) @ (zs - zm)
    K = Pxz @ np.linalg.inv(S)
    y = z - zm
    y[1] = wrap(y[1])
    return m + K @ y, C - K @ S @ K.T


mu, P = gt[0].copy(), P0.copy()
mu_u, P_u = gt[0].copy(), P0.copy()
hstd, err_e, err_u, gap = [], [], [], []
est, ratios = [], []
fig, ax = plt.subplots(figsize=(7, 6))
for k in range(N_STEPS):
    mu, P = f(mu, u, DT), F_jac(mu, u, DT) @ P @ F_jac(mu, u, DT).T + Q
    mu, P = ekf_update(mu, P, meas[k])
    mu_u, P_u = ukf_step(mu_u, P_u, meas[k])
    est.append(mu[:2].copy())
    hstd.append(np.degrees(np.sqrt(P[2, 2])))
    err_e.append(np.linalg.norm(mu[:2] - gt[k + 1][:2]))
    err_u.append(np.linalg.norm(mu_u[:2] - gt[k + 1][:2]))
    gap.append(np.linalg.norm(mu[:2] - mu_u[:2]))
    dd = LANDMARK - mu[:2]
    dd /= np.linalg.norm(dd)
    pp = np.array([-dd[1], dd[0]])
    ratios.append(np.sqrt(pp @ P[:2, :2] @ pp) / np.sqrt(dd @ P[:2, :2] @ dd))
    if k % 10 == 0 or k == N_STEPS - 1:
        e = ellipse(mu[:2], P[:2, :2])
        ax.plot(e[:, 0], e[:, 1], color="tab:red", lw=1)
        ax.plot([mu[0], LANDMARK[0]], [mu[1], LANDMARK[1]], color="tab:green", lw=0.4, alpha=0.6)
est = np.array(est)
print("demo: across/along ratio min, median, max", np.min(ratios), np.median(ratios), np.max(ratios),
      "at step", int(np.argmax(ratios)))
print("demo: EKF heading std (deg) min, max", min(hstd), max(hstd))
print("demo: mean position error EKF", np.mean(err_e), "UKF", np.mean(err_u))
print("demo: largest EKF-UKF position gap (m)", max(gap))
ax.plot(gt[:, 0], gt[:, 1], color="gray", lw=1.5, label="ground truth")
ax.plot(est[:, 0], est[:, 1], color="tab:red", lw=1, label="EKF estimate")
ax.plot(*LANDMARK, "*", color="goldenrod", ms=16, label="landmark")
ax.plot([], [], color="tab:green", lw=0.6, label="ray to landmark")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("EKF on the demo run, 1-sigma position ellipse every 10 steps")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("demo_run.png", dpi=130)
