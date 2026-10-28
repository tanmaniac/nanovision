"""Render the README figures for the world-models notes. Uses no assignment code.

pole_latent_filter.png. Our three-class pole latent (one categorical variable with classes "hanging",
"sideways", "upright") through one predict-update cycle of a swing-up: the posterior after frame t-1,
the prior for step t (the prediction before frame t arrives), and the posterior after frame t. The
KL between the last two is the quantity the world-model loss penalizes.

near_flat_gradients.png. Our near-flat reward r(a) = 1 - a^2/5 (the control factor of the
dm_control cartpole reward, swingup and balance alike, at an upright, centered, still pole) under a
Gaussian policy a = mu + sigma * eps with mu = 0.2, sigma = 0.5. Left: the reward and the policy
density. Right: the mean plus or minus one standard deviation of an estimate of d E[r] / d mu
averaged over N sampled actions, for the dynamics-backprop (reparameterized) estimator and for
REINFORCE with the exact baseline, with and without extra return noise of standard deviation 0.5
that does not depend on the action. The one-sample standard deviations are measured from 20000 draws.

Run from this directory:

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- pole latent filter
classes = ["hanging", "sideways", "upright"]
post_prev = np.array([0.10, 0.80, 0.10])
prior = np.array([0.20, 0.60, 0.20])
post = np.array([0.02, 0.08, 0.90])
kl = float((post * np.log(post / prior)).sum())

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
panels = [
    (post_prev, "posterior at t-1\n(after frame t-1: swinging up)", "tab:blue"),
    (prior, "prior at t\n(predicted, before frame t)", "tab:orange"),
    (post, "posterior at t\n(after frame t: pole up)", "tab:blue"),
]
for ax, (probs, title, color) in zip(axes, panels):
    ax.bar(classes, probs, color=color)
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("probability")
fig.text(0.355, 0.5, "predict\n(GRU step\nwith force)", ha="center", va="center", fontsize=9)
fig.text(0.68, 0.5, "update\n(encode\nframe t)", ha="center", va="center", fontsize=9)
fig.suptitle(f"Our three-class pole latent: KL[posterior || prior] at step t = {kl:.3f} nats",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.subplots_adjust(wspace=0.45)
fig.savefig("pole_latent_filter.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- near-flat reward gradients
mu, sigma, noise_sd = 0.2, 0.5, 0.5
rng = np.random.default_rng(0)
eps = rng.standard_normal(20000)
noise = noise_sd * rng.standard_normal(20000)
a = mu + sigma * eps
r = 1 - a**2 / 5
baseline = 1 - (mu**2 + sigma**2) / 5
g_path = -2 * a / 5
g_rf = (r - baseline) * eps / sigma
g_rf_noise = (r + noise - baseline) * eps / sigma
true_grad = -2 * mu / 5

fig, (ax_r, ax_g) = plt.subplots(1, 2, figsize=(11, 3.8))
xs = np.linspace(-1, 1, 200)
ax_r.plot(xs, 1 - xs**2 / 5, color="black", label="reward r(a) = 1 - a²/5")
dens = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
ax_r2 = ax_r.twinx()
ax_r2.fill_between(xs, dens, color="tab:orange", alpha=0.25)
ax_r2.set_ylabel("policy density", color="tab:orange")
ax_r2.set_ylim(0, dens.max() * 1.3)
ax_r.set_ylim(0, 1.05)
ax_r.set_xlabel("action a (force)")
ax_r.set_ylabel("reward")
ax_r.set_title("Our near-flat reward and the policy N(0.2, 0.5²)", fontsize=10)
ax_r.legend(loc="lower center", fontsize=8)

Ns = np.logspace(0, 3, 200)
for g, color, label in (
    (g_rf_noise, "tab:red", "REINFORCE with baseline, return noise 0.5"),
    (g_rf, "tab:purple", "REINFORCE with baseline, no extra noise"),
    (g_path, "tab:green", "dynamics backprop"),
):
    sd = g.std()
    ax_g.fill_between(Ns, true_grad - sd / np.sqrt(Ns), true_grad + sd / np.sqrt(Ns),
                      color=color, alpha=0.2)
    ax_g.plot(Ns, true_grad + sd / np.sqrt(Ns), color=color, lw=1,
              label=f"{label} (one-sample std {sd:.2f})")
    ax_g.plot(Ns, true_grad - sd / np.sqrt(Ns), color=color, lw=1)
ax_g.axhline(true_grad, color="black", ls="--", lw=1, label=f"true gradient {true_grad:.2f}")
ax_g.axhline(0.0, color="gray", lw=0.8)
ax_g.set_xscale("log")
ax_g.set_ylim(-1.2, 1.0)
ax_g.set_xlabel("number of sampled actions N averaged into one estimate")
ax_g.set_ylabel("estimate of dE[r]/dμ, mean ± 1 std")
ax_g.set_title("Where each band clears zero, the sign of the gradient is reliable", fontsize=10)
ax_g.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig("near_flat_gradients.png", dpi=130)
plt.close(fig)
print(f"KL = {kl:.4f}; stds: dynamics backprop {g_path.std():.3f}, REINFORCE {g_rf.std():.3f}, "
      f"REINFORCE+noise {g_rf_noise.std():.3f}")
