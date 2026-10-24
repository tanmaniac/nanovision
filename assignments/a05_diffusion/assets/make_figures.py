"""Render the README figures for the diffusion lecture notes.

noising.png. One training picture (a plus sign) carried through the forward process of the
cosine schedule with T = 1000, using one fixed noise draw, at several timesteps.

pm1_denoiser.png. Our +-1 example: one pixel whose clean value is -1 or +1 with probability
1/2 each. Left panel: the noisy-pixel density p_t(x) = 0.5 N(a, b^2) + 0.5 N(-a, b^2) at
several noise levels (a = sqrt(abar_t), b = sqrt(1 - abar_t)). Right panel: the ideal
denoiser E[x_0 | x_t = x] = tanh(a x / b^2) at the same noise levels.

pm1_sampling.png. Sampling our +-1 example with the ideal denoiser in place of a network,
cosine schedule with T = 1000. Left: DDPM ancestral sampling (noise added at every step).
Right: deterministic DDIM (eta = 0) on every 20th timestep, from the same starting noise.

Run from this directory:

    python make_figures.py
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for nanovision

from nanovision.data.toy import diffusion_image_batch  # noqa: E402


def cosine_abar(T, s=0.008):
    u = np.arange(T + 1) / T
    f = np.cos((u + s) / (1 + s) * np.pi / 2) ** 2
    return f[1:] / f[0]                      # abar_1 .. abar_T, array index i holds abar_{i+1}


T = 1000
abar = cosine_abar(T)

# ---------------------------------------------------------------- noising.png
x0_img, labels = diffusion_image_batch(n=8, seed=3)
idx = int((labels == 2).nonzero()[0, 0])     # first plus sign in the batch
x0_img = x0_img[idx, 0].numpy()
rng = np.random.default_rng(0)
eps = rng.standard_normal(x0_img.shape)
steps = [0, 250, 500, 750, 999]
fig, axes = plt.subplots(1, len(steps) + 1, figsize=(12, 2.4))
axes[0].imshow(x0_img, cmap="gray", vmin=-1, vmax=1)
axes[0].set_title("clean $x_0$")
for ax, i in zip(axes[1:], steps):
    a, b = math.sqrt(abar[i]), math.sqrt(1 - abar[i])
    ax.imshow(a * x0_img + b * eps, cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"t = {i + 1}, " + r"$\bar\alpha_t$" + f" = {abar[i]:.2f}")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig("noising.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- pm1_denoiser.png
levels = [0.95, 0.8, 0.5, 0.2, 0.05]
x = np.linspace(-3, 3, 601)
cmap = plt.get_cmap("viridis")
fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(11, 4))
for k, ab in enumerate(levels):
    a, b = math.sqrt(ab), math.sqrt(1 - ab)
    gauss = lambda m: np.exp(-(x - m) ** 2 / (2 * b * b)) / (b * math.sqrt(2 * math.pi))
    col = cmap(k / (len(levels) - 1))
    lab = r"$\bar\alpha$" + f" = {ab}"
    ax_p.plot(x, 0.5 * gauss(a) + 0.5 * gauss(-a), color=col, label=lab)
    ax_d.plot(x, np.tanh(a * x / (b * b)), color=col, label=lab)
ax_p.set_title(r"density of the noisy pixel $x_t$")
ax_p.set_xlabel("$x_t$")
ax_p.legend(fontsize=8)
ax_d.plot(x, np.sign(x), color="gray", lw=0.6, ls=":")
ax_d.set_title(r"ideal denoiser $\mathbb{E}[x_0 \mid x_t]$")
ax_d.set_xlabel("$x_t$")
ax_d.set_ylim(-1.1, 1.1)
ax_d.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("pm1_denoiser.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- pm1_sampling.png
n = 20
rng = np.random.default_rng(1)
xT = rng.standard_normal(n)
abar_prev = np.concatenate([[1.0], abar[:-1]])


def denoise(x, ab):
    a, b = math.sqrt(ab), math.sqrt(1 - ab)
    x0h = np.tanh(a * x / (b * b))
    return x0h, (x - a * x0h) / b


# DDPM ancestral, beta-tilde variance, every timestep
x = xT.copy()
ddpm = [x.copy()]
for i in reversed(range(T)):
    ab, abp = abar[i], abar_prev[i]
    beta = min(1 - ab / abp, 0.999)
    x0h, _ = denoise(x, ab)
    mean = math.sqrt(abp) * beta / (1 - ab) * x0h + math.sqrt(1 - beta) * (1 - abp) / (1 - ab) * x
    if i > 0:
        x = mean + math.sqrt((1 - abp) / (1 - ab) * beta) * rng.standard_normal(n)
    else:
        x = mean
    ddpm.append(x.copy())
ddpm = np.array(ddpm)
t_ddpm = np.arange(T, -1, -1)

# DDIM eta = 0, every 20th timestep
ts = list(range(T - 1, -1, -20))
x = xT.copy()
ddim = [x.copy()]
t_ddim = [T]
for j, i in enumerate(ts):
    prev = ts[j + 1] if j + 1 < len(ts) else -1
    abp = abar[prev] if prev >= 0 else 1.0
    x0h, eh = denoise(x, abar[i])
    x = math.sqrt(abp) * x0h + math.sqrt(1 - abp) * eh
    ddim.append(x.copy())
    t_ddim.append(prev + 1)
ddim = np.array(ddim)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for k in range(n):
    col = "tab:blue" if ddpm[-1, k] > 0 else "tab:orange"
    ax_a.plot(t_ddpm, ddpm[:, k], color=col, lw=0.8)
    col = "tab:blue" if ddim[-1, k] > 0 else "tab:orange"
    ax_b.plot(t_ddim, ddim[:, k], color=col, lw=1, marker=".", ms=3)
ax_a.set_title("DDPM ancestral: 1000 steps, noise added at each")
ax_b.set_title(r"DDIM, $\eta = 0$: 50 deterministic steps")
for ax in (ax_a, ax_b):
    ax.invert_xaxis()
    ax.set_xlabel("timestep t (sampling runs left to right, from T to 0)")
    ax.axhline(1, color="gray", lw=0.5, ls=":")
    ax.axhline(-1, color="gray", lw=0.5, ls=":")
ax_a.set_ylabel("$x_t$")
fig.tight_layout()
fig.savefig("pm1_sampling.png", dpi=110)
plt.close(fig)

print("final DDPM values", np.round(np.sort(ddpm[-1]), 3))
print("final DDIM values", np.round(np.sort(ddim[-1]), 3))
