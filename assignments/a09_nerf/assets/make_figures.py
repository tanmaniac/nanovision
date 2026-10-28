"""Render the README figures for the NeRF notes.

toy_views.png. Our toy sphere (radius 1, constant density 8, striped albedo) seen from four of
the cameras on the ring, rendered by the closed-form ray-sphere chord in nanovision.data.toy.

sphere_ray.png. Our sphere ray. Left: the camera at (4, 0, 0) looking at the sphere, seen from
above, with the central ray and the ray through pixel u = 15.5 of the 16x16 capture (focal
length 32). Middle and right: along each ray, the density sigma(t), the transmittance T(t) in
closed form, and the compositing weights w_i = T_i alpha_i of the 32 evenly spaced samples the
training loop uses between near = 2.5 and far = 5.5.

spectral_bias.png. A 1D version of the spectral-bias experiment. Left: the empirical neural
tangent kernel k(0, p) = grad f(0) . grad f(p) of a freshly initialized ReLU MLP (4 layers of
width 128, as in the provided NeRF MLP), fed either the raw coordinate p or its Fourier encoding
with L = 6 bands, averaged over 16 initializations, with the constant part removed and the peak
scaled to 1. Right: the two networks after 2000 Adam steps fitting a 1D signal with a slow and a
fast sinusoid on [-4, 4] (the range allowed by the toy's scene bound of 4), with positions divided by 4 before
encoding as in the NeRF MLP. Runs in a few minutes on CPU.

Run from this directory:

    python make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanovision.data import toy  # noqa: E402

# ---------------------------------------------------------------- toy_views.png
images, poses, K, near, far = toy.nerf_synthetic_scene(
    n_views=8, H=64, W=64, texture_freq=12.0, focal_mult=2.0
)
fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
for ax, v in zip(axes, (0, 2, 4, 6)):
    ax.imshow(images[v].numpy())
    ang = 360 * v / 8
    ax.set_title(f"camera at {ang:.0f} deg", fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig("toy_views.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- sphere_ray.png
sigma_in, radius, cam = 8.0, 1.0, np.array([4.0, 0.0, 0.0])
near, far, n = 2.5, 5.5, 32
z = near + (far - near) * np.linspace(0, 1, n)
delta = np.append(np.diff(z), 1e10)


def ray_dir(u, f=32.0, cx=7.5):
    # camera 0 of the ring: right = +z_world, down = +y_world, forward = -x_world
    d = np.array([-1.0, 0.0, (u - cx) / f])
    return d / np.linalg.norm(d)


def chord(d):
    b = 2 * (d @ cam)
    c = cam @ cam - radius**2
    disc = b * b - 4 * c
    if disc <= 0:
        return None
    s = np.sqrt(disc)
    return (-b - s) / 2, (-b + s) / 2


rays = [("central ray (u = 7.5)", ray_dir(7.5)), ("grazing ray (u = 15.5)", ray_dir(15.5))]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
th = np.linspace(0, 2 * np.pi, 200)
ax.fill(np.cos(th), np.sin(th), color="0.85")
ax.plot(np.cos(th), np.sin(th), color="0.4", lw=1)
ax.plot(cam[0], cam[2], "ks")
ax.annotate("camera", (cam[0], cam[2]), (3.3, 0.45))
for (name, d), col in zip(rays, ("tab:blue", "tab:orange")):
    p0, p1 = cam + near * d, cam + far * d
    ax.plot([cam[0], p1[0]], [cam[2], p1[2]], color=col, lw=1, ls=":")
    ax.plot([p0[0], p1[0]], [p0[2], p1[2]], color=col, lw=2, label=name)
ax.set_aspect("equal")
ax.set_xlabel("world x")
ax.set_ylabel("world z")
ax.set_title("our sphere ray, seen from above")
ax.legend(fontsize=8, loc="lower left")

tt = np.linspace(near, far, 2000)
for ax, (name, d), col in zip(axes[1:], rays, ("tab:blue", "tab:orange")):
    t0, t1 = chord(d)
    sig = np.where((tt > t0) & (tt < t1), sigma_in, 0.0)
    T = np.exp(-sigma_in * np.clip(np.minimum(tt, t1) - t0, 0, None))
    sig_s = np.where((z > t0) & (z < t1), sigma_in, 0.0)
    a = 1 - np.exp(-sig_s * delta)
    Ts = np.concatenate([[1.0], np.cumprod(1 - a)[:-1]])
    w = Ts * a
    ax.plot(tt, sig / sigma_in, color="0.5", lw=1, label=r"density $\sigma(t)/8$")
    ax.plot(tt, T, color="k", lw=1.5, label=r"transmittance $T(t)$")
    ax.vlines(z, 0, w, color=col, lw=2)
    ax.plot(z, w, "o", color=col, ms=3, label=r"weights $w_i$ (32 samples)")
    ax.set_xlabel("distance along the ray t")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(f"{name}: chord {t1 - t0:.3f}, sum of w = {w.sum():.3f}", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig("sphere_ray.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- spectral_bias.png
STEPS = 2000
N_INITS = 16
torch.manual_seed(0)


def encode(p, L):
    if L == 0:
        return p
    bands = (2.0 ** torch.arange(L)) * torch.pi
    s = p * bands
    return torch.cat([p, torch.sin(s), torch.cos(s)], dim=-1)


def mlp(in_dim):
    layers, d = [], in_dim
    for _ in range(4):
        layers += [torch.nn.Linear(d, 128), torch.nn.ReLU()]
        d = 128
    layers.append(torch.nn.Linear(128, 1))
    return torch.nn.Sequential(*layers)


def ntk_row(net, L, p0, ps):
    def grad(p):
        net.zero_grad()
        net(encode(torch.tensor([[p]]), L)).sum().backward()
        return torch.cat([q.grad.flatten() for q in net.parameters()])

    g0 = grad(p0)
    return np.array([float(g0 @ grad(float(p))) for p in ps])


def target(x):  # x in world units on [-4, 4], the range the scene bound of 4 allows
    return 0.5 + 0.25 * np.sin(0.75 * x) + 0.2 * np.sin(12.0 * x)


ps = np.linspace(-1, 1, 201)
x_train = np.linspace(-4, 4, 512)
x_fine = np.linspace(-4, 4, 2000)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
fits = {}
for L, col, name in ((0, "tab:red", "raw coordinate"), (6, "tab:green", "Fourier encoding, L = 6")):
    # Average the empirical kernel over several random initializations; a single finite-width
    # network gives a noisy estimate of the expected kernel.
    k = 0.0
    for seed in range(N_INITS):
        torch.manual_seed(100 + seed)
        k = k + ntk_row(mlp(1 if L == 0 else 1 + 2 * L), L, 0.0, ps) / N_INITS
    torch.manual_seed(0)
    net = mlp(1 if L == 0 else 1 + 2 * L)
    # Subtract the constant part of the kernel (it moves every prediction by the same amount,
    # the zero-frequency component) and normalize the peak to 1.
    kc = k - k.mean()
    axes[0].plot(ps, kc / kc[100], color=col, label=name)
    p_tr = torch.tensor(x_train / 4.0, dtype=torch.float32)[:, None]  # divide by scene_bound
    y_tr = torch.tensor(target(x_train), dtype=torch.float32)[:, None]
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    for _ in range(STEPS):
        loss = ((net(encode(p_tr, L)) - y_tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        yf = net(encode(torch.tensor(x_fine / 4.0, dtype=torch.float32)[:, None], L)).numpy()[:, 0]
    fits[name] = float(((yf - target(x_fine)) ** 2).mean())
    axes[1].plot(x_fine, yf, color=col, lw=1.5, label=f"{name} (MSE {fits[name]:.1e})")
axes[1].plot(x_fine, target(x_fine), color="k", lw=0.8, ls="--", label="target")
axes[0].set_xlabel(r"input $p$ (normalized coordinate)")
axes[0].set_ylabel(r"$k(0, p)$, constant removed, peak = 1")
axes[0].set_title(f"NTK at initialization (mean of {N_INITS} inits): spread of a correction at p = 0", fontsize=10)
axes[0].axhline(0, color="0.7", lw=0.5)
axes[0].legend(fontsize=8)
axes[1].set_xlabel("world coordinate x")
axes[1].set_title(f"fit after {STEPS} Adam steps (512 training points)", fontsize=10)
axes[1].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("spectral_bias.png", dpi=120)
plt.close(fig)
print("fit MSE", fits)
