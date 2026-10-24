"""Render the README figures for the harness assignment.

fd_error.png. Our GELU-at-1 example: the relative error of the central difference
(f(x + h) - f(x - h)) / (2h) as an estimate of GELU'(1), against the step h, computed in
float32 and float64, with the model e(h) = h^2/6 + u/h drawn for each format.

overfit_signatures.png. Overfit-one-batch on the noiseless linear-regression batch
(n = 64, d = 8, Adam at learning rate 0.1), run with a correct optimization step and with
three faults: no gradient reset, no optimizer step, and a ReLU on the model output.

Neither figure uses the assignment's hole files. Run from this directory:

    python make_figures.py
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanovision.data.toy import linreg_batch  # noqa: E402


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Exact derivative of GELU at x = 1: Phi(1) + phi(1).
exact = 0.5 * (1 + math.erf(1 / math.sqrt(2))) + math.exp(-0.5) / math.sqrt(2 * math.pi)

hs = np.logspace(-9, -0.5, 300)
fig, ax = plt.subplots(figsize=(6.5, 5.0))
for dtype, u, color, label in [
    (torch.float32, 2.0**-24, "tab:orange", "float32"),
    (torch.float64, 2.0**-53, "tab:blue", "float64"),
]:
    x = torch.tensor(1.0, dtype=dtype)
    err = []
    for h in hs:
        h_t = torch.tensor(h, dtype=dtype)
        est = (gelu(x + h_t) - gelu(x - h_t)) / (2 * h_t)
        err.append(max(abs(est.item() - exact) / exact, 1e-17))
    ax.loglog(hs, err, ".", ms=3, color=color, label=f"{label}, measured")
    ax.loglog(hs, hs**2 / 6 + u / hs, "-", lw=1, color=color, alpha=0.7,
              label=f"{label}, model $h^2/6 + u/h$")
ax.axvline(1e-6, color="gray", ls="--", lw=1)
ax.text(1.2e-6, 3e-15, "$h = 10^{-6}$\n(the helper's default)", fontsize=8, color="gray")
ax.set_xlabel("step $h$")
ax.set_ylabel("relative error of the estimate of GELU'(1)")
ax.set_ylim(1e-16, 1)
ax.set_title("Central-difference error, truncation (right) against roundoff (left)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
fig.tight_layout()
fig.savefig("fd_error.png", dpi=120)
plt.close(fig)


X, y = linreg_batch(n=64, d=8, noise=0.0, seed=0)


def run(variant, steps=500):
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 1)
    if variant == "relu":
        model = torch.nn.Sequential(torch.nn.Linear(8, 1), torch.nn.ReLU())
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    losses = []
    for _ in range(steps):
        if variant != "no_reset":
            opt.zero_grad()
        loss = F.mse_loss(model(X), y)
        loss.backward()
        if variant != "no_step":
            opt.step()
        losses.append(loss.item())
    return losses


fig, ax = plt.subplots(figsize=(6.5, 4.2))
for variant, color, label in [
    ("ok", "tab:blue", "correct step"),
    ("no_step", "tab:gray", "no optimizer step (flat)"),
    ("no_reset", "tab:red", "no gradient reset (falls, then climbs)"),
    ("relu", "tab:green", "ReLU on the output (stalls)"),
]:
    ax.semilogy(run(variant), color=color, lw=1.3, label=label)
ax.set_xlabel("step")
ax.set_ylabel("mean-squared error on the fixed batch")
ax.set_title("Overfit-one-batch with a correct step and three faults")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("overfit_signatures.png", dpi=120)
plt.close(fig)
print("wrote fd_error.png and overfit_signatures.png")
