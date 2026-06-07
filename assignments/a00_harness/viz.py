"""A0 viz: overfit a noiseless linreg batch and render the loss curve to out/.

Run with: make viz A=a00_harness  (uses the reference solution).
"""

from pathlib import Path

import torch

from nanovision.data.toy import linreg_batch
from nanovision.determinism import set_seed
from nanovision.trainer import Trainer
from nanovision.viz import plot_loss_curve


def main():
    set_seed(0)
    out = Path(__file__).parent / "out"
    X, y = linreg_batch(n=64, d=8, noise=0.0, seed=0)
    model = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    tr = Trainer(model, opt, torch.nn.functional.mse_loss, device="cpu")
    losses = tr.overfit_one_batch((X, y), steps=500)
    path = plot_loss_curve(losses, out / "loss_curve.png", title="A0 overfit linreg (MSE)")
    print(f"final loss {losses[-1]:.3e} — wrote {path}")


if __name__ == "__main__":
    main()
