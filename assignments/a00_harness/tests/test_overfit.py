"""Task 4: Trainer overfits a single linear-regression batch to ~0. Run last."""

import torch

from nanovision.data.toy import linreg_batch
from nanovision.determinism import set_seed
from nanovision.trainer import Trainer


def test_overfit_linreg():
    set_seed(0)
    X, y = linreg_batch(n=64, d=8, noise=0.0, seed=0)
    model = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    tr = Trainer(model, opt, torch.nn.functional.mse_loss, device="cpu")
    losses = tr.overfit_one_batch((X, y), steps=500)
    assert losses[-1] < 1e-4, f"expected ~0 loss on noiseless linreg, got {losses[-1]}"
    assert losses[-1] < losses[0]
