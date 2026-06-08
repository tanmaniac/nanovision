"""Overfit one fixed batch: the CFM loss drives the velocity MLP to fit the field.

The batch (x0, x1, t) is fixed, with a fresh t per row, so the network regresses distinct
(x_t, u) points rather than memorizing a single input. This checks the loop end to end
(path, target, MLP, loss). It does not assert sampling reconstructs the data.
"""

import torch

from config import FlowConfig
from flow import cfm_loss
from model import VelocityMLP

from nanovision.data import toy


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = FlowConfig()
    g = torch.Generator().manual_seed(0)
    n = 64
    x1 = toy.eight_gaussians(n, generator=g)
    x0 = torch.randn(n, cfg.data_dim, generator=g)
    t = 0.05 + 0.9 * torch.rand(n, generator=g)        # fixed per row

    model = VelocityMLP(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(3000):
        opt.zero_grad()
        loss = cfm_loss(model, x0, x1, t)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    # The loss falls ~1000x from the untrained ~E||x1-x0||^2; a small residual remains
    # where distinct pairs land near the same (x_t, t) with different velocity targets.
    assert final < 0.05, f"final {final} (start {first})"
    assert final < 0.01 * first
