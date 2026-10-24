"""The autoregressive prior memorizes one fixed batch of token grids."""

import math

import torch

from config import VQConfig
from prior import TokenPrior, ar_nll


def test_prior_overfit():
    torch.manual_seed(0)
    cfg = VQConfig()
    g = torch.Generator().manual_seed(0)
    indices = torch.randint(0, cfg.num_codes, (8, cfg.grid, cfg.grid), generator=g)

    prior = TokenPrior(cfg)
    opt = torch.optim.Adam(prior.parameters(), lr=3e-3)
    first = None
    for _ in range(1000):
        opt.zero_grad()
        loss = ar_nll(prior, indices)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    # Starts near ln(K) (uniform over K codes) and drops ~25x. It floors near ln(B)/L
    # (here ln(8)/16 ~= 0.13), not 0: a model that memorizes B distinct grids can at best give
    # each probability 1/B, so the summed cross-entropy over the L positions is at least ln(B).
    # The cost is paid where grids first differ (position 0 for most; two of these 8 grids share
    # their first token, so part of it lands at position 1), but the total is ln(B) either way.
    assert first > 0.5 * math.log(cfg.num_codes), f"start {first} not near ln(K)={math.log(cfg.num_codes):.2f}"
    assert final < 0.2, f"final {final}"
    assert final < 0.1 * first
