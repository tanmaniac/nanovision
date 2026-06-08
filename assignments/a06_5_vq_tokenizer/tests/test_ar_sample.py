"""Autoregressive sampling returns a valid, deterministic token grid."""

import torch

from config import VQConfig
from prior import TokenPrior, ar_sample


def test_ar_sample_valid_and_deterministic():
    torch.manual_seed(0)
    cfg = VQConfig()
    prior = TokenPrior(cfg).eval()

    grid = ar_sample(prior, 4, (cfg.grid, cfg.grid), cfg.num_codes,
                     generator=torch.Generator().manual_seed(7))
    assert grid.shape == (4, cfg.grid, cfg.grid)
    assert grid.min() >= 0 and grid.max() < cfg.num_codes      # real codes only, no BOS

    grid2 = ar_sample(prior, 4, (cfg.grid, cfg.grid), cfg.num_codes,
                      generator=torch.Generator().manual_seed(7))
    assert torch.equal(grid, grid2)                            # deterministic under the seed
