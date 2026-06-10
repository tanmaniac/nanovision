"""Overfit one fixed batch with the BC loss (bounded training).

BC is plain regression, so this is a quick sanity check that the conditioning reaches the head and
the chunk shapes line up: the loss drops to near zero on a fixed batch.
"""

import torch

from bc import BCPolicy, bc_loss
from config import VLAConfig


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = VLAConfig()
    g = torch.Generator().manual_seed(0)
    B = 32
    c = torch.randn(B, 6, generator=g)
    a_chunk = 0.1 * torch.randn(B, cfg.chunk, cfg.act_dim, generator=g)

    policy = BCPolicy(cfg, cond_in=6)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    first = None
    for _ in range(500):
        opt.zero_grad()
        loss = bc_loss(policy, a_chunk, c)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    assert final < 1e-4, f"final bc_loss {final} (start {first})"
