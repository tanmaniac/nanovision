"""Overfit one fixed batch with the flow-matching loss (bounded training, robust).

The PRIMARY assertion is the loss DROP: flow_loss falls to a small fraction of its untrained value
on one fixed batch within a bounded step budget. flow_loss does NOT reach zero by construction. The
target velocity is a_chunk - z0, and near t=1 the interpolant z_t collapses onto a_chunk, so the
network cannot recover z0 from its input there; that ill-conditioning leaves an irreducible
residual (the same residual the linear-path CFM loss carries in the flow-matching assignment). The
floor measured on the solution here is ~0.18 from a start of ~2.32 (ratio ~0.08); the test asserts
the ratio, not a near-zero floor.

A SECONDARY, loose check is that the few-step ODE sampler reconstructs the batch's action chunk to
within a generous tolerance (measured MAE ~0.14). The loss drop carries the test; the sampler
reconstruction is the only learned-sampler result in pytest, so its tolerance stays loose.
"""

import torch

from config import VLAConfig
from flow import FlowHead, flow_loss, flow_sample


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = VLAConfig()
    g = torch.Generator().manual_seed(0)
    B = 32
    # A fixed batch: distinct conditioning rows mapped to distinct unit-scale action chunks.
    c = torch.randn(B, 6, generator=g)
    a_chunk = torch.randn(B, cfg.chunk, cfg.act_dim, generator=g)

    head = FlowHead(cfg, cond_in=6)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = None
    for _ in range(1000):
        opt.zero_grad()
        loss = flow_loss(head, a_chunk, c, generator=g)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    # Primary: the velocity-regression loss drops to a small fraction of its untrained value.
    # Measured ratio ~0.08 (final ~0.18 from start ~2.32); 0.20 leaves margin for seed jitter.
    assert final < 0.20 * first, f"final flow_loss {final} (start {first}, ratio {final / first:.3f})"

    # Secondary, loose: the few-step ODE sampler lands near the demonstrated chunk on average.
    # Measured MAE ~0.14 against a unit-scale chunk; 0.30 is a loose bound, not a tight assertion.
    head.eval()
    with torch.no_grad():
        samp = flow_sample(head, c, cfg.chunk, cfg.n_flow_steps,
                           generator=torch.Generator().manual_seed(1))
    recon = (samp - a_chunk).abs().mean().item()
    assert recon < 0.30, f"flow_sample reconstruction MAE {recon}"
