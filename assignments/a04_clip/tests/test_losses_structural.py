"""The structural difference between InfoNCE and the sigmoid loss at small batch.

The representation-quality gap (SigLIP > InfoNCE at moderate batch size) is NOT
observable by overfitting a tiny fixed batch and is deliberately not asserted here; the
alignment-vs-N comparison lives in viz.py with an honest caption. What IS deterministic,
and is the point, is the structural difference at N=1: SigLIP is a well-defined per-pair
loss with one pair, while symmetric InfoNCE over a 1x1 similarity has no negatives and
collapses to zero, carrying no learning signal. This is exactly why InfoNCE needs a large
batch (its signal is the in-batch negatives) and the sigmoid loss does not.
"""

import torch
import torch.nn.functional as F

from losses import clip_loss, siglip_loss


def test_siglip_well_defined_at_batch_one():
    f = F.normalize(torch.randn(1, 8), dim=-1)
    loss = siglip_loss(f, f, torch.tensor(2.0), torch.tensor(-1.0))
    assert torch.isfinite(loss) and loss.item() > 0


def test_infonce_degenerate_at_batch_one():
    # A 1x1 similarity has no negatives, so the row log-softmax is 0 and clip_loss is 0:
    # there is no gradient signal at batch size 1.
    f = F.normalize(torch.randn(1, 8), dim=-1)
    loss = clip_loss(f, f, torch.tensor(2.0))
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_siglip_has_signal_where_infonce_has_none():
    # At N=1 the sigmoid loss still produces a nonzero gradient w.r.t. the features; the
    # InfoNCE loss produces none. This is the small-batch difference, made deterministic.
    f = F.normalize(torch.randn(1, 8), dim=-1).requires_grad_(True)
    siglip_loss(f, f, torch.tensor(2.0), torch.tensor(-1.0)).backward()
    g_sig = f.grad.abs().sum().item()
    f2 = F.normalize(torch.randn(1, 8), dim=-1).requires_grad_(True)
    clip_loss(f2, f2, torch.tensor(2.0)).backward()
    g_clip = f2.grad.abs().sum().item()
    assert g_sig > 1e-6 and g_clip < 1e-9
