"""Reference-value checks: the losses match their textbook definitions.

The tests may use F.cross_entropy and F.binary_cross_entropy_with_logits as the
reference (the forbidden-imports rule applies to the mechanism code, not the tests).
"""

import torch
import torch.nn.functional as F

from losses import clip_loss, siglip_loss


def test_clip_loss_matches_cross_entropy():
    torch.manual_seed(0)
    fi = F.normalize(torch.randn(5, 8), dim=-1)
    ft = F.normalize(torch.randn(5, 8), dim=-1)
    logit_scale = torch.tensor(2.0)
    logits = logit_scale.exp() * fi @ ft.T
    labels = torch.arange(5)
    ref = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    assert torch.allclose(clip_loss(fi, ft, logit_scale), ref, atol=1e-6)


def test_siglip_loss_matches_bce():
    torch.manual_seed(0)
    fi = F.normalize(torch.randn(5, 8), dim=-1)
    ft = F.normalize(torch.randn(5, 8), dim=-1)
    logit_scale = torch.tensor(2.0)
    bias = torch.tensor(-1.0)
    logits = logit_scale.exp() * fi @ ft.T + bias
    labels = 2.0 * torch.eye(5) - 1.0
    ref = -F.logsigmoid(labels * logits).sum() / 5
    assert torch.allclose(siglip_loss(fi, ft, logit_scale, bias), ref, atol=1e-6)


def test_clip_loss_low_when_aligned():
    # Identical normalized features with a high logit scale: the diagonal dominates each
    # row, so the symmetric CE is small.
    torch.manual_seed(0)
    f = F.normalize(torch.randn(6, 16), dim=-1)
    assert clip_loss(f, f, torch.tensor(4.0)).item() < 0.2
