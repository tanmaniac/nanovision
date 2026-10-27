"""Gradients. Float64 gradcheck where the function keeps the input dtype (symlog, symexp, the
actor loss); for the functions that compute in float32 by design (two-hot, critic loss), the
autograd gradient is compared with its closed form instead."""

import torch
from torch import distributions as td

from actor_critic import actor_loss, critic_loss
from nets import symexp, symexp_bins, symlog, twohot_decode, twohot_encode


def _away_from_zero(n):
    x = torch.rand(n, dtype=torch.float64) * 3 + 0.1
    return (x * torch.where(torch.rand(n) < 0.5, -1.0, 1.0).double()).requires_grad_()


def test_symlog_symexp_gradcheck():
    torch.manual_seed(0)
    assert torch.autograd.gradcheck(symlog, (_away_from_zero(8),))
    assert torch.autograd.gradcheck(symexp, (_away_from_zero(8),))


def test_actor_loss_gradcheck():
    torch.manual_seed(0)
    mean = torch.randn(3, 4, 1, dtype=torch.float64, requires_grad=True)
    log_std = (0.3 * torch.randn(3, 4, 1, dtype=torch.float64)).requires_grad_()
    actions = torch.randn(3, 4, 1, dtype=torch.float64)
    adv = torch.randn(3, 3, 1, dtype=torch.float64)
    w = torch.rand(3, 3, 1, dtype=torch.float64)

    def f(m, s):
        return actor_loss(td.Independent(td.Normal(m, s.exp()), 1), actions, adv, w, 0.1)

    assert torch.autograd.gradcheck(f, (mean, log_std))


def test_twohot_decode_gradient_closed_form():
    # d E[b] / d logit_i = p_i * (b_i - E[b]).
    bins = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    logits = torch.randn(4, 7, requires_grad=True)
    twohot_decode(logits, bins).sum().backward()
    p = torch.softmax(logits.detach(), -1)
    expect = p * (bins - (p * bins).sum(-1, keepdim=True))
    torch.testing.assert_close(logits.grad, expect, atol=1e-6, rtol=1e-5)


def test_critic_loss_gradient_closed_form():
    # Cross-entropy against a label: d/d logits = softmax - label, per term, times the weight / count.
    bins = symexp_bins(31)
    N, T = 2, 5
    logits = torch.randn(N, T, 31, requires_grad=True)
    tgt, slow = torch.randn(N, T, 1) * 20, torch.randn(N, T, 1) * 20
    w = torch.rand(N, T - 1, 1)
    critic_loss(logits, tgt, slow, w, bins).backward()
    p = torch.softmax(logits.detach(), -1)
    g = 2 * p - twohot_encode(tgt.squeeze(-1), bins) - twohot_encode(slow.squeeze(-1), bins)
    expect = torch.zeros_like(g)
    expect[:, :-1] = g[:, :-1] * w / (N * (T - 1))
    torch.testing.assert_close(logits.grad, expect, atol=1e-6, rtol=1e-4)
