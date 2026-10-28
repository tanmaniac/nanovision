"""The unimix floor on probabilities, one-hot samples with the right frequencies, and the
straight-through gradient."""

import torch

from nets import categorical_sample, unimix_logits


def test_unimix_floor_and_mixture():
    logits = torch.tensor([[100.0, 0.0, -100.0, -100.0], [0.0, 0.0, 0.0, 0.0]])
    p = unimix_logits(logits, 0.01).exp()
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    assert torch.all(p >= 0.01 / 4 - 1e-7)
    expect = 0.99 * torch.softmax(logits, -1) + 0.01 / 4
    torch.testing.assert_close(p, expect)


def test_sample_is_one_hot():
    torch.manual_seed(0)
    s = categorical_sample(torch.randn(3, 32, 16), 0.01)
    assert s.shape == (3, 32, 16)
    assert torch.all((s == 0) | (s == 1))
    assert torch.all(s.sum(-1) == 1)


def test_sample_frequencies_match_mixed_probs():
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 0.0, -1.0, -8.0])
    s = categorical_sample(logits.expand(40000, 4), 0.01)
    freq = s.mean(0)
    expect = unimix_logits(logits, 0.01).exp()
    torch.testing.assert_close(freq, expect, atol=0.01, rtol=0)


def test_straight_through_gradient_equals_probs_gradient():
    torch.manual_seed(0)
    logits = torch.randn(4, 5, 6, requires_grad=True)
    w = torch.randn(4, 5, 6)
    (categorical_sample(logits, 0.01) * w).sum().backward()
    g_st = logits.grad.clone()
    logits.grad = None
    (unimix_logits(logits, 0.01).exp() * w).sum().backward()
    torch.testing.assert_close(g_st, logits.grad, atol=1e-6, rtol=1e-5)
    assert g_st.abs().sum() > 0
