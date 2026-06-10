"""The straight-through estimator: hard one-hot forward, blended-softmax gradient backward."""

import torch

from config import WorldModelConfig
from nets import categorical_sample


def test_forward_sample_is_one_hot_per_categorical():
    cfg = WorldModelConfig()
    torch.manual_seed(0)
    logits = torch.randn(5, cfg.n_cat * cfg.n_cls)
    z, probs = categorical_sample(logits, cfg.unimix, cfg.n_cat, cfg.n_cls, greedy=True)
    assert z.shape == (5, cfg.n_cat * cfg.n_cls)
    assert probs.shape == (5, cfg.n_cat, cfg.n_cls)
    zc = z.view(5, cfg.n_cat, cfg.n_cls)
    # Each categorical row sums to 1 (the one-hot value, straight-through preserves the sum).
    assert torch.allclose(zc.sum(-1), torch.ones(5, cfg.n_cat), atol=1e-5)
    # The greedy one-hot argmax matches the blended-prob argmax.
    assert (zc.argmax(-1) == probs.argmax(-1)).all()


def test_unimix_floor():
    cfg = WorldModelConfig()
    torch.manual_seed(1)
    logits = torch.randn(4, cfg.n_cat * cfg.n_cls) * 10.0  # large logits would saturate softmax
    _, probs = categorical_sample(logits, cfg.unimix, cfg.n_cat, cfg.n_cls, greedy=True)
    floor = cfg.unimix / cfg.n_cls
    assert (probs >= floor - 1e-6).all(), probs.min()


def test_straight_through_gradient_equals_blended_prob_gradient():
    # The straight-through output z carries the gradient of the unimix-BLENDED probs, not raw
    # softmax. So d(z.sum())/d(logits) must equal d(probs.sum())/d(logits) where probs is the blend.
    cfg = WorldModelConfig()
    torch.manual_seed(2)
    logits = torch.randn(3, cfg.n_cat * cfg.n_cls, dtype=torch.float64, requires_grad=True)

    z, _ = categorical_sample(logits, cfg.unimix, cfg.n_cat, cfg.n_cls, greedy=True)
    z.sum().backward()
    g_st = logits.grad.clone()

    logits2 = logits.detach().clone().requires_grad_(True)
    lg = logits2.view(3, cfg.n_cat, cfg.n_cls)
    soft = torch.softmax(lg, dim=-1)
    probs = (1.0 - cfg.unimix) * soft + cfg.unimix / cfg.n_cls
    probs.sum().backward()
    g_probs = logits2.grad.clone()

    assert torch.allclose(g_st, g_probs, atol=1e-9), (g_st - g_probs).abs().max()
