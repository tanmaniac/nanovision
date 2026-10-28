"""KL with free bits: the value is the summed categorical KL of the unimix distributions, the clip
floors it at 1 nat with zero gradient, and the dyn / rep terms send gradient only to the prior /
posterior respectively."""

import torch
from torch.distributions import Categorical, kl_divergence

from config import Config
from rssm import kl_loss


def _mixed(logits, r=0.01):
    return 0.99 * torch.softmax(logits, -1) + r / logits.shape[-1]


def test_kl_value_and_shape():
    torch.manual_seed(0)
    post, prior = 3 * torch.randn(2, 5, 4, 6), 3 * torch.randn(2, 5, 4, 6)
    dyn, rep = kl_loss(post, prior, 0.0, 0.01)
    assert dyn.shape == rep.shape == (2, 5)
    expect = kl_divergence(Categorical(probs=_mixed(post)), Categorical(probs=_mixed(prior))).sum(-1)
    torch.testing.assert_close(dyn, expect, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rep, expect, atol=1e-5, rtol=1e-5)


def test_free_bits_floor_has_zero_gradient():
    post = torch.randn(2, 3, 4, 6, requires_grad=True)
    prior = (post.detach() + 0.01 * torch.randn(2, 3, 4, 6)).requires_grad_()
    dyn, rep = kl_loss(post, prior, 1.0, 0.01)
    assert torch.all(dyn == 1.0) and torch.all(rep == 1.0)
    (dyn.sum() + rep.sum()).backward()
    assert torch.all(post.grad == 0) and torch.all(prior.grad == 0)


def _no_grad(t):
    return t.grad is None or bool(torch.all(t.grad == 0))


def test_dyn_trains_prior_rep_trains_posterior():
    torch.manual_seed(1)
    post = (3 * torch.randn(2, 3, 4, 6)).requires_grad_()
    prior = (3 * torch.randn(2, 3, 4, 6)).requires_grad_()
    dyn, rep = kl_loss(post, prior, 1.0, 0.01)
    assert torch.all(dyn > 1.0)
    dyn.sum().backward()
    assert _no_grad(post) and prior.grad.abs().sum() > 0
    post.grad = prior.grad = None
    dyn, rep = kl_loss(post, prior, 1.0, 0.01)
    rep.sum().backward()
    assert _no_grad(prior) and post.grad.abs().sum() > 0


def test_kl_config_values():
    cfg = Config()
    assert cfg.kl_free == 1.0
    assert cfg.loss_scales["dyn"] == 1.0 and cfg.loss_scales["rep"] == 0.1
