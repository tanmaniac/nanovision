"""DreamerV3 KL: free-bits clamp on the summed KL, and the 0.5 / 0.1 prior-vs-posterior weighting.

The free-bits max is applied to the KL summed over the n_cat heads (1 nat for the whole summed
term, NOT n_cat nats). The two terms are weighted 0.5 (dynamics, gradient onto the prior) and 0.1
(representation, gradient onto the posterior), so the prior moves ~5x more than the posterior.
"""

import torch

from world_model import kl_loss

N_CAT, N_CLS = 4, 4
Z = N_CAT * N_CLS


def _equal_logits():
    # Posterior == prior => KL = 0 exactly, well below any positive free-bits floor.
    return torch.zeros(2, Z, requires_grad=True), torch.zeros(2, Z, requires_grad=True)


def test_free_bits_clamp_value_and_zero_grad_below_floor():
    post, prior = _equal_logits()
    total, parts = kl_loss(post, prior, 1.0, 0.5, 0.1, N_CAT, N_CLS)
    # KL is 0 < 1 nat, so each clipped term equals the floor of 1 nat (NOT n_cat nats).
    assert torch.allclose(parts["dyn"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(parts["rep"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(total, torch.tensor(0.5 * 1.0 + 0.1 * 1.0), atol=1e-6)
    # Below the floor the max() clips the KL, so no gradient flows to the logits.
    total.backward()
    assert post.grad is None or torch.allclose(post.grad, torch.zeros_like(post.grad), atol=1e-7)
    assert prior.grad is None or torch.allclose(prior.grad, torch.zeros_like(prior.grad), atol=1e-7)


def test_summed_free_bits_not_per_head():
    # With equal logits the per-head KL is 0, so the summed KL is 0 and the clip is exactly the
    # single floor (1.0), proving the free-bits is on the summed scalar, not n_cat * floor.
    post, prior = _equal_logits()
    _, parts = kl_loss(post, prior, 1.0, 0.5, 0.1, N_CAT, N_CLS)
    assert torch.allclose(parts["dyn"], torch.tensor(1.0), atol=1e-6)
    assert not torch.allclose(parts["dyn"], torch.tensor(float(N_CAT)), atol=1e-6)


def test_above_floor_has_nonzero_grad():
    torch.manual_seed(0)
    # Large spread => summed KL well above 1 nat, so the floor is inactive and gradient flows.
    post = (torch.randn(2, Z) * 3).requires_grad_(True)
    prior = (torch.randn(2, Z) * 3).requires_grad_(True)
    total, parts = kl_loss(post, prior, 1.0, 0.5, 0.1, N_CAT, N_CLS)
    assert parts["dyn"] > 1.0 and parts["rep"] > 1.0
    total.backward()
    assert post.grad.abs().sum() > 0
    assert prior.grad.abs().sum() > 0


def test_term_weights_are_0p5_and_0p1():
    # The implementation-independent invariant: the prior gradient is EXACTLY beta_dyn times the
    # gradient of the raw clipped dynamics term, and the posterior gradient is EXACTLY beta_rep
    # times the gradient of the raw clipped representation term. The raw KL-direction gradients
    # differ in magnitude, so the total prior/posterior ratio is NOT a clean 5; the exact 5:1 lives
    # in the weights, which this isolates. Scaling beta_dyn by 2 must scale the prior gradient by 2
    # and leave the posterior gradient untouched, and vice versa.
    torch.manual_seed(0)

    def grads(beta_dyn, beta_rep):
        post = (torch.randn(8, Z, generator=g) * 2).requires_grad_(True)
        prior = (torch.randn(8, Z, generator=g2) * 2).requires_grad_(True)
        total, parts = kl_loss(post, prior, 1.0, beta_dyn, beta_rep, N_CAT, N_CLS)
        assert parts["dyn"] > 1.0 and parts["rep"] > 1.0, "input must put both terms above the floor"
        gp, gq = torch.autograd.grad(total, (prior, post))
        return gp, gq

    g = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(11)
    gp_base, gq_base = grads(0.5, 0.1)
    g = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(11)
    gp_dyn2, gq_dyn2 = grads(1.0, 0.1)   # double beta_dyn only
    g = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(11)
    gp_rep2, gq_rep2 = grads(0.5, 0.2)   # double beta_rep only

    # Doubling beta_dyn doubles the prior gradient, leaves the posterior gradient unchanged.
    assert torch.allclose(gp_dyn2, 2.0 * gp_base, atol=1e-5)
    assert torch.allclose(gq_dyn2, gq_base, atol=1e-5)
    # Doubling beta_rep doubles the posterior gradient, leaves the prior gradient unchanged.
    assert torch.allclose(gq_rep2, 2.0 * gq_base, atol=1e-5)
    assert torch.allclose(gp_rep2, gp_base, atol=1e-5)
