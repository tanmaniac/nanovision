"""Double-precision gradcheck on a single RSSM step, the KL loss, and the lambda-returns.

All use the greedy categorical path so the graph is deterministic; multinomial sampling would make
the forward draw a different one-hot on the perturbed evaluation and break gradcheck.
"""

import torch

from actor_critic import compute_lambda_returns
from config import GradcheckConfig
from rssm import RSSMCell
from world_model import _categorical_kl


def test_forward_h_then_prior_gradcheck():
    cfg = GradcheckConfig()
    cell = RSSMCell(cfg).double()
    B = 2
    h = torch.randn(B, cfg.h_dim, dtype=torch.float64, requires_grad=True)
    z = torch.randn(B, cfg.n_cat * cfg.n_cls, dtype=torch.float64, requires_grad=True)
    a = torch.randint(0, cfg.action_dim, (B,))

    def f(h_in, z_in):
        h2 = cell.forward_h(h_in, z_in, a)
        _, z_out, _ = cell.prior(h2, greedy=True)
        return (h2.sum() + z_out.sum())

    assert torch.autograd.gradcheck(f, (h, z), eps=1e-6, atol=1e-4)


def test_categorical_kl_gradcheck():
    # The KL building block (summed over heads) must be differentiable in both arguments. The full
    # kl_loss adds stop-gradients (q.detach() / p.detach()), which finite-difference gradcheck
    # cannot validate (it ignores .detach()), so the differentiability check lives on the raw KL.
    cfg = GradcheckConfig()
    torch.manual_seed(0)
    B = 3
    q = torch.randn(B, cfg.n_cat, cfg.n_cls, dtype=torch.float64, requires_grad=True)
    p = torch.randn(B, cfg.n_cat, cfg.n_cls, dtype=torch.float64, requires_grad=True)

    def f(a, b):
        return _categorical_kl(a, b).sum()

    assert torch.autograd.gradcheck(f, (q, p), eps=1e-6, atol=1e-4)


def test_lambda_returns_gradcheck():
    torch.manual_seed(1)
    B, H = 2, 4
    rewards = torch.randn(B, H, dtype=torch.float64, requires_grad=True)
    values = torch.randn(B, H + 1, dtype=torch.float64, requires_grad=True)
    conts = torch.rand(B, H, dtype=torch.float64)  # in (0, 1), constant w.r.t. grad inputs

    def f(r, v):
        return compute_lambda_returns(r, v, conts, 0.99, 0.9).sum()

    assert torch.autograd.gradcheck(f, (rewards, values), eps=1e-6, atol=1e-4)
