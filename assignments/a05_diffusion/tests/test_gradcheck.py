"""Float64 gradcheck on the closed-form diffusion functions."""

import torch

from diffusion import q_sample, to_x0_eps, v_target
from schedule import cosine_alpha_bar


def test_q_sample_gradcheck():
    T = 20
    _, abar = cosine_alpha_bar(T)
    abar = abar.double()
    x0 = torch.randn(2, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    eps = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    t = torch.tensor([3, 11])
    assert torch.autograd.gradcheck(lambda x: q_sample(x, t, eps, abar), (x0,))


def test_v_target_gradcheck():
    abar_t = torch.rand(2, 1, 1, 1, dtype=torch.float64) * 0.9 + 0.05
    x0 = torch.randn(2, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    eps = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: v_target(x, eps, abar_t), (x0,))


def test_to_x0_eps_gradcheck():
    abar_t = torch.rand(2, 1, 1, 1, dtype=torch.float64) * 0.9 + 0.05
    x_t = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    pred = torch.randn(2, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    for kind in ("eps", "x0", "v"):
        assert torch.autograd.gradcheck(lambda p: to_x0_eps(p, x_t, abar_t, kind), (pred,)), kind
