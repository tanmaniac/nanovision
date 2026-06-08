"""q_sample produces the right Gaussian: E[x_t|x0] = sqrt(abar)*x0, Var = (1-abar)."""

import torch

from diffusion import q_sample
from schedule import cosine_alpha_bar, gather


def test_forward_moments():
    T = 100
    _, abar = cosine_alpha_bar(T)
    N = 20000
    x0 = torch.full((N, 1, 2, 2), 0.5)
    ti = 60
    t = torch.full((N,), ti, dtype=torch.long)
    eps = torch.randn(N, 1, 2, 2, generator=torch.Generator().manual_seed(0))
    x_t = q_sample(x0, t, eps, abar)

    abar_t = float(abar[ti])
    # Monte Carlo error ~ 1/sqrt(N); size tolerances to it.
    mean_tol = 5.0 * (((1 - abar_t)) / N) ** 0.5
    emp_mean = x_t.mean(dim=0)
    emp_var = x_t.var(dim=0, unbiased=False)
    assert torch.allclose(emp_mean, torch.full_like(emp_mean, abar_t ** 0.5 * 0.5), atol=mean_tol)
    assert torch.allclose(emp_var, torch.full_like(emp_var, 1 - abar_t), atol=0.05)
