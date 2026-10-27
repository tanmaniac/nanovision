"""lambda_return against a plain-Python backward recursion, and its two limits (lambda 0 and 1)."""

import torch

from actor_critic import lambda_return


def _hand(last, term, reward, boot, disc, lamb):
    N, T, _ = reward.shape
    out = torch.zeros(N, T - 1, 1, dtype=torch.float64)
    for n in range(N):
        R = float(boot[n, T - 1])
        for t in reversed(range(T - 1)):
            g = disc * (1 - float(term[n, t + 1]))
            lam = lamb * (1 - float(last[n, t + 1]))
            R = float(reward[n, t + 1]) + g * ((1 - lam) * float(boot[n, t + 1]) + lam * R)
            out[n, t, 0] = R
    return out


def _seq(N=4, T=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    reward = torch.randn(N, T, 1, generator=g, dtype=torch.float64)
    boot = torch.randn(N, T, 1, generator=g, dtype=torch.float64)
    term = (torch.rand(N, T, 1, generator=g) < 0.15).double()
    last = (torch.rand(N, T, 1, generator=g) < 0.15).double()
    return last, term, reward, boot


def test_matches_hand_recursion():
    last, term, reward, boot = _seq()
    out = lambda_return(last, term, reward, boot, 0.97, 0.9)
    assert out.shape == (4, 8, 1)
    torch.testing.assert_close(out, _hand(last, term, reward, boot, 0.97, 0.9))


def test_soft_termination():
    # Imagination passes term = 1 - P(continue), a probability rather than a flag.
    last, _, reward, boot = _seq(seed=1)
    term = torch.rand(4, 9, 1, dtype=torch.float64)
    torch.testing.assert_close(lambda_return(torch.zeros_like(last), term, reward, boot, 0.997, 0.95),
                               _hand(torch.zeros_like(last), term, reward, boot, 0.997, 0.95))


def test_lambda_zero_is_one_step_target():
    last, term, reward, boot = _seq(seed=2)
    out = lambda_return(last, term, reward, boot, 0.9, 0.0)
    torch.testing.assert_close(out, reward[:, 1:] + 0.9 * (1 - term[:, 1:]) * boot[:, 1:])


def test_lambda_one_is_discounted_sum_plus_bootstrap():
    _, _, reward, boot = _seq(seed=3)
    z = torch.zeros_like(reward)
    out = lambda_return(z, z, reward, boot, 0.9, 1.0)
    T = reward.shape[1]
    for t in range(T - 1):
        k = torch.arange(T - 1 - t, dtype=torch.float64)
        expect = (0.9 ** k * reward[:, t + 1:, 0]).sum(-1) + 0.9 ** (T - 1 - t) * boot[:, -1, 0]
        torch.testing.assert_close(out[:, t, 0], expect, rtol=1e-6, atol=1e-6)


def test_no_gradient():
    last, term, reward, boot = _seq()
    boot.requires_grad_()
    assert not lambda_return(last, term, reward, boot, 0.9, 0.9).requires_grad
