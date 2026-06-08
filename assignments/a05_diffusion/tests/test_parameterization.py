"""The three parameterizations are algebraically equivalent; the score is -eps/sqrt(1-abar)."""

import torch

from diffusion import q_sample, score_from_eps, to_x0_eps, v_target
from schedule import cosine_alpha_bar, gather


def test_conversions_recover_x0_eps():
    T = 50
    _, abar = cosine_alpha_bar(T)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 1, 8, 8, generator=g)
    eps = torch.randn(4, 1, 8, 8, generator=g)
    t = torch.tensor([5, 17, 33, 48])
    abar_t = gather(abar, t)
    x_t = q_sample(x0, t, eps, abar)

    preds = {"eps": eps, "x0": x0, "v": v_target(x0, eps, abar_t)}
    for kind, pred in preds.items():
        x0_hat, eps_hat = to_x0_eps(pred, x_t, abar_t, kind)
        assert torch.allclose(x0_hat, x0, atol=1e-4), kind
        assert torch.allclose(eps_hat, eps, atol=1e-4), kind


def test_score_from_eps():
    T = 50
    _, abar = cosine_alpha_bar(T)
    eps = torch.randn(2, 1, 4, 4, generator=torch.Generator().manual_seed(1))
    abar_t = gather(abar, torch.tensor([10, 40]))
    assert torch.allclose(score_from_eps(eps, abar_t), -eps / (1 - abar_t).sqrt(), atol=1e-6)
