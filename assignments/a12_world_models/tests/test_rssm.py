"""RSSM steps: shapes, one-hot latents, the episode-start reset, and which inputs each path reads."""

import torch

from ._helpers import tiny_cfg
from rssm import RSSM

E = 10


def _setup(B=3):
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m = RSSM(cfg, E)
    h = torch.randn(B, cfg.deter)
    z = torch.nn.functional.one_hot(torch.randint(0, cfg.discrete, (B, cfg.stoch)), cfg.discrete).float()
    a = torch.rand(B, cfg.action_dim) * 2 - 1
    return cfg, m, h, z, a, torch.randn(B, E)


def _check_onehot(z, B, cfg):
    assert z.shape == (B, cfg.stoch, cfg.discrete)
    assert torch.all(z.sum(-1) == 1) and torch.all((z == 0) | (z == 1))


def test_prior_posterior_shapes():
    cfg, m, h, z, a, e = _setup()
    zp, lp = m.prior(h)
    zq, lq = m.posterior(h, e)
    _check_onehot(zp, 3, cfg)
    _check_onehot(zq, 3, cfg)
    assert lp.shape == lq.shape == (3, cfg.stoch, cfg.discrete)


def test_posterior_reads_embed_prior_does_not():
    cfg, m, h, z, a, e = _setup()
    _, l1 = m.posterior(h, e)
    _, l2 = m.posterior(h, e + 1.0)
    assert not torch.allclose(l1, l2)
    torch.testing.assert_close(m.prior(h)[1], m._logits(m.prior_net(h)))


def test_obs_step_resets_flagged_rows():
    cfg, m, h, z, a, e = _setup()
    first = torch.tensor([True, False, True])
    h1, z1, l1 = m.obs_step(h, z, a, e, first)
    assert h1.shape == (3, cfg.deter) and l1.shape == (3, cfg.stoch, cfg.discrete)
    _check_onehot(z1, 3, cfg)
    h0, z0 = m.initial(3)
    h_zero, _, l_zero = m.obs_step(h0, z0, torch.zeros_like(a), e, torch.zeros(3, dtype=torch.bool))
    # Flagged rows behave as if the previous state and action were zero.
    torch.testing.assert_close(h1[first], h_zero[first])
    torch.testing.assert_close(l1[first], l_zero[first])
    # The unflagged row keeps its history.
    assert not torch.allclose(h1[1], h_zero[1])
    torch.testing.assert_close(h1[~first], m.forward_h(h, z, a)[~first])


def test_img_step_is_gru_then_prior():
    cfg, m, h, z, a, e = _setup()
    h1, z1 = m.img_step(h, z, a)
    torch.testing.assert_close(h1, m.forward_h(h, z, a))
    _check_onehot(z1, 3, cfg)
