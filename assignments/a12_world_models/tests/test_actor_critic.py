"""REINFORCE actor loss, advantage normalization by max(1, S), the two-hot critic loss, and the
imagination rollout."""

import torch
from torch import distributions as td
from torch import nn

from ._helpers import tiny_cfg
from actor_critic import Actor, Critic, ReturnEMA, actor_loss, critic_loss, imagination_losses, imagine
from nets import symexp_bins, twohot_encode
from world_model import WorldModel


def _policy(mean, std):
    return td.Independent(td.Normal(mean, std), 1)


def test_actor_loss_raises_logprob_of_positive_advantage():
    torch.manual_seed(0)
    N, T = 6, 5
    mean = torch.zeros(N, T, 1, requires_grad=True)
    actions = torch.randn(N, T, 1)
    adv = torch.randn(N, T - 1, 1)
    loss = actor_loss(_policy(mean, torch.full((N, T, 1), 0.5)), actions, adv, torch.ones(N, T - 1, 1), 0.0)
    assert loss.shape == ()
    loss.backward()
    with torch.no_grad():
        before = _policy(mean, 0.5).log_prob(actions)[:, :-1]
        after = _policy(mean - 0.1 * mean.grad, 0.5).log_prob(actions)[:, :-1]
    up = (after - before) > 0
    assert torch.equal(up, adv.squeeze(-1) > 0)
    # The last imagined state has no advantage, so its action gets no gradient.
    assert torch.all(mean.grad[:, -1] == 0)


def test_actor_loss_advantage_is_detached_and_weighted():
    mean = torch.zeros(2, 4, 1, requires_grad=True)
    adv = torch.ones(2, 3, 1, requires_grad=True)
    w = torch.ones(2, 3, 1)
    w[:, 1] = 0.0
    actor_loss(_policy(mean, torch.ones(2, 4, 1)), torch.ones(2, 4, 1), adv, w, 0.0).backward()
    assert adv.grad is None or torch.all(adv.grad == 0)
    assert torch.all(mean.grad[:, 1] == 0) and torch.all(mean.grad[:, 0] != 0)


def test_entropy_bonus_widens_policy():
    log_std = torch.zeros(3, 4, 1, requires_grad=True)
    pol = _policy(torch.zeros(3, 4, 1), log_std.exp())
    actor_loss(pol, torch.zeros(3, 4, 1), torch.zeros(3, 3, 1), torch.ones(3, 3, 1), 3e-4).backward()
    assert torch.all(log_std.grad[:, :-1] < 0)          # descending the loss raises the std


def _imag_setup():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    wm = WorldModel(cfg)
    feat = wm.rssm.feat_dim
    actor, critic = Actor(cfg, feat), Critic(cfg, feat)
    slow = Critic(cfg, feat)
    slow.load_state_dict(critic.state_dict())
    # The reward and critic heads start at zero output; give them random weights so returns vary.
    for lin in (wm.reward.last, critic.net.last, slow.net.last):
        nn.init.normal_(lin.weight, std=0.3)
    N = 12
    h = torch.randn(N, cfg.deter)
    z = nn.functional.one_hot(torch.randint(0, cfg.discrete, (N, cfg.stoch)), cfg.discrete).float()
    return cfg, wm, actor, critic, slow, h, z


def _adv_std(range_hi):
    cfg, wm, actor, critic, slow, h, z = _imag_setup()
    ema = ReturnEMA(alpha=0.0)                          # alpha 0: the EMA keeps the preset values
    ema.ema_vals.copy_(torch.tensor([0.0, range_hi]))
    torch.manual_seed(1)
    losses, ret, metrics = imagination_losses(wm, actor, critic, slow, ema, h, z, cfg)
    assert list(losses) == ["policy", "value"]
    assert ret.shape == (12, cfg.imag_horizon, 1)
    return float(metrics["adv_std"])


def test_advantage_normalized_by_max_one_s():
    small, one, four = _adv_std(0.5), _adv_std(1.0), _adv_std(4.0)
    assert small > 0
    assert abs(small - one) < 1e-6 * one                # S < 1: divide by 1, not by S
    assert abs(one / four - 4.0) < 1e-4


def test_imagination_losses_backward_reaches_actor_and_critic_only():
    cfg, wm, actor, critic, slow, h, z = _imag_setup()
    losses, _, _ = imagination_losses(wm, actor, critic, slow, ReturnEMA(), h, z, cfg)
    (losses["policy"] + losses["value"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    assert all(p.grad is None for p in wm.parameters())


def test_critic_loss_near_zero_at_on_bin_target():
    bins = symexp_bins(255)
    target = bins[120:130].reshape(2, 5, 1)
    logits = 40.0 * twohot_encode(target.squeeze(-1), bins)
    loss = critic_loss(logits, target, target, torch.ones(2, 4, 1), bins)
    assert loss.shape == () and 0 <= float(loss) < 1e-6


def test_critic_loss_stationary_at_twohot_target():
    bins = symexp_bins(255)
    target = torch.randn(3, 6, 1) * 50
    logits = torch.log(twohot_encode(target.squeeze(-1), bins) + 1e-12).requires_grad_()
    w = torch.ones(3, 5, 1)
    loss = critic_loss(logits, target, target, w, bins)
    loss.backward()
    assert logits.grad.abs().max() < 1e-6
    worse = critic_loss(logits.detach() + 0.5 * torch.randn_like(logits), target, target, w, bins)
    assert worse > loss


def test_critic_loss_ignores_last_step_and_zero_weight():
    bins = symexp_bins(31)
    logits = torch.randn(2, 4, 31)
    t1 = torch.randn(2, 4, 1)
    t2 = t1.clone()
    t2[:, -1] = 100.0
    w = torch.ones(2, 3, 1)
    torch.testing.assert_close(critic_loss(logits, t1, t1, w, bins), critic_loss(logits, t2, t2, w, bins))
    assert float(critic_loss(logits, t1, t1, 0 * w, bins)) == 0.0


def test_imagine_rollout():
    cfg, wm, actor, critic, slow, h, z = _imag_setup()
    feats, actions = imagine(wm.rssm, actor, h, z, 5)
    assert feats.shape == (12, 5, wm.rssm.feat_dim) and actions.shape == (12, 5, cfg.action_dim)
    assert not feats.requires_grad and not actions.requires_grad
    torch.testing.assert_close(feats[:, 0], wm.rssm.get_feat(h, z))
    zd = wm.rssm.z_dim
    # Step 1's h is the GRU applied to the start state and the first action.
    torch.testing.assert_close(feats[:, 1, zd:], wm.rssm.forward_h(h, z, actions[:, 0]))
    zs = feats[..., :zd].reshape(12, 5, cfg.stoch, cfg.discrete)
    assert torch.all(zs.sum(-1) == 1)
