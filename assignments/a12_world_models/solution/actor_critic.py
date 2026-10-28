"""Actor and critic trained on imagined rollouts, plus the critic loss on replayed states.

Ported from the actor-critic half of r2dreamer's `Dreamer._cal_grad` (`_imagine`,
`_lambda_return`, `networks.ReturnEMA`), which follows DreamerV3 v2 (arXiv 2301.04104v2):

- Imagination starts from every posterior state of the replayed batch (B*T = 1024 starts), detached,
  and rolls the prior forward 15 steps under the current actor without gradients.
- Returns are lambda-returns bootstrapped with the critic. Rewards are read off the state after the
  action, matching the world model's reward alignment.
- The actor is trained by REINFORCE: loss = -(log pi(a|s) * sg(A) + eta * H[pi]), with advantage
  A = (R^lambda - v(s)) / max(1, S) and S an EMA of the 5th-95th percentile range of returns.
- The critic is a two-hot classifier over symexp bins, regressed toward the lambda-return and
  toward its own slow EMA copy (a regularizer, weight 1).
- The critic is also trained on replayed posterior states with a replay lambda-return that
  bootstraps from the imagined returns (`repval`, scale 0.3). Its feature input is NOT detached,
  so this loss also sends gradient into the world model. The reference does this on purpose
  ("keep gradients through world model").
"""

import torch
from torch import Tensor, nn

from nets import MLPHead, actor_dist, bernoulli_dist, symexp_bins, twohot_decode, twohot_log_prob


class Actor(nn.Module):
    """pi(a | feat): Gaussian with mean tanh(mu) and std in [0.1, 1] (nets.actor_dist)."""

    def __init__(self, cfg, feat_dim: int):
        super().__init__()
        self.net = MLPHead(feat_dim, 2 * cfg.action_dim, cfg.actor_layers, cfg.units, cfg.actor_outscale)
        self.min_std, self.max_std = cfg.actor_min_std, cfg.actor_max_std

    def forward(self, feat: Tensor):
        return actor_dist(self.net(feat), self.min_std, self.max_std)


class Critic(nn.Module):
    """v(feat) as a two-hot distribution over symexp bins. forward returns logits."""

    def __init__(self, cfg, feat_dim: int):
        super().__init__()
        self.net = MLPHead(feat_dim, cfg.bins, cfg.critic_layers, cfg.units, cfg.critic_outscale)
        self.register_buffer("bins", symexp_bins(cfg.bins), persistent=False)

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)

    def value(self, feat: Tensor) -> Tensor:
        """Expected value, (..., 1)."""
        return twohot_decode(self(feat), self.bins)


@torch.no_grad()
def lambda_return(last: Tensor, term: Tensor, reward: Tensor, boot: Tensor, disc: float, lamb: float) -> Tensor:
    """Lambda-returns over a sequence, computed backward from the last step.

    R_t = r_{t+1} + gamma_{t+1} * ((1 - lam_{t+1}) * boot_{t+1} + lam_{t+1} * R_{t+1}),  R_{T-1} = boot_{T-1}
    with gamma_{t+1} = disc * (1 - term_{t+1}) and lam_{t+1} = lamb * (1 - last_{t+1}): a step
    marked `last` cuts the recursion and bootstraps fully from boot.

    Args:
        last, term, reward, boot: (N, T, 1). reward[:, t] is the reward for arriving at step t.
    Returns:
        (N, T-1, 1) returns for steps 0..T-2.
    """
    assert last.shape == term.shape == reward.shape == boot.shape
    live = (1 - term.float())[:, 1:] * disc
    cont = (1 - last.float())[:, 1:] * lamb
    interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
    out = [boot[:, -1]]
    for i in reversed(range(live.shape[1])):
        out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
    return torch.stack(list(reversed(out))[:-1], 1)


class ReturnEMA(nn.Module):
    """EMA of the 5th and 95th percentiles of the returns; scale = max(1, p95 - p5)."""

    def __init__(self, alpha: float = 1e-2, quantiles=(0.05, 0.95)):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("range", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32))

    def forward(self, x: Tensor):
        """Update the EMA with x's percentiles; return (offset, scale) as detached scalars."""
        # Sort once and interpolate the two order statistics (torch.quantile's 'linear' rule).
        # The reference avoids torch.quantile because torch.compile cannot trace it.
        srt, _ = torch.sort(torch.flatten(x.detach()))
        n = srt.shape[0]
        pos = self.range.to(srt.dtype) * (n - 1)
        lo = pos.floor().long().clamp_(0, n - 1)
        hi = pos.ceil().long().clamp_(0, n - 1)
        frac = pos - pos.floor()
        q = srt[lo] + (srt[hi] - srt[lo]) * frac
        self.ema_vals.copy_(self.alpha * q.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        return self.ema_vals[0].detach(), scale.detach()


@torch.no_grad()
def imagine(rssm, actor: Actor, h: Tensor, z: Tensor, steps: int):
    """Roll the prior forward under the actor, without gradients.

    Args:
        h, z: (N, D), (N, S, K) start states.
        steps: number of states returned, including the start (horizon + 1).
    Returns:
        feats (N, steps, F) and actions (N, steps, A); actions[:, t] is sampled at feats[:, t].
    """
    feats, actions = [], []
    for _ in range(steps):
        feat = rssm.get_feat(h, z)
        action = actor(feat).rsample()
        feats.append(feat)
        actions.append(action)
        # The step after the last action is computed and discarded, as in the reference.
        h, z = rssm.img_step(h, z, action)
    return torch.stack(feats, 1), torch.stack(actions, 1)


def actor_loss(policy, actions: Tensor, adv: Tensor, weight: Tensor, entropy_coef: float) -> Tensor:
    """REINFORCE with entropy bonus: mean(weight * -(log pi(a) * sg(adv) + eta * H)).

    Args:
        policy: distribution over the imagined feats (N, T); actions: (N, T, A).
        adv: (N, T-1, 1) normalized advantages. weight: (N, T-1, 1) cumulative discount.
    """
    logpi = policy.log_prob(actions)[:, :-1].unsqueeze(-1)
    entropy = policy.entropy()[:, :-1].unsqueeze(-1)
    return torch.mean(weight.detach() * -(logpi * adv.detach() + entropy_coef * entropy))


def critic_loss(logits: Tensor, target: Tensor, slow_target: Tensor, weight: Tensor, bins: Tensor) -> Tensor:
    """Two-hot cross-entropy toward the return target plus toward the slow critic's value.

    Args:
        logits: (N, T, bins); target, slow_target: (N, T, 1), the last step is padding and dropped.
        weight: (N, T-1, 1).
    """
    # Cast once so the two terms' gradients are summed in float32 before flowing back into the
    # bfloat16 network (casting inside each call would sum them in bfloat16).
    logits = logits.float()
    nll = -twohot_log_prob(logits, target.detach(), bins) - twohot_log_prob(logits, slow_target.detach(), bins)
    return torch.mean(weight.detach() * nll[:, :-1].unsqueeze(-1))


def imagination_losses(wm, actor: Actor, critic: Critic, slow_critic: Critic, return_ema: ReturnEMA,
                       h: Tensor, z: Tensor, cfg):
    """Actor and critic losses on rollouts imagined from detached start states h (N, D), z (N, S, K).

    Returns (losses {policy, value}, ret (N, H, 1) imagined lambda-returns, metrics).
    """
    feats, actions = imagine(wm.rssm, actor, h.detach(), z.detach(), cfg.imag_horizon + 1)
    with torch.no_grad():
        reward = twohot_decode(wm.reward(feats), wm.bins)                     # (N, H+1, 1)
        cont = bernoulli_dist(wm.cont(feats)).mean                            # P(continue)
        value = critic.value(feats)
        slow_value = slow_critic.value(feats)
    disc = 1 - 1 / cfg.horizon
    # Cumulative discount including step 0's own continuation probability; the reference starts
    # the product at cont_0 * disc rather than at 1. With cont ~ 1 the difference is a factor 0.997.
    weight = torch.cumprod(cont * disc, dim=1)
    ret = lambda_return(torch.zeros_like(cont), 1 - cont, reward, value, disc, cfg.lamb)  # (N, H, 1)
    _, ret_scale = return_ema(ret)
    adv = (ret - value[:, :-1]) / ret_scale

    policy = actor(feats)
    losses = {"policy": actor_loss(policy, actions, adv, weight[:, :-1], cfg.act_entropy)}
    tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
    losses["value"] = critic_loss(critic(feats), tar_padded, slow_value, weight[:, :-1], critic.bins)
    metrics = {
        "imag_reward": reward.mean(), "imag_value": value.mean(), "imag_return": ret.mean(),
        "ret_p05": return_ema.ema_vals[0], "ret_p95": return_ema.ema_vals[1],
        "adv_std": adv.std(), "action_entropy": policy.entropy().mean(),
        "action_std": actions.std(),
    }
    return losses, ret, metrics


def replay_value_loss(critic: Critic, slow_critic: Critic, feat: Tensor, data: dict, imag_ret: Tensor, cfg):
    """Critic loss on the replayed posterior states (the reference's `repval`).

    The replay lambda-return uses real rewards and bootstraps from the imagined return of each
    state (imag_ret[:, 0]). feat stays attached, so this term also trains the world model.
    Args:
        feat: (B, T, F) posterior features. imag_ret: (B*T, H, 1) from imagination_losses.
    """
    B, T = feat.shape[:2]
    last, term, reward = data["is_last"], data["is_terminal"], data["reward"]
    boot = imag_ret[:, 0].reshape(B, T, 1)
    with torch.no_grad():
        slow_value = slow_critic.value(feat)
    disc = 1 - 1 / cfg.horizon
    ret = lambda_return(last, term, reward, boot, disc, cfg.lamb)            # (B, T-1, 1)
    ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
    weight = 1.0 - last
    return critic_loss(critic(feat), ret_padded, slow_value, weight[:, :-1], critic.bins)
