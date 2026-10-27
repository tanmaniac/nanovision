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

Holes in this file (each raises NotImplementedError until written): lambda_return, imagine,
actor_loss, critic_loss and imagination_losses. Actor, Critic, ReturnEMA and replay_value_loss
are provided.
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
    """Lambda-returns over a sequence, computed backward from the last step (DreamerV3 v2 eq. 5).

        R_{T-1} = boot_{T-1}
        R_t     = r_{t+1} + gamma_{t+1} * ((1 - lam_{t+1}) * boot_{t+1} + lam_{t+1} * R_{t+1})

    for t = T-2 down to 0, with gamma_{t+1} = disc * (1 - term_{t+1}) and
    lam_{t+1} = lamb * (1 - last_{t+1}). A terminal step (term = 1) stops all future value; a step
    marked last (end of the stored episode, not a terminal) cuts the recursion and bootstraps fully
    from boot. Index t + 1 because reward[:, t] is the reward for ARRIVING at step t, so the return
    of state t starts with reward[:, t + 1].

    The reference groups the arithmetic as follows; use the same grouping for bit-identical
    results (it is the same math as the formula above):

        live   = (1 - term)[:, 1:] * disc                    (N, T-1, 1)
        cont   = (1 - last)[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        R      = interm[:, t] + live[:, t] * cont[:, t] * R_next,  starting from R_next = boot[:, -1]

    Args:
        last, term, reward, boot: (N, T, 1), float (cast the flags with .float()). boot is the
            value used to bootstrap: the critic for imagined rollouts, the imagined return for
            replayed sequences.
        disc: gamma (1 - 1/333). lamb: lambda (0.95).
    Returns:
        (N, T-1, 1) returns for steps 0..T-2 (the last step only supplies the bootstrap).
    """
    raise NotImplementedError("implement the lambda-return")


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

    Loop `steps` times: feat = rssm.get_feat(h, z); action = actor(feat).rsample(); record both;
    then advance h, z = rssm.img_step(h, z, action). The step after the last action is computed and
    thrown away, as in the reference (it draws random numbers, so skipping it would shift the
    random stream). Nothing here needs gradients: the actor is trained by REINFORCE, which only
    needs log pi(a | s) of the recorded actions, recomputed later from the detached feats.

    Args:
        rssm: the world model's RSSM. actor: maps feat to a distribution with rsample().
        h, z: (N, D), (N, S, K) start states.
        steps: number of states returned, including the start (imag_horizon + 1 = 16).
    Returns:
        feats (N, steps, F) and actions (N, steps, A); actions[:, t] was sampled at feats[:, t].
    """
    raise NotImplementedError("implement the imagination rollout")


def actor_loss(policy, actions: Tensor, adv: Tensor, weight: Tensor, entropy_coef: float) -> Tensor:
    """REINFORCE with an entropy bonus (DreamerV3 v2 eq. 6):

        loss = mean( weight * -( log pi(a_t | s_t) * sg(A_t) + eta * H[pi(. | s_t)] ) )

    The advantage A is detached, so the gradient is the score-function estimator: it raises the
    log-probability of actions with A > 0 and lowers it for A < 0. The entropy term pushes the
    policy to stay spread out. The weight (cumulative discount) is detached too.

    The policy covers all T imagined states but the advantages only T-1 (the last state has no
    return), so drop the last step of log_prob and entropy: policy.log_prob(actions)[:, :-1] and
    policy.entropy()[:, :-1], each unsqueezed to (N, T-1, 1).

    Args:
        policy: torch distribution over the imagined feats, batch shape (N, T), event (A,).
        actions: (N, T, A). adv: (N, T-1, 1) normalized advantages.
        weight: (N, T-1, 1) cumulative discount. entropy_coef: eta (3e-4).
    Returns:
        scalar loss.
    """
    raise NotImplementedError("implement the REINFORCE actor loss")


def critic_loss(logits: Tensor, target: Tensor, slow_target: Tensor, weight: Tensor, bins: Tensor) -> Tensor:
    """Two-hot cross-entropy toward the return target plus toward the slow critic's value.

        loss = mean( weight * ( -log p(target) - log p(slow_target) ) )

    where log p(y) = nets.twohot_log_prob(logits, y, bins). The second term regularizes the critic
    toward its own slow EMA copy (weight 1, as in the reference). Both targets are detached.

    Cast logits to float32 once, before both terms, so their gradients add in float32 before
    flowing back into a bfloat16 network. Compute the per-step loss over all T steps, then drop the
    last one (nll[:, :-1].unsqueeze(-1)): the target's last step is padding.

    Args:
        logits: (N, T, bins); target, slow_target: (N, T, 1).
        weight: (N, T-1, 1), detached inside.
    Returns:
        scalar loss.
    """
    raise NotImplementedError("implement the critic loss")


def imagination_losses(wm, actor: Actor, critic: Critic, slow_critic: Critic, return_ema: ReturnEMA,
                       h: Tensor, z: Tensor, cfg):
    """Actor and critic losses on rollouts imagined from detached start states h (N, D), z (N, S, K).

    Steps, in the reference's order (the order of random draws and EMA updates matters for
    bit-identical results):

    1. feats, actions = imagine(wm.rssm, actor, h.detach(), z.detach(), cfg.imag_horizon + 1).
    2. Under torch.no_grad, read off every imagined state (N, H+1, 1 each):
       reward = nets.twohot_decode(wm.reward(feats), wm.bins);
       cont = nets.bernoulli_dist(wm.cont(feats)).mean (the probability of continuing);
       value = critic.value(feats); slow_value = slow_critic.value(feats).
    3. disc = 1 - 1 / cfg.horizon. weight = cumprod(cont * disc, dim=1). The reference starts the
       product with step 0's own factor cont_0 * disc rather than with 1; keep that.
    4. ret = lambda_return(zeros_like(cont), 1 - cont, reward, value, disc, cfg.lamb), (N, H, 1).
       No step is marked last; the soft termination 1 - cont is the only cut.
    5. _, scale = return_ema(ret): updates the percentile EMA and returns S' = max(1, p95 - p5).
       adv = (ret - value[:, :-1]) / scale. Dividing by max(1, S) rather than by S keeps small
       returns from being blown up into noise.
    6. policy = actor(feats); losses["policy"] = actor_loss(policy, actions, adv, weight[:, :-1],
       cfg.act_entropy).
    7. Pad the target with a zero step, tar = cat([ret, 0 * ret[:, -1:]], 1), and
       losses["value"] = critic_loss(critic(feats), tar, slow_value, weight[:, :-1], critic.bins).
    8. metrics: imag_reward = reward.mean(), imag_value = value.mean(), imag_return = ret.mean(),
       ret_p05 = return_ema.ema_vals[0], ret_p95 = return_ema.ema_vals[1], adv_std = adv.std(),
       action_entropy = policy.entropy().mean(), action_std = actions.std().

    Returns:
        losses {"policy", "value"} (insert in that order), ret (N, H, 1), metrics dict.
    """
    raise NotImplementedError("implement the actor and critic losses on an imagined rollout")


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
