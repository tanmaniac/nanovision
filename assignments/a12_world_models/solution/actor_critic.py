"""Behavior learning in imagination: lambda-returns, a two-hot critic, a continuous actor. Answer key.

The actor pi(a | h, z) and critic V(h, z) train entirely on trajectories imagined inside the world
model, with no environment steps. The critic regresses lambda-returns computed along the imagined
rollout. For continuous control (cartpole-balance, a 1-D force) the actor is a reparameterized
Tanh-Normal policy trained by DYNAMICS BACKPROP: the imagined lambda-return is differentiable with
respect to the action through the learned world model, so the policy gradient is the analytic
gradient of the return, not a REINFORCE score-function estimate.

The discrete Categorical Actor + REINFORCE path (actor_loss) is kept below as the contrast used by
the README and viz; the continuous ContActor + actor_loss_dynbackprop + imagine_dynamics path is the
build target.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from nets import twohot_decode, twohot_loss, value_bins

# Tanh-Normal log-std clamp: floor at log(0.1) so the policy cannot collapse to a delta, ceiling at
# log(2.0). The floor keeps a small exploration noise even as the mean sharpens.
LOGSTD_MIN, LOGSTD_MAX = math.log(0.1), math.log(2.0)


def compute_lambda_returns(rewards: Tensor, values: Tensor, continues: Tensor,
                           gamma: float, lam: float) -> Tensor:
    """DreamerV3 lambda-returns (arXiv:2301.04104 eq. 5), bootstrapping on the NEXT value.

    Backward recursion over the horizon dimension (last axis index t in 0..H-1):
        R_t = r_t + gamma * c_t * ((1 - lam) * V_{t+1} + lam * R_{t+1}),   R_H = V_H.
    c_t (= 0 at an episode end) zeros both the bootstrap and the recursion tail at termination.

    Shapes: rewards, continues are (B, H); values are (B, H + 1) where values[:, H] is the
    bootstrap value V_H at the state after the last reward. Returns (B, H).

    Args:
        rewards: (B, H) imagined rewards r_t.
        values: (B, H + 1) critic values V_0..V_H.
        continues: (B, H) continuation flags c_t in [0, 1].
        gamma: discount.
        lam: lambda mixing weight.
    Returns:
        (B, H) lambda-returns R_0..R_{H-1}.
    """
    H = rewards.shape[1]
    returns = [None] * H
    R_next = values[:, H]                      # R_H = V_H
    for t in reversed(range(H)):
        v_next = values[:, t + 1]
        R_next = rewards[:, t] + gamma * continues[:, t] * (
            (1.0 - lam) * v_next + lam * R_next
        )
        returns[t] = R_next
    return torch.stack(returns, dim=1)


class ContActor(nn.Module):
    """Tanh-Normal policy over the 1-D continuous force. Provided body; the lesson is the loss.

    The net maps the model state (h, z) to (mean, log_std). An action is the reparameterized sample
    a = tanh(mean + std * eps), eps ~ N(0, 1), so a is in (-1, 1) and the gradient flows from a back
    into (mean, log_std) and on into the dynamics that produced (h, z). log_std is clamped to
    [log(0.1), log(2.0)] so the policy keeps exploration noise. The entropy bonus uses the pre-tanh
    Normal entropy.
    """

    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.h_dim + cfg.n_cat * cfg.n_cls
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.h_dim), nn.SiLU(),
            nn.Linear(cfg.h_dim, cfg.h_dim), nn.SiLU(),
            nn.Linear(cfg.h_dim, 2),                       # (mean, log_std) for the 1-D force
        )

    def dist(self, h: Tensor, z: Tensor):
        """Return (mean, log_std) of the pre-tanh Normal, with log_std clamped to the floor/ceiling."""
        mean, log_std = self.net(torch.cat([h, z], dim=-1)).chunk(2, dim=-1)
        return mean, torch.clamp(log_std, LOGSTD_MIN, LOGSTD_MAX)

    def sample(self, h: Tensor, z: Tensor, greedy: bool = False):
        """Reparameterized Tanh-Normal sample. Returns (action (B, 1) in (-1, 1), entropy (B,))."""
        mean, log_std = self.dist(h, z)
        std = torch.exp(log_std)
        pre = mean if greedy else mean + std * torch.randn_like(mean)
        a = torch.tanh(pre)                                # (B, 1) in (-1, 1), reparameterized
        ent = Normal(mean, std).entropy().sum(dim=-1)      # pre-tanh Normal entropy, (B,)
        return a, ent


def imagine_dynamics(model, actor, critic, h, z, cfg):
    """The DIFFERENTIABLE imagined rollout under the prior. Answer key.

    Roll the dynamics forward cfg.horizon steps with the actor in the loop, keeping the graph
    attached so the imagined lambda-return is differentiable w.r.t. the actor. The reward of a_t is
    decoded from the POST-action state s_{t+1} (states[:, 1:]). Nothing is under no_grad.

    Returns (returns (B, horizon), entropies (B, horizon), H_h (B, horizon+1, .),
    H_z (B, horizon+1, .)).
    """
    hs, zs, ents = [h], [z], []
    for _ in range(cfg.horizon):
        a, ent = actor.sample(h, z)                   # reparameterized action; gradient flows
        ents.append(ent)
        h = model.rssm.forward_h(h, z, a)             # gradient flows through the GRU dynamics
        _, z, _ = model.rssm.prior(h)                 # straight-through categorical; gradient flows
        hs.append(h)
        zs.append(z)
    H_h = torch.stack(hs, dim=1)                       # (B, horizon+1, h_dim)
    H_z = torch.stack(zs, dim=1)                       # (B, horizon+1, n_cat*n_cls)
    states = torch.cat([H_h, H_z], dim=-1)
    # Reward / continuation of a_t are read from the POST-action state s_{t+1}: states[:, 1:].
    rewards = twohot_decode(torch.softmax(model.reward_head(states[:, 1:]), dim=-1), model.bins)
    conts = torch.sigmoid(model.cont_head(states[:, 1:]).squeeze(-1))
    values = critic.value(H_h, H_z)                    # (B, horizon+1); gradient flows
    returns = compute_lambda_returns(rewards, values, conts, cfg.gamma, cfg.lam)  # (B, horizon)
    return returns, torch.stack(ents, dim=1), H_h, H_z


def actor_loss_dynbackprop(returns: Tensor, entropies: Tensor, ent_coef: float,
                           ret_range: Tensor) -> Tensor:
    """Dynamics-backprop actor loss: negative normalized return minus an entropy bonus. Answer key.

    The returns are differentiable through the dynamics, so the gradient of the return is the policy
    gradient - no log-prob, no detached advantage.

        loss = -(returns / ret_range).mean() - ent_coef * entropies.mean()
    """
    return -(returns / ret_range).mean() - ent_coef * entropies.mean()


class Actor(nn.Module):
    """MLP policy (h, z) -> action_dim logits, returning a Categorical. The discrete REINFORCE
    contrast used by the README/viz; the build target is ContActor above. Body provided."""

    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.h_dim + cfg.n_cat * cfg.n_cls
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.h_dim), nn.SiLU(),
            nn.Linear(cfg.h_dim, cfg.h_dim), nn.SiLU(),
            nn.Linear(cfg.h_dim, cfg.action_dim),
        )

    def forward(self, h: Tensor, z: Tensor) -> Categorical:
        logits = self.net(torch.cat([h, z], dim=-1))
        return Categorical(logits=logits)


class Critic(nn.Module):
    """MLP value head (h, z) -> n_bins two-hot logits. Body provided; loss is a hole."""

    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.h_dim + cfg.n_cat * cfg.n_cls
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.h_dim), nn.SiLU(),
            nn.Linear(cfg.h_dim, cfg.h_dim), nn.SiLU(),
            nn.Linear(cfg.h_dim, cfg.n_bins),
        )
        self.register_buffer("bins", value_bins(cfg))

    def logits(self, h: Tensor, z: Tensor) -> Tensor:
        return self.net(torch.cat([h, z], dim=-1))

    def value(self, h: Tensor, z: Tensor) -> Tensor:
        """Decoded scalar value: symexp of the symlog-space bin expectation (via twohot_decode)."""
        probs = F.softmax(self.logits(h, z), dim=-1)
        return twohot_decode(probs, self.bins)


def actor_loss(logprobs: Tensor, returns: Tensor, values: Tensor, entropies: Tensor,
               ent_coef: float, ret_range: Tensor) -> Tensor:
    """REINFORCE actor loss on the normalized advantage.

    adv = (returns - values).detach() / ret_range; loss = -(logprobs * adv).mean()
          - ent_coef * entropies.mean().

    The advantage is detached so only the log-prob carries the policy gradient; ret_range is
    max(1, S) where S is the EMA of the 5-95 percentile return spread (tracked in _train.py).

    Args:
        logprobs: (B, H) log-prob of each imagined action.
        returns: (B, H) lambda-returns.
        values: (B, H) critic values at the same states.
        entropies: (B, H) per-step policy entropies.
        ent_coef: entropy bonus weight.
        ret_range: scalar (or broadcastable) normalizer max(1, S).
    Returns:
        scalar actor loss.
    """
    adv = (returns - values).detach() / ret_range
    pg = -(logprobs * adv).mean()
    ent = -ent_coef * entropies.mean()
    return pg + ent


def critic_loss(logits: Tensor, returns: Tensor, bins: Tensor) -> Tensor:
    """Two-hot regression of the critic logits onto the (detached) lambda-returns.

    Args:
        logits: (B, H, n_bins) critic two-hot logits.
        returns: (B, H) lambda-return targets.
        bins: (n_bins,) value-space bin positions.
    Returns:
        scalar critic loss.
    """
    return twohot_loss(logits, returns.detach(), bins).mean()


class ReturnNormalizer:
    """EMA of the 5-95 percentile return spread; ret_range = max(1, S). DreamerV3 eq. 6-7. Provided.

    Tracked in _train.py and updated each batch; the actor divides the advantage by ret_range so
    the policy-gradient scale is invariant to return magnitude.
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.S = None

    def update(self, returns: Tensor) -> Tensor:
        lo = torch.quantile(returns.detach(), 0.05)
        hi = torch.quantile(returns.detach(), 0.95)
        spread = (hi - lo).clamp(min=0.0)
        if self.S is None:
            self.S = spread
        else:
            self.S = self.decay * self.S + (1.0 - self.decay) * spread
        return self.range()

    def range(self) -> Tensor:
        s = self.S if self.S is not None else torch.tensor(1.0)
        return torch.maximum(s, torch.ones_like(s))
