"""The RSSM cell: a deterministic GRU recurrence plus a stochastic categorical latent. Answer key.

The recurrent state-space model (RSSM, from PlaNet) runs two state components in parallel. The
deterministic state h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]) carries memory and is differentiable
end to end. The stochastic latent z_t is a set of categoricals sampled from a posterior
q(z_t | h_t, o_t) during training (it sees the observation through the encoder embedding) and from
a prior p(z_t | h_t) during imagination (no observation). The full model state is the
concatenation (h_t, z_t); every predictor conditions on it.

The KL between posterior and prior trains the prior to predict the posterior, so imagination,
which uses only the prior, stays on the manifold the posterior learned from real frames.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nets import categorical_sample


class RSSMCell(nn.Module):
    """One RSSM step: the GRU recurrence and the prior/posterior categorical heads."""

    def __init__(self, cfg):
        super().__init__()
        self.h_dim = cfg.h_dim
        self.n_cat = cfg.n_cat
        self.n_cls = cfg.n_cls
        self.action_dim = cfg.action_dim
        self.unimix = cfg.unimix
        z_dim = cfg.n_cat * cfg.n_cls

        # GRU input is the previous latent concatenated with the one-hot action, projected to width.
        self.in_proj = nn.Linear(z_dim + cfg.action_dim, cfg.h_dim)
        self.gru = nn.GRUCell(cfg.h_dim, cfg.h_dim)

        # Prior reads h only; posterior reads h and the encoder embedding.
        self.prior_mlp = nn.Sequential(
            nn.Linear(cfg.h_dim, cfg.h_dim), nn.SiLU(), nn.Linear(cfg.h_dim, z_dim)
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(cfg.h_dim + cfg.embed_dim, cfg.h_dim), nn.SiLU(), nn.Linear(cfg.h_dim, z_dim)
        )

    def forward_h(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """Advance the deterministic state: h_new = GRU(proj([z, onehot(a)]), h).

        Args:
            h: (B, h_dim) previous deterministic state.
            z: (B, n_cat*n_cls) previous flattened categorical latent.
            a: (B,) integer actions or (B, action_dim) one-hot.
        Returns:
            (B, h_dim) new deterministic state.
        """
        if a.dim() == 1:
            a = F.one_hot(a, self.action_dim).to(z.dtype)
        x = self.in_proj(torch.cat([z, a], dim=-1))
        return self.gru(x, h)

    def prior(self, h: Tensor, greedy: bool = False):
        """Transition prior p(z_t | h_t): MLP(h) -> logits, then a straight-through sample.

        Returns (logits (B, n_cat*n_cls), z (B, n_cat*n_cls), probs (B, n_cat, n_cls)).
        """
        logits = self.prior_mlp(h)
        z, probs = categorical_sample(logits, self.unimix, self.n_cat, self.n_cls, greedy=greedy)
        return logits, z, probs

    def posterior(self, h: Tensor, embed: Tensor, greedy: bool = False):
        """Posterior q(z_t | h_t, o_t): MLP([h, embed]) -> logits, then a straight-through sample.

        Returns (logits (B, n_cat*n_cls), z (B, n_cat*n_cls), probs (B, n_cat, n_cls)).
        """
        logits = self.post_mlp(torch.cat([h, embed], dim=-1))
        z, probs = categorical_sample(logits, self.unimix, self.n_cat, self.n_cls, greedy=greedy)
        return logits, z, probs

    def initial_state(self, B: int, device=None, dtype=torch.float32):
        """Zero (h, z) for a batch of size B. Provided."""
        h = torch.zeros(B, self.h_dim, device=device, dtype=dtype)
        z = torch.zeros(B, self.n_cat * self.n_cls, device=device, dtype=dtype)
        return h, z

    def observe(self, embeds: Tensor, actions: Tensor, h0: Tensor, z0: Tensor, greedy: bool = False):
        """Unroll the posterior over a real sequence. Provided loop over the per-step holes.

        At step t: advance h with the PREVIOUS action and latent, compute the prior from h (for
        the KL target) and the posterior from h and the embedding, then carry the posterior sample
        forward. Step 0 uses a zero "previous action".

        Args:
            embeds: (B, T, embed_dim) encoder embeddings of the observed frames.
            actions: (B, T) integer actions taken AT each step.
            h0, z0: (B, h_dim), (B, n_cat*n_cls) initial state.
        Returns:
            hs: (B, T, h_dim), zs: (B, T, n_cat*n_cls),
            prior_logits, post_logits: (B, T, n_cat*n_cls).
        """
        B, T, _ = embeds.shape
        h, z = h0, z0
        hs, zs, prior_l, post_l = [], [], [], []
        a_prev = torch.zeros(B, dtype=torch.long, device=embeds.device)
        for t in range(T):
            h = self.forward_h(h, z, a_prev)
            pri_logits, _, _ = self.prior(h, greedy=greedy)
            pos_logits, z, _ = self.posterior(h, embeds[:, t], greedy=greedy)
            hs.append(h)
            zs.append(z)
            prior_l.append(pri_logits)
            post_l.append(pos_logits)
            a_prev = actions[:, t]
        return (
            torch.stack(hs, 1), torch.stack(zs, 1),
            torch.stack(prior_l, 1), torch.stack(post_l, 1),
        )

    def imagine(self, policy, h0: Tensor, z0: Tensor, horizon: int, greedy: bool = False):
        """Roll the dynamics forward with the PRIOR only, no observation, no decoder. Provided.

        At each step: sample an action from policy(h, z), advance h, sample z from the prior. The
        policy returns a torch.distributions.Categorical; we record the log-prob and entropy of the
        sampled action for the actor loss.

        Args:
            policy: callable (h, z) -> Categorical over actions.
            h0, z0: starting state (typically a posterior state encoded from a real frame).
            horizon: number of imagined steps.
        Returns:
            hs, zs: (B, horizon, .) imagined states.
            actions: (B, horizon) sampled actions.
            logprobs, entropies: (B, horizon) action log-probs and per-step entropies.
        """
        h, z = h0, z0
        hs, zs, acts, logps, ents = [], [], [], [], []
        for _ in range(horizon):
            dist = policy(h, z)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
            h = self.forward_h(h, z, a)
            _, z, _ = self.prior(h, greedy=greedy)
            hs.append(h)
            zs.append(z)
            acts.append(a)
        return (
            torch.stack(hs, 1), torch.stack(zs, 1), torch.stack(acts, 1),
            torch.stack(logps, 1), torch.stack(ents, 1),
        )
