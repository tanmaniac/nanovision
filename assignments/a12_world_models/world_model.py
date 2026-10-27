"""The world model: encoder, RSSM, decoder, reward head and continuation head, and its loss.

Ported from the world-model half of r2dreamer's `Dreamer._cal_grad` (DreamerV3 mode,
`rep_loss=dreamer`). For a replayed batch of B sequences of length T:

    e_t            = encoder(o_t)
    h_t, z_t       = RSSM posterior rollout from the stored initial state
    L_dyn, L_rep   = free-bits KL terms between posterior and prior (rssm.kl_loss)
    L_image        = sum over pixels of (decoder(h_t, z_t) - o_t)^2
    L_rew          = two-hot cross-entropy of r_t from feat_t
    L_con          = Bernoulli negative log-likelihood of (1 - terminal_t) from feat_t

Reward alignment: in a replayed sequence, step t holds the frame o_t, the action a_{t-1} that led
to it, and the reward r_t earned by that action. So the reward of an action is read off the state
AFTER the action, both here and in imagination.

Hole in this file: WorldModel.loss. The networks are provided.
"""

import torch
from torch import Tensor, nn

from nets import ConvDecoder, ConvEncoder, MLPHead, bernoulli_dist, symexp_bins, twohot_log_prob
from rssm import RSSM, kl_loss


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg)
        self.rssm = RSSM(cfg, self.encoder.out_dim)
        feat = self.rssm.feat_dim
        self.reward = MLPHead(feat, cfg.bins, cfg.reward_layers, cfg.units, cfg.reward_outscale)
        self.cont = MLPHead(feat, 1, cfg.cont_layers, cfg.units, cfg.cont_outscale)
        self.decoder = ConvDecoder(cfg)
        self.register_buffer("bins", symexp_bins(cfg.bins), persistent=False)

    def loss(self, data: dict, h0: Tensor, z0: Tensor):
        """World-model losses for one batch.

        Args:
            data: image (B, T, 3, H, W) float in [0, 1]; action (B, T, A), the action before each
                frame; reward (B, T, 1); is_first (B, T) bool; is_terminal (B, T, 1) float.
            h0, z0: (B, D), (B, S, K) state before the first step of each sequence.
        Returns:
            losses: dict of scalar means (dyn, rep, image, rew, con), in the reference's order.
            h, z: (B, T, D), (B, T, S, K) posterior states, attached to the graph.
            feat: (B, T, F) concat(flatten(z), h), attached to the graph.

        Steps (each line is one call to a provided network or to a function from nets/rssm):

        1. embed = self.encoder(image), (B, T, E).
        2. h, z, post_logits = self.rssm.observe(embed, action, h0, z0, is_first): the posterior
           rollout, (B, T, D), (B, T, S, K), (B, T, S, K).
        3. _, prior_logits = self.rssm.prior(h): the prior evaluated at the posterior's h. Call
           prior (not prior_net directly) even though its sample is unused: it draws a sample like
           the reference does, which keeps the random-number stream aligned with it.
        4. dyn, rep = rssm.kl_loss(post_logits, prior_logits, cfg.kl_free, cfg.unimix); store
           their means as losses["dyn"] and losses["rep"].
        5. feat = self.rssm.get_feat(h, z).
        6. image: recon = self.decoder(h, z).float(); the squared error summed over (C, H, W),
           then averaged over (B, T). Plain pixels in [0, 1], no symlog.
        7. rew: -nets.twohot_log_prob(self.reward(feat), reward, self.bins), averaged. Feature t
           predicts reward t, the reward earned by the action that produced frame t.
        8. con: the Bernoulli negative log-likelihood of cont = 1 - is_terminal under
           nets.bernoulli_dist(self.cont(feat)), averaged.

        Insert the keys in the order dyn, rep, image, rew, con. The trainer sums the scaled
        losses in dict order (dyn x 1.0, rep x 0.1, image, rew, con x 1.0), and a different order
        changes the float rounding of the total.
        """
        raise NotImplementedError("implement the world-model loss")
