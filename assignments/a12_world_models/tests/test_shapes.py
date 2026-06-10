"""Shape contracts for the RSSM cell, the observe loop, the encoder/decoder, and the actor.

The cell and encoder/decoder are checked at the build's 64x64 / 32x32-latent sizes. forward_h is
checked with a continuous (B, action_dim) float action (the cartpole force), which is how the
dynamics-backprop path feeds the actor's reparameterized sample.
"""

import torch

from config import WorldModelConfig
from nets import Decoder, Encoder
from rssm import RSSMCell


def test_rssm_cell_step_shapes():
    cfg = WorldModelConfig()
    cell = RSSMCell(cfg)
    B = 4
    h, z = cell.initial_state(B)
    a = torch.randn(B, cfg.action_dim)              # continuous force, (B, action_dim) float
    h2 = cell.forward_h(h, z, a)
    assert h2.shape == (B, cfg.h_dim)

    pri_logits, pri_z, pri_probs = cell.prior(h2, greedy=True)
    assert pri_logits.shape == (B, cfg.n_cat * cfg.n_cls)
    assert pri_z.shape == (B, cfg.n_cat * cfg.n_cls)
    assert pri_probs.shape == (B, cfg.n_cat, cfg.n_cls)

    embed = torch.randn(B, cfg.embed_dim)
    pos_logits, pos_z, pos_probs = cell.posterior(h2, embed, greedy=True)
    assert pos_logits.shape == (B, cfg.n_cat * cfg.n_cls)
    assert pos_z.shape == (B, cfg.n_cat * cfg.n_cls)
    assert pos_probs.shape == (B, cfg.n_cat, cfg.n_cls)


def test_observe_sequence_shapes():
    cfg = WorldModelConfig()
    cell = RSSMCell(cfg)
    B, T = 3, 6
    embeds = torch.randn(B, T, cfg.embed_dim)
    actions = torch.randn(B, T, cfg.action_dim)     # continuous forces
    h0, z0 = cell.initial_state(B)
    hs, zs, prior_l, post_l = cell.observe(embeds, actions, h0, z0, greedy=True)
    assert hs.shape == (B, T, cfg.h_dim)
    assert zs.shape == (B, T, cfg.n_cat * cfg.n_cls)
    assert prior_l.shape == (B, T, cfg.n_cat * cfg.n_cls)
    assert post_l.shape == (B, T, cfg.n_cat * cfg.n_cls)


def test_cont_actor_sample_shapes():
    from actor_critic import ContActor

    cfg = WorldModelConfig()
    actor = ContActor(cfg)
    B = 5
    h = torch.randn(B, cfg.h_dim)
    z = torch.randn(B, cfg.n_cat * cfg.n_cls)
    a, ent = actor.sample(h, z)
    assert a.shape == (B, cfg.action_dim)
    assert ent.shape == (B,)


def test_encoder_decoder_roundtrip_shape():
    cfg = WorldModelConfig()
    enc = Encoder(cfg)
    dec = Decoder(cfg)
    B = 2
    obs = torch.rand(B, cfg.obs_ch, cfg.obs_size, cfg.obs_size)
    embed = enc(obs)
    assert embed.shape == (B, cfg.embed_dim)
    state = torch.randn(B, cfg.h_dim + cfg.n_cat * cfg.n_cls)
    recon = dec(state)
    assert recon.shape == (B, cfg.obs_ch, cfg.obs_size, cfg.obs_size)
