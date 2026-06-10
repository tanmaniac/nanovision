"""Forward and sampling shapes for the encoder, the three heads, and the chunker.

The action heads read a conditioning vector of width cond_in. In the pixel reacher that vector is
the image encoder's embed_dim output; the heads are agnostic to its source, so these shape tests
drive them with both an explicit cond_in and the encoder's embed_dim.
"""

import torch

from bc import BCPolicy, chunk_actions
from config import VLAConfig
from ddpm import DDPMHead, make_schedule
from flow import FlowHead, flow_sample
from nets import Encoder


def test_encoder_shape():
    cfg = VLAConfig()
    enc = Encoder(cfg)
    obs = torch.rand(5, cfg.obs_ch, cfg.obs_size, cfg.obs_size)
    c = enc(obs)
    assert c.shape == (5, cfg.embed_dim)


def test_encoder_into_flow_head():
    # The pixel path end to end at the shape level: image -> embedding -> flow chunk.
    cfg = VLAConfig()
    enc = Encoder(cfg)
    head = FlowHead(cfg, cond_in=cfg.embed_dim)
    obs = torch.rand(4, cfg.obs_ch, cfg.obs_size, cfg.obs_size)
    c = enc(obs)
    out = flow_sample(head, c, cfg.chunk, cfg.n_flow_steps)
    assert out.shape == (4, cfg.chunk, cfg.act_dim)


def test_flow_forward_and_sample_shapes():
    cfg = VLAConfig()
    cond_in = cfg.embed_dim
    head = FlowHead(cfg, cond_in)
    B = 7
    z_t = torch.randn(B, cfg.chunk, cfg.act_dim)
    t = torch.rand(B, 1, 1)
    c = torch.randn(B, cond_in)
    assert head(z_t, t, c).shape == (B, cfg.chunk, cfg.act_dim)
    out = flow_sample(head, c, cfg.chunk, cfg.n_flow_steps)
    assert out.shape == (B, cfg.chunk, cfg.act_dim)


def test_bc_forward_shape():
    cfg = VLAConfig()
    cond_in = cfg.embed_dim
    policy = BCPolicy(cfg, cond_in)
    c = torch.randn(5, cond_in)
    assert policy(c).shape == (5, cfg.chunk, cfg.act_dim)


def test_ddpm_forward_shape():
    cfg = VLAConfig()
    cond_in = cfg.embed_dim
    head = DDPMHead(cfg, cond_in, T=cfg.ddpm_T)
    a_t = torch.randn(4, cfg.chunk, cfg.act_dim)
    t = torch.randint(0, cfg.ddpm_T, (4,))
    c = torch.randn(4, cond_in)
    assert head(a_t, t, c).shape == (4, cfg.chunk, cfg.act_dim)


def test_chunk_shapes():
    B, T, A, H = 3, 10, 2, 4
    actions = torch.randn(B, T, A)
    chunks = chunk_actions(actions, H)
    assert chunks.shape == (B, T - H + 1, H, A)
