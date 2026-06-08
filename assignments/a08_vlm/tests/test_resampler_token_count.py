"""The resampler compresses any patch count to a fixed Q output tokens."""

import torch

from config import VLMConfig
from resampler import PerceiverResampler


def test_fixed_output_length():
    cfg = VLMConfig()
    res = PerceiverResampler(cfg.vit_dim, cfg.dim_l, cfg.n_queries, cfg.lm_heads)
    out16 = res(torch.randn(4, 16, cfg.vit_dim))
    out9 = res(torch.randn(4, 9, cfg.vit_dim))
    # Output length is Q regardless of how many patches came in.
    assert out16.shape == (4, cfg.n_queries, cfg.dim_l)
    assert out9.shape == (4, cfg.n_queries, cfg.dim_l)
