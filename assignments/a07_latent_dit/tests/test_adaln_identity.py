"""adaLN-Zero identity at init: a fresh block returns x unchanged, and the full DiT
predicts all zeros, regardless of the conditioning.

The zero-init adaLN Linear sets every gate to 0, so both residual branches contribute
nothing; the zero-init final adaLN and output head make the whole DiT output 0 at init.
"""

import torch

from config import DiTConfig
from dit import DiT, DiTBlock


def test_block_is_identity_at_init():
    torch.manual_seed(0)
    d, n_heads, mlp_ratio = 64, 2, 4
    block = DiTBlock(d, n_heads, mlp_ratio).eval()
    x = torch.randn(5, 16, d)
    c = torch.randn(5, d)
    out = block(x, c)
    assert torch.allclose(out, x, atol=1e-6)


def test_full_dit_predicts_zero_at_init():
    torch.manual_seed(0)
    cfg = DiTConfig()
    dit = DiT(cfg).eval()
    B, C, hw = 6, cfg.latent_dim, cfg.image_size // cfg.f
    z = torch.randn(B, C, hw, hw)
    t = torch.rand(B)
    y = torch.randint(0, cfg.num_classes, (B,))
    v = dit(z, t, y)
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)
