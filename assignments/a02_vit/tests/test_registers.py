"""Task 3, 4: register tokens enter the sequence, get gradients, and are excluded
from mean pooling.

Register tokens (Darcet et al., ICLR 2024) are extra learnable tokens appended
after the patch tokens. They receive no positional embedding, take part in
attention, and are discarded at pooling time. This test confirms they are in the
encoder sequence, that gradients flow to them, and that mean-pool reads only the N
patch tokens (not CLS, not registers).
"""

import torch
import torch.nn.functional as F

from vit import ViT


def test_registers_in_sequence():
    n_reg = 4
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4, n_registers=n_reg)
    model(torch.randn(2, 3, 32, 32))
    assert model.seq_len == 1 + 64 + n_reg


def test_register_gradients_flow():
    n_reg = 4
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4, n_registers=n_reg)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    loss = F.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    grad = model.register_tokens.grad
    assert grad is not None
    assert grad.shape == model.register_tokens.shape
    assert grad.abs().sum() > 0  # nonzero somewhere


def test_mean_pool_ignores_registers():
    n_reg = 4
    n_patches = 64
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4,
                n_registers=n_reg, pool="mean")

    B, dim = 2, 64
    seq = 1 + n_patches + n_reg
    tokens = torch.zeros(B, seq, dim)
    # CLS and register tokens carry huge values; patch tokens are a known constant.
    tokens[:, 0] = 1e6                       # CLS
    tokens[:, 1 : 1 + n_patches] = 2.0       # patch tokens
    tokens[:, 1 + n_patches :] = 1e6         # register tokens

    pooled = model._pool(tokens)
    # If registers/CLS leaked in, the mean would explode; it must equal 2.0.
    assert torch.allclose(pooled, torch.full((B, dim), 2.0), atol=1e-5)


def test_cls_pool_selects_index_zero():
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4,
                n_registers=4, pool="cls")
    B, dim = 2, 64
    seq = 1 + 64 + 4
    tokens = torch.randn(B, seq, dim)
    pooled = model._pool(tokens)
    assert torch.allclose(pooled, tokens[:, 0])
