"""Tasks 1-4 shapes. Run first."""

import torch

from primitives import ConvNeXtBlock
from vit import ViT, PatchEmbed


def test_convnext_preserves_shape():
    block = ConvNeXtBlock(16)
    x = torch.randn(2, 16, 8, 8)
    assert block(x).shape == (2, 16, 8, 8)


def test_patch_embed_shape():
    dim = 64
    pe = PatchEmbed(in_chans=3, dim=dim, patch=4)
    x = torch.randn(2, 3, 32, 32)
    out = pe(x)
    # 32 / 4 = 8 -> N = 64 patch tokens
    assert out.shape == (2, 64, dim)


def test_vit_forward_shape():
    num_classes = 10
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4,
                num_classes=num_classes, n_registers=4)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, num_classes)


def test_vit_sequence_length():
    n_registers = 4
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4, n_registers=n_registers)
    model(torch.randn(2, 3, 32, 32))
    # 1 (CLS) + 64 (patches) + n_registers
    assert model.seq_len == 1 + 64 + n_registers
