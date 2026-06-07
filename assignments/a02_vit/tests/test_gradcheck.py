"""Tasks 1, 2 gradients (float64 gradcheck). Run after shapes."""

import torch

from nanovision.gradcheck import check_gradients
from primitives import ConvNeXtBlock
from vit import PatchEmbed


def test_convnext_gradcheck():
    block = ConvNeXtBlock(4)
    x = torch.randn(1, 4, 5, 5, dtype=torch.double)
    assert check_gradients(block, (x,))


def test_patch_embed_gradcheck():
    pe = PatchEmbed(in_chans=2, dim=6, patch=2)
    x = torch.randn(1, 2, 4, 4, dtype=torch.double)
    assert check_gradients(pe, (x,))
