"""patchify/unpatchify are exact inverses, and the token count is (H/p)(W/p)."""

import torch

from dit import patchify, unpatchify


def test_roundtrip():
    B, C, H, W = 3, 4, 4, 4
    z = torch.randn(B, C, H, W)
    for p in (1, 2):
        tokens = patchify(z, p)
        n = (H // p) * (W // p)
        assert tokens.shape == (B, n, p * p * C)
        z_back = unpatchify(tokens, p, C, H, W)
        assert torch.allclose(z_back, z, atol=0.0), f"p={p}"
