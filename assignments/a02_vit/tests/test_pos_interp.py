"""Task 5: bicubic PE interpolation for resolution generalization.

Checks the shape of an upsized table, that interpolating to the same grid is close
to a no-op, and that a ViT built at 32x32 can run a 48x48 forward after swapping in
the interpolated positional embedding.
"""

import torch

from vit import ViT, interpolate_pos_embed


def test_interp_shape():
    dim = 8
    old_grid, new_grid = 8, 12
    pos = torch.randn(1, 1 + old_grid * old_grid, dim)
    out = interpolate_pos_embed(pos, old_grid, new_grid)
    assert out.shape == (1, 1 + new_grid * new_grid, dim)


def test_interp_identity():
    dim = 8
    grid = 8
    pos = torch.randn(1, 1 + grid * grid, dim)
    out = interpolate_pos_embed(pos, grid, grid)
    # Same grid is close to a no-op (bicubic on a matching grid is near-identity).
    assert torch.allclose(out, pos, atol=1e-4)


def test_vit_runs_at_new_resolution():
    # Build at 32x32 (patch 4 -> 8x8 grid), interpolate PE to the 48x48 grid.
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4,
                num_classes=10, n_registers=4)
    old_grid = model.grid          # 8
    new_side = 48
    new_grid = new_side // 4       # 12

    new_pe = interpolate_pos_embed(model.pos_embed.detach(), old_grid, new_grid)
    model.pos_embed = torch.nn.Parameter(new_pe)
    model.n_patches = new_grid * new_grid

    x = torch.randn(1, 3, new_side, new_side)
    logits = model(x)
    assert logits.shape == (1, 10)
    assert model.seq_len == 1 + new_grid * new_grid + model.n_registers
