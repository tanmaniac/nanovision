"""AnyRes token arithmetic and the tile_image reshape (LLaVA-NeXT)."""

import torch

from anyres import anyres_token_count, tile_image


def test_token_count_336px_2x2():
    # 4 crops of 576 + a 576-token overview = 2880; without the overview = 2304.
    assert anyres_token_count(576, grid=2, with_overview=True) == 2880
    assert anyres_token_count(576, grid=2, with_overview=False) == 2304


def test_token_count_general():
    # grid*grid crops, each n_patches_per_crop tokens, plus the overview.
    assert anyres_token_count(100, grid=3, with_overview=True) == 1000
    assert anyres_token_count(100, grid=3, with_overview=False) == 900


def test_tile_image_shapes():
    img = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
    tiles = tile_image(img, grid=2)
    assert tiles.shape == (2, 4, 3, 8, 8)
    # The top-left crop is the first 8x8 block of the original.
    assert torch.equal(tiles[:, 0], img[:, :, :8, :8])
    # The bottom-right crop is the last 8x8 block.
    assert torch.equal(tiles[:, 3], img[:, :, 8:, 8:])
