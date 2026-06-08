"""AnyRes tiling arithmetic and the image-tiling reshape (LLaVA-NeXT). Provided.

A fixed-resolution ViT processes one image size. To handle higher resolution, LLaVA-NeXT
(AnyRes) splits the image into a grid of sub-crops, encodes each crop through the same ViT,
and concatenates the per-crop patch grids. A low-resolution overview of the full image is
also encoded and concatenated, so the LM sees both the global layout and the high-resolution
detail. The visual token count scales with the crop count, which makes it a first-class cost
knob: more crops means more context.

This file is a shape exercise. anyres_token_count gives the token arithmetic; tile_image
does the reshape for the tiling figure. No encoding happens here.
"""

from torch import Tensor


def anyres_token_count(n_patches_per_crop: int, grid: int, with_overview: bool = True) -> int:
    """Total visual tokens for an AnyRes grid.

    n_patches_per_crop: tokens one ViT crop produces (e.g. 576 for a 336px image at patch 14).
    grid: the tiling is grid x grid sub-crops, so grid*grid crops.
    with_overview: if True, add one more full overview image, also encoded at
        n_patches_per_crop tokens (the overview is a whole image, NOT a single token).

    Example (the LLaVA-NeXT 336px case): a 2x2 grid is 4 crops at 576 tokens each (2304) plus
    a 336px overview at 576 tokens, total 2880.
    """
    count = grid * grid * n_patches_per_crop
    if with_overview:
        count += n_patches_per_crop
    return count


def tile_image(img: Tensor, grid: int) -> Tensor:
    """Split (B, C, H, W) into grid*grid non-overlapping crops.

    Returns (B, grid*grid, C, H/grid, W/grid). H and W must be divisible by grid. Crops are
    in row-major order (top-left first, scanning left to right then top to bottom).
    """
    B, C, H, W = img.shape
    assert H % grid == 0 and W % grid == 0, "H and W must be divisible by grid"
    h, w = H // grid, W // grid
    # (B, C, grid, h, grid, w) -> (B, grid, grid, C, h, w) -> (B, grid*grid, C, h, w)
    out = img.reshape(B, C, grid, h, grid, w)
    out = out.permute(0, 2, 4, 1, 3, 5).reshape(B, grid * grid, C, h, w)
    return out
