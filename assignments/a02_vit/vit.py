"""A2 - the ViT. Fill the four holes (Tasks 2-5), then run the tests.

The ViT is assignment-local glue: it chains the A1 transformer encoder (classic
ViT config: LayerNorm + GELU MLP + absolute PE, non-causal) over image patch
tokens. You implement the patch tokenizer, the sequence assembly (CLS + PE +
register tokens), the pooling choice, and the bicubic PE interpolation. The module
construction and the forward plumbing are given. The reference lives in
`solution/vit.py` (read it if you get stuck).

Shapes: images are (B, C, H, W); patch tokens are (B, N, dim) with N = (H/p)^2;
the encoder sees (B, 1 + N + n_registers, dim); logits are (B, num_classes).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanovision.primitives import LayerNorm
from nanovision.transformer import TransformerEncoder


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and linearly project each.

    A Conv2d with kernel_size = stride = patch applies one shared linear map to
    every p x p patch, which is exactly the ViT patch projection. The spatial grid
    of conv outputs is then flattened to a token sequence.

    forward(x): x is (B, in_chans, H, W); output is (B, N, dim) with
    N = (H / patch) * (W / patch).
    """

    def __init__(self, in_chans: int, dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_chans, H, W) -> (B, N, dim).

        Implement:
            1. x = self.proj(x)        # (B, dim, H/p, W/p)
            2. flatten the spatial dims: (B, dim, N)
            3. transpose to (B, N, dim)
        """
        raise NotImplementedError("A2 Task 2: implement PatchEmbed.forward")


def interpolate_pos_embed(pos_embed: Tensor, old_grid: int, new_grid: int) -> Tensor:
    """Bicubically resize the patch part of a learned PE table to a new grid.

    The table is (1, 1 + old_grid^2, dim): row 0 is the [CLS] positional vector,
    the remaining old_grid^2 rows are the patch positions in row-major order. Keep
    the CLS row unchanged; reshape the patch rows to an (old_grid, old_grid)
    spatial map per channel and resize to (new_grid, new_grid) with bicubic
    interpolation. This is the trick that lets a ViT trained at one resolution run
    at another.

    Returns a (1, 1 + new_grid^2, dim) table.

    Implement:
        1. split off cls_pe = pos_embed[:, :1] and patch_pe = pos_embed[:, 1:]
        2. reshape patch_pe to (1, old_grid, old_grid, dim) then permute to
           (1, dim, old_grid, old_grid)
        3. F.interpolate(..., size=(new_grid, new_grid), mode="bicubic",
           align_corners=False)
        4. permute/reshape back to (1, new_grid^2, dim)
        5. torch.cat([cls_pe, patch_pe], dim=1)
    """
    raise NotImplementedError("A2 Task 5: implement interpolate_pos_embed")


class ViT(nn.Module):
    """Vision transformer assembled from the A1 transformer encoder.

    Pipeline: patch embed -> prepend [CLS] -> add learned absolute PE over
    [CLS] + patches -> append n_registers learnable register tokens (no PE) ->
    LayerNorm + GELU-MLP transformer encoder -> final LayerNorm -> pool (CLS or
    mean over patch tokens) -> linear classifier head.

    forward(x): x is (B, in_chans, H, W); returns (B, num_classes) logits. The
    encoder sequence length is recorded on `self.seq_len` after a forward.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch: int = 4,
        in_chans: int = 3,
        dim: int = 64,
        depth: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        n_registers: int = 4,
        pool: str = "cls",
    ):
        super().__init__()
        assert pool in ("cls", "mean"), "pool must be 'cls' or 'mean'"
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        self.n_registers = n_registers
        self.pool = pool
        self.seq_len = None  # set on forward, for tests/introspection

        self.patch_embed = PatchEmbed(in_chans, dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Learned absolute PE covering [CLS] + N patch tokens (not the registers).
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, n_registers, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # Classic ViT block: LayerNorm + GELU MLP + non-causal; PE added above.
        self.encoder = TransformerEncoder(
            dim, n_heads, depth, mlp_ratio=mlp_ratio, norm="layer", ffn="mlp", pos="none"
        )
        self.norm = LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def _assemble_tokens(self, patches: Tensor) -> Tensor:
        """Build the encoder input sequence from patch tokens.

        patches: (B, N, dim). Returns (B, 1 + N + n_registers, dim).

        Implement:
            1. expand self.cls_token to (B, 1, dim) and cat in front of patches
               -> (B, 1 + N, dim)
            2. add self.pos_embed (covers the CLS + patch rows)
            3. if self.n_registers > 0: expand self.register_tokens to
               (B, n_registers, dim) and cat them on the end (no PE added)
        """
        raise NotImplementedError("A2 Task 3: assemble CLS + PE + register tokens")

    def _pool(self, tokens: Tensor) -> Tensor:
        """Reduce encoder output (B, 1 + N + n_reg, dim) to (B, dim).

        "cls": the [CLS] token at index 0. "mean": the mean over the N patch
        tokens only (indices 1 .. 1 + N), excluding CLS and the register tokens.

        Implement:
            if self.pool == "cls": return tokens[:, 0]
            else: return the mean over tokens[:, 1 : 1 + self.n_patches]
        """
        raise NotImplementedError("A2 Task 4: implement _pool (cls vs mean)")

    def forward(self, x: Tensor) -> Tensor:
        patches = self.patch_embed(x)        # (B, N, dim)
        tokens = self._assemble_tokens(patches)
        self.seq_len = tokens.shape[1]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        rep = self._pool(tokens)
        return self.head(rep)
