"""Reference ViT: patch embedding, CLS + register tokens, learned PE, pooling.

The ViT is assignment-local glue, not part of the shared library: it chains the
A1 transformer encoder (classic ViT config: LayerNorm + GELU MLP + absolute PE,
non-causal) over image patch tokens. The taught mechanisms are the patch
tokenizer, the sequence assembly (CLS + PE + register tokens), the pooling choice,
and the bicubic PE interpolation. Everything else is provided plumbing.

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
        x = self.proj(x)               # (B, dim, H/p, W/p)
        x = x.flatten(2)               # (B, dim, N)
        x = x.transpose(1, 2)          # (B, N, dim)
        return x


def interpolate_pos_embed(pos_embed: Tensor, old_grid: int, new_grid: int) -> Tensor:
    """Bicubically resize the patch part of a learned PE table to a new grid.

    The table is (1, 1 + old_grid^2, dim): row 0 is the [CLS] positional vector,
    the remaining old_grid^2 rows are the patch positions in row-major order. The
    CLS row is kept unchanged; the patch rows are reshaped to an
    (old_grid, old_grid) spatial map per channel and resized to
    (new_grid, new_grid) with bicubic interpolation. This is the trick that lets a
    ViT trained at one resolution run at another.

    Returns a (1, 1 + new_grid^2, dim) table.
    """
    dim = pos_embed.shape[-1]
    cls_pe = pos_embed[:, :1]                      # (1, 1, dim)
    patch_pe = pos_embed[:, 1:]                    # (1, old_grid^2, dim)
    # (1, old_grid, old_grid, dim) -> (1, dim, old_grid, old_grid) for interpolate.
    patch_pe = patch_pe.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(
        patch_pe, size=(new_grid, new_grid), mode="bicubic", align_corners=False
    )
    # back to (1, new_grid^2, dim)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
    return torch.cat([cls_pe, patch_pe], dim=1)


class ViT(nn.Module):
    """Vision transformer assembled from the A1 transformer encoder.

    Pipeline: patch embed -> prepend [CLS] -> add learned absolute PE over
    [CLS] + patches -> append n_registers learnable register tokens (no PE) ->
    LayerNorm + GELU-MLP transformer encoder -> final LayerNorm -> pool (CLS or
    mean over patch tokens) -> linear classifier head.

    Args:
        img_size, patch, in_chans, dim, depth, n_heads, mlp_ratio: standard ViT
            sizes. num_classes: classifier outputs. n_registers: register tokens
            (Darcet et al., 2024). pool: "cls" or "mean".

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

        patches: (B, N, dim). Prepend [CLS], add the learned PE to the
        [CLS] + patch rows, then append the register tokens (which get no PE).
        Returns (B, 1 + N + n_registers, dim).
        """
        B = patches.shape[0]
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, dim)
        tokens = torch.cat([cls, patches], dim=1)       # (B, 1 + N, dim)
        tokens = tokens + self.pos_embed                # add PE to CLS + patches
        if self.n_registers > 0:
            regs = self.register_tokens.expand(B, -1, -1)  # (B, n_reg, dim) no PE
            tokens = torch.cat([tokens, regs], dim=1)
        return tokens

    def _pool(self, tokens: Tensor) -> Tensor:
        """Reduce encoder output (B, 1 + N + n_reg, dim) to (B, dim).

        "cls": the [CLS] token at index 0. "mean": the mean over the N patch
        tokens only (indices 1 .. 1 + N), excluding CLS and the register tokens.
        """
        if self.pool == "cls":
            return tokens[:, 0]
        patch_tokens = tokens[:, 1 : 1 + self.n_patches]
        return patch_tokens.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        patches = self.patch_embed(x)        # (B, N, dim)
        tokens = self._assemble_tokens(patches)
        self.seq_len = tokens.shape[1]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        rep = self._pool(tokens)
        return self.head(rep)
