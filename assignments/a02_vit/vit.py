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

    The projection is the provided Conv2d (kernel_size = stride = patch); the hole
    turns its spatial output grid into a token sequence.

    forward(x): x is (B, in_chans, H, W); output is (B, N, dim) with
    N = (H / patch) * (W / patch).

    See the patch embedding section of the README.
    """

    def __init__(self, in_chans: int, dim: int, patch: int):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_chans, H, W) -> (B, N, dim).

        See the patch embedding section of the README.
        """
        batch_size = x.shape[0]
        y = self.proj(x).reshape((batch_size, self.dim, -1)).permute((0, 2, 1))
        return y


def interpolate_pos_embed(pos_embed: Tensor, old_grid: int, new_grid: int) -> Tensor:
    """Bicubically resize the patch part of a learned PE table to a new grid.

    The table is (1, 1 + old_grid^2, dim): row 0 is the [CLS] positional vector,
    the remaining old_grid^2 rows are the patch positions in row-major order. The
    CLS row is kept unchanged; only the patch rows are resized, from an old_grid x
    old_grid spatial map to new_grid x new_grid. This is the trick that lets a ViT
    trained at one resolution run at another.

    Returns a (1, 1 + new_grid^2, dim) table.

    See the positional-embedding interpolation section of the README.
    """
    b, _, d = pos_embed.shape
    spatial = pos_embed[:, 1 : 1 + old_grid**2]
    spatial = spatial.reshape([b, old_grid, old_grid, d]).permute([0, 3, 1, 2])
    spatial = F.interpolate(spatial, (new_grid, new_grid), mode="bicubic")
    out = spatial.permute([0, 2, 3, 1]).reshape([b, -1, d])
    out = torch.concat([pos_embed[:, 0].unsqueeze(1), out], dim=1)
    return out


class ViT(nn.Module):
    """Vision transformer assembled from the transformer encoder.

    Patch tokens plus a prepended [CLS] token, a learned absolute positional
    embedding, and appended register tokens are run through the encoder and pooled
    to one vector per image for a linear classifier head. The construction and the
    forward plumbing are given; the holes are the token assembly and the pooling.
    Pooling is selected by the `pool` option ("cls" or "mean").

    forward(x): x is (B, in_chans, H, W); returns (B, num_classes) logits. The
    encoder sequence length is recorded on `self.seq_len` after a forward.

    See the token sequence and pooling sections of the README.
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
            dim,
            n_heads,
            depth,
            mlp_ratio=mlp_ratio,
            norm="layer",
            ffn="mlp",
            pos="none",
        )
        self.norm = LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def _assemble_tokens(self, patches: Tensor) -> Tensor:
        """Build the encoder input sequence from patch tokens.

        patches: (B, N, dim). Returns (B, 1 + N + n_registers, dim). The positional
        embedding covers the [CLS] and patch rows only; the register tokens (appended
        when n_registers > 0) get no positional embedding.

        See the token sequence section of the README.
        """
        b = patches.shape[0]
        y = torch.cat([patches, self.cls_token.expand([b, -1, -1])], dim=1)
        y = torch.cat(
            [y + self.pos_embed, self.register_tokens.expand([b, -1, -1])], dim=1
        )
        return y

    def _pool(self, tokens: Tensor) -> Tensor:
        """Reduce encoder output (B, 1 + N + n_reg, dim) to (B, dim).

        "cls": the [CLS] token at index 0. "mean": the mean over the N patch
        tokens only (indices 1 .. 1 + N), excluding CLS and the register tokens.

        See the pooling section of the README.
        """
        if self.pool == "cls":
            return tokens[:, 0]
        else:
            return torch.mean(tokens[:, 1 : 1 + self.n_patches], dim=1)

    def forward_features(self, x: Tensor) -> Tensor:
        """Patch-grid features for downstream use (VLM visual tokens, dense prediction).

        Runs the patch embed, encoder, and final norm, then returns only the N patch
        tokens (dropping the [CLS] at index 0 and any register tokens): (B, n_patches,
        dim). Unlike forward, it stops before pooling and the classification head, so a
        downstream model gets one feature vector per image patch. Does not touch _pool,
        so it works whether or not _pool is implemented.
        """
        patches = self.patch_embed(x)
        tokens = self._assemble_tokens(patches)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 1 : 1 + self.n_patches]

    def forward(self, x: Tensor) -> Tensor:
        patches = self.patch_embed(x)  # (B, N, dim)
        tokens = self._assemble_tokens(patches)
        self.seq_len = tokens.shape[1]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        rep = self._pool(tokens)
        return self.head(rep)
