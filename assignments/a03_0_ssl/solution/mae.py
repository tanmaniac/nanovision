"""MAE: random masking, mask-token assembly, and the masked-pixel loss.

The three taught mechanisms (Tasks 1-3) are the bodies of `random_masking`,
`append_mask_tokens`, and `mae_loss`. The MAE module that chains them through the
provided encoder and a light decoder is given as plumbing.

Shapes: patch tokens are (B, N, D); the encoder sees only the visible tokens
(B, n_keep, D); the decoder sees the full grid (B, N, D_dec) after mask tokens are
appended; the reconstruction target and prediction are (B, N, p*p*C).
"""

import torch
from torch import Tensor, nn

from backbone import ViTEncoder, patchify, per_patch_normalize
from nanovision.primitives import LayerNorm
from nanovision.transformer import TransformerEncoder


def random_masking(x: Tensor, mask_ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """Per-sample random masking via the shuffle-keep-unshuffle trick (Task 1).

    x: (B, N, D) patch tokens. Keep n_keep = round((1 - mask_ratio) * N) tokens per
    sample, chosen by a random permutation. Returns:
        x_kept: (B, n_keep, D) the visible tokens, gathered in shuffled order.
        mask: (B, N) binary in ORIGINAL patch order, 0 = kept, 1 = masked.
        ids_restore: (B, N) the argsort of the shuffle, so that gathering a tensor
            in shuffled order by ids_restore puts it back into original order.

    The encoder will see only x_kept; the decoder uses ids_restore to unshuffle the
    reconstructed full set back to grid order.
    """
    B, N, D = x.shape
    n_keep = round((1 - mask_ratio) * N)

    noise = torch.rand(B, N, device=x.device)        # random score per token
    ids_shuffle = torch.argsort(noise, dim=1)        # ascending -> random permutation
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse permutation

    ids_keep = ids_shuffle[:, :n_keep]               # (B, n_keep) which to keep
    x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    # Mask in shuffled order is [0]*n_keep + [1]*(N-n_keep); unshuffle to original.
    mask = torch.ones(B, N, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_kept, mask, ids_restore


def append_mask_tokens(x_enc: Tensor, ids_restore: Tensor, mask_token: Tensor) -> Tensor:
    """Append shared mask tokens and unshuffle to grid order (Task 2).

    x_enc: (B, n_keep, D_dec) the encoded visible tokens projected to decoder dim.
    ids_restore: (B, N) the inverse permutation from random_masking.
    mask_token: (1, 1, D_dec) one shared learned embedding.

    Build the full set by broadcasting mask_token to the N - n_keep masked slots,
    concatenating [visible; mask] in shuffled order, then gathering by ids_restore
    so position i holds either its encoded visible token or a mask token, in
    original patch order. Returns (B, N, D_dec). The decoder positional embedding is
    added by the caller after this assembly.
    """
    B, n_keep, D = x_enc.shape
    N = ids_restore.shape[1]
    n_mask = N - n_keep
    masks = mask_token.expand(B, n_mask, D)          # (B, n_mask, D_dec)
    x_full = torch.cat([x_enc, masks], dim=1)        # shuffled order
    x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
    return x_full


def mae_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """MSE on masked patches only, per-patch-normalized target (Task 3).

    pred, target: (B, N, p*p*C). target is the per-patch-normalized pixels. mask:
    (B, N) with 1 on masked patches. Compute the mean squared error per patch
    (mean over the pixel dim), then average over the masked patches only, using
    mask as the weight. Visible patches do not contribute to the loss.
    """
    per_patch = ((pred - target) ** 2).mean(dim=-1)  # (B, N) MSE per patch
    return (per_patch * mask).sum() / mask.sum()


class MAE(nn.Module):
    """Asymmetric masked autoencoder over patch tokens (provided plumbing).

    Pipeline: patch embed (B, N, enc_dim) + encoder PE -> random_masking keeps
    n_keep visible tokens -> encoder runs on the visible tokens only -> project to
    decoder dim -> append_mask_tokens unshuffles to (B, N, dec_dim) -> add decoder
    PE -> decoder transformer -> linear predictor to (B, N, p*p*C). The target is
    the per-patch-normalized pixels; the loss is on masked patches only.

    forward(img): img is (B, C, H, W). Returns (loss, pred, mask) with
        pred (B, N, p*p*C), mask (B, N), and loss a scalar.
    """

    def __init__(self, img_size: int = 32, patch: int = 4, in_chans: int = 3,
                 enc_dim: int = 64, enc_depth: int = 4, enc_heads: int = 4,
                 dec_dim: int = 32, dec_depth: int = 2, dec_heads: int = 4,
                 mlp_ratio: float = 4.0, mask_ratio: float = 0.75):
        super().__init__()
        self.patch = patch
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.encoder = ViTEncoder(img_size, patch, in_chans, enc_dim, enc_depth,
                                  enc_heads, mlp_ratio)
        self.grid = self.encoder.grid
        self.n_patches = self.encoder.n_patches
        patch_dim = patch * patch * in_chans

        # Decoder: project encoded tokens to dec_dim, a shared mask token, a decoder
        # PE over the full grid, the transformer stack, and a linear pixel predictor.
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dec_dim))
        self.decoder = TransformerEncoder(
            dec_dim, dec_heads, dec_depth, mlp_ratio=mlp_ratio,
            norm="layer", ffn="mlp", pos="none"
        )
        self.dec_norm = LayerNorm(dec_dim)
        self.predictor = nn.Linear(dec_dim, patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

    def forward_encoder(self, img: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.encoder.embed(img) + self.encoder.pos_embed  # (B, N, enc_dim)
        x_kept, mask, ids_restore = random_masking(tokens, self.mask_ratio)
        x_enc = self.encoder.forward_tokens(x_kept)                # (B, n_keep, enc_dim)
        return x_enc, mask, ids_restore

    def forward_decoder(self, x_enc: Tensor, ids_restore: Tensor) -> Tensor:
        x = self.enc_to_dec(x_enc)                                 # (B, n_keep, dec_dim)
        x = append_mask_tokens(x, ids_restore, self.mask_token)    # (B, N, dec_dim)
        x = x + self.dec_pos_embed
        x = self.dec_norm(self.decoder(x))
        return self.predictor(x)                                   # (B, N, p*p*C)

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_enc, mask, ids_restore = self.forward_encoder(img)
        pred = self.forward_decoder(x_enc, ids_restore)
        target = per_patch_normalize(patchify(img, self.patch))
        loss = mae_loss(pred, target, mask)
        return loss, pred, mask
