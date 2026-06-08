"""Reference: video MAE with tube masking.

The two taught mechanisms (Tasks 2-3) are `tube_masking` and `video_mae_loss`. The
decoder-side reassembly (`_append_mask_tokens`) and the `VideoMAE` module that chains
everything through the provided video ViT encoder and a light decoder are given as
plumbing, so the new content is the tubelet embed (Task 1, in tubelet.py) and the
structured tube mask.

Shapes: tubelet tokens are (B, N, D) with N = T' * S' (S' = spatial tokens); the
encoder sees only the visible tubelets (B, n_keep, D); the decoder sees the full grid
(B, N, D_dec); the reconstruction target and prediction are (B, N, t*p*p*C).
"""

import torch
from torch import Tensor, nn

from backbone import VideoViTEncoder, per_tubelet_normalize, tubeletify
from nanovision.primitives import LayerNorm
from nanovision.transformer import TransformerEncoder


def tube_masking(x: Tensor, t_prime: int, mask_ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """Tube masking: one spatial keep set, shared across all temporal steps (Task 2).

    x: (B, N, D) tubelet tokens, N = t_prime * S' in temporal-outermost order
    (idx = t' * S' + s). Draw one spatial keep set per sample of size
    n_keep_spatial = round((1 - mask_ratio) * S') and apply the SAME spatial pattern to
    every temporal step, so the visible tokens form spatiotemporal tubes. This is not
    A3's per-token argsort shuffle; the visible set is structured, so the indices are
    built explicitly so the append/unshuffle reassembly still works.

    Returns:
        x_kept: (B, n_keep, D) visible tubelets, n_keep = t_prime * n_keep_spatial.
        mask: (B, N) binary in original token order, 0 = kept, 1 = masked.
        ids_restore: (B, N) inverse of the [keep; drop] index ordering.
    """
    B, N, D = x.shape
    tp = t_prime
    S = N // tp
    n_keep_spatial = round((1 - mask_ratio) * S)

    # Per-sample random spatial permutation; the first n_keep_spatial are the tube.
    noise = torch.rand(B, S, device=x.device)
    ids_spatial = torch.argsort(noise, dim=1)          # (B, S)
    keep_s = ids_spatial[:, :n_keep_spatial]           # (B, n_keep_spatial)
    drop_s = ids_spatial[:, n_keep_spatial:]           # (B, S - n_keep_spatial)

    # Lift spatial indices to full-token indices for every temporal step (idx = t*S + s).
    t_off = (torch.arange(tp, device=x.device).view(1, tp, 1)) * S   # (1, tp, 1)
    ids_keep = (t_off + keep_s.unsqueeze(1)).reshape(B, -1)          # (B, tp*n_keep_spatial)
    ids_drop = (t_off + drop_s.unsqueeze(1)).reshape(B, -1)
    ids_shuffle = torch.cat([ids_keep, ids_drop], dim=1)            # (B, N)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    n_keep = ids_keep.shape[1]
    x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    # Mask in shuffled order is [0]*n_keep + [1]*(N-n_keep); unshuffle to original.
    mask = torch.ones(B, N, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_kept, mask, ids_restore


def video_mae_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """MSE on masked tubelets only, per-tubelet-normalized target (Task 3).

    pred, target: (B, N, t*p*p*C). target is the per-tubelet-normalized pixels. mask:
    (B, N) with 1 on masked tubelets. This is A3's masked-patch MSE with the patch
    enlarged from p*p to t*p*p: mean squared error per tubelet (mean over the pixel
    dim), averaged over the masked tubelets only.
    """
    per_tubelet = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
    return (per_tubelet * mask).sum() / mask.sum()


def _append_mask_tokens(x_enc: Tensor, ids_restore: Tensor, mask_token: Tensor) -> Tensor:
    """Append shared mask tokens and unshuffle to grid order (provided, as in A3).

    Identical contract to A3's append_mask_tokens: broadcast mask_token to the masked
    slots, concatenate [visible; mask] in shuffled order, gather by ids_restore so
    every position holds its encoded visible token or a mask token in original order.
    """
    B, n_keep, D = x_enc.shape
    N = ids_restore.shape[1]
    n_mask = N - n_keep
    masks = mask_token.expand(B, n_mask, D)
    x_full = torch.cat([x_enc, masks], dim=1)
    x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
    return x_full


class VideoMAE(nn.Module):
    """Asymmetric video masked autoencoder over tubelet tokens (provided plumbing).

    Pipeline: tubelet embed (B, N, enc_dim) + encoder PE -> tube_masking keeps the
    visible tubes -> encoder runs on the visible tokens only -> project to decoder dim
    -> append mask tokens, unshuffle to (B, N, dec_dim) -> add decoder PE -> decoder
    -> linear predictor to (B, N, t*p*p*C). The target is the per-tubelet-normalized
    pixels; the loss is on masked tubelets only.

    forward(video): video is (B, C, T, H, W). Returns (loss, pred, mask) with
        pred (B, N, t*p*p*C), mask (B, N), and loss a scalar.
    """

    def __init__(self, img_size: int = 16, patch: int = 4, tubelet_t: int = 2,
                 n_frames: int = 6, in_chans: int = 3, enc_dim: int = 64,
                 enc_depth: int = 4, enc_heads: int = 4, dec_dim: int = 48,
                 dec_depth: int = 2, dec_heads: int = 4, mlp_ratio: float = 4.0,
                 mask_ratio: float = 0.875):
        super().__init__()
        self.patch = patch
        self.tubelet_t = tubelet_t
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.encoder = VideoViTEncoder(img_size, patch, tubelet_t, n_frames, in_chans,
                                       enc_dim, enc_depth, enc_heads, mlp_ratio)
        self.t_prime = self.encoder.t_prime
        self.grid = self.encoder.grid
        self.n_tokens = self.encoder.n_tokens
        tubelet_dim = tubelet_t * patch * patch * in_chans

        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dec_dim))
        self.decoder = TransformerEncoder(
            dec_dim, dec_heads, dec_depth, mlp_ratio=mlp_ratio,
            norm="layer", ffn="mlp", pos="none",
        )
        self.dec_norm = LayerNorm(dec_dim)
        self.predictor = nn.Linear(dec_dim, tubelet_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

    def forward_encoder(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.encoder.embed(video) + self.encoder.pos_embed   # (B, N, enc_dim)
        x_kept, mask, ids_restore = tube_masking(tokens, self.t_prime, self.mask_ratio)
        x_enc = self.encoder.forward_tokens(x_kept)                    # (B, n_keep, enc_dim)
        return x_enc, mask, ids_restore

    def forward_decoder(self, x_enc: Tensor, ids_restore: Tensor) -> Tensor:
        x = self.enc_to_dec(x_enc)                                     # (B, n_keep, dec_dim)
        x = _append_mask_tokens(x, ids_restore, self.mask_token)       # (B, N, dec_dim)
        x = x + self.dec_pos_embed
        x = self.dec_norm(self.decoder(x))
        return self.predictor(x)                                       # (B, N, t*p*p*C)

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_enc, mask, ids_restore = self.forward_encoder(video)
        pred = self.forward_decoder(x_enc, ids_restore)
        target = per_tubelet_normalize(tubeletify(video, self.tubelet_t, self.patch))
        loss = video_mae_loss(pred, target, mask)
        return loss, pred, mask
