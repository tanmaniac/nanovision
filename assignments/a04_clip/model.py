"""Provided dual encoder for A4: a tiny image tower and text tower into a shared space.

None of this file is a hole. A4's taught mechanisms are the contrastive losses
(losses.py) and zero-shot inference (inference.py); the encoders are small stand-ins
built on the shared transformer, the same way A3 and A3.5 provide their own ViT rather
than importing a02's local vit.py. The towers pool to one (B, D) vector and L2-normalize
into a shared embedding space, with a learnable temperature (logit_scale) and a SigLIP
bias.

Text pooling follows CLIP: take the hidden state at the EOS position, found as
`tokens.argmax(dim=-1)` because EOS is the largest token id and sequences are padded, so
it is NOT the last position. This is a known bug source and is provided here. (SigLIP's
real text tower is bidirectional with a different pool; this toy uses the CLIP causal-EOS
convention for both losses.)
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanovision.primitives import LayerNorm
from nanovision.transformer import TransformerDecoder, TransformerEncoder


class ImageTower(nn.Module):
    """Tiny ViT image tower: patch embed, learned PE, encoder, mean-pool, project."""

    def __init__(self, img_size: int, patch: int, in_chans: int, dim: int, depth: int,
                 n_heads: int, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = TransformerEncoder(
            dim, n_heads, depth, mlp_ratio=mlp_ratio, norm="layer", ffn="mlp", pos="none"
        )
        self.norm = LayerNorm(dim)
        self.head = nn.Linear(dim, embed_dim)

    def forward(self, images: Tensor) -> Tensor:
        x = self.proj(images).flatten(2).transpose(1, 2)   # (B, N, dim)
        x = x + self.pos_embed
        x = self.norm(self.encoder(x))
        return self.head(x.mean(dim=1))                    # (B, embed_dim)


class TextTower(nn.Module):
    """Tiny causal text tower: token embed, learned PE, causal stack, EOS-pool, project."""

    def __init__(self, vocab_size: int, max_len: int, dim: int, depth: int, n_heads: int,
                 embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.decoder = TransformerDecoder(
            dim, n_heads, depth, mlp_ratio=mlp_ratio, norm="layer", ffn="mlp", pos="none"
        )
        self.norm = LayerNorm(dim)
        self.head = nn.Linear(dim, embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.token_embed(tokens) + self.pos_embed[:, : tokens.shape[1]]
        x = self.norm(self.decoder(x))                     # causal self-attention
        # EOS pooling: the EOS token has the largest id, so argmax finds its position.
        eos_pos = tokens.argmax(dim=-1)                    # (B,)
        pooled = x[torch.arange(x.shape[0]), eos_pos]      # (B, dim)
        return self.head(pooled)                           # (B, embed_dim)


class CLIPModel(nn.Module):
    """Dual encoder into a shared L2-normalized space, with a temperature and a bias.

    encode_image / encode_text return L2-normalized (B, embed_dim) features.
    forward(images, tokens) returns (image_features, text_features). logit_scale is a
    learnable scalar in log space; logit_scale_clamped() applies CLIP's stability clamp.
    bias is the learnable SigLIP bias.
    """

    def __init__(self, cfg):
        super().__init__()
        self.image_tower = ImageTower(cfg.img_size, cfg.patch, cfg.in_chans, cfg.img_dim,
                                      cfg.img_depth, cfg.img_heads, cfg.embed_dim, cfg.mlp_ratio)
        self.text_tower = TextTower(cfg.vocab_size, cfg.max_len, cfg.txt_dim, cfg.txt_depth,
                                    cfg.txt_heads, cfg.embed_dim, cfg.mlp_ratio)
        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.init_logit_scale)))
        self.bias = nn.Parameter(torch.tensor(float(cfg.init_bias)))
        self._logit_scale_max = float(cfg.logit_scale_max)

    def encode_image(self, images: Tensor) -> Tensor:
        return F.normalize(self.image_tower(images), dim=-1, p=2)

    def encode_text(self, tokens: Tensor) -> Tensor:
        return F.normalize(self.text_tower(tokens), dim=-1, p=2)

    def logit_scale_clamped(self) -> Tensor:
        """The log-space temperature parameter, clamped at log(100) for stability."""
        return self.logit_scale.clamp(max=self._logit_scale_max)

    def forward(self, images: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode_image(images), self.encode_text(tokens)
