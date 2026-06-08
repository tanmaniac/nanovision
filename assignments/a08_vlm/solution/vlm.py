"""The LLaVA-style bridge: frozen ViT -> projector -> prepend to text -> decoder-only LM.

The VLM turns an image into a sequence of tokens that live in the LM's embedding space and
prepends them to the caption's text embeddings, then runs a causal decoder over the combined
sequence and predicts the caption autoregressively. The two holes are the interface itself:
prepend_visual (where the visual tokens go) and vlm_loss (which positions are supervised).

The reused decoder-only LM (nanovision.transformer.TransformerDecoder) operates in embedding
space and uses sequence-order RoPE applied inside attention. Visual tokens get positions
0..N-1 and text follows at N.., so train-time and generation-time positions agree. The LM
here is randomly initialized; it carries no language prior. The lesson is the visual-token
interface and the freeze curriculum, not exploiting pretrained linguistic knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanovision.transformer import TransformerDecoder, build_causal_mask
from nanovision.vit import ViT

from projector import MLPProjector
from resampler import PerceiverResampler


def prepend_visual(visual_tokens: Tensor, text_embeds: Tensor) -> Tensor:
    """Concatenate visual then text along the sequence axis.

    visual_tokens (B, N, d) + text_embeds (B, L, d) -> (B, N + L, d). The order is fixed:
    visual tokens occupy positions 0..N-1, text follows at N.. This is the LLaVA injection
    point - the visual tokens are ordinary positions in the LM's context, no LM change.
    """
    return torch.cat([visual_tokens, text_embeds], dim=1)


def vlm_loss(logits: Tensor, token_ids: Tensor, n_visual: int) -> Tensor:
    """Next-token cross-entropy supervised ONLY on the text positions.

    logits (B, N+L, V) over the combined sequence [visual_0..visual_{N-1}, text_0..text_{L-1}].
    token_ids (B, L) are the caption tokens (0 = pad). n_visual = N.

    Teacher forcing predicts position i+1 from positions <= i. The supervised targets are
    the text tokens: the last visual position predicts text_0 (the class token), text_k
    predicts text_{k+1}, and pad targets are ignored. No BOS or separator is used, because
    generation also starts from an empty text side, so train and inference agree.

    Build a label tensor of length N+L:
      1. fill (B, N+L) with -100 (ignore index),
      2. write the true tokens into the text slice: labels[:, N:N+L] = token_ids,
      3. re-mask pads WITHIN that slice: labels[:, N:N+L][token_ids == 0] = -100.
    Do NOT index labels[token_ids == 0] directly - labels has length N+L and token_ids has
    length L, so the boolean mask would mismatch. Then shift and reduce:
      F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                      ignore_index=-100).
    """
    B, _, V = logits.shape
    N = n_visual
    L = token_ids.shape[1]
    labels = torch.full((B, N + L), -100, dtype=torch.long, device=logits.device)
    labels[:, N:N + L] = token_ids
    labels[:, N:N + L][token_ids == 0] = -100
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100
    )


class VLM(nn.Module):
    """Frozen ViT + projector + token embedding + causal decoder LM + output head.

    forward(img, token_ids) returns (logits (B, N+L, V), n_visual). The ViT runs through
    forward_features to get the per-patch grid; the projector maps it into the LM embedding
    dim; prepend_visual puts the visual tokens before the text token embeddings; the decoder
    runs over the combined sequence under a causal mask; the head maps the final hidden
    states to vocabulary logits.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vit = ViT(
            img_size=cfg.img_size, patch=cfg.patch, in_chans=cfg.in_chans,
            dim=cfg.vit_dim, depth=cfg.vit_depth, n_heads=cfg.vit_heads,
            num_classes=cfg.vocab_size, n_registers=cfg.n_registers, pool="mean",
        )
        self.vit.requires_grad_(False)

        if cfg.connector == "resampler":
            self.connector = PerceiverResampler(cfg.vit_dim, cfg.dim_l, cfg.n_queries, cfg.lm_heads)
            self.n_visual = cfg.n_queries
        else:
            self.connector = MLPProjector(cfg.vit_dim, cfg.dim_l)
            self.n_visual = cfg.n_patches

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim_l)
        self.decoder = TransformerDecoder(cfg.dim_l, cfg.lm_heads, cfg.lm_depth, cross_attn=False)
        self.norm = nn.LayerNorm(cfg.dim_l)
        self.head = nn.Linear(cfg.dim_l, cfg.vocab_size)

    def _seq_len(self, n_text: int) -> int:
        return self.n_visual + n_text

    def forward(self, img: Tensor, token_ids: Tensor) -> tuple[Tensor, int]:
        feats = self.vit.forward_features(img)          # (B, n_patches, vit_dim), grads off
        vis = self.connector(feats)                     # (B, N, dim_l)
        txt = self.embed(token_ids)                     # (B, L, dim_l)
        seq = prepend_visual(vis, txt)                  # (B, N+L, dim_l)
        mask = build_causal_mask(seq.shape[1]).to(seq.device)
        h = self.decoder(seq, mask=mask)
        logits = self.head(self.norm(h))                # (B, N+L, V)
        return logits, self.n_visual

    def set_stage(self, stage: int) -> None:
        """Toggle requires_grad for the two-stage freeze curriculum.

        Stage 1: only the projector/connector, the token embedding, and the output head
        train; the ViT and the decoder are frozen. Stage 2: the decoder also unfreezes; the
        ViT stays frozen. Real LLaVA freezes the LM's embedding and unembedding in stage 1
        and moves only the projector, but with a randomly initialized decoder there is no
        fixed embedding geometry to map visual features into, so the embedding and head must
        train here too - a course-scale deviation.
        """
        assert stage in (1, 2)
        self.vit.requires_grad_(False)                  # ViT always frozen
        self.connector.requires_grad_(True)
        self.embed.requires_grad_(True)
        self.head.requires_grad_(True)
        self.norm.requires_grad_(True)
        self.decoder.requires_grad_(stage == 2)

    @torch.no_grad()
    def generate(self, img: Tensor, max_new_tokens: int) -> Tensor:
        """Greedy autoregressive decode starting from an empty text side.

        Prepend the visual tokens, then argmax one token at a time and append it to the text
        side. Returns the generated token ids (B, max_new_tokens). No BOS: the first text
        token is predicted from the last visual position, matching the training shift.
        """
        self.eval()
        feats = self.vit.forward_features(img)
        vis = self.connector(feats)                     # (B, N, dim_l)
        B = vis.shape[0]
        generated = torch.zeros(B, 0, dtype=torch.long, device=img.device)
        for _ in range(max_new_tokens):
            if generated.shape[1] == 0:
                seq = vis
            else:
                seq = prepend_visual(vis, self.embed(generated))
            mask = build_causal_mask(seq.shape[1]).to(seq.device)
            h = self.decoder(seq, mask=mask)
            logits = self.head(self.norm(h))            # (B, S, V)
            next_tok = logits[:, -1].argmax(dim=-1)     # predict from the last position
            generated = torch.cat([generated, next_tok[:, None]], dim=1)
        return generated
