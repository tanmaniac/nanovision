"""Hyperparameters for the VLM toy.

The core of A8 is the visual-token interface: take the frozen ViT's per-patch features,
project them into the language model's embedding dimension, prepend them to the text token
embeddings, and run the decoder-only LM over the combined sequence. These numbers fix the
toy. The image is (3, 16, 16); a patch of 4 gives a 4x4 = 16-patch grid, so N = 16 visual
tokens. The LM embedding dim is 64 and the vocabulary is 32 tokens (matching the toy
captioner: pad, class tokens, attribute tokens, EOS).
"""

from dataclasses import dataclass


@dataclass
class VLMConfig:
    # Image and frozen ViT encoder.
    img_size: int = 16
    patch: int = 4
    in_chans: int = 3
    vit_dim: int = 32
    vit_depth: int = 2
    vit_heads: int = 2
    n_registers: int = 0        # keep the encoder simple; no register tokens

    # Language model and projector.
    dim_l: int = 64             # LM embedding dim; the projector maps vit_dim -> dim_l
    vocab_size: int = 32
    lm_depth: int = 2
    lm_heads: int = 4

    # Toy caption layout (nanovision.data.toy.image_text_batch).
    n_classes: int = 4
    n_attrs: int = 4
    max_len: int = 8            # caption length L; the text side is (B, L)

    # Connector: "mlp" (LLaVA-1.5 2-layer MLP) or "resampler" (Perceiver miniature).
    connector: str = "mlp"
    n_queries: int = 8          # Q learned queries for the resampler

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch) ** 2
