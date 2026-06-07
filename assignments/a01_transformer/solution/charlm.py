"""Decoder-only character LM assembled from the A1 transformer (reference).

Thin wrapper: token embedding -> stack of causal RoPE/RMSNorm/SwiGLU blocks ->
final norm -> linear projection to vocab logits. RoPE injects position inside
attention, so there is no separate positional-encoding module. The transformer
pieces import from the package; this assembly is local assignment glue, not part
of the shared-library surface.
"""

import torch
from torch import Tensor, nn

from primitives import RMSNorm
from transformer import TransformerDecoder


class CharLM(nn.Module):
    """Decoder-only LM. forward(idx): (B, S) ids -> (B, S, vocab_size) logits."""

    def __init__(self, vocab_size: int, dim: int = 64, n_heads: int = 4,
                 depth: int = 2, n_kv_heads=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.decoder = TransformerDecoder(dim, n_heads, depth, pos="rope",
                                          norm="rms", ffn="swiglu", n_kv_heads=n_kv_heads)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        x = self.embed(idx)
        x = self.decoder(x)
        x = self.norm(x)
        return self.head(x)
