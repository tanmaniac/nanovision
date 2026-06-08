"""Transformer blocks and positional schemes, sourced from A1.

Loaded from assignments/a01_transformer/transformer.py (or solution/ under
NANOVISION_IMPL=solution) through nanovision/_student.py. Import as
`from nanovision.transformer import TransformerEncoder`, etc.
"""

from nanovision._student import load

_m = load("a01_transformer", "transformer")

build_causal_mask = _m.build_causal_mask
apply_rope = _m.apply_rope
SinusoidalPositionalEncoding = _m.SinusoidalPositionalEncoding
LearnedPositionalEncoding = _m.LearnedPositionalEncoding
_RoPEAttention = _m._RoPEAttention
TransformerBlock = _m.TransformerBlock
TransformerEncoder = _m.TransformerEncoder
TransformerDecoder = _m.TransformerDecoder

__all__ = [
    "build_causal_mask",
    "apply_rope",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "_RoPEAttention",
    "TransformerBlock",
    "TransformerEncoder",
    "TransformerDecoder",
]
