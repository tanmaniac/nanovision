"""Transformer blocks and positional schemes (A1) plus the tubelet embed (A3.5).

Most symbols are sourced from assignments/a01_transformer/transformer.py;
TubeletEmbedding is sourced from assignments/a03_5_video/tubelet.py. Both go through
nanovision/_student.py (or solution/ under NANOVISION_IMPL=solution). Import as
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

# Spatiotemporal tubelet embed for video, built in A3.5.
_v = load("a03_5_video", "tubelet")
TubeletEmbedding = _v.TubeletEmbedding

__all__ = [
    "build_causal_mask",
    "apply_rope",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "_RoPEAttention",
    "TransformerBlock",
    "TransformerEncoder",
    "TransformerDecoder",
    "TubeletEmbedding",
]
