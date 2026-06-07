"""Reference solution: the canonical implementation lives in the shared library."""

from nanovision.transformer import (  # noqa: F401
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    TransformerDecoder,
    TransformerEncoder,
    apply_rope,
    build_causal_mask,
)
