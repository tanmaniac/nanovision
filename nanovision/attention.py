"""Attention, sourced from A1 where the student builds it.

`scaled_dot_product_attention` and `MultiHeadAttention` are loaded from
assignments/a01_transformer/attention.py (or solution/ under NANOVISION_IMPL=solution)
through nanovision/_student.py. Import them as
`from nanovision.attention import MultiHeadAttention`. Every later assignment (ViT,
SSL, CLIP, DiT, BEVFormer, VLM) reuses this.
"""

from nanovision._student import load

_m = load("a01_transformer", "attention")
scaled_dot_product_attention = _m.scaled_dot_product_attention
MultiHeadAttention = _m.MultiHeadAttention

__all__ = ["scaled_dot_product_attention", "MultiHeadAttention"]
