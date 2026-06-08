"""Shared primitives, sourced from the assignments where the student builds them.

This module holds no implementations. Each symbol is loaded from the assignment
that owns it through nanovision/_student.py (which honors NANOVISION_IMPL):

    gelu, LayerNorm, MLP  -> A0   (assignments/a00_harness/primitives.py)
    RMSNorm, SwiGLU       -> A1   (assignments/a01_transformer/primitives.py)
    ConvNeXtBlock         -> A2   (assignments/a02_vit/convnext.py)

Import these as `from nanovision.primitives import LayerNorm`, etc. Do not import
the owning files by bare name; that would create a second module identity.
"""

from nanovision._student import load

_a0 = load("a00_harness", "primitives")
gelu = _a0.gelu
LayerNorm = _a0.LayerNorm
MLP = _a0.MLP

_a1 = load("a01_transformer", "primitives")
RMSNorm = _a1.RMSNorm
SwiGLU = _a1.SwiGLU

_a2 = load("a02_vit", "convnext")
ConvNeXtBlock = _a2.ConvNeXtBlock

__all__ = ["gelu", "LayerNorm", "MLP", "RMSNorm", "SwiGLU", "ConvNeXtBlock"]
