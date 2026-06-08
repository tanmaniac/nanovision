"""Vector quantization, sourced from A6.5.

Loaded from assignments/a06_5_vq_tokenizer/quantize.py (or solution/ under
NANOVISION_IMPL=solution) through nanovision/_student.py. Import as
`from nanovision.quantize import VectorQuantizer`. This is the shared codebook +
straight-through estimator used by the VQ-VAE and any downstream discrete tokenizer.
"""

from nanovision._student import load

_m = load("a06_5_vq_tokenizer", "quantize")

VectorQuantizer = _m.VectorQuantizer
codebook_perplexity = _m.codebook_perplexity

__all__ = ["VectorQuantizer", "codebook_perplexity"]
