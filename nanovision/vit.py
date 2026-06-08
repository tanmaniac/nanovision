"""The vision transformer, sourced from A2 where the student builds it.

Loaded from assignments/a02_vit/vit.py (or solution/ under NANOVISION_IMPL=solution)
through nanovision/_student.py. Import as `from nanovision.vit import ViT`. Downstream
assignments (the VLM, detection/segmentation, the AV BEV stack) reuse it as a frozen or
fine-tuned image encoder; `ViT.forward_features` returns the per-patch grid (B, n_patches,
dim) those models consume, while `ViT.forward` returns classification logits.
"""

from nanovision._student import load

_m = load("a02_vit", "vit")

ViT = _m.ViT
PatchEmbed = _m.PatchEmbed

__all__ = ["ViT", "PatchEmbed"]
