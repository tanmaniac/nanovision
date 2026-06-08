"""The time-embedded U-Net, sourced from A5.

Loaded from assignments/a05_diffusion/unet.py (or solution/ under NANOVISION_IMPL=solution)
through nanovision/_student.py. Import as `from nanovision.unet import TimeEmbeddedUNet`.
A6 reuses this backbone for the image-scale flow-matching demo (swap the diffusion
objective for the CFM objective), which runs in solution mode where A5's holes are filled.
"""

from nanovision._student import load

_m = load("a05_diffusion", "unet")

TimeEmbeddedUNet = _m.TimeEmbeddedUNet
timestep_embedding = _m.timestep_embedding

__all__ = ["TimeEmbeddedUNet", "timestep_embedding"]
