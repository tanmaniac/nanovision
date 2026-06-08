"""BEVFormer-style query-pull view transform, sourced from A11.5c where the student builds it.

Loaded from assignments/a11_5c_bevformer/bevformer.py (or solution/ under
NANOVISION_IMPL=solution) through nanovision/_student.py. Import as
`from nanovision.bevformer import SpatialCrossAttention, bev_reference_points`. The BEV reference
points, the projection to camera coordinates, the spatial cross-attention (bilinear sample at
projected pillars), the ego-motion BEV warp, and the temporal self-attention are shared OWNED
infrastructure: A11.5d (occupancy) and A11.5e (map / prediction) consume the dense BEV feature
grid this encoder produces.
"""

from nanovision._student import load

_m = load("a11_5c_bevformer", "bevformer")

bev_reference_points = _m.bev_reference_points
project_reference_points = _m.project_reference_points
SpatialCrossAttention = _m.SpatialCrossAttention
warp_bev = _m.warp_bev
TemporalSelfAttention = _m.TemporalSelfAttention
BEVFormerEncoder = _m.BEVFormerEncoder
BEVFormerSeg = _m.BEVFormerSeg

__all__ = [
    "bev_reference_points",
    "project_reference_points",
    "SpatialCrossAttention",
    "warp_bev",
    "TemporalSelfAttention",
    "BEVFormerEncoder",
    "BEVFormerSeg",
]
