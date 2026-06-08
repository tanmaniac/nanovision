"""Lift-Splat-Shoot view transform, sourced from A11.5b where the student builds it.

Loaded from assignments/a11_5b_lift_splat_shoot/lift_splat.py (or solution/ under
NANOVISION_IMPL=solution) through nanovision/_student.py. Import as
`from nanovision.lift_splat import DepthLift, frustum_points, cumsum_pool`. The depth-lift,
frustum coordinate generation, and the sort+cumsum pooling are shared OWNED infrastructure:
A11.5d (3D occupancy) reuses the depth lift and the frustum-to-3D step without modification,
passing a 3D voxel index to cumsum_pool instead of a 2D BEV pillar index.
"""

from nanovision._student import load

_m = load("a11_5b_lift_splat_shoot", "lift_splat")

DepthLift = _m.DepthLift
frustum_points = _m.frustum_points
pillar_index = _m.pillar_index
cumsum_pool = _m.cumsum_pool
LiftSplatShoot = _m.LiftSplatShoot
bevdepth_depth_loss = _m.bevdepth_depth_loss

__all__ = [
    "DepthLift",
    "frustum_points",
    "pillar_index",
    "cumsum_pool",
    "LiftSplatShoot",
    "bevdepth_depth_loss",
]
