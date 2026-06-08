"""Camera geometry and BEV, sourced from A11.5a.

Loaded from assignments/a115a_camera_geometry_bev/geometry.py (or solution/ under
NANOVISION_IMPL=solution) through nanovision/_student.py. Import as
`from nanovision.geometry import project_points`, etc.
"""

from nanovision._student import load

_m = load("a115a_camera_geometry_bev", "geometry")

project_points = _m.project_points
unproject = _m.unproject
make_transform = _m.make_transform
apply_transform = _m.apply_transform
invert_transform = _m.invert_transform
compose_transforms = _m.compose_transforms
BEVGrid = _m.BEVGrid
CameraRig = _m.CameraRig
ipm_to_bev = _m.ipm_to_bev

__all__ = [
    "project_points",
    "unproject",
    "make_transform",
    "apply_transform",
    "invert_transform",
    "compose_transforms",
    "BEVGrid",
    "CameraRig",
    "ipm_to_bev",
]
