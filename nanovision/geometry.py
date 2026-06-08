"""Camera geometry and BEV (A11.5a), plus pointmap/depth utilities (A10.5).

The camera primitives are sourced from assignments/a11_5a_camera_geometry_bev/geometry.py and
the DUSt3R-style pointmap utilities from assignments/a10_5_geometry_fm/geometry_fm.py (or each
assignment's solution/ under NANOVISION_IMPL=solution), through nanovision/_student.py. Import as
`from nanovision.geometry import project_points, depth_to_pointmap`, etc.
"""

from nanovision._student import load

_m = load("a11_5a_camera_geometry_bev", "geometry")

project_points = _m.project_points
unproject = _m.unproject
make_transform = _m.make_transform
apply_transform = _m.apply_transform
invert_transform = _m.invert_transform
compose_transforms = _m.compose_transforms
BEVGrid = _m.BEVGrid
CameraRig = _m.CameraRig
ipm_to_bev = _m.ipm_to_bev

# Pointmap / depth utilities, owned by A10.5 (DUSt3R-style geometry foundation models).
_pm = load("a10_5_geometry_fm", "geometry_fm")
depth_to_pointmap = _pm.depth_to_pointmap
pointmap_to_depth = _pm.pointmap_to_depth
reproject_pointmap = _pm.reproject_pointmap

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
    "depth_to_pointmap",
    "pointmap_to_depth",
    "reproject_pointmap",
]
