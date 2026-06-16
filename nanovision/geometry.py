"""Camera geometry primitives with three owners, re-exported under one import path.

The base pinhole model and SE(3) toolkit (project_points, unproject, make_transform,
apply_transform, invert_transform, compose_transforms) are sourced from
assignments/a09_nerf/geometry.py; the autonomous-driving objects (BEVGrid, CameraRig,
ipm_to_bev) from assignments/a11_5a_camera_geometry_bev/geometry.py; and the DUSt3R-style
pointmap utilities from assignments/a10_5_geometry_fm/geometry_fm.py (or each assignment's
solution/ under NANOVISION_IMPL=solution), through nanovision/_student.py. Import as
`from nanovision.geometry import project_points, depth_to_pointmap`, etc.

Load order matters: the a11_5a and a10_5 modules import the base primitives back through this
shim at import time, so the base primitives must be assigned on this module before those
modules are loaded.
"""

from nanovision._student import load

# Base pinhole + SE(3) primitives, owned by A9 (the first 3D assignment; NeRF ray
# generation needs back-projection). Assigned first so a11_5a/a10_5 can import them
# back through this shim while they load.
_base = load("a09_nerf", "geometry")
project_points = _base.project_points
unproject = _base.unproject
make_transform = _base.make_transform
apply_transform = _base.apply_transform
invert_transform = _base.invert_transform
compose_transforms = _base.compose_transforms

# Autonomous-driving objects, owned by A11.5a (built on the base primitives above).
_m = load("a11_5a_camera_geometry_bev", "geometry")
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
