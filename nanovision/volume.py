"""Volume rendering and ray generation, sourced from A9 where the student builds them.

Loaded from assignments/a09_nerf/render.py and rays.py (or solution/ under
NANOVISION_IMPL=solution) through nanovision/_student.py. Import as
`from nanovision.volume import volume_render`. The discretized emission-absorption renderer
and the pinhole ray generation are shared OWNED infrastructure: A10 (Gaussian splatting)
reuses the front-to-back alpha-compositing, and A11.5d (occupancy / neural SDF) reuses the
whole ray + render stack with the density swapped for an SDF-derived one.
"""

from nanovision._student import load

_render = load("a09_nerf", "render")
_rays = load("a09_nerf", "rays")

volume_render = _render.volume_render
stratified_sample_rays = _rays.stratified_sample_rays
sample_along_rays = _rays.sample_along_rays
deltas_from_z = _rays.deltas_from_z

__all__ = [
    "volume_render",
    "stratified_sample_rays",
    "sample_along_rays",
    "deltas_from_z",
]
