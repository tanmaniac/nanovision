"""Synthetic point clouds for ICP. Provided.

`terrain_cloud` is a bumpy surface with analytic normals - locally planar with normals that
vary across the cloud, which is exactly the setting where point-to-plane beats point-to-point.
`planar_cloud` is a flat sheet (all normals equal), the degenerate case that exercises the
reflection-avoiding determinant correction in the point-to-point solve. `make_pair` applies a
known transform to a cloud to produce a source/target pair whose true alignment is known.
"""

import numpy as np

from _helpers import rng, rotvec_to_R, make_transform, apply_transform, transform_normals
import config as C


def terrain_cloud(seed=0, n=None):
    """A bumpy height field z = a sin(fx x) sin(fy y) over a square, with exact unit normals
    from the surface gradient. Returns (points N x 3, normals N x 3)."""
    gen = rng(seed)
    n = n if n is not None else C.N_POINTS
    a, fx, fy = 0.6, 1.1, 0.9
    x = gen.uniform(-3, 3, n)
    y = gen.uniform(-3, 3, n)
    z = a * np.sin(fx * x) * np.sin(fy * y)
    pts = np.stack([x, y, z], axis=1)
    # Normal proportional to (-dz/dx, -dz/dy, 1).
    dzdx = a * fx * np.cos(fx * x) * np.sin(fy * y)
    dzdy = a * fy * np.sin(fx * x) * np.cos(fy * y)
    nrm = np.stack([-dzdx, -dzdy, np.ones_like(x)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm


def planar_cloud(seed=0, n=None):
    """A flat sheet at z = 0 (all normals = +z). The degenerate case for point-to-point."""
    gen = rng(seed)
    n = n if n is not None else C.N_POINTS
    x = gen.uniform(-3, 3, n)
    y = gen.uniform(-3, 3, n)
    pts = np.stack([x, y, np.zeros_like(x)], axis=1)
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    return pts, nrm


def make_pair(points, normals, rotvec=None, trans=None):
    """Build a source/target ICP pair from a single cloud. The target is the cloud itself;
    the source is the cloud moved by a known transform G (source = G applied to target). The
    true alignment (source -> target) is G^{-1}. Returns a dict with source, target, the
    target normals, and the ground-truth alignment T_align (= G^{-1})."""
    R = rotvec_to_R(C.PERTURB_ROTVEC if rotvec is None else rotvec)
    t = (C.PERTURB_TRANS if trans is None else np.asarray(trans))
    G = make_transform(R, t)
    source = apply_transform(G, points)
    T_align = np.linalg.inv(G)  # maps source back onto the target
    return {
        "source": source,
        "target": points,
        "target_normals": normals,
        "T_align": T_align,
    }
