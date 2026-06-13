"""Point-to-plane Gauss-Newton step: iterating the linearized step on matched
correspondences recovers a known transform, confirming the linearization (the [n; p x n]
Jacobian and the right step direction)."""

import numpy as np

from _impl import point_to_plane_step
from _helpers import transform_error, apply_transform
import sim


def test_step_iterates_to_known_transform():
    pts, nrm = sim.terrain_cloud(seed=1)
    # A modest known motion so the small-angle linearization is valid; correspondences are
    # matched 1:1 (no nearest-neighbor step), isolating the solver from association.
    pair = sim.make_pair(pts, nrm, rotvec=[0.05, -0.07, 0.05], trans=[0.1, -0.08, 0.05])
    source, target, normals = pair["source"], pair["target"], pair["target_normals"]

    T = np.eye(4)
    for _ in range(12):
        P_now = apply_transform(T, source)
        delta = np.asarray(point_to_plane_step(P_now, target, normals))
        T = delta @ T
    rot_err, trans_err = transform_error(T, pair["T_align"])
    assert rot_err < 1e-6
    assert trans_err < 1e-6


def test_single_step_reduces_error():
    # One step from the identity already shrinks the alignment error (the step descends).
    pts, nrm = sim.terrain_cloud(seed=4)
    pair = sim.make_pair(pts, nrm, rotvec=[0.04, -0.05, 0.04], trans=[0.08, -0.06, 0.04])
    source, target, normals = pair["source"], pair["target"], pair["target_normals"]
    e0 = transform_error(np.eye(4), pair["T_align"])
    delta = np.asarray(point_to_plane_step(source, target, normals))
    e1 = transform_error(delta, pair["T_align"])
    assert e1[0] < e0[0] and e1[1] < e0[1]
