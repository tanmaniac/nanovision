"""Oracle cross-check: the from-scratch ICP transform agrees with Open3D's registration on
the same clouds. Open3D is a labeled comparison only - the graded solver is the C++ one. The
test is skipped when Open3D is not installed."""

import numpy as np
import pytest

from _impl import icp
from config import MAX_ITER, MAX_CORR_DIST
from _helpers import transform_error
import sim

o3d = pytest.importorskip("open3d")


def _to_pcd(points, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals))
    return pcd


def test_agrees_with_open3d_point_to_point():
    pts, nrm = sim.terrain_cloud(seed=1)
    pair = sim.make_pair(pts, nrm)

    T_ours, _, _ = icp(pair["source"], pair["target"], pair["target_normals"],
                       np.eye(4), MAX_ITER, MAX_CORR_DIST, "point")

    reg = o3d.pipelines.registration.registration_icp(
        _to_pcd(pair["source"]), _to_pcd(pair["target"]), MAX_CORR_DIST, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    rot_err, trans_err = transform_error(np.asarray(T_ours), np.asarray(reg.transformation))
    assert np.degrees(rot_err) < 0.5
    assert trans_err < 1e-2
