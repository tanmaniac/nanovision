"""The Sampson distance is near zero for correspondences consistent with the true
fundamental matrix and large for wrong matches, and it matches its closed-form definition."""

import numpy as np

from _impl import eight_point, sampson_distance
from config import RANSAC_THRESH_PX
import sim


def _sampson_reference(F, u1, u2):
    d = []
    for i in range(len(u1)):
        x1 = np.array([*u1[i], 1.0])
        x2 = np.array([*u2[i], 1.0])
        a = F @ x1
        b = F.T @ x2
        r = x2 @ a
        denom = np.sqrt(a[0] ** 2 + a[1] ** 2 + b[0] ** 2 + b[1] ** 2)
        d.append(abs(r) / denom)
    return np.array(d)


def test_matches_reference_formula():
    sc = sim.two_view_scene(seed=1, noise_px=0.5)
    F = eight_point(sc["u1"], sc["u2"])
    d = np.asarray(sampson_distance(F, sc["u1"], sc["u2"]))
    assert np.allclose(d, _sampson_reference(F, sc["u1"], sc["u2"]), atol=1e-9)


def test_small_for_inliers_large_for_outliers():
    sc = sim.two_view_scene(seed=2, noise_px=0.5, outlier_frac=0.35)
    # Fit F on the ground-truth inliers only, then score everyone.
    inl = sc["inlier"]
    F = eight_point(sc["u1"][inl], sc["u2"][inl])
    d = np.asarray(sampson_distance(F, sc["u1"], sc["u2"]))
    # Inliers sit under the RANSAC pixel threshold; outliers are far above it.
    assert np.median(d[inl]) < RANSAC_THRESH_PX
    assert np.median(d[~inl]) > 5.0 * RANSAC_THRESH_PX
