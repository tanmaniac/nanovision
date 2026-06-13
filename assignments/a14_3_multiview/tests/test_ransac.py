"""RANSAC and the composed two-view front-end: with a third of the matches wrong, RANSAC
recovers the inlier set and the robust pose stays accurate, while a plain least-squares fit
on all correspondences does not. The robust-vs-naive ordering holds across seeds."""

import numpy as np

from _impl import eight_point, ransac_fundamental, recover_pose, two_view_relative_pose
from config import CX, CY, FX, FY, RANSAC_THRESH_PX, RANSAC_ITERS
from _helpers import rotation_angle, translation_angle
import sim


def _normalized(u):
    return (u - [CX, CY]) / [FX, FY]


def test_ransac_recovers_inlier_set():
    sc = sim.two_view_scene(seed=1, noise_px=0.5, outlier_frac=0.35)
    F, inliers = ransac_fundamental(sc["u1"], sc["u2"], RANSAC_THRESH_PX, RANSAC_ITERS, 0)
    inliers = set(int(i) for i in inliers)
    true_inl = set(np.where(sc["inlier"])[0].tolist())
    precision = len(inliers & true_inl) / max(len(inliers), 1)
    recall = len(inliers & true_inl) / len(true_inl)
    assert precision > 0.9
    assert recall > 0.8


def test_front_end_recovers_pose_up_to_scale():
    sc = sim.two_view_scene(seed=2, noise_px=0.5, outlier_frac=0.35)
    T, pts, inliers = two_view_relative_pose(
        sc["K"], sc["u1"], sc["u2"], RANSAC_THRESH_PX, RANSAC_ITERS, 0)
    assert np.degrees(rotation_angle(T[:3, :3], sc["R"])) < 1.0
    # Translation is recovered only up to direction (monocular scale gauge).
    assert np.degrees(translation_angle(T[:3, 3], sc["t"])) < 10.0


def test_robust_beats_naive_across_seeds():
    for seed in range(5):
        sc = sim.two_view_scene(seed=seed, noise_px=0.5, outlier_frac=0.35)
        x1, x2 = _normalized(sc["u1"]), _normalized(sc["u2"])
        # Naive: eight-point on ALL correspondences (outliers included), then pose.
        E_naive = eight_point(x1, x2)
        R_naive, _ = recover_pose(E_naive, x1, x2)
        naive_err = np.degrees(rotation_angle(R_naive, sc["R"]))
        # Robust: the RANSAC front-end.
        T, _, _ = two_view_relative_pose(
            sc["K"], sc["u1"], sc["u2"], RANSAC_THRESH_PX, RANSAC_ITERS, 0)
        robust_err = np.degrees(rotation_angle(T[:3, :3], sc["R"]))
        assert robust_err < 1.0
        assert naive_err > 3.0 * robust_err
