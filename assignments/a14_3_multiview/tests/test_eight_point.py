"""The normalized eight-point algorithm: the recovered fundamental matrix satisfies the
epipolar constraint on pixels and is rank 2, and the same solver on normalized rays yields
an essential matrix satisfying the constraint on rays."""

import numpy as np

from _impl import eight_point
from config import K, CX, CY, FX, FY
import sim


def _normalized(u):
    return (u - [CX, CY]) / [FX, FY]


def test_fundamental_epipolar_constraint_and_rank2():
    sc = sim.two_view_scene(seed=1, noise_px=0.0)
    u1, u2 = sc["u1"], sc["u2"]
    F = eight_point(u1, u2)
    # x2^T F x1 = 0 on every (noise-free) correspondence.
    res = [abs(np.array([*u2[i], 1.0]) @ F @ np.array([*u1[i], 1.0])) for i in range(len(u1))]
    assert max(res) < 1e-8
    # Rank 2: the smallest singular value is zero (det F = 0).
    s = np.linalg.svd(F, compute_uv=False)
    assert s[2] / s[0] < 1e-10


def test_essential_on_normalized_rays():
    sc = sim.two_view_scene(seed=3, noise_px=0.0)
    x1, x2 = _normalized(sc["u1"]), _normalized(sc["u2"])
    E = eight_point(x1, x2)
    res = [abs(np.array([*x2[i], 1.0]) @ E @ np.array([*x1[i], 1.0])) for i in range(len(x1))]
    assert max(res) < 1e-10


def test_recovers_under_pixel_noise():
    # With mild noise the constraint is no longer exactly zero, but normalization keeps the
    # mean residual small; an unnormalized solver would be orders of magnitude worse.
    sc = sim.two_view_scene(seed=5, noise_px=0.5)
    F = eight_point(sc["u1"], sc["u2"])
    u1, u2 = sc["u1"], sc["u2"]
    res = np.array([abs(np.array([*u2[i], 1.0]) @ F @ np.array([*u1[i], 1.0]))
                    for i in range(len(u1))])
    assert np.median(res) < 1e-2
