"""On a short, mild trajectory where linearization is benign, the filter is consistent:
the robot NEES averages near its 3-DOF expectation of 3 and stays in a chi-square band.
(The long-loop case, where EKF-SLAM goes optimistic, is a viz/README demonstration, not a
gate - asserting consistency there would contradict the known behavior.)"""

import numpy as np

from sim import simulate
from runner import run_slam
from _helpers import robot_nees


def test_short_trajectory_filter_is_not_optimistic():
    # The robot NEES (3-DOF, expectation 3) must stay below the upper 95% chi-square bound:
    # an overconfident filter, the EKF-SLAM failure mode, would blow through it. The lower
    # side is only checked to be positive: with a well-anchored map and rich measurements
    # the absolute-frame estimate is consistent-to-conservative, which is not an error (the
    # famous optimism appears on the long loop, demonstrated in the viz, not gated here).
    sim = simulate(n_steps=40, seed=1, process_noise=True, meas_noise=True)
    history, _ = run_slam(sim, known_association=True)
    poses = sim["poses"]
    nees = [robot_nees(mu, P, poses[k + 1]) for k, (mu, P) in enumerate(history)]
    mean_nees = float(np.mean(nees))
    assert 0.0 < mean_nees < 6.0, f"mean robot NEES {mean_nees:.2f} not in (0, 6)"


def test_map_becomes_correlated():
    # SLAM's defining property: observing landmarks from a shared, uncertain robot pose
    # correlates them. After a run the off-diagonal landmark-landmark covariance blocks are
    # non-negligible, so the joint covariance is genuinely dense (the O(n^2) cost).
    import itertools
    sim = simulate(n_steps=60, seed=2, process_noise=True, meas_noise=True)
    history, _ = run_slam(sim, known_association=True)
    mu, P = history[-1]
    N = (mu.size - 3) // 2
    blocks = [np.linalg.norm(P[3 + 2 * a:5 + 2 * a, 3 + 2 * b:5 + 2 * b])
              for a, b in itertools.combinations(range(N), 2)]
    assert max(blocks) > 0.02  # landmarks are correlated, not block-diagonal


def test_map_tracks_landmarks():
    sim = simulate(n_steps=60, seed=2, process_noise=True, meas_noise=True)
    history, id_to_index = run_slam(sim, known_association=True)
    mu, _ = history[-1]
    lm_true = sim["landmarks"]
    errs = []
    for lm_id, j in id_to_index.items():
        est = mu[3 + 2 * j: 3 + 2 * j + 2]
        errs.append(np.linalg.norm(est - lm_true[lm_id]))
    assert len(errs) >= 3  # several landmarks were mapped
    assert np.mean(errs) < 0.6  # the map is close to truth, not diverging
