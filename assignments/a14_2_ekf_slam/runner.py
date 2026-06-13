"""Drive the EKF-SLAM filter over a simulated run. Provided (tests and viz both use it).

The filter loop is glue around the four C++ holes: predict on each control, then for each
observation either initialize a new landmark or update an existing one. With known
association the true landmark id picks the map slot; with unknown association
slam_associate does, and a gated-out reading starts a new landmark.
"""

import numpy as np

import _impl as slam
from config import DT, Q_DIAG, R_DIAG, INIT_P_DIAG, GATE


def run_slam(sim, known_association=True, gate=GATE):
    Q = np.diag(Q_DIAG)
    R = np.diag(R_DIAG)
    poses = sim["poses"]
    controls = sim["controls"]
    obs = sim["observations"]

    mu = poses[0].copy()
    P = np.diag(INIT_P_DIAG).copy()
    id_to_index = {}  # true landmark id -> slam map index (known association only)

    def fold(mu, P, reading):
        lm_id, z = reading
        if known_association:
            if lm_id not in id_to_index:
                mu, P = slam.slam_add_landmark(mu, P, z, R)
                id_to_index[lm_id] = (mu.size - 3) // 2 - 1
            else:
                mu, P = slam.slam_update(mu, P, z, id_to_index[lm_id], R)
        else:
            j = slam.slam_associate(mu, P, z, R, gate)
            if j < 0:
                mu, P = slam.slam_add_landmark(mu, P, z, R)
            else:
                mu, P = slam.slam_update(mu, P, z, j, R)
        return mu, P

    for reading in obs[0]:
        mu, P = fold(mu, P, reading)

    history = []  # (mu, P) after the update at each pose, aligned with poses[1:]
    for k in range(len(controls)):
        mu, P = slam.slam_predict(mu, P, controls[k], DT, Q)
        for reading in obs[k + 1]:
            mu, P = fold(mu, P, reading)
        history.append((mu.copy(), P.copy()))
    return history, id_to_index
