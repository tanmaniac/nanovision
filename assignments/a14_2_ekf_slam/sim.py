"""The simulated world for EKF-SLAM. Provided.

A unicycle robot drives a circular loop around the origin among fixed point landmarks,
observing range and bearing to every landmark within SENSOR_RANGE each step. The loop
closes: near the end the robot re-sees the landmarks it mapped first, which is where a
correct filter tightens the whole map and where EKF-SLAM's linearization inconsistency
shows up.

simulate() returns ground-truth poses (with process noise when enabled, so NEES is
meaningful), the controls the filter also receives, and the noisy observations.
"""

import numpy as np

from _helpers import rng, wrap_angle
from config import DT, V, SENSOR_RANGE, LANDMARKS, Q_DIAG, R_DIAG

LOOP_RADIUS = 5.0
OMEGA = V / LOOP_RADIUS  # turn rate that closes a circle of radius LOOP_RADIUS


def _observe(pose, landmarks, r_noise, rng_):
    """Range-bearing to every landmark within range, as a list of (id, z)."""
    obs = []
    for i, lm in enumerate(landmarks):
        dx, dy = lm[0] - pose[0], lm[1] - pose[1]
        rng_dist = np.hypot(dx, dy)
        if rng_dist > SENSOR_RANGE:
            continue
        bearing = wrap_angle(np.arctan2(dy, dx) - pose[2])
        z = np.array([rng_dist, bearing])
        if r_noise:
            z = z + rng_.normal(0.0, np.sqrt(R_DIAG))
            z[1] = wrap_angle(z[1])
        obs.append((i, z))
    return obs


def simulate(n_steps, seed=0, process_noise=True, meas_noise=True):
    r = rng(seed)
    # Start on the loop heading tangent (counter-clockwise around the origin).
    pose = np.array([LOOP_RADIUS, 0.0, np.pi / 2])
    u = np.array([V, OMEGA])

    poses = [pose.copy()]
    controls = []
    observations = [_observe(pose, LANDMARKS, meas_noise, r)]
    for _ in range(n_steps):
        pose = np.array([
            pose[0] + V * DT * np.cos(pose[2]),
            pose[1] + V * DT * np.sin(pose[2]),
            wrap_angle(pose[2] + OMEGA * DT),
        ])
        if process_noise:
            pose = pose + r.normal(0.0, np.sqrt(Q_DIAG))
            pose[2] = wrap_angle(pose[2])
        poses.append(pose.copy())
        controls.append(u.copy())
        observations.append(_observe(pose, LANDMARKS, meas_noise, r))
    return {
        "poses": np.array(poses),
        "controls": np.array(controls),
        "observations": observations,  # observations[k] are the readings at poses[k]
        "landmarks": LANDMARKS.copy(),
    }
