"""EKF-SLAM on the loop, visualized in Rerun.

A unicycle drives a loop among point landmarks, observing range-bearing. The filter builds
the map online. The 2D view shows ground truth, the robot estimate and its covariance
ellipse, the true and estimated landmarks with their ellipses, and the active measurement
rays, all on a timeline you can scrub. Two scalar panels tell the EKF-SLAM story: the robot
NEES against its 3-sigma (chi-square) consistency bound, and the average mapped-landmark
error. Watch the loop closure near the end re-see the first landmarks and tighten the map,
and watch the NEES creep up as linearization error accumulates - EKF-SLAM going optimistic.

Run headless (writes out/ekf_slam.rrd):   make viz A=a14_2_ekf_slam
Open the interactive viewer:               make viz A=a14_2_ekf_slam SHOW=1
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).parent))
import rerun as rr

from sim import simulate
from runner import run_slam
from _helpers import covariance_ellipse, robot_nees
from config import LANDMARKS

SHOW = os.environ.get("NANOVISION_VIZ_SHOW") == "1"
_OUT = Path(__file__).parent / "out"
NEES_BOUND = float(chi2.ppf(0.95, df=3))  # 3-DOF robot NEES 95% bound


def _set_step(k):
    if hasattr(rr, "set_time"):
        rr.set_time("k", sequence=k)
    else:
        rr.set_time_sequence("k", k)


def _scalars(path, values):
    if hasattr(rr, "Scalars"):
        rr.log(path, rr.Scalars(values))
    else:
        for i, v in enumerate(values):
            rr.log(f"{path}/{i}", rr.Scalar(v))


def main():
    sim = simulate(n_steps=300, seed=0, process_noise=True, meas_noise=True)
    history, id_to_index = run_slam(sim, known_association=True)
    poses = sim["poses"]

    rr.init("a14_2_ekf_slam", spawn=SHOW)
    rr.log("world/landmarks_true", rr.Points2D(LANDMARKS, radii=0.12, colors=[[120, 120, 120]]),
           static=True)

    gt_path, est_path = [], []
    for k, (mu, P) in enumerate(history):
        _set_step(k)
        g = poses[k + 1]
        gt_path.append(g[:2]); est_path.append(mu[:2])

        rr.log("world/ground_truth", rr.LineStrips2D([gt_path], colors=[[160, 160, 160]]))
        rr.log("world/estimate/path", rr.LineStrips2D([est_path], colors=[[228, 26, 28]]))
        rr.log("world/estimate/robot_cov",
               rr.LineStrips2D([covariance_ellipse(mu[:2], P[:2, :2], n_sigma=3.0)],
                               colors=[[228, 26, 28]]))

        # Mapped landmarks and their (3-sigma) ellipses.
        N = (mu.size - 3) // 2
        lm_xy = [mu[3 + 2 * j: 5 + 2 * j] for j in range(N)]
        if lm_xy:
            rr.log("world/estimate/landmarks",
                   rr.Points2D(np.array(lm_xy), radii=0.1, colors=[[55, 126, 184]]))
            ell = [covariance_ellipse(mu[3 + 2 * j:5 + 2 * j],
                                      P[3 + 2 * j:5 + 2 * j, 3 + 2 * j:5 + 2 * j], n_sigma=3.0)
                   for j in range(N)]
            rr.log("world/estimate/landmark_cov",
                   rr.LineStrips2D(ell, colors=[[55, 126, 184]]))

        # Measurement rays from the estimated robot to each observed landmark.
        rays = []
        for lm_id, _ in sim["observations"][k + 1]:
            j = id_to_index.get(lm_id, -1)
            if 0 <= j < N:  # the landmark exists in the map at this step
                rays.append([mu[:2], mu[3 + 2 * j: 5 + 2 * j]])
        if rays:
            rr.log("world/estimate/rays", rr.LineStrips2D(rays, colors=[[80, 180, 80]]))

        # Consistency and map-error panels.
        _scalars("metrics/robot_nees", [robot_nees(mu, P, g), NEES_BOUND])
        lm_err = [np.linalg.norm(mu[3 + 2 * j:5 + 2 * j] - LANDMARKS[lid])
                  for lid, j in id_to_index.items() if j < N]
        _scalars("metrics/mean_landmark_error", [float(np.mean(lm_err)) if lm_err else 0.0])

    if not SHOW:
        _OUT.mkdir(parents=True, exist_ok=True)
        rrd = _OUT / "ekf_slam.rrd"
        rr.save(str(rrd))
        print(f"wrote {rrd}")
        print("open it with:  rerun", rrd)
        print("or render live with:  make viz A=a14_2_ekf_slam SHOW=1")


if __name__ == "__main__":
    main()
