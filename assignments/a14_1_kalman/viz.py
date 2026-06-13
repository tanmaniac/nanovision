"""EKF vs UKF tracking a noisy unicycle, visualized in Rerun.

A unicycle drives a gentle arc, observed only by range-bearing readings to one fixed
landmark. Two filters run on the same measurements: an EKF (linearize-by-Jacobian) and a
UKF (linearize-by-sigma-points). The 2D view shows ground truth, both estimates, the
landmark, and each filter's current 1-sigma position-covariance ellipse, all on a
timeline you can scrub. Scalar plots compare the two filters' position error and the
trace of their position covariance.

Run headless (writes out/kalman_track.rrd):   make viz A=a14_1_kalman
Open the interactive viewer:                  make viz A=a14_1_kalman SHOW=1
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # so `import _impl` resolves under -m
import rerun as rr

import _impl as kal
from _helpers import simulate_unicycle, ukf_predict, ukf_update, wrap_angle, covariance_ellipse
from config import (DT, V, OMEGA, N_STEPS, LANDMARK, Q_DIAG, R_DIAG, SEED,
                    UKF_ALPHA, UKF_BETA, UKF_KAPPA)

SHOW = os.environ.get("NANOVISION_VIZ_SHOW") == "1"
_OUT = Path(__file__).parent / "out"


def _set_step(k):
    if hasattr(rr, "set_time"):
        rr.set_time("k", sequence=k)
    else:  # older rerun
        rr.set_time_sequence("k", k)


def _bearing_residual(z, z_pred):
    d = z - z_pred
    d[1] = wrap_angle(d[1])
    return d


def main():
    gt, meas = simulate_unicycle(V, OMEGA, DT, N_STEPS, LANDMARK, Q_DIAG, R_DIAG, seed=SEED)
    Q, R = np.diag(Q_DIAG), np.diag(R_DIAG)
    u = np.array([V, OMEGA])

    mu_e, P_e = gt[0].copy(), np.diag([0.2, 0.2, 0.1])
    mu_u, P_u = gt[0].copy(), np.diag([0.2, 0.2, 0.1])

    f = lambda s: np.asarray(kal.ekf_f(s, u, DT))
    h = lambda s: np.asarray(kal.ekf_h(s, LANDMARK))

    rr.init("a14_1_kalman", spawn=SHOW)
    rr.log("world/landmark", rr.Points2D([LANDMARK], radii=0.15, colors=[[255, 215, 0]]),
           static=True)

    gt_path, e_path, u_path = [], [], []
    for k in range(N_STEPS):
        _set_step(k)
        z = meas[k]

        mu_e, P_e = kal.ekf_predict(mu_e, P_e, u, DT, Q)
        mu_e, P_e = kal.ekf_update(mu_e, P_e, z, LANDMARK, R)

        mu_u, P_u = ukf_predict(mu_u, P_u, f, Q, UKF_ALPHA, UKF_BETA, UKF_KAPPA)
        mu_u, P_u = ukf_update(mu_u, P_u, z, h, R, UKF_ALPHA, UKF_BETA, UKF_KAPPA,
                               residual=_bearing_residual)
        mu_u = np.asarray(mu_u)

        g = gt[k + 1]
        gt_path.append(g[:2]); e_path.append(mu_e[:2]); u_path.append(mu_u[:2])
        rr.log("world/ground_truth", rr.LineStrips2D([gt_path], colors=[[160, 160, 160]]))
        rr.log("world/ekf/path", rr.LineStrips2D([e_path], colors=[[228, 26, 28]]))
        rr.log("world/ukf/path", rr.LineStrips2D([u_path], colors=[[55, 126, 184]]))
        rr.log("world/ekf/cov",
               rr.LineStrips2D([covariance_ellipse(mu_e[:2], P_e[:2, :2])],
                               colors=[[228, 26, 28]]))
        rr.log("world/ukf/cov",
               rr.LineStrips2D([covariance_ellipse(mu_u[:2], P_u[:2, :2])],
                               colors=[[55, 126, 184]]))

        # Shared-axis scalar comparisons (EKF red, UKF blue).
        if hasattr(rr, "Scalars"):
            rr.log("metrics/position_error",
                   rr.Scalars([float(np.linalg.norm(mu_e[:2] - g[:2])),
                               float(np.linalg.norm(mu_u[:2] - g[:2]))]))
            rr.log("metrics/position_cov_trace",
                   rr.Scalars([float(np.trace(P_e[:2, :2])),
                               float(np.trace(P_u[:2, :2]))]))

    if not SHOW:
        _OUT.mkdir(parents=True, exist_ok=True)
        rrd = _OUT / "kalman_track.rrd"
        rr.save(str(rrd))
        print(f"wrote {rrd}")
        print("open it with:  rerun", rrd)
        print("or render live with:  make viz A=a14_1_kalman SHOW=1")


if __name__ == "__main__":
    main()
