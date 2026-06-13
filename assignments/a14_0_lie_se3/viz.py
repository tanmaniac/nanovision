"""Manifold vs naive pose interpolation, visualized in Rerun.

Interpolating between two poses T0 and T1 two ways:
  manifold:  T(s) = T0 |+ (s * (T1 |- T0)) = T0 * exp(s * log(T0^-1 T1))  - a screw motion
  naive:     R(s) = (1-s) R0 + s R1,  t(s) = (1-s) t0 + s t1            - leaves SO(3)

The naive rotation block stops being a rotation: its columns lose orthonormality and
det(R) drifts from 1. The viewer shows both coordinate frames moving over a timeline,
plus scalar plots of det(R) and the orthonormality error.

Run headless (writes out/lie_interp.rrd):   make viz A=a14_0_lie_se3
Open the interactive viewer:                make viz A=a14_0_lie_se3 SHOW=1
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # so `import _impl` resolves under -m
import rerun as rr

import _impl as lie
from config import VIZ_STEPS

SHOW = os.environ.get("NANOVISION_VIZ_SHOW") == "1"
_OUT = Path(__file__).parent / "out"
_AXIS_COLORS = [[228, 26, 28], [77, 175, 74], [55, 126, 184]]  # x red, y green, z blue


def _set_step(s_idx):
    if hasattr(rr, "set_time"):
        rr.set_time("s", sequence=s_idx)
    else:  # older rerun
        rr.set_time_sequence("s", s_idx)


def _log_frame(path, R, t, scale=0.6):
    origins = np.tile(t, (3, 1))
    vectors = (R * scale).T  # rows = scaled axis directions
    rr.log(path, rr.Arrows3D(origins=origins, vectors=vectors, colors=_AXIS_COLORS))


def _log_pair(path, manifold_value, naive_value):
    # Two series in one view share a y-axis, so the manifold line reads as flat while
    # the naive line deviates - much clearer than two autoscaled single-value panels
    # (a constant 1.0 autoscales into its 1e-16 float-noise band and looks like noise).
    if hasattr(rr, "Scalars"):
        rr.log(path, rr.Scalars([manifold_value, naive_value]))
    else:
        rr.log(path + "/manifold", rr.Scalar(manifold_value))
        rr.log(path + "/naive", rr.Scalar(naive_value))


def _name_series(path, names):
    try:  # set legend names once; guard against rerun API drift across versions
        rr.log(path, rr.SeriesLines(names=names), static=True)
    except Exception:
        pass


def _ortho_error(R):
    return float(np.linalg.norm(R.T @ R - np.eye(3)))


def main():
    # Two poses far enough apart that the naive path visibly breaks.
    T0 = lie.se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    T1 = lie.se3_exp(np.array([2.0, 1.0, 0.5, 0.4, -1.6, 2.2]))
    R0, t0 = T0[:3, :3], T0[:3, 3]
    R1, t1 = T1[:3, :3], T1[:3, 3]

    xi = lie.se3_boxminus(T1, T0)  # the screw twist from T0 to T1
    naive_offset = np.array([0.0, 3.0, 0.0])  # separate the two frames for viewing

    rr.init("a14_0_lie_se3", spawn=SHOW)
    _name_series("metrics/determinant", ["manifold", "naive"])
    _name_series("metrics/orthonormality_error", ["manifold", "naive"])
    for i in range(VIZ_STEPS + 1):
        s = i / VIZ_STEPS
        _set_step(i)

        T = lie.se3_boxplus(T0, s * xi)
        Rm = T[:3, :3]
        _log_frame("world/manifold", Rm, T[:3, 3])

        Rn = (1 - s) * R0 + s * R1
        tn = (1 - s) * t0 + s * t1 + naive_offset
        _log_frame("world/naive", Rn, tn)

        # manifold stays at det 1 / orthonormality error 0; naive deviates from both.
        _log_pair("metrics/determinant", float(np.linalg.det(Rm)), float(np.linalg.det(Rn)))
        _log_pair("metrics/orthonormality_error", _ortho_error(Rm), _ortho_error(Rn))

    if not SHOW:
        _OUT.mkdir(parents=True, exist_ok=True)
        rrd = _OUT / "lie_interp.rrd"
        rr.save(str(rrd))
        print(f"wrote {rrd}")
        print("open it with:  rerun", rrd)
        print("or render live with:  make viz A=a14_0_lie_se3 SHOW=1")


if __name__ == "__main__":
    main()
