# a14_classical_slam - module plan

Status: draft for expert review. Six C++ assignments adding the classical geometric
SLAM / localization canon the deep-learning course deliberately skips. This is the
largest single module in the repo and the only one in C++.

## Why this module exists

The course teaches 2020-2026 deep-learning vision. It omits the classical geometric
estimation that perception/robotics/AV interviews still test hardest: Lie-group state
representation, Kalman filtering, filter-based SLAM, multi-view geometry estimators,
point-cloud registration, and factor-graph optimization. This module fills that gap.

Two things are different from the rest of the repo, by explicit learner request:

1. The code is C++17 with Eigen, not Python/PyTorch. For deep learning Python is
   right; for localization and SLAM, C++ is what the field ships and what interviews
   expect, so implementing the mechanisms in C++ builds fluency in the delivery
   language. Reimplementing SO(3)/SE(3) in C++ even though a11_5a has a Python version
   is intended practice, not duplication.
2. The emphasis is relearning, not just verification. The learner has prior SLAM
   experience (stereo VO, loosely-coupled EKF fusion) gone rusty and is interview-prep.
   They want to see and understand things visually. So every assignment is
   visualization-first: an animated, steppable view of the estimator running is the
   default way to "run" it, with the headless artifact as the CI fallback - the inverse
   of the rest of the repo's headless-first emphasis. Each README also carries an "In
   interviews" depth section that the rest of the repo does not have.

The pedagogical contract from the rest of the course is preserved: load-bearing lines
are holes the learner fills, `solution/` holds the answer key in plain sight, tests run
the workflow order (shape/type -> analytic-Jacobian or numerical check -> reference
value -> convergence), READMEs are standalone lecture notes with verified paper links,
and the `make test` / `make verify` / `make viz` green-red surface is unchanged.

## Toolchain and harness design (the novel part)

The repo today has no build step: 21 Python assignments run from root via pytest +
`pythonpath`. This module adds C++ without disturbing that. Decisions (locked with the
learner 2026-06-11):

- Build driver: `make` stays the command surface. CMake under each `a14` assignment
  compiles the C++. No Bazel (it would force a dual-build or a gratuitous Bazelify of
  21 working Python assignments, and the SLAM oracle libraries are all CMake-native, so
  Bazel fights their linkage). The make target runs `cmake --build` then the test runner.
- Integration: pybind11 exposes the student's C++ mechanism to the existing pytest
  green-red bar. The only C++ that must build and link is the student's mechanism plus
  Eigen plus pybind11. Viz and the oracle libraries come in as Python bindings, so no
  C++ linkage against them.
- Viewer: Rerun (Python SDK), interactive primary, `.rrd`/PNG headless export for CI.
  Its timeline/streaming model fits the "steppable, watch-it-evolve" requirement
  directly (covariance ellipses shrinking, clouds converging, trajectory snapping on
  loop closure are all natural on a Rerun timeline).
- Oracles: pip-installable Python bindings, used as labeled side-by-sides, never as the
  graded implementation. GTSAM (`gtsam`) for pose-graph/BA, Open3D (`open3d`) for
  ICP/GICP, `pyceres` optional. Oracle-comparison tests skip cleanly when the library is
  absent (the nuScenes-loader pattern from a11_5a).
- Data: sim-first, zero new downloads. A controllable 2D range-bearing world for the
  filters, a lightweight 3D sim for ICP/multiview/pose-graph, reusing the in-repo
  nuScenes-mini where it adds realism. Noise levels and loop-closing trajectories are
  config knobs. Real-data hooks (KITTI-odometry, TUM RGB-D) can be added later but are
  not required to run the module.

### The student-vs-solution switch in C++

This is the part with no precedent in the repo. The Python harness keys on
`NANOVISION_IMPL` and swaps which `.py` the `conftest` puts on `sys.path`. The C++
analog:

- Each assignment has top-level C++ sources (e.g. `se3.cpp` / `se3.hpp`) where the
  load-bearing lines are replaced by `throw std::logic_error("NOT_IMPLEMENTED: <contract>")`
  - the C++ analog of `raise NotImplementedError` - with a comment stating input/output
  types, shapes, and the formula. `solution/` holds the filled copy of each source. The
  provided helpers/classes are identical in both copies; only the hole bodies differ.
- CMake builds two pybind extension modules from the same `CMakeLists.txt`:
  `_a14_x_student` from the top-level sources and `_a14_x_solution` from `solution/`.
  Both can coexist under `build/`.
- A tiny provided Python shim `_impl.py` in the assignment imports the right extension
  by reading `NANOVISION_IMPL` (unset -> student module, `solution` -> solution module).
  Tests and `viz.py` import the mechanism through that shim, so default mode loads the
  student `.so` (which throws at the holes -> a clean pytest failure, not a collection
  error) and `verify` mode loads the solution `.so` (green).
- `make test A=a14_x` runs `cmake --build` of the student target then pytest; `make
  verify` builds the solution target then pytest with `NANOVISION_IMPL=solution`. A
  first-build-from-cold compiles in seconds for these small targets; incremental builds
  are sub-second. The make targets gain a `cmake --build` prerequisite step guarded so
  the 21 Python assignments (no `CMakeLists.txt`) skip it.
- Forbidden-imports analog: a static scan test greps the C++ sources (top-level +
  solution) for `#include <gtsam`, `#include <open3d`, `#include <ceres`, etc., so the
  student cannot implement the mechanism by calling an oracle. Mirrors the Python
  `test_forbidden_imports.py`, scanning `.cpp`/`.hpp` with comment/string stripping.

### Per-assignment directory layout

```
assignments/a14_x_name/
  README.md            lecture notes + lean assignment section + "In interviews"
  CMakeLists.txt       builds _a14_x_student and _a14_x_solution via pybind11
  <module>.cpp/.hpp    top-level C++ the student edits (holes)
  bindings.cpp         pybind11 wrapper exposing the C++ to Python (provided)
  _impl.py             provided shim: imports the right extension per NANOVISION_IMPL
  conftest.py          builds the extension if missing, puts build dir on sys.path
  config.py            sim/noise/threshold parameters (provided)
  sim.py               the synthetic world / data generator (provided, Python)
  viz.py               Rerun visualization, mechanism + oracle side-by-side (provided)
  solution/            filled C++ answer key (<module>.cpp/.hpp) + __init__.py
  tests/               pytest: type/shape -> Jacobian/numeric -> reference -> convergence
  out/                 headless .rrd / .png artifacts
```

Rationale for Python sim/viz/config glue around C++ mechanism: it keeps the C++ surface
to just the mechanism (what the learner is here to write), lets viz use Rerun's mature
Python SDK and the oracle Python bindings directly, and reuses the repo's Python
nuScenes loader. The learner writes C++ for every graded hole; Python is only the
provided scaffolding, consistent with "the student writes the interesting lines, the
boilerplate is given."

### Dependency additions

- `environment.yml`: add a commented optional `a14` block - `cmake>=3.22`,
  `pybind11>=2.11`, `eigen` (header-only, via conda-forge `eigen`), `rerun-sdk>=0.18`,
  and optional oracle bindings `gtsam`, `open3d`, `pyceres`. A C++17 compiler (system
  `g++`/`clang++`) is a documented prerequisite, not a conda package.
- Top-level `README.md`: add the module to the assignment table with a note that it is
  C++ and lists its build prerequisites; add a short "C++ module" paragraph to the
  running-tests section.
- `BUILD_ORDER.md`: add Phase 5 with the six assignments and their dependency edges.

## The six assignments

Build in dependency order. Conventions match a11_5a where they overlap: `T_b_a` names an
a-to-b transform, frames are stated explicitly, quaternions are scalar-first. a14_0
introduces the Lie/manifold layer fresh (a11_5a is matrix-level only) and fixes the
right-perturbation convention used by the rest of the module.

### a14_0_lie_se3 - Lie groups for state estimation  [Core]

Depends on: a11_5a (match frame/naming conventions; no code shared across the language
boundary).

Holes (in `so3.cpp`/`se3.cpp`):
- `hat`/`vee` for so(3) (3-vector <-> skew) and se(3) (6-vector <-> 4x4).
- `SO3::exp`/`log` (Rodrigues and its inverse, with the small-angle Taylor branch).
- `SE3::exp`/`log` (the closed form; the translation block is `t = V ρ` where `V` is the
  SO(3) left-Jacobian `J_l(θ)`). `V`, `J_l`, and `J_r` are three different series and each
  needs its own small-angle branch, not a shared one.
- Left and right Jacobians `J_l`, `J_r` of SO(3) (and their inverses), with their own
  small-angle branches. State the defining property the downstream assignments rely on:
  `Exp(ξ + δ) ≈ Exp(ξ) · Exp(J_r(ξ) δ)` (and `J_r(ξ) = J_l(-ξ)`). a14_5 cites `J_r⁻¹`.
- Adjoint `Ad_T` of SE(3). State its direction explicitly: `Ad_T` maps a body/right twist
  to a spatial/left one, `T · Exp(ξ) = Exp(Ad_T ξ) · T`. Getting the direction (Ad_T vs
  Ad_{T⁻¹}) right is what the a14_5 Jacobians depend on.
- Box-plus / box-minus on the manifold: `T ⊞ ξ = T · SE3::exp(ξ)` (right perturbation)
  and `T2 ⊟ T1 = SE3::log(T1⁻¹ · T2)`. State this convention; everything downstream uses
  it.

Tests: round-trip `log(exp(ξ)) == ξ` and `exp(log(T)) == T` to ~1e-10 on random small
and large twists; `hat`/`vee` inverse; `J_r(ξ)⁻¹ J_r(ξ) == I`; numerical-vs-analytic
Jacobian check (perturb `exp`, finite-difference, compare to `J_r`) to ~1e-6; adjoint
identity `T exp(ξ) T⁻¹ == exp(Ad_T ξ)` AND the direction check `T·Exp(ξ) == Exp(Ad_T ξ)·T`;
box-plus/box-minus inverse consistency.

Viz: animate a pose interpolating from `T0` to `T1` along the manifold
(`T0 · exp(s · (T0 ⊟ T1))`, s in [0,1]) versus naive componentwise linear interpolation
of R and t. The naive path leaves SO(3) (R stops being a rotation, shown by `det != 1`
and shear), the manifold path stays a clean screw motion. Rerun shows both frames moving
in 3D with a scrubber over s.

Interview section: why rotations need a manifold not a vector space; exp/log vs
quaternion; left vs right Jacobian and why the choice matters for which side the
covariance lives on; the adjoint as the tool that moves a twist between frames;
retraction / box-plus as the basis of on-manifold optimization. Classic on-the-spot ask:
implement Rodrigues and its small-angle limit.

### a14_1_kalman - KF / EKF / UKF  [Core]

Depends on: a14_0.

Holes:
- Linear KF `predict` (`mu = F mu + B u`, `P = F P Fᵀ + Q`) and `update` (innovation,
  Kalman gain, Joseph-form covariance update).
- EKF on a nonlinear motion-and-measurement model: the nonlinear process model `f(x, u)`
  with its Jacobian `F_x` (used in predict) AND a range-bearing `h(x)` with its analytic
  Jacobian `H` (used in update). The nonlinear process model is core EKF and is what
  a14_2's motion update reuses, so it lives here, not only in the measurement.
- UKF unscented transform: sigma-point set (2n+1), mean weights `Wm` and covariance
  weights `Wc` from `λ = α²(n+κ) - n`, with the `(1 - α² + β)` term on `Wc⁽⁰⁾` (β=2 is
  Gaussian-optimal; dropping it is a real bug). Propagate through `f`/`h`, recover mean
  and covariance, cross-covariance for the gain.
- Information form: the dual `Ω = P⁻¹`, `η = Ω mu`, and the additive measurement update
  `Ω⁺ = Ω⁻ + Hᵀ R⁻¹ H`, `η⁺ = η⁻ + Hᵀ R⁻¹ z`. (Only the measurement update is additive;
  the prediction step is the expensive one in information form. State that.)

Tests: on a linear-Gaussian system KF, EKF (with an affine model), and UKF return the
same posterior to ~1e-6. The agreement is exact in exact arithmetic (the UKF is exact for
affine `f`/`h`); the ~1e-6 is a floating-point/conditioning tolerance, not a linearization
gap. The test fixes `α, β, κ` to a benign, well-conditioned set (κ = 3−n can drive
negative covariance weights). Analytic `F_x`/`H` match numerical to ~1e-6; the information
update equals the covariance update on the same data; a multi-step run keeps `P`
symmetric positive-definite.

Viz: a noisy target on a path; watch the estimate and the 1-sigma covariance ellipse
track it in real time on a Rerun timeline; switch the measurement model to range-bearing
and watch EKF vs UKF differ as nonlinearity grows.

Interview section: KF vs EKF vs UKF (linearization-by-Jacobian vs linearization-by-
sigma-points, and when each breaks); why the Joseph form; the information filter and why
it is natural for multi-sensor fusion and sparse SLAM; observability. On-the-spot ask:
write the 1D KF update.

### a14_2_ekf_slam - EKF-SLAM  [Core]

Depends on: a14_1.

The world (provided `sim.py`): a 2D robot on a loop-closing trajectory among point
landmarks, range-bearing observations with controllable noise, known or unknown data
association.

Holes (the motion model and its Jacobian are reused from a14_1's EKF; data association is
self-contained here, a14_1 does not teach it):
- State augmentation on landmark initialization: the inverse measurement model placing a
  new landmark in the map from the first observation, and its Jacobians, growing the
  joint mean and covariance.
- The joint EKF predict/update over `[robot; landmarks]`, including the full covariance
  cross-terms (the O(n²) fill-in is the point).
- Data association: nearest-neighbor with a Mahalanobis gate (chi-square threshold over
  the 2-DOF range-bearing innovation) over the predicted measurements.

Tests (favoring the truly implementation-independent statements; EKF-SLAM is known to be
optimistic, so do not gate on the property it famously violates):
- The joint covariance determinant is non-increasing under a measurement update and the
  information matrix Ω is non-decreasing in the Loewner order (`Ω⁺ ⪰ Ω⁻`). Assert this,
  NOT a strict per-landmark marginal-determinant decrease - a single landmark's marginal
  is a Schur complement of the joint and is not monotone under a near-non-informative
  update, so the strict marginal claim can fail for correct code.
- On a short, mild trajectory (where linearization is benign) the estimates are consistent:
  the NEES stays within its chi-square band.
- The Mahalanobis gate rejects a deliberately spurious measurement; the joint covariance
  stays symmetric PSD throughout.
- The long loop case is a DEMONSTRATION, not a gate: the viz/README shows the
  linearization-induced inconsistency (NEES rising above the chi-square bound on the loop),
  per the no-fragile-test gate - do not assert "inside 3-sigma after the loop", which
  contradicts the known behavior.

Viz: animate robot pose, landmark estimates, and their covariance ellipses evolving
frame by frame; show an ellipse shrink on re-observation; visualize the dense covariance
matrix filling in as landmarks correlate. The loop-closure moment (re-seeing the first
landmarks) visibly tightens the whole map.

Interview section: why EKF-SLAM is O(n²) in the map size (the dense joint covariance) and
what that killed; consistency and the linearization-induced inconsistency of EKF-SLAM;
data association as the real failure mode; the filter-vs-smoother divide that motivates
the factor-graph assignment. On-the-spot ask: what goes in the state and why the
off-diagonal covariance blocks matter.

### a14_3_multiview - Multi-view geometry estimators  [Core]

Depends on: a14_0, a11_5a (camera intrinsics/projection conventions).

Conventions to pin in the README and solution (each is a classic silent transpose/sign
trap): the unprimed image is the first/reference image (the frame the points start in);
`(R, t) = T_2_1` is the transform taking frame-1 coordinates into frame-2, and
`E = [t]_× R`; the essential constraint `x'ᵀ E x = 0` holds for NORMALIZED rays
`x = K⁻¹ u`, while the fundamental constraint `x'ᵀ F x = 0` holds for PIXEL coordinates
`u`, with `F = K'⁻ᵀ E K⁻¹`. Leaving the primed/unprimed choice or the `(R,t)` direction
unstated silently transposes `E` and flips the sign of `t`, after which cheirality
"fails" on correct code.

Holes:
- Triangulation: linear DLT (the `A x = 0` SVD form) and a nonlinear refinement
  (Gauss-Newton on reprojection).
- Eight-point essential/fundamental: the normalized eight-point algorithm (Hartley
  normalization, the `f` null-vector via SVD, rank-2 enforcement), and decomposing `E`
  into the four `(R, t)` candidates with cheirality to pick the physical one.
- PnP: DLT initialization plus nonlinear refinement (Gauss-Newton minimizing
  reprojection over SE(3), using a14_0's `J_r`).
- A robust RANSAC/PROSAC wrapper: minimal-sample fit, inlier scoring by the Sampson
  distance (the first-order geometric error, NOT the raw algebraic `|x'ᵀ F x|`, which is
  not metric), refit on the consensus set.
- A composed two-view relative-pose front-end: chain RANSAC eight-point -> E ->
  cheirality -> triangulation into a single estimated `T_2_1` plus the 3D points. This is
  the actual VO front-end and the thing an interview asks you to assemble.

Tests: on noise-free synthetic correspondences, triangulation reproduces the 3D points
and PnP recovers the camera pose to ~1e-6; the recovered `E` satisfies `x'ᵀ E x ≈ 0` on
normalized inlier rays (and `F` on pixels); the composed front-end recovers `T_2_1` up to
the inherent monocular scale; with injected outliers, RANSAC recovers the inlier set and
the pose stays accurate while a plain least-squares fit does not (the robust-vs-naive gap
is the lesson, asserted as an ordering that holds across seeds).

Viz: draw epipolar lines on an image pair, color inliers vs outliers, and render the
recovered camera frusta plus triangulated points in interactive 3D. Watch RANSAC's
consensus set form.

Interview section: why RANSAC is mandatory (the breakdown point of least squares under
outliers); the eight-point algorithm and why normalization matters; essential vs
fundamental (calibrated vs not), and that the five-point algorithm (Nister 2004) is the
modern minimal relative-pose solver - eight-point is the pedagogical route, not state of
the art; the four-fold `E` ambiguity and cheirality; PnP variants (P3P, EPnP) and
degeneracies. On-the-spot ask: outline normalized eight-point.

### a14_4_icp - Point-cloud registration  [Core]

Depends on: a14_0.

Holes:
- Point-to-point ICP: the closed-form rotation+translation via SVD (Umeyama/Kabsch).
  Center both clouds, `H = Σ (p_i - p̄)(q_i - q̄)ᵀ`, `H = U Σ Vᵀ`, then
  `R = V diag(1, 1, det(V Uᵀ)) Uᵀ` - the `det(V Uᵀ)` correction on the LAST singular
  direction prevents a reflection on planar/degenerate data; `t = q̄ - R p̄`. State the
  exact form and that `R` maps the source `p`-cloud onto the target `q`-cloud (matching
  the module's `T_b_a` source-to-target naming).
- Point-to-plane ICP: the linearized small-angle system minimizing point-to-tangent-
  plane distance, solving the 6x6 normal equations per iteration (using a14_0's
  retraction to update the pose on SE(3)).
- The ICP outer loop: associate (nearest neighbor), reject (max-distance / trimmed),
  solve, update, iterate to convergence.

Provided (not a hole): the kd-tree for nearest-neighbor correspondence. A correct
balanced kd-tree is a 1-2 hour build that is not the registration lesson, so it is given
(a thin wrapper over a provided implementation or `scipy.cKDTree`-equivalent); the learner
writes the correspondence use and outlier rejection, not the tree.

Oracle: cross-check the converged transform against Open3D's ICP/GICP on the same clouds;
visualize the residual difference. The from-scratch solver is graded; Open3D is the
labeled side-by-side.

Tests: on a cloud transformed by a known SE(3), point-to-point recovers the inverse to
~1e-6 from a close initialization (including a planar-cloud case that exercises the
reflection-avoiding det-correction); point-to-plane converges faster (fewer iterations to
the same residual) on a surface-rich cloud; with the oracle present, the from-scratch and
Open3D transforms agree within tolerance (skipped if Open3D absent).

Viz: watch the two clouds converge iteration by iteration in interactive 3D - the moving
cloud sliding onto the fixed one, residual shrinking on a Rerun plot beside it. Point-to-
point vs point-to-plane convergence side by side.

Interview section: point-to-point vs point-to-plane (and why the latter converges faster
on structured scenes); the SVD solution to the orthogonal Procrustes problem; ICP local
minima and initialization; GICP/NDT as the production variants; correspondence cost.
On-the-spot ask: derive the point-to-plane linearization.

### a14_5_factor_graph - Pose-graph and bundle adjustment  [Core]

Depends on: a14_0, a14_3 (the BA reprojection residual reuses a14_3's projection and
a11_5a's pinhole intrinsics, transitively; cite the pinhole model explicitly).

Holes:
- SE(2)/SE(3) pose-graph residuals: the between-factor error
  `r_ij = Log(T_meas_ij⁻¹ · (T_i⁻¹ T_j))`, where under the repo's `T_b_a` naming the
  measured relative pose must be `T_meas_ij = T_i_j` (so that `T_i⁻¹ T_j = T_i_j` and the
  residual is `Log(T_i_j,meas⁻¹ · T_i_j)`). The solution commits to one published
  derivation (Barfoot / GTSAM `BetweenFactor`) for the analytic Jacobians under right
  perturbation: `∂r/∂δ_i = -J_r⁻¹(r) · Ad_{T_j⁻¹ T_i}` and `∂r/∂δ_j = J_r⁻¹(r)`. The
  minus sign on the `i`-side and the `Ad` argument are the single most common hand-rolled-
  pose-graph bug; the numerical-vs-analytic test gates it, but the reference solution must
  carry the exact form, not "use J_r⁻¹ and the adjoint".
- The Gauss-Newton and Levenberg-Marquardt loops: assemble the sparse normal equations
  `H = Σ Jᵀ Ω J`, `b = Σ Jᵀ Ω r`, solve `H δ = -b` (Eigen `LDLT`/`SimplicialLLT` on the
  assembled `H` - a full AMD-ordered sparse Cholesky is out of scope), retract on the
  manifold, iterate; LM adds the damping term and the accept/reject step. Anchor one pose
  (or add a prior) to fix the gauge freedom.
- The Schur complement for BA-style landmark marginalization: partition `H` into pose and
  landmark blocks, marginalize the landmark block (which is block-diagonal, one block per
  landmark - that structure is what makes it cheap and is the lesson) to a reduced pose
  system, back-substitute.
- A loop-closure edge added to a drifted odometry chain that corrects the trajectory.

Oracle: cross-check the optimized trajectory against g2o or GTSAM on the same graph;
overlay both. From-scratch is graded; the production solver is the labeled side-by-side.

Tests: on a synthetic pose graph with a known solution, Gauss-Newton drives the total
residual to ~0 and recovers the ground-truth poses up to gauge freedom to ~1e-6; the
analytic edge Jacobians match numerical to ~1e-6; the Schur-complement BA solve equals
the dense solve on the same problem; adding the loop-closure edge reduces end-of-
trajectory drift by the expected large factor; with the oracle present, the from-scratch
and g2o/GTSAM trajectories agree within tolerance (skipped if absent).

Viz: watch the trajectory snap into place the instant the loop-closure edge fires, side
by side with the production solver's result; show the sparse `H` structure and the
arrowhead fill-in. The drift-then-correct moment is the centerpiece.

Interview section: filter vs smoother (why factor graphs / iSAM replaced EKF-SLAM); the
Schur complement / marginalization trick and why BA exploits the pose-landmark block
structure; Gauss-Newton vs LM vs dogleg; gauge freedom and fixing it; sparsity and why it
makes large BA tractable; g2o vs GTSAM vs Ceres. On-the-spot ask: write the relative-pose
residual and one of its Jacobians.

## Canonical sources to cite (per assignment README)

The README author verifies each link by fetching the arXiv/DOI page before citing.
- a14_0: Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics"
  (arXiv:1812.01537); Barfoot, *State Estimation for Robotics* (2017), ch. 7.
- a14_1: Kalman (1960); Julier & Uhlmann, "Unscented filtering and nonlinear estimation"
  (Proc. IEEE 2004) and Wan & van der Merwe (2000) for the UKF/sigma weights; Thrun,
  Burgard, Fox, *Probabilistic Robotics* (2005) for EKF / information form.
- a14_2: Smith, Self, Cheeseman (1990, the stochastic map); Dissanayake et al. (2001);
  Bailey & Durrant-Whyte SLAM tutorial I/II (2006); Huang & Dissanayake (2007) on
  EKF-SLAM inconsistency.
- a14_3: Hartley & Zisserman, *Multiple View Geometry* (2004, ch. 9-11); Hartley, "In
  defense of the eight-point algorithm" (PAMI 1997); Longuet-Higgins (1981); Nister, "An
  efficient solution to the five-point relative pose problem" (PAMI 2004); Lepetit et al.,
  "EPnP" (IJCV 2009); Fischler & Bolles, "RANSAC" (1981).
- a14_4: Besl & McKay (PAMI 1992, point-to-point ICP); Chen & Medioni (1991/92, point-to-
  plane); Umeyama (1991, the SVD least-squares with det-correction); Segal, Haehnel,
  Thrun, "Generalized-ICP" (RSS 2009).
- a14_5: Lu & Milios (1997); Kümmerle et al., "g2o" (ICRA 2011); Kaess et al., "iSAM2"
  (IJRR 2012); Triggs et al., "Bundle adjustment - a modern synthesis" (2000); Dellaert &
  Kaess, "Factor Graphs for Robot Perception" (2017).

## Reading-only notes: a14_classical_slam/notes/ (not built)

In the repo's `notes/` style, reading-only:
- IMU pre-integration and visual-inertial fusion (the manifold pre-integration trick,
  why VIO needs it).
- Time-sync and extrinsic/temporal calibration (the cost of a few ms of skew, tying back
  to a11_5a's temporal-offset exercise).
- Place recognition / loop-closure detection (bag-of-words DBoW2, and learned global
  descriptors NetVLAD / modern variants) - this module assumes a loop closure is given;
  this note covers how it is found.
- A one-page bridge from classical to learned geometry: where this module meets the
  learned geometry foundation models already in the repo (DUSt3R/VGGT in a10_5) and BEV
  perception (a11_5), and where classical and learned SLAM are converging in 2026.

## Build sequencing and gates

1. Viewer smoke test (learner runs the Rerun script). DONE 2026-06-12: window opens under
   WSL2 (Wayland/WSLg) on a software rasterizer, orbit/zoom and timeline scrub all work;
   software rendering is fine at these scene sizes.
2. Expert review of this plan (CV/SLAM correctness gate, orchestrator-owned). DONE
   2026-06-12: corrections folded into a14_0 (adjoint direction, separate Jacobian series
   branches, J_r defining property), a14_1 (nonlinear process model + F_x hole, the β term
   on Wc⁽⁰⁾, agreement reframed as floating-point not linearization), a14_2 (replaced the
   strict marginal-determinant test with the Loewner-monotone joint statement; the loop is
   a demonstration of inconsistency, not a consistency gate), a14_3 (pinned the two-view
   conventions, Sampson inliers, added the composed front-end hole, flagged five-point),
   a14_4 (kd-tree now provided, exact det-correction form and source/target direction),
   a14_5 (committed to the explicit Jacobian form with the i-side minus sign and the
   T_meas_ij = T_i_j naming; H_ll block-diagonal; gauge anchor). Canonical sources added.
3. Toolchain skeleton: stand up the CMake + pybind11 + `_impl.py` + conftest pattern on
   a14_0 first, prove both modes green and the Rerun viz renders, get the learner to
   eyeball the first real animation. This de-risks the harness before the other five
   lean on it.
4. Build a14_0 -> a14_5 in dependency order, each: solution first, carve holes, write
   tests against the solution, write the Rerun viz, write the README (lecture notes +
   assignment + interview section), run the context-less style pass, verify both modes on
   disk. Pause to compact when context fills (~55-60%).
5. The `notes/` writeup after the assignments.
6. Top-level README, BUILD_ORDER, environment.yml updates; `make verify-all` stays green
   (the a14 targets build then run; Python assignments unaffected).

## Open risks

- WSL2 interactive GL for Rerun. Retired or characterized by gate 1 before building.
- C++ build time inside `make test`. Targets are small; expected sub-second incremental,
  seconds cold. If a cold build is slow, cache the configured build dir under `build/`.
- Oracle library availability (GTSAM/Open3D pip wheels on this platform). Oracle tests
  skip cleanly when absent, so a missing wheel never breaks the green bar; only the
  side-by-side viz degrades.
- Test determinism. Favor analytic exact checks (round-trips, Jacobian-vs-numerical,
  noise-free recovery to 1e-6) over convergence-of-a-run thresholds, per the
  no-fragile-test gate. Where a robust/naive ordering is the lesson, assert the ordering
  across seeds, not a pinned number.
- Scope. Six C++ assignments with viz and oracles is multi-session. Build incrementally;
  do not autopilot all six before the a14_0 harness is proven.
```

