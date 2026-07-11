# a14_5 - Pose-graph and bundle adjustment

The last assignment of the classical SLAM module, in C++17 with Eigen. You build the
factor-graph back-end: the smoothing optimizer that replaced filtering as the standard SLAM
estimator. You implement the between-factor residual and its analytic Jacobians, the
Gauss-Newton loop that optimizes a pose graph, and the Schur complement that makes bundle
adjustment tractable. The centerpiece is loop closure - watching a drifted trajectory snap
into place the instant a single loop-closure edge is added.

It builds on the Lie-group assignment: poses live on SE(3), the residual is a logarithm, the
Jacobians use the right perturbation and the inverse right Jacobian, and the updates retract on
the manifold. The reprojection-based bundle-adjustment view also reuses the pinhole projection
from the multi-view assignment. The SE(3) machinery - the exponential and logarithm, the
adjoint, and the 6x6 inverse right Jacobian - is carried over and provided in `models.cpp`, so
your work is the factors, the normal equations, and the marginalization on top of it.

Required reading before you start:
- Grisetti, Kümmerle, Stachniss, Burgard, "A tutorial on graph-based SLAM" (2010) - the
  pose-graph formulation, the error function, and the Gauss-Newton solution.
- Dellaert and Kaess, "Factor graphs for robot perception" (2017) - the factor-graph view,
  smoothing, and the sparsity that makes it scale.
- Triggs et al., "Bundle adjustment - a modern synthesis" (2000) - bundle adjustment and the
  Schur complement.

## Lecture notes

### Filtering versus smoothing

The EKF-SLAM assignment kept a single Gaussian over the current robot pose and the map, and
marginalized out every past pose as the robot moved. That is filtering, and it has two costs
the assignment named: a dense covariance that makes each update quadratic in the map size, and
linearization errors that are baked in permanently, because a marginalized pose can never be
re-linearized when later evidence arrives.

Smoothing keeps all the poses (and optionally the landmarks) as free variables and solves for
all of them at once, every time. The variables and the measurements connecting them form a
factor graph: a node per pose, a factor per measurement (an odometry edge, a loop closure, a
landmark observation). The estimate is the configuration that minimizes the total weighted
squared residual over all factors. Two things make this practical despite keeping everything:
the problem is sparse (each factor touches only two nodes), and the variables can be
re-linearized at every iteration, so linearization error does not accumulate the way it does in
a filter. This is why factor graphs and incremental smoothing (iSAM) replaced EKF-SLAM.

### The pose graph and the between-factor

A pose graph has a node per robot pose $T_i \in SE(3)$ and an edge per relative-pose
measurement. An edge between $i$ and $j$ carries a measured relative pose $T_{\text{meas}} =
T_{i \leftarrow j}$ (frame $j$ expressed in frame $i$, in the module's source-to-target naming)
and an information matrix $\Omega$ (the inverse measurement covariance). The residual is the
difference, on the manifold, between the measured relative pose and the one the current
estimates predict:

$$r_{ij} = \mathrm{Log}\big(T_{\text{meas}}^{-1}\, (T_i^{-1} T_j)\big) \in \mathbb{R}^6.$$

It is zero exactly when the estimated relative pose $T_i^{-1} T_j$ equals the measurement.
The total cost summed over edges is $\sum_{ij} r_{ij}^\top \Omega_{ij}\, r_{ij}$.

The analytic Jacobians of $r_{ij}$ with respect to the right perturbations $\delta_i, \delta_j$
of the two poses (where $T \leftarrow T\, \mathrm{exp}(\delta)$) are the part to get exactly
right. Using the published derivation (Barfoot; GTSAM's `BetweenFactor`):

$$\frac{\partial r_{ij}}{\partial \delta_i} = -\,\mathcal{J}_r^{-1}(r_{ij})\, \mathrm{Ad}_{T_j^{-1} T_i}, \qquad \frac{\partial r_{ij}}{\partial \delta_j} = \mathcal{J}_r^{-1}(r_{ij}),$$

where $\mathcal{J}_r^{-1}$ is the $6 \times 6$ inverse right Jacobian of SE(3) (provided) and
$\mathrm{Ad}$ is the adjoint. The minus sign on the $i$-side and the adjoint argument
$T_j^{-1} T_i$ are the single most common hand-rolled-pose-graph bug; the
numerical-versus-analytic test gates them. The inverse right Jacobian $\mathcal{J}_r^{-1}(r)$
appears because the residual is a logarithm: perturbing $T_j$ by $\delta_j$ changes the
argument of $\mathrm{Log}$ by a right multiplication, and the derivative of $\mathrm{Log}(\exp(r)
\exp(\delta))$ with respect to $\delta$ at $\delta = 0$ is exactly $\mathcal{J}_r^{-1}(r)$. Near
convergence $r \to 0$ and $\mathcal{J}_r^{-1} \to I$, but the test runs at nonzero residual, so
the full form is required.

### Gauss-Newton, the normal equations, and the gauge

With residuals and Jacobians, Gauss-Newton minimizes the total cost. Each iteration stacks the
per-edge contributions into the sparse normal equations over the 6-DOF-per-pose state:

$$H = \sum_{ij} J_{ij}^\top \Omega_{ij} J_{ij}, \qquad g = \sum_{ij} J_{ij}^\top \Omega_{ij}\, r_{ij},$$

where each edge writes its $J_i, J_j$ blocks into the four positions $(i,i), (i,j), (j,i),
(j,j)$ of $H$ and the $i, j$ positions of $g$. Solve $H\,\delta = -g$ (a Cholesky factorization,
Eigen's `LDLT` here; a production solver adds a fill-reducing ordering, which is out of scope),
then retract each pose $T_i \leftarrow T_i\, \mathrm{exp}(\delta_i)$, and iterate.

One subtlety: a pure pose graph has a gauge freedom. Nothing in the relative-pose measurements
fixes where the whole trajectory sits in the world - translating and rotating every pose
together leaves every residual unchanged - so $H$ is singular (it has a six-dimensional null
space). Fixing the gauge means pinning one pose: this assignment anchors pose 0 by holding its
6 DOF fixed (equivalently, adding an infinitely strong prior on it), which removes the null
space and makes $H$ invertible on the remaining poses. Levenberg-Marquardt adds a damping term
$H + \lambda \, \mathrm{diag}(H)$ and an accept/reject step on $\lambda$, which makes the solve
robust to a poor initialization; Gauss-Newton is the undamped $\lambda = 0$ case and is enough
for the well-initialized graphs here.

### Loop closure: the moment that matters

Odometry drifts. Integrating noisy relative-pose measurements along a trajectory accumulates
error, so a robot that drives a loop comes back to a start pose that does not match where it
began. The odometry edges alone cannot fix this - they are all individually satisfied by the
drifted trajectory, so the optimizer leaves it untouched. A loop-closure edge is a measurement
that the robot has returned to a place it has been (from re-recognizing a landmark or matching a
scan), tying a late pose back to an early one. That single edge has a large residual on the
drifted trajectory, and Gauss-Newton distributes its correction back along the whole loop,
snapping the trajectory into a consistent shape. Running this assignment's loop, adding the
loop-closure edge cuts the end-of-trajectory drift by an order of magnitude; removing it leaves
the drift exactly in place. This is the drift-then-correct moment that the visualization is
built around.

### Bundle adjustment and the Schur complement

Bundle adjustment is the same machinery with landmarks added: the variables are camera poses
and 3D points, and the factors are reprojection residuals (a point projected into a camera
should land on its observed pixel, reusing the pinhole projection from the multi-view
assignment). The system matrix $H$ now has a pose block and a much larger landmark block. The
structure that makes BA tractable is that landmarks do not connect to each other - a landmark
factor touches one pose and one point - so the landmark-landmark block $H_{\ell\ell}$ is
block-diagonal, one small block per landmark.

The Schur complement exploits exactly this. Partition the system

$$\begin{bmatrix} H_{pp} & H_{p\ell} \\ H_{\ell p} & H_{\ell\ell} \end{bmatrix} \begin{bmatrix} \delta_p \\ \delta_\ell \end{bmatrix} = \begin{bmatrix} b_p \\ b_\ell \end{bmatrix},$$

and marginalize the landmarks. Because $H_{\ell\ell}$ is block-diagonal, inverting it is cheap
(invert each landmark's block independently), and the reduced system on the poses alone is

$$\underbrace{(H_{pp} - H_{p\ell} H_{\ell\ell}^{-1} H_{\ell p})}_{\text{reduced camera system}}\, \delta_p = b_p - H_{p\ell} H_{\ell\ell}^{-1} b_\ell.$$

Solve this much smaller system for the pose update, then back-substitute $\delta_\ell =
H_{\ell\ell}^{-1}(b_\ell - H_{\ell p}\, \delta_p)$ for the landmarks. The result is identical to
solving the full dense system, but the expensive solve is over the poses only (a few hundred
variables) instead of the poses plus thousands of points. This is the trick that makes
large-scale structure-from-motion and BA feasible.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the declaration and comments in each file give the signature and shapes.

1. [`between_residual()`](factor_graph.cpp) in `factor_graph.cpp`
2. [`between_jacobians()`](factor_graph.cpp) in `factor_graph.cpp`
3. [`optimize_pose_graph()`](factor_graph.cpp) in `factor_graph.cpp`
4. [`schur_solve()`](factor_graph.cpp) in `factor_graph.cpp`

### Building and running

Same toolchain as the rest of the module (C++17, CMake, pybind11, Eigen). You never call CMake
by hand.

```
make verify A=a14_5_factor_graph   # build + run the reference (solution/); the green target
make test   A=a14_5_factor_graph   # build + run YOUR code; red until the holes are filled
make viz    A=a14_5_factor_graph        # render the loop-closure correction (reference)
make viz-mine A=a14_5_factor_graph      # the same, from YOUR code, once the holes are filled
```

The tests favor implementation-independent statements. The residual is zero when the estimate
matches the measurement; the analytic edge Jacobians match numerical differentiation to ~1e-5
(the test that catches the sign and adjoint-argument bug); Gauss-Newton recovers the
ground-truth trajectory on a noise-free graph (up to the anchored gauge) and drives the cost to
zero; the anchored pose does not move; adding the loop-closure edge reduces end-of-trajectory
drift by a large factor while removing it leaves the drift in place; and the Schur-complement
solve equals the dense solve on a BA-structured system. If GTSAM is installed,
`test_oracle_gtsam.py` cross-checks the optimized trajectory against GTSAM's optimizer; it is
skipped when GTSAM is absent.

`make viz` writes `out/pose_graph.rrd`. Add `SHOW=1` for the interactive viewer: the
ground-truth loop, the drifted trajectory snapping onto it as the Gauss-Newton iterations
replay, the loop-closure edge, a cost panel falling to zero, and the block-sparsity of $H$ (the
band plus the off-corner the loop-closure edge creates). Scrub the timeline to watch the
correction propagate.

The pose graph is implemented on SE(3) (the trajectory here is planar, a loop with yaw, so it
reads as a 2D plot, but the code is the full 3D machinery and a SE(2) graph is the same
construction in three dimensions instead of six). The canonical real datasets for this method
are the g2o pose-graph benchmarks (Intel, MIT Killian Court, the sphere and torus graphs); the
tests use a synthetic loop because it gives the deterministic ground truth they need.

## In interviews

Factor-graph SLAM and bundle adjustment are standard interview topics for SLAM, AR, and
3D-reconstruction roles, both as derivations and as system-design discussions.

Filter versus smoother. Expect "why did factor graphs replace the EKF for SLAM?" The answer is
two things: smoothing keeps all poses and re-linearizes every iteration, so linearization error
does not get baked in the way it does when a filter marginalizes a pose; and the problem is
sparse, so keeping everything is affordable. Incremental smoothing (iSAM/iSAM2) makes it
real-time by updating the factorization rather than refactoring from scratch.

The relative-pose residual and one of its Jacobians. A common on-the-spot ask is to write
$r_{ij} = \mathrm{Log}(T_{\text{meas}}^{-1} T_i^{-1} T_j)$ and one of its Jacobians. Knowing that
the Jacobian carries the inverse right Jacobian $\mathcal{J}_r^{-1}(r)$ (because the residual is
a logarithm) and the adjoint, with the minus sign on the $i$-side, is a strong signal.

The Schur complement and why BA exploits structure. Be ready to explain that the landmark block
is block-diagonal (landmarks do not see each other), so marginalizing it is cheap and reduces
the solve to the camera poses, identical in result to the dense solve. This is the single most
important idea in scalable BA.

Gauge freedom and how to fix it. A pure pose graph (or a BA problem) has an unobservable global
transform, so the system matrix is singular; you fix it by anchoring one pose or adding a prior.
Not handling the gauge is a common bug - the solve fails or wanders. Related: Gauss-Newton
versus Levenberg-Marquardt versus dogleg (damping for robustness to a bad initialization), and
the production solvers g2o, GTSAM, and Ceres.

## Further reading

- Grisetti et al., "A tutorial on graph-based SLAM" (2010) - the pose-graph formulation and
  Gauss-Newton solution.
- Dellaert and Kaess, "Factor graphs for robot perception" (2017) - the factor-graph view and
  incremental smoothing.
- Kaess et al., "iSAM2: incremental smoothing and mapping using the Bayes tree" (IJRR 2012) -
  the incremental solver behind modern systems.
- Triggs et al., "Bundle adjustment - a modern synthesis" (2000) - BA and the Schur complement.
- Barfoot, *State Estimation for Robotics* (2017), chapter 7 - the SE(3) Jacobians used for the
  edge derivatives.
