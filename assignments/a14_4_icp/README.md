# a14_4 - Point-cloud registration (ICP)

The fifth assignment of the classical SLAM module, in C++17 with Eigen. You build iterative
closest point (ICP): given two point clouds of the same scene from different viewpoints, find
the rigid transform that aligns them. ICP is the workhorse of LiDAR odometry and depth-camera
registration, the step that turns overlapping scans into a single consistent reconstruction,
and a standard interview topic in its own right.

It builds on the Lie-group assignment: the point-to-plane step updates the pose on the SE(3)
manifold by the same retraction. The SE(3) exponential, the skew operator, and the
nearest-neighbor correspondence search are carried over and provided here in `models.cpp`, so
your work is the two solvers and the registration loop.

Required reading before you start:
- Besl and McKay, "A method for registration of 3-D shapes" (PAMI 1992) - the original ICP
  paper (point-to-point).
- Chen and Medioni, "Object modelling by registration of multiple range images" (1992) - the
  point-to-plane variant.
- Umeyama, "Least-squares estimation of transformation parameters between two point patterns"
  (PAMI 1991) - the closed-form solution with the reflection-avoiding determinant correction.

## Lecture notes

### The registration problem and the ICP loop

Two clouds, a source $P$ and a target $Q$, sample the same surface from different poses. The
goal is the rigid transform $T = (R, t)$ that maps the source onto the target, $q \approx R p
+ t$. If the correspondences were known (which source point matches which target point), this
would be a one-shot least-squares solve. They are not known, so ICP alternates two steps until
it converges:

1. Assign correspondences: for each (currently transformed) source point, its nearest target
   point.
2. Solve for the transform that best aligns those correspondences, apply it, and repeat.

Each iteration strictly decreases the mean correspondence distance, so the loop converges -
but to a local minimum, which is why ICP needs a reasonable initialization. From a pose that
is already roughly right, the nearest-neighbor assignment is mostly correct and ICP polishes
it to high accuracy; from a bad start it can lock onto the wrong alignment. The outer loop
also rejects correspondences farther apart than a gate (a maximum distance, or keeping only
the closest fraction), so that points with no real match in the other cloud - from partial
overlap or clutter - do not drag the solve.

In this module's source-to-target naming, the estimate is $T_{\text{target} \leftarrow
\text{source}}$, and a transform applied to the cloud uses $q = R p + t$.

### Point-to-point: the closed-form solve

Given matched correspondences, the transform that minimizes $\sum_i \lVert R p_i + t - q_i
\rVert^2$ has a closed form - this is the orthogonal Procrustes problem, solved by Umeyama and
Kabsch. Center both clouds at their centroids $\bar p, \bar q$, form the $3 \times 3$
cross-covariance

$$H = \sum_i (p_i - \bar p)(q_i - \bar q)^\top,$$

take its SVD $H = U \Sigma V^\top$, and read off the rotation and translation:

$$R = V \begin{bmatrix} 1 & & \\ & 1 & \\ & & \det(V U^\top) \end{bmatrix} U^\top, \qquad t = \bar q - R \bar p.$$

The translation just lines up the centroids once the rotation is known. The
$\det(V U^\top)$ term in the middle is the part people forget. Without it, $R = V U^\top$ is
the orthogonal matrix closest to the data, but on planar or otherwise degenerate clouds (where
$H$ is rank-deficient) that closest orthogonal matrix can be a reflection, $\det R = -1$,
which is not a rotation and warps the cloud. Replacing the last diagonal entry with
$\det(V U^\top)$ forces $\det R = +1$ while changing nothing when the data already prefers a
proper rotation. The planar case in this assignment's tests exists to catch an implementation
that drops this correction.

### Point-to-plane: minimize distance to the surface, not to the point

Point-to-point pays the full distance between a source point and its matched target point,
even the component that slides along the surface - but two scans of the same surface rarely
sample the same physical points, so that sliding component is not real error. Point-to-plane
fixes this by measuring only the distance along the target's surface normal $n_i$:

$$\min_{R, t} \sum_i \big( n_i \cdot (R p_i + t - q_i) \big)^2.$$

A point is then free to slide within the local tangent plane at no cost, which is exactly the
freedom real scans have, and the result is that point-to-plane converges in far fewer
iterations on structured surfaces. Running this assignment's loop on a bumpy surface,
point-to-plane reaches a tight alignment in two or three iterations where point-to-point needs
ten or more.

The cost is nonlinear in the rotation, so it is solved by Gauss-Newton with a small-motion
linearization. Parametrize the update as the retraction $T \leftarrow \mathrm{se3\_exp}(\xi)\,
T$ with $\xi = [\rho; \theta]$ (translation part first, the Lie-group assignment's twist
ordering). To first order the moved point is $p_i' \approx p_i - [p_i]_\times \theta + \rho$,
so the residual linearizes to

$$r_i \approx (p_i - q_i)\cdot n_i + n_i \cdot \rho + (p_i \times n_i)\cdot\theta = (p_i - q_i)\cdot n_i + a_i^\top \xi, \qquad a_i = \begin{bmatrix} n_i \\ p_i \times n_i \end{bmatrix}.$$

Setting the derivative to zero gives the $6 \times 6$ normal equations $\big(\sum_i a_i
a_i^\top\big)\,\xi = \sum_i a_i b_i$ with $b_i = -(p_i - q_i)\cdot n_i$, solved once per
iteration; the increment is $\mathrm{se3\_exp}(\xi)$, composed onto the current pose. The
normals are a property of the target surface (provided here; in practice they are estimated
from the target cloud by fitting a plane to each point's neighbors). Deriving this
linearization on the spot is a common interview ask.

### Why the kd-tree is given, not built

The correspondence step needs, for each source point, its nearest target point. A brute-force
scan is $O(nm)$ and fine at teaching cloud sizes; production uses a kd-tree (or an
approximate-nearest-neighbor structure) to make each query $O(\log m)$. A correct balanced
kd-tree is a one-to-two-hour build that is not the registration lesson, so the search is
provided (`nearest_neighbors` in `models.cpp`, brute force). Your loop calls it and decides
which correspondences to keep; you do not write the tree. The dominant cost of real ICP is
this correspondence search, which is why the data structure matters in practice.

### Local minima, initialization, and the production variants

ICP minimizes a non-convex cost by alternating assignment and solving, so it finds the nearest
local minimum, not the global one. A coarse initialization (from odometry, IMU, or a global
registration like feature matching or RANSAC) is what keeps it in the right basin. The
production variants relax the hard nearest-point assignment: generalized ICP (GICP) models each
point as a small Gaussian disk and minimizes a plane-to-plane cost, and NDT (the normal
distributions transform) replaces the target cloud with a grid of Gaussians and registers
against that. Both are more robust to sampling differences than vanilla ICP, and both are
still local methods that need an initialization.

## The assignment

Implement the two solvers and the registration loop in C++. The SE(3) retraction, the skew
operator, the nearest-neighbor search, the pybind11 bindings, the CMake build, the synthetic
clouds, the tests, and the Rerun visualization are provided.

### Files to modify

`icp.cpp` holds three holes:

- `align_point_to_point` - the closed-form Umeyama/Kabsch solve, including the determinant
  correction that keeps the rotation proper.
- `point_to_plane_step` - one Gauss-Newton step of the point-to-plane cost: assemble the
  $6 \times 6$ normal equations from the $[n_i; p_i \times n_i]$ rows and return the
  incremental transform.
- `icp` - the outer loop: transform, find and gate correspondences, solve (point-to-point or
  point-to-plane), compose, iterate.

`models.cpp` (the SE(3) `se3_exp`, the skew `hat3`, and the brute-force `nearest_neighbors`,
carried over from the Lie-group assignment and standing in for a kd-tree) is provided and
compiled in both builds; call those, do not reimplement them. Each hole's contract is in the
comment at the hole and in `icp.hpp`; the math is in the lecture notes. The reference is in
`solution/icp.cpp`. You may not include an existing registration or solver library in the C++;
a test scans the sources (Open3D is allowed only as the Python oracle below).

### Building and running

Same toolchain as the rest of the module (C++17, CMake, pybind11, Eigen). You never call CMake
by hand.

```
make verify A=a14_4_icp   # build + run the reference (solution/); the green target
make test   A=a14_4_icp   # build + run YOUR code; red until the holes are filled
make viz    A=a14_4_icp        # render the ICP convergence (reference)
make viz-mine A=a14_4_icp      # the same, from YOUR code, once the holes are filled
```

The tests favor implementation-independent statements. The closed-form solve recovers a known
transform from matched correspondences to numerical precision and keeps the rotation proper
(det $+1$) on a planar cloud; the point-to-plane step, iterated on matched correspondences,
recovers a known transform, and a single step from the identity already reduces the error; the
full loop recovers the transform from an identity start in both modes; point-to-plane converges
in far fewer iterations than point-to-point on a surface-rich cloud (asserted as a wide-margin
ordering); and the maximum-distance gate rejects unmatched outlier points so the result stays
accurate.

If Open3D is installed, `test_oracle_open3d.py` cross-checks the from-scratch transform against
Open3D's registration on the same clouds; it is skipped when Open3D is absent. The from-scratch
solver is the graded code; Open3D is the labeled side-by-side, never the answer.

`make viz` writes `out/icp.rrd`. Add `SHOW=1` for the interactive viewer: the fixed target
cloud and the source cloud sliding onto it, point-to-point and point-to-plane side by side,
with a scalar panel plotting each method's RMS correspondence distance as it shrinks. Scrub the
timeline to step through the iterations and watch point-to-plane converge first.

The clouds are synthetic: a bumpy height field with analytic surface normals (the surface-rich
case where point-to-plane shines) and a flat sheet (the planar degenerate case for the
determinant correction). A real pipeline would add a normal-estimation step on the target and a
kd-tree for the search, but the registration math is identical, and synthetic clouds with a
known transform give the deterministic ground truth the tests need.

## In interviews

ICP is a standard interview topic for LiDAR, depth-camera, and SLAM roles, both as a derivation
and as a "how would you register two scans" discussion.

Point-to-point versus point-to-plane, and why the latter converges faster. The answer is that
point-to-point penalizes the full point-to-point distance including the tangential slide, while
point-to-plane penalizes only the distance along the surface normal, letting points slide
within the tangent plane at no cost - which matches how real scans sample different physical
points on the same surface, so it converges in far fewer iterations on structured scenes.

The orthogonal Procrustes solution. Be ready to derive or state the SVD solution for the
point-to-point rotation, and specifically the $\det(V U^\top)$ correction and why it is there
(to prevent a reflection on degenerate or planar data). This is the single most asked detail.

ICP's local minima and the role of initialization. ICP is a local method that descends to the
nearest minimum of a non-convex cost, so it needs a coarse initialization (odometry, IMU, or a
global registration) to land in the right basin. Knowing that this is the failure mode, and
that GICP and NDT are the more robust production variants, is a strong signal.

The cost of correspondence. The dominant cost of real ICP is the nearest-neighbor search, which
is why it runs on a kd-tree or an approximate-nearest-neighbor structure rather than a
brute-force scan. A common follow-up: how would you speed up ICP on a million-point cloud (the
answer involves the search structure, downsampling, and point selection). The standard
on-the-spot ask is to derive the point-to-plane linearization.

## Further reading

- Besl and McKay (1992) and Chen and Medioni (1992) - the two original ICP variants.
- Umeyama (1991) - the closed-form point-to-point solution with the determinant correction.
- Segal, Haehnel, Thrun, "Generalized-ICP" (RSS 2009) - the plane-to-plane production variant.
- Rusinkiewicz and Levoy, "Efficient variants of the ICP algorithm" (3DIM 2001) - a survey of
  the design choices (point selection, weighting, rejection, the error metric).
