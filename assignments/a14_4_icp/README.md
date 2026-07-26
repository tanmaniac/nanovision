# a14_4 - Point-cloud registration (ICP)

The fifth assignment of the classical SLAM module, in C++17 with Eigen. It builds iterative
closest point (ICP): given two point clouds of the same scene from different viewpoints, find
the rigid transform that aligns them. ICP is the workhorse of LiDAR odometry and depth-camera
registration, the step that turns overlapping scans into a single consistent reconstruction,
and a standard interview topic in its own right.

The point-to-plane solver steps the pose with the SE(3) exponential map from the Lie-group
assignment. That map and the skew operator are carried over into `models.cpp`, which also
provides the nearest-neighbor correspondence search, so the work here is the two solvers and
the outer loop.

Required reading before you start:
- Besl and McKay, "A method for registration of 3-D shapes" (PAMI 1992) - the original ICP
  paper (point-to-point).
- Chen and Medioni, "Object modelling by registration of multiple range images" (1992) - the
  point-to-plane variant.
- Umeyama, "Least-squares estimation of transformation parameters between two point patterns"
  (PAMI 1991) - the closed-form solution with the reflection-avoiding determinant correction.

## Lecture notes

### The registration problem

A point cloud is an unordered set of 3D points sampled off the surfaces in a scene: a LiDAR
sweep, the back-projected pixels of a depth image, a set of triangulated image features. Two
clouds, a source $P = \{p_i\}$ and a target $Q = \{q_j\}$, sample the same scene from two
sensor poses. Registration asks for the rigid transform $T = (R, t)$, with $R \in SO(3)$ and
$t \in \mathbb{R}^3$, that carries the source onto the target, $q \approx R p + t$. In the
module's frame naming that estimate is $T_{\text{target} \leftarrow \text{source}}$, and
applying it to a cloud means $q = R p + t$ on every row.

If the pairing were given - if row $i$ of $P$ were known to be the same physical point as row
$i$ of $Q$ - the problem would be a single least-squares solve,

$$\min_{R,\,t}\ \sum_i \lVert R p_i + t - q_i \rVert^2,$$

which has a closed form derived below. The difficulty is entirely in the pairing, and the
pairing does not exist. Two sweeps of the same wall do not hit the same physical spots; the
beam that lands on one patch of plaster in the first sweep lands centimeters away in the
second. The honest statement of the problem is surface to surface, with points only a sampling
of those surfaces. ICP handles that by pretending a pairing exists, computing it from the
current pose estimate, and recomputing it as the estimate improves.

### The ICP loop and why it converges

ICP alternates over the two unknowns, the pairing and the pose:

1. Assignment. Transform the source by the current $T$, and for each transformed source point
   take the nearest target point as its correspondence.
2. Solve. Holding those pairs fixed, compute the transform that minimizes the summed squared
   distance over the pairs, compose it onto $T$, and repeat.

Convergence follows from writing both unknowns into one cost. Let $c$ be the assignment map,
$c(i)$ the index of the target point paired with source point $i$, and

$$E(T, c) = \sum_i \lVert R p_i + t - q_{c(i)} \rVert^2.$$

Step 1 minimizes $E$ over $c$ with $T$ held fixed, and it does so exactly, because the terms
are independent and each is minimized by picking that point's nearest target. Step 2 minimizes
$E$ over $T$ with $c$ held fixed, again exactly, by the closed-form solve below. Neither step
can increase $E$, and $E \ge 0$, so the sequence of costs is non-increasing and bounded below,
and therefore converges. This is the argument in Besl and McKay (1992), and it is ordinary
alternating minimization, the same structure as k-means or expectation-maximization. It covers
the point-to-point mode as written. The point-to-plane mode takes one Gauss-Newton step on a
different cost instead of solving its own exactly, so the same guarantee does not transfer to
it; in practice it converges faster than the mode that has the proof.

What that argument does not give is a correct answer. It says the cost settles, not that the
pose is right. Minimizing $c$ out leaves $E(T) = \min_c E(T, c)$, a pointwise minimum over the
finitely many assignment maps, so it is piecewise smooth and thoroughly non-convex, and
alternating minimization walks downhill into whichever basin it started in. A concrete failure:
scan a corridor with pillars every 2 m and initialize the pose 1.5 m off along the corridor.
Every source point's nearest target point sits on the neighboring pillar, each solve is happy,
the loop converges in a handful of iterations, and the answer is one pillar out. Nothing in the
cost is violated - the cost really is small at that pose.

The gate is the other piece of the loop. Partial overlap and clutter mean some source points
have no counterpart in the target at all, but the assignment step still hands each of them a
nearest neighbor, and that wrong pair pulls on the solve with the same weight as a good one.
Two standard remedies: drop any pair farther apart than a fixed distance (this assignment, with
`MAX_CORR_DIST` in `config.py`, in the units of the cloud), or keep only the closest fraction of
pairs, which is trimmed ICP (Chetverikov et al. 2002). Gating costs the convergence proof, since
consecutive iterations then minimize sums over different point sets and the monotone argument no
longer applies. It is standard practice anyway, because ungated phantom correspondences do more
damage than the lost guarantee.

The loop in this assignment stops when the incremental transform is negligible - its
translation norm and $\lVert R_{\text{inc}} - I \rVert$ both below about $10^{-10}$ - or when
`max_iter` is reached.

### Least squares under a rotation constraint

Before the closed form, it is worth being clear about why the closed form is not just the normal
equations. Fitting a general affine map $p \mapsto A p + t$ to matched pairs is unconstrained
linear least squares in the 12 entries of $(A, t)$, and the usual $A^\top A x = A^\top b$ solve
handles it. But the minimizing $A$ is not a rotation. It will happily scale and shear to shave
off residual, which changes the shape of the cloud, and a cloud's shape is exactly what should
not change when the sensor moves.

Rigid registration constrains $R^\top R = I$ and $\det R = +1$: three degrees of freedom rather
than nine, and a curved, non-convex constraint set that the normal equations cannot express.
The constrained version has a name, the orthogonal Procrustes problem - find the orthogonal
matrix that best maps one set of vectors onto another - and it has an exact solution through the
SVD, found by Kabsch (1976) for molecular structures and given in the form used here, with the
reflection handled, by Umeyama (1991).

### The point-to-point solve

Take the pairs as given, so row $i$ of $P$ matches row $i$ of $Q$, and minimize
$\sum_i \lVert R p_i + t - q_i \rVert^2$. Three steps.

Separate the translation. For any fixed $R$, the cost is quadratic in $t$; setting its gradient
to zero gives $t = \bar q - R \bar p$, with $\bar p, \bar q$ the two centroids. So the
translation only lines up the centroids, and substituting it back leaves a problem in $R$ alone
over the centered clouds $\tilde p_i = p_i - \bar p$ and $\tilde q_i = q_i - \bar q$:

$$\min_R\ \sum_i \lVert R \tilde p_i - \tilde q_i \rVert^2.$$

Turn it into a trace maximization. Expanding the square gives
$\sum_i (\lVert \tilde p_i \rVert^2 + \lVert \tilde q_i \rVert^2 - 2\,\tilde q_i^\top R \tilde p_i)$,
using $\lVert R \tilde p_i \rVert = \lVert \tilde p_i \rVert$ because a rotation preserves
length. The first two terms do not involve $R$, so minimizing the cost is maximizing
$\sum_i \tilde q_i^\top R \tilde p_i$. That sum is a trace:

$$\sum_i \tilde q_i^\top R \tilde p_i = \operatorname{tr}\!\big(R H\big), \qquad H = \sum_i (p_i - \bar p)(q_i - \bar q)^\top.$$

$H$ is the $3 \times 3$ cross-covariance of the two centered clouds (up to the $1/n$ that does
not affect the maximizer): entry $(a, b)$ is how strongly coordinate $a$ of the source co-varies
with coordinate $b$ of the target across the pairs.

Maximize the trace with the SVD. Write $H = U \Sigma V^\top$ with $\Sigma = \operatorname{diag}(\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0)$.
Then $\operatorname{tr}(R H) = \operatorname{tr}(R U \Sigma V^\top)$, and cycling the trace turns
that into $\operatorname{tr}(M \Sigma)$ with $M = V^\top R U$, which is orthogonal whenever $R$
is. Since $\Sigma$ is diagonal, $\operatorname{tr}(M\Sigma) = \sum_k \sigma_k M_{kk}$, and every
diagonal entry of an orthogonal matrix satisfies $|M_{kk}| \le 1$ because its columns are unit
vectors. The sum is therefore at most $\sigma_1 + \sigma_2 + \sigma_3$, attained at $M = I$,
that is $R = V U^\top$.

Nothing in that chain enforced $\det R = +1$; it only enforced orthogonality. Since
$\det M = \det(V)\det(R)\det(U) = \det(R)\,\det(V U^\top)$, asking for a proper rotation means
asking for $\det M = \det(V U^\top)$. When that is $+1$ the unconstrained maximizer $M = I$ is
admissible and $R = V U^\top$ stands. When it is $-1$ the maximizer must have determinant $-1$,
and the best such $M$ flips the sign on the smallest singular value,
$M = \operatorname{diag}(1, 1, -1)$, which lowers the maximized trace by $2\sigma_3$ and nothing
more. Both cases collapse into one formula:

$$R = V \begin{bmatrix} 1 & & \\ & 1 & \\ & & \det(V U^\top) \end{bmatrix} U^\top, \qquad t = \bar q - R \bar p.$$

Dropping the middle matrix is the classic bug. Without it the solve can return a reflection,
$\det R = -1$, which is not a motion any rigid body can perform: it flips handedness, turning a
right-handed cloud into its mirror image. The penalty for correcting it is $2\sigma_3$, so the
correction is free exactly when $\sigma_3 = 0$, which is when $H$ is rank-deficient - a cloud
that is flat, or collinear, or otherwise has no extent in some direction.

The planar test in this assignment is built to expose it. `planar_cloud` is a flat sheet at
$z = 0$, and `test_planar_cloud_det_correction` mirrors the target across $x$, so the data is
literally a reflection of itself. Running the pieces on those 800 points, $H$ has singular values
$(2494,\ 2340,\ 0)$ - rank 2, as a flat sheet must be - and $\det(V U^\top) = -1$. The
uncorrected answer is $V U^\top = \operatorname{diag}(-1, 1, 1)$, a mirror across the $yz$ plane.
The corrected answer is $\operatorname{diag}(-1, 1, -1)$, which is a 180-degree rotation about
the $y$ axis. On the plane $z = 0$ the two agree exactly, so both fit the data perfectly and the
correction costs nothing; only one of them is a rotation. The test checks both halves of that:
$\det R = +1$, and the corrected transform still maps source onto target to $10^{-9}$.

### Surface normals and the point-to-plane cost

Point-to-point pays the whole vector between a source point and its matched target point,
including the part that slides along the surface. That sliding part is not error. Two scans
sample different physical points on the same wall, so a source point's true position relative to
its nearest target point includes an offset within the wall that is as large as the point
spacing, and no pose correction can remove it. Point-to-point tries anyway, and the sum of those
tugs is what drags the estimate around.

Point-to-plane measures only the component along the target surface normal. A surface normal at
a target point is the unit vector perpendicular to the local surface there. Real pipelines
estimate it: take the point's $k$ nearest neighbors, form the $3 \times 3$ covariance of those
neighbors about their mean, and take the eigenvector of the smallest eigenvalue, the direction
in which the neighborhood has the least spread, which is the normal of the best-fit plane
through them. The sign is ambiguous and is resolved by orienting normals toward the sensor. This
assignment provides them analytically instead: `terrain_cloud` is a height field
$z = a\sin(f_x x)\sin(f_y y)$, whose normal is the normalized $(-\partial z/\partial x,\, -\partial z/\partial y,\, 1)$.

With normals $n_i$ on the target, the cost is

$$\min_{R,\,t}\ \sum_i \big( n_i \cdot (R p_i + t - q_i) \big)^2 .$$

Geometrically, the target point $q_i$ is replaced by the tangent plane through $q_i$ with normal
$n_i$, and the residual is the signed distance from the moved source point to that plane. A
source point can slide anywhere within the tangent plane at zero cost, which is the freedom real
scans actually have. The directions that carried the sampling offsets are gone from the
objective, and the solve is left pulling only on the components that are genuine misalignment.

### Gauss-Newton and the normal equations

The point-to-plane cost is nonlinear in $R$, so there is no closed form and it is minimized
iteratively. Gauss-Newton is the standard method for a sum of squared residuals
$\sum_i r_i(x)^2$ where each $r_i$ depends nonlinearly on the parameters $x$. Linearize each
residual about the current estimate,

$$r_i(x + \delta) \approx r_i(x) + J_i\,\delta, \qquad J_i = \frac{\partial r_i}{\partial x},$$

a row vector with one entry per parameter. The cost in the increment $\delta$ is then an honest
quadratic, $\sum_i (r_i + J_i \delta)^2$, and setting its gradient to zero gives

$$\Big(\sum_i J_i^\top J_i\Big)\,\delta = -\sum_i J_i^\top r_i .$$

These are the normal equations: the same $A^\top A\,x = A^\top b$ that solves an overdetermined
linear system $A x \approx b$ in the least-squares sense, with $A$ the stacked Jacobian rows and
$b$ the negated residuals. Solve for $\delta$, apply it, relinearize at the new estimate, repeat.
Gauss-Newton is Newton's method with the term involving second derivatives of the residuals
dropped, which is a good approximation precisely when the residuals are small at the solution -
which is the situation when registering two scans of the same surface.

One wrinkle: the parameters here are a pose, and poses do not add. The increment is a
six-dimensional twist $\xi = [\rho; \theta]$ with the translation part first, following the
Lie-group assignment's ordering, and it reaches the group through the exponential map
`se3_exp`. Where it goes matters. This loop composes on the left, $T \leftarrow \exp(\widehat\xi)\,T$,
so $\xi$ is a motion expressed in the target frame acting on source points that have already
been moved by the current $T$. The Lie-group assignment's box-plus retraction is the right-hand
version, $T\exp(\widehat\xi)$, a correction in the body frame. The two express the same set of
updates and are related by the adjoint,
$\exp(\widehat\xi)\,T = T\,\exp(\widehat{\mathrm{Ad}_{T^{-1}}\xi})$; the left version is used here
because it keeps the increment in the same frame as the residuals.

### Linearizing the point-to-plane step

Let $p_i$ now denote the current transformed source point, $q_i$ its matched target point, and
$n_i$ that target point's normal. To first order $\exp(\widehat\xi) \approx I + \widehat\xi$, so
the increment moves the point to

$$p_i' \approx p_i + \rho + \theta \times p_i = p_i + \rho - \widehat{p_i}\,\theta,$$

where $\widehat{a}$ is the skew (hat) operator from the Lie-group assignment, defined by
$\widehat{a}\,b = a \times b$. Dotting with the normal and using the scalar triple product
identity $n \cdot (p \times \theta) = \theta \cdot (n \times p)$:

$$r_i = n_i \cdot (p_i' - q_i) \approx (p_i - q_i)\cdot n_i + n_i \cdot \rho + (p_i \times n_i)\cdot\theta = (p_i - q_i)\cdot n_i + a_i^\top \xi, \qquad a_i = \begin{bmatrix} n_i \\ p_i \times n_i \end{bmatrix}.$$

So the Jacobian row is $a_i^\top$ and the constant term is the current signed distance from the
point to its target's tangent plane. Feeding those into the normal equations above gives a
$6 \times 6$ system,

$$\Big(\sum_i a_i a_i^\top\Big)\,\xi = \sum_i a_i b_i, \qquad b_i = -(p_i - q_i)\cdot n_i,$$

solved once per iteration; the increment is `se3_exp(xi)`, composed onto the current pose. The
matrix is symmetric and positive semi-definite, and the reference factors it with Eigen's
`LDLT`, a Cholesky-type factorization for symmetric matrices. Assembly is a single pass over the
correspondences accumulating a $6\times 6$ and a 6-vector, which is free next to the
nearest-neighbor search.

The two halves of $a_i$ read directly. The top half $n_i$ is how fast a pure translation changes
the point-to-plane distance, and the bottom half $p_i \times n_i$ is the moment of the normal
about the origin, how fast a rotation changes it - the same force-and-torque pairing as a wrench
in rigid-body mechanics. Deriving this on the spot is a common interview ask.

### When point-to-plane goes singular

Removing the tangential directions from the cost is the whole point, and it is also the failure
mode. If every normal in the overlap points the same way, nothing in the cost constrains sliding
within that plane, and $\sum_i a_i a_i^\top$ loses rank.

The two clouds in this assignment show both sides of it. On the flat sheet from `planar_cloud`
(800 points, all normals $+z$), the eigenvalues of $\sum_i a_i a_i^\top$ come out as
$(0,\ 0,\ 0,\ 796,\ 2388,\ 2478)$: exactly three zeros, which are the two in-plane translations
and the rotation about the sheet normal. On the bumpy `terrain_cloud` the same matrix has
eigenvalues spread from roughly 40 to 2000 with no null space, because the normals vary across
the cloud. That is why the point-to-plane tests run on the terrain cloud and the determinant
test runs on the sheet.

The real-world version is a tunnel, a long featureless corridor, or a highway with no structure
ahead: the along-axis direction is unconstrained and the solve drifts along it while reporting a
small residual. Production LiDAR systems detect this from the spectrum of the same matrix -
watching the smallest eigenvalue, or its ratio to the largest - and either suppress the update
in that direction or lean on another sensor (Zhang, Kaess and Singh 2016).

### What convergence looks like here

Running this assignment's loop on the terrain cloud with the motion used in the convergence test
(a rotation of about 11.4 degrees and a translation of about 0.31 in cloud units), point-to-plane
drives the rotation error under 0.01 degrees in 2 iterations; point-to-point needs 12. The first
iteration is where the gap opens: point-to-plane goes from 11.4 degrees to 0.66 degrees, while
point-to-point reaches 5.4 degrees and then creeps, 4.6, 4.1, 3.6, 3.1, and so on, because each
solve is still being tugged by the tangential sampling offsets. `test_point_to_plane_converges_faster`
asserts only the wide-margin ordering (plane within 4 iterations, point at least twice that), so
it does not depend on the exact counts, which shift with the seed and the size of the initial
motion.

These are toy clouds with analytic normals and no sensor noise. What they reproduce is the
qualitative result from Chen and Medioni (1992): on locally planar structure the plane metric
converges in far fewer iterations than the point metric.

### The correspondence search

Every iteration needs, for each source point, its nearest target point. A brute-force scan is
$O(nm)$ for $n$ source and $m$ target points. At the sizes here, 800 against 800, that is 640k
distance evaluations per iteration and nobody notices. A 100k-point LiDAR sweep against a
100k-point local map is $10^{10}$ per iteration, and it is the whole runtime.

A kd-tree is the standard fix. It is a binary tree over the target points where each internal
node splits its points with a plane perpendicular to one coordinate axis, cycling through the
axes or picking the widest one, cutting at the median so the tree stays balanced at depth
$\log_2 m$. A query descends to the leaf containing the query point and records the best distance
found. On the way back up it visits a sibling subtree only if the distance from the query to that
node's splitting plane is less than the best distance so far, since no point on the far side can
beat it otherwise. Most siblings fail that test and are never opened, which is where the speed
comes from: average cost logarithmic in $m$ for low-dimensional data, worst case still linear.
The fraction of subtrees that can be pruned collapses as dimension grows, which is why kd-trees
are used at $d = 3$ and abandoned for high-dimensional descriptor matching.

Approximate nearest neighbor goes one step further: cap the number of nodes visited or the number
of backtracks, and accept a small probability of returning something that is not quite the
nearest point. For ICP that trade is cheap, because the correspondence is a guess that gets
recomputed next iteration anyway. FLANN and nanoflann are the usual libraries (nanoflann is on
this assignment's forbidden-import list, since the search would then not be from scratch).

A correct balanced kd-tree with the backtracking bound is its own exercise, and it is not the
registration lesson, so the search is provided in `models.cpp` as `nearest_neighbors`, a brute
force scan. The loop calls it and decides which correspondences to keep.

### Initialization and the basin of convergence

The corridor example above generalizes: ICP descends to the nearest local minimum of a non-convex
cost, and the set of starting poses from which it reaches the right one is its basin of
convergence. Roughly, the initialization has to be good enough that most source points' nearest
target points are the right piece of surface. In LiDAR odometry that comes for free from a
constant-velocity prediction off the previous scan pair, from wheel odometry, or from integrating
an IMU across the scan interval.

When there is no such prediction - relocalization, loop closure, merging two scans of unknown
relative pose - the initialization comes from global registration, which finds a coarse transform
with no prior at all. The standard recipe computes a local descriptor at each point, a vector
summarizing the shape of its neighborhood that does not change when the cloud is moved. FPFH
(Rusu et al. 2009) is the classic hand-designed one: a histogram of the angles between a point's
normal and the normals of its neighbors. Matching descriptors between the two clouds proposes
candidate pairs, and RANSAC, from the multi-view assignment, samples triples of pairs and keeps
the transform the most other pairs agree with. The result is coarse, typically degrees and
centimeters off, which is exactly the input ICP wants. Learned descriptors have largely replaced
hand-designed ones in this step (Choy et al. 2019).

### Production variants

The variants that replace vanilla ICP soften the hard nearest-point assignment.

Generalized ICP (Segal et al. 2009) attaches a covariance to every point, computed from its
neighbors the same way normals are, and then reshapes it: a locally planar neighborhood has one
small eigenvalue, and GICP replaces the eigenvalues with $(1, 1, \epsilon)$, turning each point
into a flat disk that is uncertain along the surface and certain across it. The residual
$d_i = q_i - (R p_i + t)$ is scored by its Mahalanobis distance under the combined covariance,

$$\sum_i d_i^\top \big(C_i^Q + R\,C_i^P R^\top\big)^{-1} d_i,$$

the same "squared error divided by how uncertain that error is" used for data association in the
EKF-SLAM assignment. Both earlier costs are special cases: identity covariances everywhere give
back point-to-point, and a zero source covariance with a flat-disk target covariance gives back
point-to-plane, since the inverse then blows up along the normal and the cost keeps only that
component. GICP sits between them, and because the disks on both clouds carry the local surface
orientation, it tolerates the two clouds sampling different points better than either.

The normal distributions transform (Biber and Straßer 2003 in 2D, Magnusson 2009 in 3D) drops
point correspondences altogether. Divide the target volume into voxels, and in each voxel compute
the mean and covariance of the points that fall inside, so the target becomes a piecewise
Gaussian density rather than a set of points. The score of a candidate pose is the sum, over
source points, of that density evaluated at the transformed point, maximized by Newton's method.
There is no nearest-neighbor search and no discrete assignment step, so the cost is smooth in the
pose within a voxel, and the voxel size sets how far off the initialization can be and still be
pulled in.

Both are still local methods that need an initialization. And the arms race is not one-directional:
KISS-ICP (Vizzo et al. 2023) is plain point-to-point ICP with a voxel-grid data structure instead
of a tree, an adaptive correspondence threshold, and a constant-velocity motion prediction, and it
is competitive with the more elaborate methods on LiDAR odometry benchmarks. Rusinkiewicz and
Levoy (2001) is the map of the design space these all live in: which points to select, how to
weight them, which to reject, and which error metric to minimize.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the declaration and comments in each file give the signature and shapes.

1. [`align_point_to_point()`](icp.cpp) in `icp.cpp`
2. [`point_to_plane_step()`](icp.cpp) in `icp.cpp`
3. [`icp()`](icp.cpp) in `icp.cpp`

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
accurate. A static scan of the C++ sources also checks that no registration or solver library
(Open3D, PCL, nanoflann, Sophus, Ceres, GTSAM, g2o) has been included.

If Open3D is installed, `test_oracle_open3d.py` cross-checks the from-scratch transform against
Open3D's registration on the same clouds; it is skipped when Open3D is absent. The from-scratch
solver is the graded code; Open3D is the labeled side-by-side, never the answer.

`make viz` writes `out/icp.rrd`. Add `SHOW=1` for the interactive viewer: the fixed target cloud
in gray, with the source cloud logged twice into the same world view as it slides on, red for
point-to-point and blue for point-to-plane, plus a scalar panel plotting each method's RMS
correspondence distance as it shrinks. Scrub the timeline to step through the iterations and
watch point-to-plane converge first. The RMS at a given step is measured on the correspondences
that step used, before its own update is applied, which is why both methods report the same value
at the first step: both start from the identity.

The clouds are synthetic: a bumpy height field with analytic surface normals (the surface-rich
case where point-to-plane wins) and a flat sheet (the planar degenerate case for the determinant
correction). A real pipeline would add a normal-estimation step on the target and a kd-tree for
the search, but the registration math is identical, and synthetic clouds with a known transform
give the deterministic ground truth the tests need.

## In interviews

ICP is a standard interview topic for LiDAR, depth-camera, and SLAM roles, both as a derivation
and as a "how would you register two scans" discussion.

Point-to-point versus point-to-plane, and why the latter converges faster. Point-to-point
penalizes the full distance between paired points, tangential slide included, and that slide is
sampling rather than error, since two scans never hit the same physical points. Point-to-plane
penalizes only the distance along the target normal, so points slide within the tangent plane for
free and the solve is pulled only by real misalignment.

The orthogonal Procrustes solution. Be ready to derive or state the SVD solution for the
point-to-point rotation, and specifically the $\det(V U^\top)$ correction and why it is there:
maximizing the trace alone constrains $R$ to be orthogonal, not to be a rotation, and on
rank-deficient data the best orthogonal fit can be a reflection. This is the single most asked
detail.

The point-to-plane linearization. The standard on-the-spot ask is to derive it: first-order
motion of a point under a twist, dot with the normal, read off the Jacobian row
$[n;\ p \times n]$, assemble the $6\times6$ normal equations.

ICP's local minima and the role of initialization. ICP descends to the nearest minimum of a
non-convex cost, so it needs a coarse initialization - constant-velocity prediction, odometry,
IMU, or a global registration from descriptor matching and RANSAC - to land in the right basin.
Knowing that this is the failure mode, and that GICP and NDT are the production variants that
soften the hard assignment, is a strong signal.

The cost of correspondence. The nearest-neighbor search dominates real ICP, which is why it runs
on a kd-tree, a voxel grid, or an approximate-nearest-neighbor structure rather than a
brute-force scan. A common follow-up: how would you speed up ICP on a million-point cloud (the
answer involves the search structure, downsampling, and point selection).

## Further reading

- Besl and McKay (1992) and Chen and Medioni (1992) - the two original ICP variants.
- Kabsch (1976) and Umeyama (1991) - the closed-form point-to-point solution, the latter with
  the determinant correction.
- Chetverikov, Svirko, Stepanov and Krsek, "The trimmed iterative closest point algorithm"
  (ICPR 2002) - the closest-fraction rejection rule.
- Rusinkiewicz and Levoy, "Efficient variants of the ICP algorithm" (3DIM 2001) - a survey of
  the design choices (point selection, weighting, rejection, the error metric).
- Segal, Haehnel and Thrun, "Generalized-ICP" (RSS 2009) - the plane-to-plane production variant.
- Biber and Straßer (IROS 2003) and Magnusson (PhD thesis, 2009) - the normal distributions
  transform in 2D and 3D.
- Rusu, Blodow and Beetz, "Fast point feature histograms (FPFH) for 3D registration" (ICRA 2009)
  - the descriptor behind classic global registration.
- Zhang, Kaess and Singh, "On degeneracy of optimization-based state estimation problems"
  (ICRA 2016) - detecting the unconstrained directions of the point-to-plane solve.
- Vizzo, Guadagnino, Mersch, Wiesmann, Behley and Stachniss, "KISS-ICP: in defense of
  point-to-point ICP" (RA-L 2023) - how far careful engineering takes the simplest variant.
- Choy, Park and Koltun, "Fully convolutional geometric features" (ICCV 2019) - learned
  descriptors in place of FPFH.
