# a14_3 - Multi-view geometry estimators

The fourth assignment of the classical SLAM module, in C++17 with Eigen. You build the
geometric estimators that turn pixel correspondences into 3D structure and camera motion:
triangulation, the normalized eight-point algorithm for the fundamental and essential
matrices, essential-matrix decomposition with the cheirality test, PnP, the Sampson distance,
RANSAC, and the composed two-view relative-pose front-end. This is the front end of every
classical visual-odometry and structure-from-motion system, and the part of SLAM that the
deep-learning assignments never touch.

It builds on the Lie-group assignment (the SE(3) retraction used to step a pose during PnP
refinement) and the camera-geometry assignment (the OpenCV pinhole model). The SE(3)
exponential and the pinhole projection and its Jacobian are carried over and provided here in
`models.cpp`, so your work is the multi-view estimators on top of them.

Required reading before you start:
- Hartley and Zisserman, *Multiple View Geometry in Computer Vision* (2004), chapters 9-11 -
  epipolar geometry, the fundamental and essential matrices, triangulation, and the
  estimation algorithms. This is the reference the whole assignment follows.
- Hartley, "In defense of the eight-point algorithm" (PAMI 1997) - why the normalization
  step is what makes the eight-point algorithm usable.
- Fischler and Bolles, "Random sample consensus" (1981) - the original RANSAC paper.

## Lecture notes

### The conventions, pinned first

Multi-view geometry is full of silent transpose and sign traps: a result that is correct up
to a transpose or a sign flip looks like a bug somewhere else entirely (cheirality "fails" on
correct code, an epipolar line lands on the wrong side). The fix is to fix the conventions
once and never deviate.

- The unprimed image is the first, reference image - the frame the points start in. The
  primed image is the second. Subscripts 1 and 2 mean first and second.
- The relative pose $(R, t) = T_{2 \leftarrow 1}$ takes a frame-1 coordinate into frame-2:
  $X_2 = R X_1 + t$. The essential matrix is $E = [t]_\times R$, where $[t]_\times$ is the
  skew-symmetric matrix with $[t]_\times v = t \times v$.
- The essential constraint $x_2^\top E x_1 = 0$ holds for normalized rays
  $x = K^{-1} u$ (homogeneous, third coordinate 1). The fundamental constraint
  $x_2^\top F x_1 = 0$ holds for pixel coordinates $u$, with $F = K_2^{-\top} E K_1^{-1}$.
- A camera pose used for projection and PnP is $T_{\text{cam} \leftarrow \text{world}}$: a
  world point projects as $u = \pi(K (R X + t))$, the same OpenCV pinhole as the
  camera-geometry assignment ($+x$ right, $+y$ down, $+z$ forward, $u = f_x X/Z + c_x$).

### Triangulation

Given a 3D point seen in two cameras with known projection matrices $P_1, P_2$ (each
$3 \times 4$, mapping a homogeneous world point to a homogeneous image point), and its two
image points, recover the 3D point. Each view gives the relation $x \simeq P X$ (equality up
to scale), and the scale is removed by a cross product: $x \times (P X) = 0$. Writing the
rows of $P$ as $P^{(0)}, P^{(1)}, P^{(2)}$ and the image point as $(x, y)$, two of the three
cross-product rows are independent:

$$x\, P^{(2)} - P^{(0)} = 0, \qquad y\, P^{(2)} - P^{(1)} = 0.$$

Stacking these two rows from each view gives a $4 \times 4$ homogeneous system $A X = 0$. The
solution is the right singular vector of $A$ with the smallest singular value (the last column
of $V$ in $A = U \Sigma V^\top$), dehomogenized by dividing by its fourth entry. This is the
direct linear transform (DLT). It minimizes an algebraic error, not the geometric reprojection
error, so a nonlinear refinement follows: Gauss-Newton on the summed reprojection residual
over the three coordinates of $X$, started from the DLT point. On noise-free correspondences
the DLT already nails the point; the refinement matters once there is pixel noise.

### Epipolar geometry and the eight-point algorithm

When the same scene point is seen in two images, its two image points are not independent:
the point, the two camera centers, and the two image points are coplanar. That coplanarity,
written in coordinates, is the epipolar constraint $x_2^\top E x_1 = 0$ on normalized rays
(or $x_2^\top F x_1 = 0$ on pixels). The matrix $E$ (or $F$) is a $3 \times 3$ matrix of rank
2 that encodes the entire relative geometry of the two views; recovering it from
correspondences is the first step of two-view reconstruction.

The constraint is linear in the entries of $F$. Each correspondence $((u, v), (u', v'))$
contributes one row to a homogeneous system in the 9-vector $f = \mathrm{vec}(F)$:

$$[\,u'u,\ u'v,\ u',\ v'u,\ v'v,\ v',\ u,\ v,\ 1\,]\, f = 0.$$

With eight or more correspondences, $f$ is the smallest right singular vector of the stacked
$N \times 9$ matrix - hence "eight-point". Two details make the difference between a textbook
formula and a usable estimator:

Rank-2 enforcement. The raw solution is a full-rank $3 \times 3$ matrix, but a true
fundamental matrix has rank 2 (its null space is the epipole). After solving, take the SVD
$F = U \Sigma V^\top$, zero the smallest singular value, and rebuild. Without this, the
epipolar lines of a pencil do not meet at a single epipole.

Normalization. This is the content of Hartley's "In defense" paper. Pixel coordinates run to
the hundreds, so the entries of the $N \times 9$ matrix span many orders of magnitude
($u'u \sim 10^5$ against the constant $1$), and the SVD is numerically dominated by the large
entries. The fix is to precondition: translate each image's points so their centroid is at
the origin and scale so their mean distance to the origin is $\sqrt{2}$, via a similarity
transform $T$ (so $\hat{x} = T x$). Solve on the normalized points to get $\hat{F}$, then
undo the transform: $F = T_2^\top \hat{F}\, T_1$. Running the unnormalized and normalized
solvers on the same noisy correspondences, the normalized one is orders of magnitude more
accurate; the unnormalized eight-point algorithm has a deserved bad reputation that the
normalization step removes.

Fed pixel coordinates, the algorithm returns $F$. Fed normalized rays $x = K^{-1} u$, the
same algorithm returns the essential matrix $E$ (up to the rank-2 cleanup). The essential
matrix carries the metric relative pose; the fundamental matrix does not, because it folds in
the unknown intrinsics.

### From the essential matrix to a pose, and cheirality

The essential matrix factors as $E = [t]_\times R$, and the goal is to read $(R, t)$ back out.
A true essential matrix has two equal nonzero singular values and one zero. Take the SVD
$E = U \Sigma V^\top$; because the decomposition below uses only the singular vectors $U$ and
$V$, that rank structure is what makes it valid, and the singular values themselves need not
be touched. With both $U$ and $V$ made proper rotations and the fixed matrix
$W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$, the decomposition
gives two rotation candidates and a translation direction:

$$R_1 = U W V^\top, \qquad R_2 = U W^\top V^\top, \qquad t = U_{:,3}.$$

The sign of $t$ is not determined by $E$ (both $\pm t$ satisfy $E = [t]_\times R$), so there
are four candidate poses: $(R_1, +t), (R_1, -t), (R_2, +t), (R_2, -t)$. Exactly one of them
places the scene in front of both cameras. This is the cheirality test (from "chiral",
handedness): triangulate the correspondences against each candidate and count the points with
positive depth ($z > 0$) in both cameras; keep the candidate with the most. The other three
put some or all of the scene behind a camera, which is physically impossible. The translation
is recovered only as a direction: monocular two-view geometry has no absolute scale, because
scaling the whole scene and the baseline together reprojects identically. That missing scale
is the monocular gauge freedom, and it is why a single camera cannot measure distance without
an external reference.

### PnP: pose from known 3D points

The perspective-$n$-point problem is the other direction: given $n$ known 3D world points and
their pixels in one calibrated camera, find the camera pose $T_{\text{cam} \leftarrow
\text{world}}$. It is the relocalization step in SLAM (place a new frame against the existing
map) and the standard way to track against known structure.

A linear initialization comes from the DLT again. With normalized rays $x = K^{-1} u$, the
relation $x \simeq M [X; 1]$ for the $3 \times 4$ matrix $M = [R \mid t]$ is linear in the
12 entries of $M$; the cross product $x \times M[X;1] = 0$ gives two independent rows per
point, and six points suffice. The catch is that the recovered $M$ is only approximately
$[R \mid t]$: its left $3 \times 3$ block is not exactly a rotation, and its overall scale is
arbitrary. Peel off the scale (the cube root of the determinant of the left block, which
preserves sign) and project the rotation block onto $SO(3)$ with an SVD.

The DLT minimizes an algebraic error, so a nonlinear refinement follows, and this is where
the Lie-group machinery enters. Gauss-Newton minimizes the summed reprojection error
$\sum_i \lVert u_i - \pi(K(R X_i + t)) \rVert^2$ over the pose, updating on the manifold by
the right perturbation $T \leftarrow T\, \exp(\xi^\wedge)$ with $\xi = [\rho; \theta]$
(translation part first, matching the Lie-group assignment's twist ordering). The Jacobian of
the camera-frame point under that perturbation is $[\,R \mid -R\,[X_i]_\times\,]$ ($3 \times
6$), and the measurement Jacobian is the pinhole derivative composed with it:

$$A_i = \frac{\partial \pi}{\partial X_{\text{cam}}}\Big[\,R \ \big| \ -R\,[X_i]_\times\,\Big] \in \mathbb{R}^{2 \times 6}.$$

Accumulate $H = \sum_i A_i^\top A_i$ and $g = \sum_i A_i^\top e_i$ with the residual
$e_i = u_i - \pi(K(R X_i + t))$, solve $H\,\delta\xi = g$, retract, and iterate. Because the
perturbation is taken afresh around the current estimate each step (relinearization at the
identity), the right Jacobian $J_r$ does not appear explicitly here - it is absorbed into the
re-evaluated $A_i$; it resurfaces in the pose-graph assignment, where residuals are
accumulated in a fixed tangent space.

### RANSAC and the Sampson distance

Everything above assumed the correspondences are correct. Real feature matching produces
wrong matches (outliers), and least squares has a breakdown point of zero: a single gross
outlier can move the fit arbitrarily far, because the squared-error cost lets one large
residual dominate. An eight-point fit on correspondences that are a third outliers is not
slightly wrong, it is useless.

RANSAC (random sample consensus) is the standard answer. Repeatedly draw a minimal sample (8
correspondences for the eight-point algorithm), fit a model to just that sample, and score
how many of all the correspondences agree with it (the consensus set). After enough rounds,
the model with the largest consensus set is the one that a sample of all-inliers produced;
refit it on its full consensus set. The logic is that a minimal sample free of outliers
yields a model that many other inliers agree with, while any sample containing an outlier
yields a model that few agree with.

Scoring needs a distance. The raw algebraic residual $x_2^\top F x_1$ is not a usable
threshold because it has no units - it scales with $F$ and with the pixel magnitudes. The
Sampson distance is the first-order approximation to the true geometric reprojection error
and does have pixel units:

$$d = \frac{\lvert x_2^\top F x_1 \rvert}{\sqrt{(F x_1)_0^2 + (F x_1)_1^2 + (F^\top x_2)_0^2 + (F^\top x_2)_1^2}}.$$

A correspondence is an inlier when $d$ is below a threshold of a pixel or two. Running this
assignment's front-end on a scene that is a third outliers, RANSAC recovers essentially the
full inlier set and a pose within a fraction of a degree of truth, while the plain eight-point
fit on all correspondences lands tens of degrees off - the robust-versus-naive gap is the
whole lesson, and the tests assert it as an ordering that holds across seeds.

### The composed front-end

The pieces assemble into the actual visual-odometry front-end: RANSAC the fundamental matrix
from the pixel correspondences, convert it to the essential matrix $E = K_2^\top F K_1$,
recover $(R, t)$ by cheirality on the inlier rays, and triangulate the inliers into 3D points
expressed in frame 1. The output is a relative pose (up to the monocular scale) and a sparse
point cloud - one camera's worth of motion and structure, the unit a SLAM back-end then
stitches and optimizes. Assembling this from its parts is a common interview ask.

### Where this sits

The eight-point algorithm is the pedagogical route, not the state of the art. The minimal
relative-pose solver is the five-point algorithm (Nister 2004), which uses the two extra
constraints that the essential matrix carries (the equal-singular-value structure) to solve
from five correspondences instead of eight - fewer points per RANSAC sample means far fewer
iterations to hit an all-inlier sample. The eight-point algorithm is linear and easy to
derive, which is why it is taught first; production systems use the five-point solver inside
RANSAC. The estimators here are the geometric front-end; the factor-graph and bundle-adjustment
assignment is the back-end that refines many poses and points jointly.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the declaration and comments in each file give the signature and shapes.

1. [`triangulate_dlt()`](multiview.cpp) in `multiview.cpp`
2. [`triangulate_refine()`](multiview.cpp) in `multiview.cpp`
3. [`eight_point()`](multiview.cpp) in `multiview.cpp`
4. [`decompose_essential()`](multiview.cpp) in `multiview.cpp`
5. [`recover_pose()`](multiview.cpp) in `multiview.cpp`
6. [`pnp_dlt()`](multiview.cpp) in `multiview.cpp`
7. [`pnp_refine()`](multiview.cpp) in `multiview.cpp`
8. [`sampson_distance()`](multiview.cpp) in `multiview.cpp`
9. [`ransac_fundamental()`](multiview.cpp) in `multiview.cpp`
10. [`two_view_relative_pose()`](multiview.cpp) in `multiview.cpp`

### Building and running

Same toolchain as the rest of the module (C++17, CMake, pybind11, Eigen). You never call
CMake by hand.

```
make verify A=a14_3_multiview   # build + run the reference (solution/); the green target
make test   A=a14_3_multiview   # build + run YOUR code; red until the holes are filled
make viz    A=a14_3_multiview        # render the two-view geometry (reference)
make viz-mine A=a14_3_multiview      # the same, from YOUR code, once the holes are filled
```

The tests favor implementation-independent statements. Triangulation reproduces the 3D points
on noise-free correspondences and the refinement reduces error from a perturbed start; the
recovered $F$ satisfies the epipolar constraint and is rank 2, and the essential matrix
satisfies it on rays; the decomposition yields proper rotations and cheirality recovers the
true pose (rotation exactly, translation up to its sign and the scale) with all points in
front of both cameras; PnP recovers a known pose to numerical precision and the refinement
drives the reprojection error to zero; the Sampson distance matches its closed form and
separates inliers from outliers; and RANSAC recovers the inlier set while the robust pose
beats the naive least-squares pose by a wide margin across seeds.

`make viz` writes `out/multiview.rrd`. Add `SHOW=1` for the interactive viewer: the
triangulated points and the two recovered camera frusta in 3D, the correspondences colored by
inlier versus outlier in the two image views, and a few epipolar lines drawn in the second
image (every inlier sits on its line, every outlier does not).

The scene is synthetic: a random point cloud viewed by two cameras a short baseline apart,
with the world frame chosen as camera 1 so the ground-truth relative pose is exactly
$T_{2 \leftarrow 1}$. A real image pair would add a feature detector and descriptor matcher in
front of these estimators, but the geometry being estimated is identical, and synthetic
correspondences give the deterministic ground truth the tests need.

## In interviews

Multi-view geometry is a standard interview topic for perception, SLAM, and 3D-vision roles,
both as derivations on a whiteboard and as "assemble the front-end" questions.

Outline the normalized eight-point algorithm. This is a frequent on-the-spot ask. The answer
is the four steps: normalize each image's points (centroid to origin, mean distance
$\sqrt{2}$), solve the linear system for the null vector, enforce rank 2 by zeroing the
smallest singular value, denormalize. Be ready to say why normalization matters (the
conditioning of the linear system) and why rank-2 enforcement matters (a real fundamental
matrix is rank 2; its null space is the epipole).

Essential versus fundamental. Know that the essential matrix is the calibrated version
($x_2^\top E x_1 = 0$ on normalized rays, carries the metric pose) and the fundamental matrix
is the uncalibrated one ($x_2^\top F x_1 = 0$ on pixels, $F = K_2^{-\top} E K_1^{-1}$). The
follow-up is usually the decomposition: four candidate poses from $E$, resolved by cheirality.

Why RANSAC is mandatory. The breakdown point of least squares is zero - one outlier ruins the
fit - so any estimator running on real correspondences needs a robust wrapper. Know the loop
(minimal sample, consensus scoring, refit) and that the score must be a metric distance (the
Sampson distance), not the unitless algebraic residual. A strong answer mentions the
five-point algorithm as the modern minimal solver and why fewer points per sample means fewer
RANSAC iterations.

PnP and its variants. Know that PnP recovers a calibrated camera's pose from 3D-2D
correspondences, that the DLT gives a linear initialization that a nonlinear refinement over
$SE(3)$ then improves, and the named minimal/efficient solvers: P3P (the minimal three-point
solver, used inside RANSAC) and EPnP (an efficient $O(n)$ solver). The standard degeneracy
question: coplanar points, or points and the camera center in a critical configuration.

The monocular scale ambiguity. Expect "why can't a single moving camera measure absolute
distance?" The answer is that scaling the scene and the baseline together reprojects
identically, so two-view monocular geometry recovers translation only as a direction; absolute
scale needs a second sensor, a known object size, or stereo.

## Further reading

- Hartley and Zisserman, *Multiple View Geometry in Computer Vision* (2004), chapters 9-11 -
  the definitive treatment of epipolar geometry, triangulation, and the estimators.
- Hartley, "In defense of the eight-point algorithm" (PAMI 1997) - normalization.
- Nister, "An efficient solution to the five-point relative pose problem" (PAMI 2004) - the
  modern minimal relative-pose solver.
- Lepetit, Moreno-Noguer, Fua, "EPnP: an accurate O(n) solution to the PnP problem" (IJCV
  2009), and the P3P literature - the production PnP solvers.
- Fischler and Bolles, "Random sample consensus" (1981) - RANSAC.
