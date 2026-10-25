# a14_4 - Point-cloud registration (ICP)

The fifth assignment of the classical SLAM module, in C++17 with Eigen. It builds iterative
closest point (ICP): given two point clouds of the same scene from different viewpoints, find
the rigid transform that aligns them. ICP is the standard registration method in LiDAR odometry and depth-camera
registration, the step that turns overlapping scans into a single consistent reconstruction,
and a standard interview topic in its own right.

The point-to-plane solver steps the pose with the SE(3) exponential map from the Lie-group
assignment. That map and the skew operator are carried over into `models.cpp`, which also
provides the nearest-neighbor correspondence search, so the work here is the two solvers and
the outer loop. The Python around the C++ (the build, the synthetic clouds, the tests, and the
Rerun visualization) is provided, as in the earlier assignments of the module.

Required reading before starting:
- Besl and McKay, "A method for registration of 3-D shapes" (PAMI 1992) - the original ICP
  paper (point-to-point).
- Chen and Medioni, "Object modelling by registration of multiple range images" (1992) - the
  point-to-plane variant.
- Umeyama, "Least-squares estimation of transformation parameters between two point patterns"
  (PAMI 1991) - the closed-form solution with the reflection-avoiding determinant correction.

## Lecture notes

### What registration is for

#### The task

A car drives down a street with a spinning LiDAR on its roof. Each sweep returns a point cloud:
an unordered set of 3D points sampled off the surfaces around the car (the road, building
fronts, parked cars, poles), each point measured in the sensor's frame at the time of the sweep.
Between two consecutive sweeps the car has moved, so the same building front appears in the
second cloud shifted and slightly turned. LiDAR odometry asks how the car moved between the
sweeps, and the answer is the rigid transform that lays the second cloud on top of the first.
The same question comes up when a depth camera's frames are fused into one model, when a new scan
is matched against a prebuilt map to localize, and when two scans of a room taken from
different spots are merged.

In the notation used throughout, the cloud that moves is the source $P = \{p_i\}$ and the cloud
that stays fixed is the target $Q = \{q_j\}$. Registration asks for the rigid transform
$T = (R, t)$, with $R \in SO(3)$ and $t \in \mathbb{R}^3$, that carries the source onto the
target, $q \approx R p + t$. In the module's frame naming that estimate is
$T_{\text{target} \leftarrow \text{source}}$, and applying it to a cloud means $q = R p + t$ on
every row.

The assignment's instance is smaller: the target is 800 points scattered over a bumpy surface,
a height field over a 6 by 6 square with bumps of height up to 0.6, and the source is the same
cloud moved by a rotation of several degrees and a translation of a few tenths, both in cloud
units. The job is to recover that motion from the two clouds alone.

#### Why the obvious approaches fail

If the pairing were given (if row $i$ of $P$ were known to be the same physical point as row
$i$ of $Q$), the problem would be a single least-squares solve,

$$\min_{R,\,t}\ \sum_i \lVert R p_i + t - q_i \rVert^2,$$

which has a closed form derived below. What is missing is the pairing,
and real scans do not provide one. Two sweeps of the same wall do not hit the same physical spots; the
beam that lands on one patch of plaster in the first sweep lands centimeters away in the second.
The two clouds are two samplings of the same surfaces, and the problem is really surface to
surface.

A search over all poses is the other obvious move. The pose has six degrees of freedom, and
scoring one candidate pose means checking every source point against the target. A grid fine
enough to reach centimeters and tenths of a degree over a range of meters and tens of degrees
has far too many cells to score each one.

#### The idea

ICP pretends a pairing exists, computes it from the current pose estimate, and recomputes it as
the estimate improves:

1. Pair each source point, moved by the current pose estimate, with its nearest target point.
2. Solve the least-squares problem above for those pairs, which is cheap, and update the pose.
3. Repeat until the pose stops changing.

When the starting pose is close, most nearest points lie on the right piece of surface, the
solve moves the cloud closer, and the next pairing is better. Two choices then decide how fast
this goes and when it breaks. The first is the cost minimized in step 2: the distance between
the paired points (point-to-point), or only the distance along the target surface's normal
(point-to-plane). The second is the starting pose, because ICP only finds the alignment near
where it starts.

#### Why it matters

Every LiDAR odometry and mapping pipeline, and every depth-camera fusion system in the
KinectFusion line, registers each new scan with some version of this loop. The variants differ
in the cost, the correspondence search, and how bad pairs are rejected; the loop is the same.

### Our corner example (a 2D scan of a room corner, moved by a pure translation)

The assignment's cloud has 800 points in 3D, too many to follow by hand. We use a
stripped-down version that we call our corner example: a 2D scanner, seen from above, in the
corner of a room. The target has three points on the wall $x = 0$, at $(0, 1)$, $(0, 2)$,
$(0, 3)$, and three on the wall $y = 0$, at $(1, 0)$, $(2, 0)$, $(3, 0)$, in meters. The
unit normals of the two walls, pointing into the room, are $(1, 0)$ and $(0, 1)$. The source is
the same six points moved by $g = (0.2, 0.7)$, so the true alignment is the pure translation
$t = -g = (-0.2, -0.7)$ with $R = I$.

![Our corner example. Left: the six target points on the two walls (gray), the six source points (black squares), and the nearest-neighbor pair of each source point (arrows); each pair's gap is split into its part along the target normal (blue) and its part along the wall (red, dotted). Right: the source after one point-to-point step (red dots), which comes out rotated by about 6.7 degrees and still off the walls, and after one point-to-plane step (blue circles), which lands exactly on the target](assets/corner_example.png)

The first pairing already shows the problem ICP has to live with. On the wall $y = 0$ each
source point sits 0.7 m above its own target point and 0.2 m to the side, and its nearest target
point is its own: those three pairs are correct. On the wall $x = 0$ the source points sit at
$y = 1.7$, $2.7$, $3.7$, and the nearest target points are $(0, 2)$, $(0, 3)$, and $(0, 3)$. The
first two are paired with the target point one place up the wall. Each of those pairs has the
right gap across the wall (0.2 m in $x$, the true offset from the wall) and a wrong gap along the
wall (0.3 m up, where the true offset is 0.7 m down).

The right panel shows what each cost does with those pairs. Point-to-point fits the whole gap,
including the wrong part along the wall, and comes out rotated and short of the target.
Point-to-plane fits only the parts along the normals, which are correct for every pair, and lands
on the target in one step. The later sections derive both solves and come back to this example in
the exercises.

Our corner example keeps the nearest-neighbor pairing, the fact that a nearest neighbor is
usually on the right surface but at the wrong place along it, the two costs, and the loss of a
direction when only one wall is visible. It drops the third dimension, and its true motion has
no rotation. Like the assignment's clouds, its source and target are the same samples, so at the
true pose every pair is exact; two real scans sample different points, and the gaps along the
surface never vanish.

### Overview of the rest of the notes

1. The ICP loop and why it converges. Alternating minimization, the same structure as k-means;
   what the convergence argument does and does not promise; the distance gate.
2. Least squares under a rotation constraint. Why the normal equations of ordinary linear least
   squares do not apply.
3. The point-to-point solve. The orthogonal Procrustes problem, solved with the SVD, and the
   determinant correction.
4. The point-to-plane cost. Surface normals and why measuring only along them helps.
5. Gauss-Newton. The linearize-and-solve step of the EKF, iterated, with the pose increment
   composed on the left.
6. Linearizing the point-to-plane step. The $6 \times 6$ system and its Jacobian row.
7. When point-to-plane goes singular. The surfaces that leave a direction unconstrained.
8. The correspondence search, initialization, and the variants used in production.

Five exercises, placed after the sections they test, are meant to be worked by hand: two on our
corner example, one on a one-dimensional corridor, and two on small 3D point sets. Each has a
worked answer that can be expanded.

### The ICP loop and why it converges

ICP alternates over the two unknowns, the pairing and the pose:

1. Correspondence step. Transform the source by the current $T$, and for each transformed
   source point take the nearest target point as its correspondence.
2. Solve step. Holding those pairs fixed, compute the transform that minimizes the summed
   squared distance over the pairs, compose it onto $T$, and repeat.

The convergence argument writes both unknowns into one cost. Let $c$ be the correspondence map,
$c(i)$ the index of the target point paired with source point $i$, and

$$E(T, c) = \sum_i \lVert R p_i + t - q_{c(i)} \rVert^2.$$

The correspondence step minimizes $E$ over $c$ with $T$ held fixed, and it does so exactly,
because the terms are independent and each is minimized by picking that point's nearest target.
The solve step minimizes $E$ over $T$ with $c$ held fixed, again exactly, by the closed-form
solve below. Neither step can increase $E$, and $E \ge 0$, so the sequence of costs is
non-increasing and bounded below, and therefore converges. This is the argument in Besl and
McKay (1992), and it is ordinary alternating minimization, the same structure as k-means or
expectation-maximization. It covers the point-to-point mode as written. The point-to-plane mode
takes one Gauss-Newton step (an iterative linearize-and-solve method, introduced below) on a
different cost instead of solving its own exactly, so the same
guarantee does not transfer to it; in practice it converges faster than the mode that has the
proof.

The argument says the cost settles. It does not say the pose is right. Minimizing $c$ out leaves
$E(T) = \min_c E(T, c)$, a pointwise minimum over the finitely many correspondence maps, so it
is piecewise smooth and non-convex, and alternating minimization walks downhill into whichever
basin it started in. A concrete failure, which we call our pillar-corridor example: scan a
corridor with pillars every 2 m and initialize the pose 1.5 m off along the corridor. Every
source point's nearest target point sits on the neighboring pillar, each solve fits those pairs,
the loop converges in a handful of iterations, and the answer is one pillar out. Nothing in the
cost is violated: the cost is small at that pose.

The distance gate is the other piece of the loop. Partial overlap and clutter mean some source
points have no counterpart in the target at all (the far end of the street that only the second
sweep reached, a car that drove away), but the correspondence step still hands each of them a
nearest neighbor, and that wrong pair pulls on the solve with the same weight as a good one. Two
standard remedies: drop any pair farther apart than a fixed distance, which is the gate, or keep
only the closest fraction of pairs, which is trimmed ICP (Chetverikov et al. 2002). Gating gives
up the convergence proof, since consecutive iterations then minimize sums over different point
sets and the monotone argument no longer applies. It is standard practice anyway, because
ungated phantom correspondences do more damage than the lost guarantee.

The loop stops when the incremental transform is negligible (its translation norm and
$\lVert R_{\text{inc}} - I \rVert$ both below about $10^{-10}$) or when an iteration limit is
reached.

> **Exercise 1 (the wrong pillar).** Treat our pillar-corridor example in one dimension along the
> corridor, far from its ends: target pillars at $0, 2, 4, \dots$ m, and the source is the same
> pillars observed from a pose that is wrong by $+1.5$ m, so each source pillar sits at its true
> position plus 1.5. With rotation ignored, the point-to-point solve for a translation $s$ is
> the mean of $q_{c(i)} - p_i$. (a) Run the correspondence step and the solve step once. (b) What
> does the next iteration do, and what is the cost $E$ at the final pose? (c) For which initial
> errors does the loop end at the correct pose?

<details>
<summary>Answer</summary>

(a) A source pillar at $x_k + 1.5$ is 0.5 m from the next target pillar $x_k + 2$ and 1.5 m from
its own $x_k$, so every correspondence is the next pillar, and every gap is
$q_{c(i)} - p_i = +0.5$. The mean is $s = +0.5$: the solve moves the source further from the
truth, to an error of $+2$ m.

(b) Every source pillar now sits exactly on a target pillar (the next one along). The
correspondence step pairs each with the pillar it sits on, all gaps are 0, the increment is 0,
and the loop stops. The cost at that pose is $E = 0$, the same as at the true pose. The cost
cannot tell the two apart because the scene repeats every 2 m.

(c) The nearest target pillar is the correct one when the error is less than half the spacing,
$|e| < 1$ m; then every gap is $-e$, one step removes the error, and the loop stops at the right
pose. For $1 < |e| < 2$ the loop converges one pillar out, as above. That interval of starting
errors, $|e| < 1$ m, is the basin of convergence of the correct pose, discussed at the end of the
notes.

</details>

### Least squares under a rotation constraint

The solve step is not the normal equations of ordinary least squares, and the reason is the
rotation. Fitting a general affine map $p \mapsto A p + t$ to matched pairs is unconstrained
linear least squares in the 12 entries of $(A, t)$, and the usual $A^\top A x = A^\top b$ solve
handles it. But the minimizing $A$ is not a rotation. It will scale and shear to reduce the
residual, which changes the shape of the cloud, and a cloud's shape is exactly what should not
change when the sensor moves.

Rigid registration constrains $R^\top R = I$ and $\det R = +1$: three degrees of freedom rather
than nine, and a curved, non-convex constraint set that the normal equations cannot express.
The constrained version has a name, the orthogonal Procrustes problem (find the orthogonal
matrix that best maps one set of vectors onto another), and it has an exact solution through the
SVD, found by Kabsch (1976) for molecular structures and given in the form used here, with the
reflection handled, by Umeyama (1991).

### The point-to-point solve

Take the pairs as given, so row $i$ of $P$ matches row $i$ of $Q$, and minimize
$\sum_i \lVert R p_i + t - q_i \rVert^2$. The solve takes three steps.

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
more. Both cases are covered by one formula:

$$R = V \begin{bmatrix} 1 & & \\ & 1 & \\ & & \det(V U^\top) \end{bmatrix} U^\top, \qquad t = \bar q - R \bar p.$$

Dropping the middle matrix is the classic bug. Without it the solve can return a reflection,
$\det R = -1$, which is not a motion any rigid body can perform: it flips handedness, turning a
right-handed cloud into its mirror image. The penalty for correcting it is $2\sigma_3$, so the
correction is free exactly when $\sigma_3 = 0$, which is when $H$ is rank-deficient - a cloud
that is flat, or collinear, or otherwise has no extent in some direction. On such data a mirror
and a rotation can fit equally well, and only the correction guarantees the rotation.

> **Exercise 2 (a mirrored flat cross).** The source is four points in the plane $z = 0$:
> $(1, 0, 0)$, $(-1, 0, 0)$, $(0, 2, 0)$, $(0, -2, 0)$. The target is the same four points with
> $x$ negated, paired in the same order. (a) Compute $H$ and its singular values. (b) One valid
> SVD has $U = [e_2, e_1, e_3]$ and $V = [e_2, -e_1, e_3]$ (columns, with $e_k$ the unit
> coordinate vectors). Compute $V U^\top$ and its determinant, and say what it does geometrically.
> (c) Apply the determinant correction. Does the corrected $R$ still map every source point
> exactly onto its target?

<details>
<summary>Answer</summary>

(a) Both centroids are at the origin, so $H = \sum_i p_i q_i^\top$. The first two points
contribute $1 \cdot (-1) + (-1) \cdot 1 = -2$ to entry $(1,1)$, the last two contribute
$2 \cdot 2 + (-2)(-2) = 8$ to entry $(2,2)$, and nothing else is nonzero:
$H = \operatorname{diag}(-2, 8, 0)$. Its singular values are $(8, 2, 0)$. The zero comes from the
flat cloud: no point has any extent in $z$.

(b) Check the factorization: $U \Sigma V^\top = 8\,e_2 e_2^\top + 2\,e_1(-e_1)^\top = \operatorname{diag}(-2, 8, 0)$.
Then $V U^\top = e_2 e_2^\top - e_1 e_1^\top + e_3 e_3^\top = \operatorname{diag}(-1, 1, 1)$, with
determinant $-1$. It is the mirror across the $yz$ plane, which is exactly how the target was
made, so it fits the data perfectly. It is not a rotation.

(c) The correction replaces the middle matrix with $\operatorname{diag}(1, 1, -1)$:
$R = \operatorname{diag}(-1, 1, -1)$, determinant $+1$, a 180-degree rotation about the $y$ axis.
It sends $(x, y, 0)$ to $(-x, y, 0)$, so it maps every source point exactly onto its target too.
Both matrices give $\operatorname{tr}(RH) = 2 + 8 = 10$: the penalty $2\sigma_3$ is zero because
$\sigma_3 = 0$. Because $\sigma_3 = 0$, the signs of the third columns of $U$ and $V$ are also
arbitrary, and a different SVD routine could return a $V U^\top$ with determinant $+1$ directly.
The corrected formula returns a rotation for every valid SVD.

</details>

### Surface normals and the point-to-plane cost

Point-to-point penalizes the whole vector between a source point and its matched target point,
including the component that slides along the surface. That component is not pose error, for
two reasons. In real scans, two sweeps sample different physical points on the same wall, so
even at the true pose a source point and its nearest target point are apart by an offset within
the wall about as large as the point spacing, and no pose correction can remove it. And at any
pose other than the true one, the nearest target point is usually on the right surface but at
the wrong place along it, as on the wall $x = 0$ in our corner example. Point-to-point tries to
close those tangential gaps anyway, and the sum of those pulls moves the estimate off course.

Point-to-plane measures only the component along the target surface normal. A surface normal at
a target point is the unit vector perpendicular to the local surface there. Real pipelines
estimate it: take the point's $k$ nearest neighbors, form the $3 \times 3$ covariance of those
neighbors about their mean, and take the eigenvector of the smallest eigenvalue, the direction
in which the neighborhood has the least spread, which is the normal of the best-fit plane
through them. The sign is ambiguous and is resolved by orienting normals toward the sensor. For a
surface given as a height field $z = f(x, y)$ the normal is available in closed form, the
normalized $(-\partial z/\partial x,\, -\partial z/\partial y,\, 1)$.

With normals $n_i$ on the target, the cost is

$$\min_{R,\,t}\ \sum_i \big( n_i \cdot (R p_i + t - q_i) \big)^2 .$$

Geometrically, the target point $q_i$ is replaced by the tangent plane through $q_i$ with normal
$n_i$, and the residual is the signed distance from the moved source point to that plane. A
source point can slide anywhere within the tangent plane at zero cost, which is the freedom real
scans actually have. The directions that carried the sampling offsets and the pairing errors are
gone from the objective, and the solve responds only to the components that are actual
misalignment.

> **Exercise 3 (one step on our corner example).** Use the first pairing of our corner example
> (source points $(0.2, 1.7)$, $(0.2, 2.7)$, $(0.2, 3.7)$, $(1.2, 0.7)$, $(2.2, 0.7)$,
> $(3.2, 0.7)$, paired with $(0, 2)$, $(0, 3)$, $(0, 3)$, $(1, 0)$, $(2, 0)$, $(3, 0)$). Restrict
> both solves to a pure translation $s$. (a) The point-to-point solve for a translation is the
> mean gap, $s = \frac{1}{n}\sum_i (q_i - p_i)$. Compute it. (b) The point-to-plane solve for a
> translation minimizes $\sum_i (n_i \cdot (p_i + s - q_i))^2$, whose normal equations are
> $(\sum_i n_i n_i^\top)\, s = -\sum_i n_i\, (n_i \cdot (p_i - q_i))$. Compute $s$. (c) Compare
> both with the true correction $(-0.2, -0.7)$.

<details>
<summary>Answer</summary>

(a) The gaps $q_i - p_i$ are $(-0.2, 0.3)$, $(-0.2, 0.3)$, $(-0.2, -0.7)$ on the wall $x = 0$ and
$(-0.2, -0.7)$ three times on the wall $y = 0$. The mean is
$s = (-0.2,\ (0.3 + 0.3 - 4 \cdot 0.7)/6) = (-0.2, -0.367)$.

(b) The three normals $(1, 0)$ and three normals $(0, 1)$ give
$\sum_i n_i n_i^\top = \operatorname{diag}(3, 3)$. The normal residuals $n_i \cdot (p_i - q_i)$
are $0.2$ on the wall $x = 0$ (the $x$ offset from that wall) and $0.7$ on the wall $y = 0$, so the
right-hand side is $-(3 \cdot 0.2,\ 3 \cdot 0.7) = (-0.6, -2.1)$ and $s = (-0.2, -0.7)$.

(c) Point-to-plane recovers the true correction exactly in one step. Point-to-point gets $x$
right and $y$ only about half right: the two mispaired points on the wall $x = 0$ contribute
$+0.3$ to the $y$ average where the truth is $-0.7$. Their gaps across the wall (0.2 in $x$) were
correct; only the part along the wall was wrong, and point-to-plane never looks at it. If the
point-to-point solve is also allowed a rotation, as in the full Procrustes solve, the same pairs
make it rotate the cloud by about $-6.7$ degrees, although the true motion has no rotation at
all; that is the red cloud in the right panel of the figure. In this small example the next
point-to-point iteration finds the correct pairs and finishes. On a larger cloud the wrong
tangential gaps recur at every iteration, and point-to-point creeps in over many iterations.

</details>

### Gauss-Newton and the normal equations

The point-to-plane cost is nonlinear in $R$, so there is no closed form and it is minimized
iteratively with Gauss-Newton, the standard method for a sum of squared residuals
$\sum_i r_i(x)^2$ where each $r_i$ depends nonlinearly on the parameters $x$. It is the EKF's
linearize-and-solve step, repeated until it stops moving. Linearize each residual about the
current estimate,

$$r_i(x + \delta) \approx r_i(x) + J_i\,\delta, \qquad J_i = \frac{\partial r_i}{\partial x},$$

a row vector with one entry per parameter. The cost in the increment $\delta$ is then an exact
quadratic, $\sum_i (r_i + J_i \delta)^2$, and setting its gradient to zero gives

$$\Big(\sum_i J_i^\top J_i\Big)\,\delta = -\sum_i J_i^\top r_i .$$

These are the normal equations: the same $A^\top A\,x = A^\top b$ that solves an overdetermined
linear system $A x \approx b$ in the least-squares sense, with $A$ the stacked Jacobian rows and
$b$ the negated residuals. Solve for $\delta$, apply it, relinearize at the new estimate, repeat.
Gauss-Newton is Newton's method with the term involving second derivatives of the residuals
dropped, which is a good approximation when the residuals are small at the solution, as they are
when registering two scans of the same surface.

The parameters here are a pose, and poses do not add. The increment is a six-dimensional twist
$\xi = [\rho; \theta]$, translation part first, following the Lie-group assignment's ordering,
and it reaches the group through the $SE(3)$ exponential map. Which side of $T$ it is composed
on matters, and ICP differs from the rest of the module here. ICP composes on the left,

$$T \leftarrow \exp(\widehat\xi)\,T,$$

while the PnP refinement in the multi-view assignment and the pose-graph optimizer use the
Lie-group assignment's box-plus retraction (box-plus applies a twist to a pose by multiplying
by its exponential) on the right, $T \leftarrow T\exp(\widehat\xi)$, a correction in the body
frame. ICP uses the left because its residuals are computed in the target frame. For this
paragraph only, write $s_i$ for an original source point and $p_i = T s_i$ for the same point
moved into the target frame by the current $T$; everywhere else in the notes $p_i$ means the
point the solve step works on, which is this moved point. The solve step works on the $p_i$, and the increment is a motion of those moved points:
$p_i \mapsto \exp(\widehat\xi)\,p_i = \exp(\widehat\xi)\,T\,s_i$. So the increment is a
target-frame (left) perturbation, and the pose that produces the moved cloud is
$\exp(\widehat\xi)\,T$. The two conventions express the same set of updates and are related by
the adjoint from the Lie-group assignment (the $6 \times 6$ matrix that re-expresses a twist
in another frame), $\exp(\widehat\xi)\,T = T\,\exp(\widehat{\mathrm{Ad}_{T^{-1}}\xi})$; the left
version is used here because it keeps the increment in the same frame as the residuals, so the
Jacobian below contains no $R$.

### Linearizing the point-to-plane step

Let $p_i$ now denote the current transformed source point, $q_i$ its matched target point, and
$n_i$ that target point's normal. To first order $\exp(\widehat\xi) \approx I + \widehat\xi$, so
the increment moves the point to

$$p_i' \approx p_i + \rho + \theta \times p_i = p_i + \rho - \widehat{p_i}\,\theta,$$

where $\widehat{a}$ is the skew (hat) operator from the Lie-group assignment, defined by
$\widehat{a}\,b = a \times b$. Dotting with the normal and using the scalar triple product
identity $n \cdot (\theta \times p) = \theta \cdot (p \times n)$:

$$r_i = n_i \cdot (p_i' - q_i) \approx (p_i - q_i)\cdot n_i + n_i \cdot \rho + (p_i \times n_i)\cdot\theta = (p_i - q_i)\cdot n_i + a_i^\top \xi, \qquad a_i = \begin{bmatrix} n_i \\ p_i \times n_i \end{bmatrix}.$$

So the Jacobian row is $a_i^\top$ and the constant term is the current signed distance from the
point to its target's tangent plane. Feeding those into the normal equations above gives a
$6 \times 6$ system,

$$\Big(\sum_i a_i a_i^\top\Big)\,\xi = \sum_i a_i b_i, \qquad b_i = -(p_i - q_i)\cdot n_i,$$

solved once per iteration; the increment is $\exp(\widehat\xi)$, composed onto the current pose
on the left. The matrix is symmetric and positive semi-definite, and when the surface
constrains all six directions it is positive definite, so a Cholesky-type factorization solves
it. Assembly is a single pass over the correspondences accumulating a
$6\times 6$ matrix and a 6-vector, which costs little next to the nearest-neighbor search.

The two halves of $a_i$ have a direct reading. The top half $n_i$ is how fast a pure translation
changes the point-to-plane distance, and the bottom half $p_i \times n_i$ is the moment of the
normal about the origin, how fast a rotation changes it, the same force-and-torque pairing as a
wrench in rigid-body mechanics. Deriving this on the spot is a common interview ask.

> **Exercise 4 (one Jacobian row).** A moved source point is $p = (1, 2, 0)$, its matched target
> point is $q = (1, 2, -0.1)$, and the target normal is $n = (0, 0, 1)$: the point is 0.1 above
> the target's tangent plane $z = -0.1$. (a) Compute the residual and $a$. (b) Using the
> linearization, find the pure translation, the pure rotation about $x$, and the pure rotation
> about $y$ that each bring the residual to zero. Which single-axis motions cannot change it?
> (c) Apply the rotation about $x$ exactly and compute the remaining residual.

<details>
<summary>Answer</summary>

(a) The residual is $(p - q) \cdot n = 0.1$. The bottom half of $a$ is
$p \times n = (2 \cdot 1 - 0 \cdot 0,\ 0 \cdot 0 - 1 \cdot 1,\ 1 \cdot 0 - 2 \cdot 0) = (2, -1, 0)$,
so $a = (0, 0, 1, 2, -1, 0)$.

(b) The linearized residual is $0.1 + \rho_z + 2\theta_x - \theta_y$. A translation
$\rho_z = -0.1$ zeroes it. So does $\theta_x = -0.05$: the point is 2 from the $x$ axis, so a
small rotation about $x$ moves it in $z$ twice as fast. So does $\theta_y = +0.1$: the point is 1
from the $y$ axis, and a positive rotation about $y$ moves a point on the $+x$ side downward.
Translations along $x$ and $y$ and rotation about $z$ have zero entries in $a$: they slide the
point within the tangent plane or spin it about an axis parallel to the plane's normal, which
the cost does not see to first order.

(c) Rotating $(1, 2, 0)$ about $x$ by $-0.05$ gives $z = 2\sin(-0.05) = -0.09996$ and
$y = 2\cos(0.05) = 1.9975$. The new residual is $-0.09996 - (-0.1) = 4.2 \times 10^{-5}$, equal to
$2(\theta - \sin\theta) \approx \theta^3/3$, third order in the angle, as a first-order
linearization promises. The 0.0025 drop in $y$ is a slide within the
tangent plane, which costs nothing.

</details>

### When point-to-plane goes singular

Removing the tangential directions from the cost makes point-to-plane fast, and it also makes it
fail. If every normal in the overlap points the same way, nothing in the cost
constrains sliding within that plane, and $\sum_i a_i a_i^\top$ loses rank. For a flat sheet
whose normals are all $+z$, every $a_i$ has zero entries for $\rho_x$, $\rho_y$, and $\theta_z$,
so the matrix has a three-dimensional null space: the two in-plane translations and the
rotation about the sheet normal. A surface whose normals vary from point to point, such as a
bumpy terrain, constrains all six directions.

The real-world version is a tunnel, a long featureless corridor, or a highway with no structure
ahead: the along-axis direction is unconstrained and the solve drifts along it while reporting a
small residual. Production LiDAR systems detect this from the spectrum of the same matrix
(the smallest eigenvalue, or its ratio to the largest) and either suppress the update
in that direction or lean on another sensor (Zhang, Kaess and Singh 2016).

> **Exercise 5 (our one-wall variant).** In 2D the twist is $(\rho_x, \rho_y, \theta)$ and the
> Jacobian row is $a_i = (n_x, n_y,\ p_x n_y - p_y n_x)$, the last entry being the 2D cross
> product $p \times n$. Our one-wall variant keeps only the wall $x = 0$ of our corner example:
> points $(0, 1)$, $(0, 2)$, $(0, 3)$ with normal $(1, 0)$. (a) Compute $\sum_i a_i a_i^\top$ and
> find its null space. What motion is it? (b) Add back the three points on the wall $y = 0$ with
> normal $(0, 1)$ and show the matrix becomes invertible.

<details>
<summary>Answer</summary>

(a) Each row is $a_i = (1, 0, -y_i)$, with $y_i = 1, 2, 3$. Summing $a_i a_i^\top$ gives

$$\sum_i a_i a_i^\top = \begin{bmatrix} 3 & 0 & -6 \\ 0 & 0 & 0 \\ -6 & 0 & 14 \end{bmatrix},$$

using $\sum y_i = 6$ and $\sum y_i^2 = 14$. The middle row and column are zero, so
$(0, 1, 0)$ is a null vector: a translation along the wall. The remaining $2 \times 2$ block has
determinant $3 \cdot 14 - 36 = 6 \ne 0$, so the null space is exactly that one direction (the
eigenvalues are $0$, $0.36$, and $16.64$). One wall pins the distance to the wall and the
rotation, and says nothing about where along the wall the scanner is: the corridor failure in
miniature.

(b) The wall $y = 0$ points $(x_i, 0)$ with normal $(0, 1)$ give rows $(0, 1, x_i)$, with
$x_i = 1, 2, 3$. Adding their outer products gives

$$\sum_i a_i a_i^\top = \begin{bmatrix} 3 & 0 & -6 \\ 0 & 3 & 6 \\ -6 & 6 & 28 \end{bmatrix},$$

with determinant $3(84 - 36) - 6 \cdot 18 = 36 \ne 0$. All three directions are constrained. The
smallest eigenvalue is about $0.39$, against $30.6$ for the largest, so one combination of
rotation and translation is constrained much more weakly than the rest: six points within 3 m of
the corner give a short lever arm.

</details>

### The correspondence search

Every iteration needs, for each source point, its nearest target point. A brute-force scan is
$O(nm)$ for $n$ source and $m$ target points. At the sizes in this assignment, 800 against 800,
that is 640k distance evaluations per iteration and nobody notices. A 100k-point LiDAR sweep
against a 100k-point local map is $10^{10}$ per iteration, and it is the whole runtime.

A kd-tree is the standard fix. It is a binary tree over the target points where each internal
node splits its points with a plane perpendicular to one coordinate axis, cycling through the
axes or picking the widest one, cutting at the median so the tree stays balanced at depth
$\log_2 m$. A query descends to the leaf containing the query point and records the best distance
found. On the way back up it visits a sibling subtree only if the distance from the query to that
node's splitting plane is less than the best distance so far, since no point on the far side can
beat it otherwise. Most siblings fail that test and are never opened, so the
average cost is logarithmic in $m$ for low-dimensional data, worst case still linear.
The fraction of subtrees that can be pruned shrinks quickly as dimension grows, which is why
kd-trees are used at $d = 3$ and abandoned for high-dimensional descriptor matching.

Approximate nearest neighbor goes one step further: cap the number of nodes visited or the number
of backtracks, and accept a small probability of returning something that is not quite the
nearest point. For ICP that trade is cheap, because the correspondence is a guess that gets
recomputed next iteration anyway. FLANN and nanoflann are the usual libraries.

### Initialization and the basin of convergence

Our pillar-corridor example generalizes: ICP descends to the nearest local minimum of a
non-convex cost, and the set of starting poses from which it reaches the right one is its basin
of convergence. Roughly, the initialization has to be good enough that most source points'
nearest target points are the right piece of surface. In LiDAR odometry that comes for free from
a constant-velocity prediction off the previous scan pair, from wheel odometry, or from
integrating an IMU across the scan interval.

When there is no such prediction (relocalization, loop closure, merging two scans of unknown
relative pose), the initialization comes from global registration, which finds a coarse transform
with no prior at all. The standard recipe computes a local descriptor at each point, a vector
summarizing the shape of its neighborhood that does not change when the cloud is moved. FPFH
(Rusu et al. 2009) is the classic hand-designed one: a histogram of the angles between a point's
normal and the normals of its neighbors. Matching descriptors between the two clouds proposes
candidate pairs, and RANSAC, from the multi-view assignment, samples triples of pairs and keeps
the transform the most other pairs agree with. The result is coarse, typically degrees and
centimeters off, which is exactly the input ICP wants. Learned descriptors have largely replaced
hand-designed ones in this step (Choy et al. 2019).

### Production variants

The variants that replace vanilla ICP soften the hard nearest-point correspondence.

Generalized ICP (Segal et al. 2009) attaches a covariance to every point, computed from its
neighbors the same way normals are, and then reshapes it: a locally planar neighborhood has one
small eigenvalue, and GICP replaces the eigenvalues with $(1, 1, \epsilon)$, turning each point
into a flat disk that is uncertain along the surface and certain across it. The residual
$d_i = q_i - (R p_i + t)$ is scored by its Mahalanobis distance under the combined covariance,

$$\sum_i d_i^\top \big(C_i^Q + R\,C_i^P R^\top\big)^{-1} d_i,$$

the same "squared error divided by how uncertain that error is" used for data association in the
EKF-SLAM assignment. Both earlier costs are special cases: identity covariances everywhere give
back point-to-point, and a zero source covariance with a flat-disk target covariance gives back
point-to-plane, since the inverse then grows without bound along the normal and the cost keeps
only that component. GICP sits between them, and because the disks on both clouds carry the
local surface orientation, it tolerates the two clouds sampling different points better than
either.

The normal distributions transform (Biber and Straßer 2003 in 2D, Magnusson 2009 in 3D) drops
point correspondences altogether. Divide the target volume into voxels, and in each voxel compute
the mean and covariance of the points that fall inside, so the target becomes a piecewise
Gaussian density rather than a set of points. The score of a candidate pose is the sum, over
source points, of that density evaluated at the transformed point, maximized by Newton's method.
There is no nearest-neighbor search and no discrete correspondence step, so the cost is smooth in
the pose within a voxel, and the voxel size sets how far off the initialization can be and still
be pulled in.

Both are still local methods that need an initialization. And the elaborate variants have not
simply won: KISS-ICP (Vizzo et al. 2023) is plain point-to-point ICP with a voxel-grid data
structure instead of a tree, an adaptive correspondence threshold, and a constant-velocity motion
prediction, and it is competitive with the more elaborate methods on LiDAR odometry benchmarks.
Rusinkiewicz and Levoy (2001) map the design space these all live in: which points to select,
how to weight them, which to reject, and which error metric to minimize.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the
declarations and comments in `icp.hpp` and `icp.cpp` give the signatures and shapes.

| # | Function | File | Concept from the notes |
|---|---|---|---|
| 1 | [`align_point_to_point()`](icp.cpp) | `icp.cpp` | the point-to-point solve: centroids, $H$, SVD, the $\det(VU^\top)$ correction |
| 2 | [`point_to_plane_step()`](icp.cpp) | `icp.cpp` | one Gauss-Newton step: rows $a_i = [n_i;\ p_i \times n_i]$, the $6 \times 6$ solve, $\exp(\widehat\xi)$ |
| 3 | [`icp()`](icp.cpp) | `icp.cpp` | the ICP loop: correspondence step, distance gate, solve step, left composition, stopping |

You may not include Open3D, PCL, nanoflann, Sophus, Ceres, GTSAM, or g2o; a test scans the
sources.

Notes on specific holes:
- The provided primitives in `models.cpp` are `hat3` (the skew matrix), `se3_exp` (the $SE(3)$
  exponential, $\xi = [\rho; \theta]$), and `nearest_neighbors`, a brute-force scan that returns,
  for each source row, the index of the closest target row and that distance. Call these; do not
  reimplement them. A correct balanced kd-tree with the backtracking bound is its own exercise,
  and it is not the registration lesson, so the search is provided; nanoflann is on the
  forbidden list because the search would then not be from scratch.
- Clouds are row-major $N \times 3$ Eigen matrices; the target normals are unit rows on the
  target cloud.
- `align_point_to_point`: $H = \sum_i (p_i - \bar p)(q_i - \bar q)^\top$ is
  `Pc.transpose() * Qc` on the centered matrices. Use `Eigen::JacobiSVD` with
  `Eigen::ComputeFullU | Eigen::ComputeFullV`.
- `point_to_plane_step`: the inputs are already matched and already moved by the current pose.
  Accumulate $\sum_i a_i a_i^\top$ and $\sum_i a_i b_i$, solve with `.ldlt().solve(...)` (Eigen's
  `LDLT`, a Cholesky-type factorization for symmetric matrices), and return `se3_exp(xi)`.
- `icp`: keep pairs with distance at most `max_corr_dist` (`MAX_CORR_DIST = 1.0` in
  `config.py`, in the units of the cloud), stop early if fewer than 3 pairs survive, compute
  the RMS correspondence distance over the kept pairs before applying the update, compose on the
  left (`T = delta * T`), and stop when the increment's translation norm and
  $\lVert R_{\text{inc}} - I \rVert$ are both below $10^{-10}$ or after `max_iter` iterations.
  Return the pose, the number of iterations run, and that RMS.

### Building and running

Same toolchain as the rest of the module: a C++17 compiler plus CMake, pybind11, and Eigen (the
`environment.yml` a14 block). You never call CMake by hand: importing `_impl.py` (which the
tests and `viz.py` do) configures and builds a pybind11 extension with CMake, from the top-level
`icp.cpp` by default or from `solution/icp.cpp` under `NANOVISION_IMPL=solution`, compiled
together with `models.cpp` and `bindings.cpp` in both cases. The build is cached under
`build/student/` or `build/solution/`, so only the first run or a source change compiles.

Activate the environment first (`conda activate nanovision`); the Makefile calls the `python`
on the path. Then:

```
make verify A=a14_4_icp   # build + run the reference (solution/); the green target
make test   A=a14_4_icp   # build + run YOUR code; red until the holes are filled
make viz    A=a14_4_icp        # render the ICP convergence (reference)
make viz-mine A=a14_4_icp      # the same, from YOUR code, once the holes are filled
```

`make test` also accepts `K="plane"` to run only matching tests and `SKIP_HOLES=1` to report
unfilled holes as skips instead of failures.

The tests favor implementation-independent statements:
- `test_point_to_point.py`: the closed-form solve recovers a known transform from matched
  correspondences to within $10^{-7}$, returns a proper rotation, and keeps the rotation proper
  (det $+1$) on a planar cloud.
- `test_point_to_plane.py`: the point-to-plane step, iterated on matched correspondences,
  recovers a known transform to within $10^{-6}$, and a single step from the identity already
  reduces both the rotation and the translation error.
- `test_icp.py`: the full loop recovers the transform from an identity start in both modes;
  point-to-plane converges in far fewer iterations than point-to-point on a surface-rich cloud
  (asserted as a wide-margin ordering); and the maximum-distance gate rejects 40 unmatched
  outlier points placed far outside the cloud so the result stays accurate.
- `test_forbidden_imports.py`: a static scan of the C++ sources checks that no registration or
  solver library (Open3D, PCL, nanoflann, Sophus, Ceres, GTSAM, g2o) has been included.
- `test_oracle_open3d.py`: if Open3D is installed, it cross-checks the from-scratch transform
  against Open3D's registration on the same clouds; it is skipped when Open3D is absent. The
  from-scratch solver is the graded code; Open3D is the labeled side-by-side, never the answer.

`make viz` writes `out/icp.rrd`. Add `SHOW=1` for the interactive viewer: the fixed target cloud
in gray, with the source cloud logged twice into the same world view as it slides on, red for
point-to-point and blue for point-to-plane, plus a scalar panel plotting each method's RMS
correspondence distance as it shrinks. Scrub the timeline to step through the iterations and
watch point-to-plane converge first. The RMS at a given step is measured on the correspondences
that step used, before its own update is applied, which is why both methods report the same value
at the first step: both start from the identity.

The figure in the notes is generated by `assets/make_figures.py` (run it from `assets/`), which
computes our corner example in NumPy.

What you should see when you run this. On the terrain cloud with the motion used in the
convergence test (a rotation of about 11.4 degrees and a translation of about 0.31 in cloud
units), point-to-plane drives the rotation error under 0.01 degrees in 2 iterations;
point-to-point needs 12. The first iteration is where the gap opens: point-to-plane goes from
11.4 degrees to 0.66 degrees, while point-to-point reaches 5.4 degrees and then creeps, 4.6, 4.1,
3.6, 3.1, and so on, because each solve is still being pulled by the tangential gaps of
mispaired points. Run to the stopping rule, the loop ends after 5 iterations in point-to-plane
mode and 13 in point-to-point mode. `test_point_to_plane_converges_faster` asserts only the
wide-margin ordering (point-to-plane within 4 iterations, and point-to-point needing at least
twice as many iterations as point-to-plane did), so it does not
depend on the exact counts, which shift with the seed and the size of the initial motion.

The planar test is built to expose the determinant correction. `planar_cloud` is a flat sheet at
$z = 0$, and `test_planar_cloud_det_correction` mirrors the target across $x$, so the data is
literally a reflection of itself, the 800-point version of Exercise 2. On those points, $H$ has
singular values $(2494,\ 2340,\ 0)$ - rank 2, as a flat sheet must be - and
$\det(V U^\top) = -1$. The uncorrected answer is $V U^\top = \operatorname{diag}(-1, 1, 1)$, a
mirror across the $yz$ plane. The corrected answer is $\operatorname{diag}(-1, 1, -1)$, which is
a 180-degree rotation about the $y$ axis. On the plane $z = 0$ the two agree exactly, so both fit
the data perfectly and the correction costs nothing; only one of them is a rotation. The test
checks both halves of that: $\det R = +1$, and the corrected transform still maps source onto
target to $10^{-9}$.

The two clouds also show both sides of the point-to-plane singularity. On the flat sheet from
`planar_cloud` (800 points, all normals $+z$, seed 0), the eigenvalues of $\sum_i a_i a_i^\top$ come out
as $(0,\ 0,\ 0,\ 796,\ 2388,\ 2478)$ (the determinant test uses seed 3, where the
nonzero ones are 798, 2345, and 2495): exactly three zeros, which are the two in-plane
translations and the rotation about the sheet normal. On the bumpy `terrain_cloud` the same
matrix has eigenvalues spread from roughly 40 to 2000 with no null space, because the normals
vary across the cloud. That is why the point-to-plane tests run on the terrain cloud and the
determinant test runs on the sheet. These are toy clouds with analytic normals and no sensor
noise. What they reproduce is the qualitative result from Chen and Medioni (1992): on locally
planar structure the plane metric converges in far fewer iterations than the point metric.

The clouds are synthetic: `terrain_cloud` is a bumpy height field
$z = a\sin(f_x x)\sin(f_y y)$ with analytic surface normals (the surface-rich case where
point-to-plane wins), and `planar_cloud` is a flat sheet (the planar degenerate case for the
determinant correction). `sim.make_pair` builds the source by moving the target by a known
transform, so the ground-truth alignment is that transform's inverse. A real pipeline would add
a normal-estimation step on the target and a kd-tree for the search, but the registration math
is identical, and synthetic clouds with a known transform give the deterministic ground truth
the tests need.

## In interviews

ICP is a standard interview topic for LiDAR, depth-camera, and SLAM roles, both as a derivation
and as a "how would you register two scans" discussion.

Point-to-point versus point-to-plane, and why the latter converges faster. Point-to-point
penalizes the full distance between paired points, tangential slide included, and that slide is
sampling or pairing error rather than pose error, since two scans never hit the same physical
points and a nearest neighbor is rarely the true counterpart. Point-to-plane penalizes only the
distance along the target normal, so points slide within the tangent plane for free and the
solve responds only to real misalignment.

The orthogonal Procrustes solution. Be ready to derive or state the SVD solution for the
point-to-point rotation, and specifically the $\det(V U^\top)$ correction and why it is there:
maximizing the trace alone constrains $R$ to be orthogonal, not to be a rotation, and on
rank-deficient data the best orthogonal fit can be a reflection. Interviewers often ask about this detail
specifically.

The point-to-plane linearization. The standard on-the-spot ask is to derive it: first-order
motion of a point under a twist, dot with the normal, read off the Jacobian row
$[n;\ p \times n]$, assemble the $6\times6$ normal equations.

ICP's local minima and the role of initialization. ICP descends to the nearest minimum of a
non-convex cost, so it needs a coarse initialization (constant-velocity prediction, odometry,
IMU, or a global registration from descriptor matching and RANSAC) to land in the right basin.
Knowing that this is the failure mode, and that GICP and NDT are the production variants that
soften the hard correspondence, is a strong signal.

The cost of correspondence. The nearest-neighbor search dominates real ICP, which is why it runs
on a kd-tree, a voxel grid, or an approximate-nearest-neighbor structure rather than a
brute-force scan. A common follow-up: how would you speed up ICP on a million-point cloud (the
answer involves the search structure, downsampling, and point selection).

## Where this goes next

A LiDAR odometry system runs this loop on every consecutive pair of scans, and the resulting
relative poses drift, because each one carries a small error and they are chained. The
factor-graph assignment takes such relative poses as edges of a pose graph, adds loop-closure
edges (often themselves from ICP, initialized by global registration), and optimizes all the
poses jointly with the same right-perturbation Gauss-Newton loop as the multi-view assignment's
PnP refinement.

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
