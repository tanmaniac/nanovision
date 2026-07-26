# a14_3 - Multi-view geometry estimators

The fourth assignment of the classical SLAM module, in C++17 with Eigen. This assignment
builds the geometric estimators that turn pixel correspondences into 3D structure and camera
motion: triangulation, the normalized eight-point algorithm for the fundamental and essential
matrices, essential-matrix decomposition with the cheirality test, PnP, the Sampson distance,
RANSAC, and the composed two-view relative-pose front-end. This is the front end of every
classical visual-odometry and structure-from-motion system, and the part of SLAM that the
deep-learning assignments never touch.

It builds on the Lie-group assignment (the SE(3) retraction used to step a pose during PnP
refinement) and the camera-geometry assignment (the OpenCV pinhole model). The SE(3)
exponential and the pinhole projection and its Jacobian are carried over and provided here in
`models.cpp`, so the work is the multi-view estimators on top of them.

Required reading before you start:
- Hartley and Zisserman, *Multiple View Geometry in Computer Vision* (2004), chapters 9-12 -
  epipolar geometry, the fundamental and essential matrices, two-view reconstruction, and
  triangulation. Chapter 7, computation of the camera matrix, is the DLT behind PnP. This is
  the reference the whole assignment follows.
- Hartley, "In defense of the eight-point algorithm" (PAMI 1997) - what the normalization step
  does to the conditioning of the linear system, and how much accuracy that buys.
- Fischler and Bolles, "Random sample consensus" (1981) - the original RANSAC paper.

## Lecture notes

### The conventions, pinned first

Multi-view geometry is full of silent transpose and sign traps. A result that is correct up
to a transpose or a sign flip does not announce itself as a transpose bug; it looks like a bug
somewhere else entirely (cheirality "fails" on correct code, an epipolar line lands on the
wrong side of the image). The remedy is to pin the conventions once and never deviate.

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
- Vectors are columns, and indices in these notes are 1-based: $U_{:,3}$ is the third column
  of $U$ and $(F x)_1$ is the first component of $F x$. The C++ is 0-based, so those are
  `U.col(2)` and `(F * x)(0)`.

### Homogeneous coordinates and equality up to scale

A pinhole camera divides by depth, and division is not a linear operation on $(X, Y, Z)$.
Homogeneous coordinates buy the linearity back by adding one coordinate and declaring that
scaling the whole vector changes nothing: the 3-vector $(a, b, w)$ with $w \neq 0$ names the
image point $(a/w,\, b/w)$, and $(2a, 2b, 2w)$ names the same image point. Recovering
$(a/w,\, b/w)$ is dehomogenizing. A world point becomes the 4-vector $\tilde{X} = (X, Y, Z, 1)$
and the whole projection collapses into one matrix multiply followed by a dehomogenize:

$$\tilde{x} = P\,\tilde{X}, \qquad P = K\,[\,R \mid t\,] \in \mathbb{R}^{3 \times 4}.$$

Because the scale of $\tilde{x}$ carries no information, the honest way to write the relation
is $\tilde{x} \simeq P \tilde{X}$, read "equal up to some nonzero scale factor". That symbol
appears throughout these notes.

Two consequences shape every estimator here. First, a matrix that is only defined up to scale
($P$, $F$, $E$) has one fewer degree of freedom than it has entries, and an estimator has to
fix the scale by convention rather than measure it; the convention used below is unit norm,
which is what an SVD hands back. Second, $\tilde{x} \simeq P \tilde{X}$ is not usable as a
linear constraint, because the unknown scale factor sits in the middle of it. The standard way
to remove it is the cross product: two vectors are parallel exactly when their cross product
vanishes, so $\tilde{x} \times (P \tilde{X}) = 0$ says the same thing with no scale factor
left, and it is linear in whichever of $\tilde{X}$ or $P$ is unknown. Both the triangulation
and PnP linear solves are that one trick.

Lines get homogeneous coordinates too. The 3-vector $\ell = (a, b, c)$ names the image line
$\{(u, v) : au + bv + c = 0\}$, and a point lies on that line exactly when
$\ell^\top \tilde{x} = 0$. That reading turns the epipolar constraint into a statement about a
point lying on a line, which is how it gets used.

### Solving a homogeneous linear system with the SVD

Three of the estimators in this assignment reduce to the same numerical problem: stack the
measurements into a matrix $A \in \mathbb{R}^{m \times n}$ and find the $n$-vector $x$ with
$A x = 0$. With noisy measurements no exact null vector exists, and the scale-free version of
the question is

$$\hat{x} = \arg\min_{\lVert x \rVert = 1} \lVert A x \rVert^2 .$$

The unit-norm constraint is not cosmetic: without it $x = 0$ wins and says nothing.

Take the SVD $A = U \Sigma V^\top$ with singular values $\sigma_1 \ge \dots \ge \sigma_n \ge 0$
and $V$ orthonormal. Since the columns of $V$ are an orthonormal basis, write
$x = \sum_j c_j v_j$ with $\sum_j c_j^2 = 1$. Then $\lVert A x \rVert^2 = \sum_j \sigma_j^2 c_j^2$,
a weighted average of the $\sigma_j^2$ with weights summing to one, and the way to minimize a
weighted average is to put all the weight on the smallest entry. So $c_n = 1$ and
$\hat{x} = v_n$, the last column of $V$. That is the entire justification for the instruction
"take the right singular vector with the smallest singular value", which appears three times
below.

Two properties of that solve matter later. The residual at the optimum is $\sigma_n$, so
$\sigma_n$ measures how nearly the constraints are consistent. And the sensitivity of
$\hat{x}$ to noise in $A$ is set by how well separated $\sigma_n$ is from $\sigma_{n-1}$:
perturbing $A$ by $\Delta$ rotates $v_n$ by roughly
$\lVert \Delta \rVert / (\sigma_{n-1} - \sigma_n)$. Measurement noise typically scales
$\lVert \Delta \rVert$ with the overall size of $A$, which is $\sigma_1$, and by construction
$\sigma_n \approx 0$, so the number that predicts accuracy is the ratio
$\sigma_1 / \sigma_{n-1}$. A large ratio means an ill-conditioned solve: the null direction is
barely distinguishable from its neighbors, and a little noise moves it a lot. That ratio is
exactly what the normalization step in the eight-point algorithm attacks.

### Algebraic error, geometric error, and Gauss-Newton

Every $A x = 0$ estimator minimizes $\lVert A x \rVert$, and the entries of $A x$ are the
residuals of whatever algebraic identity was stacked into the rows. Those residuals are
whatever the algebra happened to produce. They carry mixed units, they weight correspondences
by where in the image the points sit, and multiplying one row by a constant would change the
answer without changing what the row asserts. This is the algebraic error, and minimizing it
is a convenience that makes the problem linear, not a statement about measurement noise.

The geometric error is the one worth minimizing: the sum of squared distances, in pixels,
between where a feature was measured and where the estimated model predicts it. The standard
form is the reprojection error $\sum_i \lVert u_i - \pi(K(R X_i + t)) \rVert^2$. Under the
usual assumption that pixel measurements carry independent isotropic Gaussian noise,
minimizing it is maximum likelihood. It is nonlinear in the unknowns, so no SVD solves it.

The pattern throughout this assignment, and in classical structure from motion generally, is
therefore two stages: a linear method minimizing algebraic error to get a starting point, then
a nonlinear refinement minimizing geometric error from there.

The refinement is Gauss-Newton. Write the residual as measurement minus prediction,
$r(\theta) = z - h(\theta)$, and let $J = \partial h / \partial \theta$ be the Jacobian of the
prediction. Replacing $h$ by its first-order expansion turns the cost into a linear
least-squares problem in the step $\delta$,

$$\lVert r(\theta + \delta) \rVert^2 \approx \lVert r(\theta) - J\delta \rVert^2 ,$$

whose minimizer solves the normal equations

$$H\,\delta = g, \qquad H = J^\top J, \qquad g = J^\top r(\theta).$$

Update $\theta \leftarrow \theta + \delta$, re-evaluate $h$ and $J$ at the new $\theta$, and
repeat. $H$ is the Gauss-Newton approximation to the Hessian of the cost: it drops the term
containing second derivatives of $h$, which is negligible when the residuals are small. It is
symmetric positive semidefinite, so the step is a Cholesky solve (Eigen's `ldlt`) rather than
a general matrix inverse. Near a solution with zero residual the iteration converges
quadratically, which is why the noise-free tests here reach machine precision in a handful of
steps.

Both refinements in this assignment are this loop with a different $\theta$: the three
coordinates of a point in `triangulate_refine`, and a six-dimensional pose step in
`pnp_refine`.

### Triangulation

Given a 3D point seen in two cameras with known projection matrices $P_1, P_2$ (each
$3 \times 4$) and its two image points, recover the 3D point. Each view contributes
$\tilde{x} \simeq P \tilde{X}$, and the cross product removes the scale. Writing the rows of
$P$ as $P^{(1)}, P^{(2)}, P^{(3)}$, the image point as $(x, y)$, and $p = P\tilde{X}$, the
three components of $\tilde{x} \times p = 0$ are

$$x\,p_3 - p_1 = 0, \qquad y\,p_3 - p_2 = 0, \qquad x\,p_2 - y\,p_1 = 0,$$

and the third is $y$ times the first minus $x$ times the second, so only two are independent.
Keeping the first two and substituting $p_k = P^{(k)} \tilde{X}$ gives two linear rows in
$\tilde{X}$ per view. Stacking both views gives a $4 \times 4$ homogeneous system
$A \tilde{X} = 0$, solved by the smallest right singular vector and dehomogenized by dividing
by the fourth entry. This is the direct linear transform (DLT), the name for the whole family
of "remove the scale, stack the rows, take the null vector" constructions.

What is being minimized is worth spelling out, because it explains why a refinement follows.
The first row's residual is $x\,p_3 - p_1 = p_3\,(x - p_1/p_3)$, which is the reprojection
offset in one image coordinate multiplied by $p_3$, the projective depth of the point in that
camera. So the DLT minimizes reprojection error weighted by depth, which up-weights distant
points for no good reason. That is the algebraic-versus-geometric gap in concrete form.

The refinement is the Gauss-Newton loop above with $\theta = X \in \mathbb{R}^3$ and the
4-vector residual stacking both views' $(x - p_1/p_3,\ y - p_2/p_3)$. The Jacobian of the
prediction follows from the quotient rule,

$$\frac{\partial}{\partial X}\left(\frac{p_k}{p_3}\right)
= \frac{1}{p_3}\left(P^{(k)}_{1:3} - \frac{p_k}{p_3}\,P^{(3)}_{1:3}\right), \qquad k = 1, 2,$$

where the subscript $1{:}3$ takes the first three entries of the row (the fourth multiplies the
constant 1 of $\tilde{X}$ and contributes nothing to the derivative). On noise-free
correspondences the DLT already lands on the point to machine precision, so the test for the
refinement starts it from a deliberately perturbed 3D point and checks it converges back. The
refinement earns its keep once there is pixel noise and the depth weighting starts to bite.

### Epipolar geometry

When the same scene point is seen in two images, its two image points are not independent. The
point, the two camera centers, and the two rays all lie in one plane, the epipolar plane of
that point: it contains the baseline joining the two centers, and it contains the point.
Coplanarity of three vectors is the vanishing of their triple product, so write the three in
frame 2. The vector from camera 2's center to the point is $X_2$; the vector from camera 2's
center to camera 1's center is $t$ (camera 1's origin maps to $R\cdot 0 + t$); and the ray
seen by camera 1, expressed in frame 2, is $R X_1 = X_2 - t$. Their triple product vanishes,
and since $x_1 \simeq X_1$ and $x_2 \simeq X_2$ the scale factors drop out:

$$x_2 \cdot (t \times R x_1) = 0 \quad \Longleftrightarrow \quad x_2^\top [t]_\times R\, x_1 = 0
\quad \Longleftrightarrow \quad x_2^\top E\, x_1 = 0, \qquad E = [t]_\times R .$$

That is where the essential matrix comes from: it is the coplanarity condition written as a
bilinear form.

Fix $x_1$ and read the constraint as a condition on $x_2$. It says
$(E x_1)^\top x_2 = 0$, which by the line convention above says $x_2$ lies on the line
$\ell_2 = E x_1$. This is the epipolar line of $x_1$: the matching point in image 2 must lie
on it, so correspondence search drops from a 2D window to a 1D line, and a candidate match can
be scored by its distance from that line. Symmetrically $\ell_1 = E^\top x_2$ is a line in
image 1.

The epipole is where the other camera's center projects. Camera 1's center lands at $e_2$ in
image 2, camera 2's center lands at $e_1$ in image 1. Every epipolar plane contains the
baseline, so every epipolar line in image 2 passes through $e_2$; a family of lines through a
common point is called a pencil, so the epipolar lines of image 2 form the pencil through
$e_2$. Algebraically, $e_2$ lies on $E x_1$ for every $x_1$, which forces $E^\top e_2 = 0$,
and symmetrically $E e_1 = 0$. So $E$ has a null vector on both sides and is singular. Its
rank is exactly 2, and the reason is visible in the factorization: a $3 \times 3$
skew-symmetric matrix $[t]_\times$ always has $t$ in its null space (because $t \times t = 0$)
and rank 2 otherwise, and multiplying by the invertible $R$ preserves rank.

Everything transfers to pixels by substituting $x = K^{-1}u$:
$u_2^\top K_2^{-\top} E K_1^{-1} u_1 = 0$, which defines the fundamental matrix
$F = K_2^{-\top} E K_1^{-1}$. Same rank 2, same epipoles, now in pixel coordinates.

Counting degrees of freedom explains what the estimators can and cannot exploit. $F$ has 9
entries, minus 1 for the free overall scale, minus 1 for the constraint $\det F = 0$: 7 free
parameters. $E$ knows the intrinsics, so its content is a rotation (3) plus a translation
direction (2): 5. Both are below 8, so the eight-point algorithm is not a minimal solver, and
$E$ carries structure beyond rank 2 that the linear method throws away. Concretely, the two
nonzero singular values of a true essential matrix are equal, which follows from
$E = [t]_\times R$: the singular values of $[t]_\times$ are $(\lVert t \rVert, \lVert t \rVert, 0)$,
and right-multiplying by an orthogonal matrix does not change them.

### The eight-point algorithm

The constraint $x_2^\top F x_1 = 0$ is linear in the entries of $F$, so each correspondence
$((u, v), (u', v'))$ gives one row of a homogeneous system in the 9-vector
$f = \mathrm{vec}(F)$, stacking $F$ row by row:

$$[\,u'u,\ u'v,\ u',\ v'u,\ v'v,\ v',\ u,\ v,\ 1\,]\, f = 0.$$

With eight or more correspondences, $f$ is the smallest right singular vector of the stacked
$N \times 9$ matrix, by the argument in the SVD section - hence "eight-point". Two steps
separate the textbook formula from a usable estimator.

Rank-2 enforcement. The linear solve has no way to impose $\det F = 0$, so it returns a
full-rank matrix. Fix it afterwards: take the SVD $F = U \Sigma V^\top$, set the smallest
singular value to zero, and rebuild $U\,\mathrm{diag}(\sigma_1, \sigma_2, 0)\,V^\top$. Zeroing
the smallest singular value is the closest rank-2 matrix in Frobenius norm, by the
Eckart-Young theorem, so this is a projection rather than a hack. Skipping it leaves $F$
invertible, so it has no null vector and therefore no epipole: the lines $F x_1$ for different
points fan out across the image instead of meeting in a pencil.

Normalization and conditioning. Pixel coordinates on this scene's 640x480 image run to about
640, so the columns of the $N \times 9$ matrix have wildly different magnitudes - the $u'u$
column reaches roughly $4 \times 10^5$ while the last column is exactly 1. By the sensitivity
argument above, the accuracy of the null vector is governed by $\sigma_1 / \sigma_8$, and on
this scene's raw matrix that ratio comes out around $6 \times 10^4$.

The fix is to precondition the data. Replace each image's points by a similarity transform of
them chosen to make all the numbers order 1: translate so the centroid is at the origin, and
scale so the mean distance from the origin is $\sqrt{2}$, which puts a typical point near
$(\pm 1, \pm 1)$ so that all three homogeneous coordinates are comparable. Write the transform
as the $3 \times 3$ matrix $T$ acting on homogeneous points, $\hat{x} = T x$. Solve on the
normalized points for $\hat{F}$. Then undo it: $\hat{x}_2^\top \hat{F} \hat{x}_1 = 0$ with
$\hat{x} = Tx$ is $x_2^\top (T_2^\top \hat{F} T_1) x_1 = 0$, so
$F = T_2^\top \hat{F}\, T_1$. Measured on this scene, the same $\sigma_1 / \sigma_8$ ratio
after normalization is about 30, three orders of magnitude better than the raw one.

How much accuracy that conditioning buys depends on the configuration. This scene is generous
- 120 correspondences spread across the whole frame - and both solvers land close to the truth
on it. Hartley (1997) is the reference for the general case: the conditioning improvement is
large enough that the normalized linear method performs comparably to iterative methods that
minimize geometric error, while the unnormalized one is very sensitive to noise. The bad
reputation the eight-point algorithm carried before that paper belongs to the unnormalized
version.

Fed pixel coordinates, the algorithm returns $F$. Fed normalized rays $x = K^{-1}u$, the same
code returns the essential matrix. The rank-2 cleanup only zeros the third singular value and
leaves the first two unequal, so what comes back in the second case is close to an essential
matrix without being exactly one; the decomposition below is built to tolerate that. The
essential matrix carries the metric relative pose and the fundamental matrix does not, because
$F$ folds the unknown intrinsics in with the motion.

Two configurations break the algorithm outright, and both come up in practice. If the camera
only rotates, the baseline is zero, $E = [0]_\times R = 0$, and there is no epipolar geometry
to estimate - the two images are related by a homography instead. If every scene point lies on
a plane, the correspondences are again related by a homography, and any $F$ of the form
$[e_2]_\times H$ satisfies them, so the linear system loses rank and $F$ is not unique. Both
are why monocular VO initialization usually fits a homography and a fundamental matrix in
parallel and picks whichever the data supports.

### From the essential matrix to a pose, and cheirality

The essential matrix factors as $E = [t]_\times R$, and the goal is to read $(R, t)$ back out.
Take the SVD $E = U \Sigma V^\top$ and make both $U$ and $V$ proper rotations by negating the
third column of either one whose determinant is $-1$. That negation is free: it changes
$U \Sigma V^\top$ by $\pm \sigma_3 u_3 v_3^\top$, and $\sigma_3 = 0$ for a rank-2 matrix, so it
changes nothing. With the fixed matrix

$$W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix},$$

the two factorizations of $E$ into a skew-symmetric matrix times a rotation (Hartley and
Zisserman derive that there are exactly two) give two rotation candidates and one translation
direction:

$$R_1 = U W V^\top, \qquad R_2 = U W^\top V^\top, \qquad t = U_{:,3}.$$

Both $R_1$ and $R_2$ are products of rotations, so both are proper rotations, and $t$ is a
column of an orthogonal matrix, so it already has unit length.

The formulas use only $U$ and $V$ and discard $\Sigma$ entirely, which matters in practice. An
$E$ estimated by the eight-point algorithm has been forced to rank 2, but its two nonzero
singular values are not equal, so it is not exactly an essential matrix. Throwing $\Sigma$
away amounts to replacing it with $\mathrm{diag}(1, 1, 0)$, which is the projection onto the
set of essential matrices; the $(R_1, R_2, t)$ that come back define a genuine essential
matrix near the estimated one. No separate cleanup step is needed.

The sign of $t$ is not determined by $E$, since $[t]_\times R$ and $[-t]_\times R$ differ only
by an overall sign and $E$ is defined up to scale anyway. So there are four candidate poses:
$(R_1, +t), (R_1, -t), (R_2, +t), (R_2, -t)$. The epipolar constraint cannot tell them apart
because it is a statement about lines through camera centers, and a line does not distinguish
a point in front of the camera from its mirror image behind it. Exactly one of the four
candidates puts the scene in front of both cameras, and picking it is the cheirality test
("cheirality" from the Greek for hand, the same root as "chiral": it is the handedness
information that the algebraic constraint discards). The test is direct: triangulate the
correspondences against each candidate, with $P_1 = [\,I \mid 0\,]$ and $P_2 = [\,R \mid t\,]$,
count the points whose depth $z$ is positive in both cameras, and keep the candidate with the
highest count. On noise-free data the true candidate scores every point and the other three
score almost none.

The translation comes back only as a direction, never a length. Scale the entire scene and the
baseline by the same factor - every 3D point to $sX$, the baseline to $st$ - and every
projection is unchanged, because the pinhole divides by depth and the factor cancels top and
bottom. No image measurement can determine $s$. A direction in parameter space that the
measurements cannot see is called a gauge freedom, and monocular two-view geometry has exactly
one, this scale. Estimators deal with a gauge by fixing it arbitrarily; here it is fixed by
returning $\lVert t \rVert = 1$. That is why the tests compare translations by angle rather
than by vector difference, and why a single moving camera cannot report metric distance
without an outside reference: a second camera at a known baseline, an object of known size,
wheel odometry, or an accelerometer.

### PnP, pose from known 3D points

The perspective-$n$-point problem runs the other direction: given $n$ known 3D world points and
their pixels in one calibrated camera, find the camera pose
$T_{\text{cam} \leftarrow \text{world}}$. It is the relocalization step in SLAM (place a new
frame against the existing map) and the standard way to track a camera against known structure.

A linear initialization comes from the DLT again. With normalized rays $x = K^{-1}u$
dehomogenized so the third coordinate is 1, the relation $x \simeq M \tilde{X}$ for the
$3 \times 4$ matrix $M = [\,R \mid t\,]$ is linear in the 12 entries of $M$. The same cross
product as in triangulation gives two independent rows per point, now with $\tilde{X}$ known
and $M$ unknown, so six points give 12 rows and suffice for a 12-vector defined up to scale.

The catch is that the recovered $M$ is only approximately $[\,R \mid t\,]$. The linear solve
cannot impose $R^\top R = I$, and the answer carries an arbitrary scale. Fix both, in order.
Scale first: if $M$ is scaled by $s$ then its left $3 \times 3$ block is too, and that block's
determinant by $s^3$, while a true $[\,R \mid t\,]$ has $\det R = 1$. So the real cube root of
that determinant recovers $s$, and taking the real cube root rather than a magnitude keeps the
sign - a negative determinant means the solver returned $-M$, and dividing by a negative $s$
flips it back so the points end up in front of the camera rather than behind it. Rotation
second: divide out $s$ and project the left block $B$ onto $SO(3)$. The closest rotation to
$B$ in Frobenius norm is $U V^\top$ from the SVD $B = U \Sigma V^\top$, which is the
orthogonal Procrustes solution, with the third column of $U$ negated if that product comes out
with determinant $-1$, so the result is a rotation and not a reflection.

The DLT minimizes an algebraic error, so a geometric refinement follows, and this is where the
Lie-group machinery enters. The unknown is a pose, which lives on the six-dimensional manifold
$SE(3)$; adding a step to the 12 entries of a pose matrix would leave the manifold and produce
something that is no longer a rigid transform. The retraction from the Lie-group assignment
handles it: parameterize the step by a six-vector twist $\xi = [\rho; \theta]$ (translation
part first, rotation part second, the ordering used throughout this module), map it onto the
group with the exponential $\exp(\xi^\wedge)$, and apply it on the right,

$$T \leftarrow T\, \exp(\xi^\wedge),$$

which reads as "move by $\xi$ expressed in the current body frame". The optimization is then
the Gauss-Newton loop from above with $\theta = \xi$, re-parameterized around the current pose
at every iteration.

The measurement Jacobian is a chain rule with two links. The first link is how the camera-frame
point moves with the twist. Substitute $\exp(\xi^\wedge) \approx I + \xi^\wedge$ into
$T \exp(\xi^\wedge) \tilde{X}_i$ and drop second-order terms:
$X_{\text{cam}} \approx (R X_i + t) + R\rho - R\,[X_i]_\times \theta$, where $X_i$ is the world
point and the last term uses $\theta \times X_i = -[X_i]_\times \theta$. So

$$\frac{\partial X_{\text{cam}}}{\partial \xi} = \big[\,R \ \big| \ -R\,[X_i]_\times\,\big] \in \mathbb{R}^{3 \times 6}.$$

The second link is the pinhole derivative $\partial \pi / \partial X_{\text{cam}}$, the
$2 \times 3$ matrix provided as `pinhole_jacobian`. Composing them,

$$A_i = \frac{\partial \pi}{\partial X_{\text{cam}}}\Big[\,R \ \big| \ -R\,[X_i]_\times\,\Big] \in \mathbb{R}^{2 \times 6}.$$

Accumulate $H = \sum_i A_i^\top A_i$ and $g = \sum_i A_i^\top e_i$ with the residual
$e_i = u_i - \pi(K(R X_i + t))$, solve $H\,\delta\xi = g$, retract, and iterate.

One Lie-group detail is worth naming because it shows up in the next assignment. The right
Jacobian $J_r(\xi)$ is the correction that relates a small change of the twist to a small
motion on the group,
$\exp((\xi + \delta)^\wedge) \approx \exp(\xi^\wedge)\exp((J_r(\xi)\delta)^\wedge)$, and it is
not the identity for a general $\xi$. It never appears in this loop because each iteration
starts its parameterization over: the twist is always measured from the current estimate, so
the linearization point is $\xi = 0$, and $J_r(0) = I$. The pose-graph and bundle-adjustment
assignment keeps residuals that are themselves logarithms of pose errors at nonzero twists, and
differentiating those brings $J_r^{-1}$ in explicitly.

### Outliers and the breakdown point

Everything above assumed the correspondences are correct. Real feature matching does not
deliver that: repeated texture, occlusion boundaries, and independently moving objects all
produce confident matches between points that are not the same scene point.

The breakdown point of an estimator is the largest fraction of arbitrarily corrupted
measurements it can absorb before its output can be pushed arbitrarily far from the truth. For
least squares that fraction is zero, and the reason is in the cost function: a residual enters
as $r^2$, so a match that is 300 pixels wrong contributes $10^4$ times what a 3-pixel error
does, and the fit will happily ruin its agreement with every good measurement to reduce that
one term. A single gross outlier is enough in principle.

The effect is not subtle on this assignment's scene. With 35% of the 120 correspondences
replaced by wrong matches, the plain eight-point fit on all of them returns a rotation 8 to 36
degrees from the truth across the first eight seeds, while the true relative rotation is only
about 7 degrees. The error is larger than the signal.

### The Sampson distance

Any robust wrapper has to decide whether a correspondence agrees with a candidate $F$, and
that needs a distance with units. The algebraic residual $x_2^\top F x_1$ has none: it scales
with the arbitrary scale of $F$ and with the magnitudes of the pixel coordinates, so a fixed
threshold on it does not mean the same thing for two different candidate matrices, let alone
for two different parts of the image.

The quantity that does have units is geometric: how far, in pixels, would the two measured
points have to move for the pair to satisfy the epipolar constraint exactly? Collect the four
measured coordinates as $m = (u, v, u', v')$ and write the constraint as the scalar function
$C(m) = x_2^\top F x_1$ with $x_1 = (u, v, 1)$ and $x_2 = (u', v', 1)$. The exact answer is the
constrained minimization $\min \lVert \delta \rVert$ subject to $C(m + \delta) = 0$, which has
no closed form because $C$ is quadratic in $m$.

Sampson's approximation replaces the constraint surface by its tangent plane at the
measurement: $C(m + \delta) \approx C(m) + J\delta$ with $J = \partial C / \partial m$ a row
4-vector. Minimizing $\lVert \delta \rVert^2$ subject to the single linear constraint
$J\delta = -C(m)$ is the least-norm problem, whose solution is the step perpendicular to that
plane, $\delta = -J^\top C(m) / (J J^\top)$, of length

$$d = \frac{\lvert C(m) \rvert}{\lVert J \rVert}.$$

Differentiating $C$ one coordinate at a time gives $\partial C/\partial u = (F^\top x_2)_1$,
$\partial C/\partial v = (F^\top x_2)_2$, $\partial C/\partial u' = (F x_1)_1$, and
$\partial C/\partial v' = (F x_1)_2$; the third homogeneous components never appear because
they are held at 1 and are not measurements. Substituting,

$$d = \frac{\lvert x_2^\top F x_1 \rvert}{\sqrt{(F x_1)_1^2 + (F x_1)_2^2 + (F^\top x_2)_1^2 + (F^\top x_2)_2^2}}.$$

Both numerator and denominator are linear in $F$, so $d$ does not change when $F$ is rescaled,
and it comes out in the units of $m$, which are pixels. It is a first-order estimate of the
smallest total pixel correction that would make the pair consistent, so a threshold of a pixel
or two is a meaningful statement about feature-localization noise and transfers across images
and across candidate models. The same function scores an essential matrix on normalized rays,
in which case the units are those of the normalized image plane rather than pixels, and the
threshold has to be scaled accordingly.

### RANSAC

RANSAC (random sample consensus, Fischler and Bolles 1981) inverts the usual approach. Instead
of fitting to all the data and hoping the outliers average out, it fits to as little data as
possible and lets the rest vote.

One round: draw a minimal sample uniformly at random (8 correspondences for the eight-point
algorithm), fit a model to just those, score every correspondence in the full set by its
Sampson distance to that model, and collect the ones below the threshold. That collection is
the consensus set. Repeat for a fixed budget of rounds, keep the model with the largest
consensus set, and refit on that set at the end, so the returned $F$ is a least-squares fit to
everything RANSAC judged to be an inlier rather than to the eight points of the winning sample.

The logic is asymmetric, which is the whole idea. A sample that happens to contain only
inliers produces a model near the truth, and every other inlier then agrees with it, so its
consensus set is large. A sample containing even one outlier is constrained by a correspondence
with no geometric meaning, so the model it produces satisfies few other points and its
consensus set is small. Maximizing consensus therefore selects, indirectly, for having drawn a
clean sample.

The iteration budget comes from a probability calculation. With an inlier fraction $w$ and a
sample size $s$, one draw is all-inliers with probability $w^s$ (treating draws as independent,
which is close enough when $N$ is much larger than $s$). Getting at least one clean sample with
probability $p$ takes

$$N = \frac{\log(1 - p)}{\log(1 - w^s)}$$

rounds. This assignment's scene is 35% outliers, so $w = 0.65$; at $s = 8$ and $p = 0.99$ the
formula asks for about 142 rounds, and `config.py` budgets 500. The exponent is what hurts:
$w^s$ decays geometrically in $s$, so a five-point solver at the same $w$ needs about 37 rounds
for the same confidence. At 50% outliers the same comparison is roughly 1180 rounds against
145. That gap is the argument for minimal solvers, and it widens as conditions get worse.

Running the front-end on this scene (120 correspondences, 35% of them wrong matches, 0.5 px
noise on the rest), what you will see is a recovered inlier set with essentially no false
positives and a handful of true inliers missed: across the first eight seeds, precision runs
0.99 to 1.00 and recall 0.87 to 0.99. The recovered rotation lands within 0.06 to 0.23 degrees
of the truth and the translation direction within 0.4 to 3.3 degrees, against the plain
eight-point fit's 8 to 36 degrees of rotation error on the same data. The tests assert the
ordering rather than the numbers - robust rotation error under 1 degree, and naive error more
than three times robust - so the check survives different seeds and different implementations.
These are toy figures from a synthetic scene: they show the mechanism working, not how a real
front-end behaves on real features.

### The composed front-end

The pieces assemble into the actual visual-odometry front-end. RANSAC the fundamental matrix
from the pixel correspondences, convert it to the essential matrix with
$E = K_2^\top F K_1$ (the inverse of $F = K_2^{-\top} E K_1^{-1}$), recover $(R, t)$ by
cheirality on the inlier rays, and triangulate the inliers into 3D points expressed in frame 1.
The output is a relative pose up to the monocular scale plus a sparse point cloud: one camera's
worth of motion and structure. That is the unit a SLAM back-end consumes. Front end here means
the per-frame-pair geometry that produces measurements; back end means the optimization that
takes many such measurements and solves for all the poses and points at once. Assembling this
front end from its parts is a common interview ask.

### Where this sits

The eight-point algorithm is the pedagogical route, not the state of the art. The minimal
relative-pose solver is the five-point algorithm (Nister 2004). Since $E$ has 5 degrees of
freedom, five correspondences are enough in principle, but using the extra structure means
imposing the equal-singular-value condition, which as a polynomial identity in $E$ is
$2EE^\top E - \mathrm{tr}(EE^\top)E = 0$. Those constraints are cubic rather than linear, so
the solver ends in the roots of a degree-10 polynomial and returns up to ten candidate
essential matrices, disambiguated by cheirality and by consensus scoring. The payoff is the
exponent in the RANSAC budget above: five points per sample instead of eight is roughly a
quarter of the rounds at this scene's outlier rate, and a much larger factor at higher ones.
The eight-point algorithm is linear and easy to derive, which is why it is taught first;
production systems put the five-point solver inside RANSAC.

RANSAC itself has been refined in ways worth knowing by name. LO-RANSAC (Chum, Matas and
Kittler 2003) adds a local optimization step whenever a new best model is found, refitting on
its consensus set before scoring again, which reaches a given accuracy in far fewer rounds.
MAGSAC (Barath, Matas and Noskova 2019) removes the inlier threshold as a tuned parameter by
marginalizing the scoring over a range of plausible noise scales.

The estimators here are the geometric layer, and modern systems mostly change the layers around
them rather than replacing them. Learned detectors and matchers produce the correspondences
that feed these solvers, and the pose is still recovered by the same essential-matrix
machinery. The pose-graph and bundle-adjustment assignment is the back end that takes many
two-view results and refines all the poses and points jointly. The geometry
foundation-model assignment goes the other way and predicts geometry directly from images,
skipping the correspondence-then-solve pipeline entirely.

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
on noise-free correspondences, and the refinement converges back to the true point from a
perturbed start. The recovered $F$ satisfies the epipolar constraint and is rank 2, and the
same solver on normalized rays gives an essential matrix satisfying the constraint on rays.
The decomposition yields proper rotations and a unit translation, and cheirality recovers the
true pose (rotation and translation direction to better than $10^{-4}$ degrees, translation
length not compared, since it is the free gauge) with all points in front of both cameras. PnP
recovers a known pose from noise-free data and the refinement drives the reprojection error
below $10^{-8}$ pixels RMS. The Sampson distance matches its closed form and separates inliers
from outliers by more than a factor of five in the median. RANSAC recovers the inlier set with
high precision and recall, and the robust pose beats the naive least-squares pose by more than
a factor of three across seeds. A static scan of the C++ sources also blocks the shortcut of
including OpenCV, Ceres, GTSAM, g2o, Sophus, manif, or OpenGV; it passes with the holes still
in place.

`make viz` writes `out/multiview.rrd`, a recording for the Rerun viewer. Add `SHOW=1` to open
it interactively instead: the triangulated points and the two recovered camera frusta (the
pyramid of space each camera sees) in 3D, the correspondences colored by inlier versus outlier
in the two image views, and eight epipolar lines drawn in the second image, where every inlier
sits on its line and every outlier does not.

The scene is synthetic: 120 random points at depths of 4 to 12 m viewed by two 640x480 pinhole
cameras with $f = 500$ px, separated by a baseline of about 0.84 m and a relative rotation of
about 7 degrees, with the world frame chosen as camera 1 so the ground-truth relative pose is
exactly $T_{2 \leftarrow 1}$. A real image pair would put a feature detector and descriptor
matcher in front of these estimators, but the geometry being estimated is identical, and
synthetic correspondences give the deterministic ground truth the tests need.

## In interviews

Multi-view geometry is a standard interview topic for perception, SLAM, and 3D-vision roles,
both as derivations on a whiteboard and as "assemble the front-end" questions.

Outline the normalized eight-point algorithm. This is a frequent on-the-spot ask. The answer
is the four steps: normalize each image's points (centroid to origin, mean distance
$\sqrt{2}$), solve the linear system for the null vector, enforce rank 2 by zeroing the
smallest singular value, denormalize. Be ready to say why normalization matters (the
conditioning of the linear system, and what conditioning means for how noise propagates to the
null vector) and why rank-2 enforcement matters (a real fundamental matrix is rank 2; its null
space is the epipole, and without the constraint the epipolar lines do not meet).

Essential versus fundamental. Know that the essential matrix is the calibrated version
($x_2^\top E x_1 = 0$ on normalized rays, carries the metric pose, 5 degrees of freedom) and
the fundamental matrix is the uncalibrated one ($x_2^\top F x_1 = 0$ on pixels,
$F = K_2^{-\top} E K_1^{-1}$, 7 degrees of freedom). The follow-up is usually the
decomposition: four candidate poses from $E$, resolved by cheirality.

Why RANSAC is mandatory. The breakdown point of least squares is zero - one outlier ruins the
fit - so any estimator running on real correspondences needs a robust wrapper. Know the loop
(minimal sample, consensus scoring, refit) and that the score must be a metric distance (the
Sampson distance), not the unitless algebraic residual. A strong answer works the iteration
count out loud from $N = \log(1-p)/\log(1-w^s)$ and uses it to explain why the five-point
solver replaced the eight-point one inside RANSAC.

PnP and its variants. Know that PnP recovers a calibrated camera's pose from 3D-2D
correspondences, and that the DLT gives a linear initialization that a nonlinear refinement
over $SE(3)$ then improves. Two named solvers come up. P3P is the minimal one: three
correspondences give up to four geometrically valid poses, and a fourth point picks among them,
which makes it the natural fit inside RANSAC. EPnP (Lepetit, Moreno-Noguer and Fua 2009)
writes the $n$ world points as weighted combinations of four virtual control points and solves
for those instead, so the problem size stops growing with $n$ and the cost is $O(n)$. The
standard follow-up is degeneracy: a critical configuration is one where the estimator's linear
system loses rank so the solution is no longer unique, and for the DLT PnP here the common case
is all the points lying on a plane, which needs a dedicated planar solver.

The monocular scale ambiguity. Expect "why can't a single moving camera measure absolute
distance?" The answer is that scaling the scene and the baseline together reprojects
identically, so it is a gauge freedom, not a weakness of a particular estimator: two-view
monocular geometry recovers translation only as a direction, and absolute scale needs a second
sensor, a known object size, or stereo.

## Further reading

- Hartley and Zisserman, *Multiple View Geometry in Computer Vision* (2004), chapters 9-12 -
  the definitive treatment of epipolar geometry, two-view reconstruction, and triangulation;
  chapter 7 for camera resection.
- Hartley, "In defense of the eight-point algorithm" (PAMI 1997) - normalization.
- Nister, "An efficient solution to the five-point relative pose problem" (PAMI 2004) - the
  modern minimal relative-pose solver.
- Lepetit, Moreno-Noguer, Fua, "EPnP: an accurate O(n) solution to the PnP problem" (IJCV
  2009), and the P3P literature - the production PnP solvers.
- Fischler and Bolles, "Random sample consensus" (1981) - RANSAC.
- Chum, Matas and Kittler, "Locally optimized RANSAC" (2003), and Barath, Matas and Noskova,
  "MAGSAC" (2019) - the refinements that production robust estimators actually use.
