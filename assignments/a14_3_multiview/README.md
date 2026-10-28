# a14_3 - Multi-view geometry estimators

The fourth assignment of the classical SLAM module, in C++17 with Eigen. This assignment
builds the geometric estimators that turn pixel correspondences into 3D structure and camera
motion: triangulation, the normalized eight-point algorithm for the fundamental and essential
matrices, essential-matrix decomposition with the cheirality test, PnP, the Sampson distance,
RANSAC, and the composed two-view relative-pose front end. This is the front end of every
classical visual-odometry and structure-from-motion system, and the part of SLAM that the
deep-learning assignments never touch.

It builds on the Lie-group assignment (the SE(3) retraction used to step a pose during PnP
refinement) and the camera-geometry assignment (the OpenCV pinhole model). The SE(3)
exponential and the pinhole projection and its Jacobian are carried over and provided here in
`models.cpp`, so the work is the multi-view estimators on top of them. The Python around the
C++ (the build, the synthetic scenes, the tests, and the Rerun visualization) is provided, as
in the Lie-group, Kalman-filter, and EKF-SLAM assignments.

Required reading before starting:
- Hartley and Zisserman, *Multiple View Geometry in Computer Vision* (2004), chapters 9-12 -
  epipolar geometry, the fundamental and essential matrices, two-view reconstruction, and
  triangulation. Chapter 7, computation of the camera matrix, is the DLT behind PnP. This is
  the reference the whole assignment follows.
- Hartley, "In defense of the eight-point algorithm" (PAMI 1997) - what the normalization step
  does to the conditioning of the linear system, and how much accuracy that buys.
- Fischler and Bolles, "Random sample consensus" (1981) - the original RANSAC paper.

## Lecture notes

### What two-view geometry is for

#### The task

A single camera moves through a static scene and takes two 640x480 pictures. A feature matcher
has paired up 120 points between the two images: for each pair it claims that pixel $u_1$ in
the first image and pixel $u_2$ in the second are views of the same scene point. Such a pair is
a correspondence. The pixel positions carry about half a pixel of localization noise, and 35%
of the correspondences are wrong: the matcher paired two points that are not the same scene
point. The camera intrinsics $K$ are known (focal length 500 px, principal point at the image
center).

The job is to recover, from these 240 pixel positions alone, how the camera moved between the
two shots (a rotation $R$ and a translation $t$, together the relative pose) and where the
correctly matched scene points are in 3D. In the assignment's scene camera 2 sits about 0.82 m
to the left of camera 1 and 0.15 m behind it (a baseline of about 0.84 m), turned by about 7
degrees, mostly a yaw; the points sit 4 to 12 m away. A second, related task comes up once a
map exists: given known 3D points and their pixels in a new image, find where that camera is.
That is the perspective-$n$-point (PnP) problem.

This is what a visual-odometry front end does for every new frame, and what a monocular SLAM
system does to initialize its map from its first two frames.

#### Why the obvious approaches fail

The quantity to minimize is clear: choose a pose and 3D points so that the points, projected
through both cameras, land on the measured pixels. That is a nonlinear least-squares problem in
6 pose parameters and 3 coordinates per point, and Gauss-Newton solves it well from a good
starting guess. There is no good starting guess. The cost is nonconvex in the rotation and in
the depths, and from an arbitrary start the iteration can settle into a wrong local minimum or
put the points behind the camera.

The other obvious move is a linear least-squares fit to all 120 correspondences, which needs no
starting guess. The 42 wrong correspondences ruin it. A least-squares cost squares every
residual, so a correspondence that is 300 pixels wrong outweighs everything else, and the fitted
rotation can end up further from the truth than the 7 degrees the camera actually turned.

Finally, one part of the answer cannot be recovered at all. Doubling every 3D point and the
baseline produces exactly the same two images, so no estimator can report the length of $t$.

#### The idea

Classical two-view geometry solves the problem in stages, each one simple:

1. Write a constraint that every correct correspondence satisfies and that is linear in the
   unknowns. The two camera centers and the scene point form a triangle, so the two viewing
   rays and the baseline lie in one plane. That coplanarity condition is linear in the nine
   entries of a $3 \times 3$ matrix, the essential matrix $E$, which encodes $R$ and the
   direction of $t$. Eight correspondences determine it by a singular value decomposition (SVD).
2. Project the linear answer back onto what it is supposed to be (rank 2 for the fundamental
   matrix, two equal singular values for the essential matrix, a rotation for PnP), because the
   linear solve cannot impose those constraints itself.
3. Read the pose out of $E$. Four poses fit the same $E$; only one puts the points in front of
   both cameras, and the cheirality test picks it.
4. Wrap the linear solver in RANSAC (random sample consensus): fit to many random minimal
   samples (just enough correspondences to determine a model, eight here), score every correspondence against each fit by a distance in pixels, and keep the
   fit that most correspondences agree with. This rejects the wrong correspondences.
5. Where accuracy matters, use the linear answer as the starting guess for Gauss-Newton on the
   reprojection error, the cost the obvious approach wanted to minimize in the first place.

#### Why it matters

Every geometric SLAM and structure-from-motion system runs this pipeline or a refinement of it:
monocular ORB-SLAM initializes its map this way, COLMAP starts its reconstruction from a
two-view pose and triangulation, and a PnP solver inside RANSAC is how a tracker relocalizes
against a map. Learned feature matchers changed where the correspondences come from; what
happens to them afterward is the same.

### Our sideways-step example (two cameras one meter apart, no rotation)

The real scene has 120 points, a rotation, and a baseline in three directions, too much to
follow by hand. We use a stripped-down version that we call our sideways-step example. Camera 1
sits at the origin looking down $+z$. Camera 2 has the same orientation ($R = I$) and the
relative pose $t = (1, 0, 0)$, which puts its center 1 m to the left of camera 1, at
$(-1, 0, 0)$ in camera 1's frame. (The relative pose maps camera-1 coordinates to camera-2
coordinates, so a camera center at $c_2$ satisfies $R c_2 + t = 0$.) This is a rectified
stereo pair, the configuration of stereo depth estimation.

The main point is $X = (0, 0, 4)$, 4 m straight ahead of camera 1. In camera 2's frame it is
$X + t = (1, 0, 4)$. Dividing by depth gives the normalized image coordinates
$x_1 = (0, 0)$ and $x_2 = (0.25, 0)$, and with the assignment's intrinsics ($f = 500$ px,
principal point $(320, 240)$) the pixels $u_1 = (320, 240)$ and $u_2 = (445, 240)$. The
horizontal shift of $125$ px is the disparity, $f b / Z = 500 \cdot 1 / 4$.

![Our sideways-step example. Left: top view of the two cameras and three scene points with the rays from each camera center, and in gray the same scene with every point and the baseline doubled. Right: image 2 with the three matched points (dots), their positions in image 1 (squares), and the epipolar line of each image-1 point, which here is the horizontal scanline through it](assets/sideways_step.png)

The figure shows the three facts the rest of the notes derive. In image 2 each match lies on a
horizontal line through its image-1 position, so a correspondence can be checked by a
one-dimensional test (is it on the line?). The depth follows from the disparity and the
baseline. And the doubled scene in gray projects to exactly the same pixels, so the baseline's
length is invisible.

In the real scene the translation $t = (0.8, -0.05, 0.25)$ m is also mostly sideways: camera 2
sits about 0.82 m to the left of camera 1 and 0.15 m behind it. Our example
keeps the epipolar constraint, triangulation from two rays, the four-way ambiguity of the pose
read out of $E$, and the unobservable scale. It drops the rotation, and it drops the
component of the baseline along the optical axis, which in the real scene makes the epipolar lines converge on a point
(far outside the image) instead of being parallel. It also has no noise and no wrong
correspondences; those enter with the Sampson distance and RANSAC.

### Overview of the rest of the notes

1. Conventions, fixed once, because transposes and sign flips are the usual bugs here.
2. Homogeneous coordinates. The projective geometry behind the pinhole model, used to turn
   "equal up to scale" into a linear equation with a cross product.
3. The homogeneous least-squares solve with the SVD. The tool behind three of the estimators.
4. Algebraic versus geometric error, and Gauss-Newton. The EKF's linearize-and-solve step,
   iterated; this is the refinement stage.
5. Triangulation. The DLT (direct linear transform) on two views, then Gauss-Newton.
6. Epipolar geometry. The coplanarity constraint, the essential and fundamental matrices.
7. The eight-point algorithm. The SVD solve applied to the epipolar constraint, with Hartley
   normalization for conditioning.
8. From $E$ to a pose. An SVD factorization, four candidates, and the cheirality test.
9. PnP. The DLT again, then Gauss-Newton on $SE(3)$ with the right-perturbation retraction from
   the Lie-groups notes.
10. Outliers, the Sampson distance, and RANSAC. Robust estimation; RANSAC does for
    correspondences what the Mahalanobis gate does for data association in EKF-SLAM.
11. The composed front end, and later developments.

Five exercises, placed after the sections they test, are meant to be worked by hand: four on
our sideways-step example and one on the assignment's scene. Each has a worked answer that can be expanded.

### The conventions, pinned first

Multi-view geometry is full of silent transpose and sign traps. A result that is correct up
to a transpose or a sign flip does not announce itself as a transpose bug; it looks like a bug
somewhere else entirely (cheirality "fails" on correct code, an epipolar line lands on the
wrong side of the image). The remedy is to pin the conventions once and never deviate.

- The unprimed image is the first, reference image - the frame the points start in. The
  primed image is the second. Subscripts 1 and 2 mean first and second.
- The relative pose $(R, t) = T_{2 \leftarrow 1}$ takes a frame-1 coordinate into frame-2:
  $X_2 = R X_1 + t$. The essential matrix is $E = [t]_\times R$, where $[t]_\times$ is the
  skew-symmetric matrix with $[t]_\times v = t \times v$ (the hat operator of the Lie-groups
  notes).
- A normalized ray is $x = K^{-1} u$, the pixel with the intrinsics removed, written
  homogeneous with third coordinate 1; its first two entries are the normalized image
  coordinates. The essential constraint $x_2^\top E x_1 = 0$ holds for normalized rays. The
  fundamental constraint $x_2^\top F x_1 = 0$ holds for pixel coordinates $u$, with
  $F = K_2^{-\top} E K_1^{-1}$.
- A camera pose used for projection and PnP is $T_{\text{cam} \leftarrow \text{world}}$: a
  world point projects as $u = \pi(K (R X + t))$, the same OpenCV pinhole as the
  camera-geometry assignment ($+x$ right, $+y$ down, $+z$ forward, $u = f_x X/Z + c_x$).
- Vectors are columns, and indices in these notes are 1-based: $U_{:,3}$ is the third column
  of $U$ and $(F x)_1$ is the first component of $F x$. The C++ is 0-based, so those are
  `U.col(2)` and `(F * x)(0)`.

### Homogeneous coordinates and equality up to scale

A pinhole camera divides by depth, and division is not a linear operation on $(X, Y, Z)$.
Homogeneous coordinates restore linearity by adding one coordinate and declaring that scaling
the whole vector changes nothing: the 3-vector $(a, b, w)$ with $w \neq 0$ names the image
point $(a/w,\, b/w)$, and $(2a, 2b, 2w)$ names the same image point. Recovering
$(a/w,\, b/w)$ is dehomogenizing. A world point becomes the 4-vector $\tilde{X} = (X, Y, Z, 1)$
and the whole projection becomes one matrix multiply followed by a dehomogenize:

$$\tilde{x} = P\,\tilde{X}, \qquad P = K\,[\,R \mid t\,] \in \mathbb{R}^{3 \times 4}.$$

Because the scale of $\tilde{x}$ carries no information, the precise way to write the relation
is $\tilde{x} \simeq P \tilde{X}$, read "equal up to some nonzero scale factor". That symbol
appears throughout these notes. In our sideways-step example, camera 2 in normalized
coordinates (with $K$ left out) has $P_2 = [\,I \mid t\,]$, and
$P_2 \tilde{X} = (1, 0, 4) \simeq (0.25, 0, 1)$.

Two consequences shape every estimator here. First, a matrix that is only defined up to scale
($P$, $F$, $E$) has one fewer degree of freedom than it has entries, and an estimator has to
fix the scale by convention rather than measure it; the convention used below is unit norm,
which is what an SVD returns. Second, $\tilde{x} \simeq P \tilde{X}$ is not usable as a
linear constraint, because the unknown scale factor sits in the middle of it. The standard way
to remove it is the cross product: two vectors are parallel exactly when their cross product
vanishes, so $\tilde{x} \times (P \tilde{X}) = 0$ says the same thing with no scale factor
left, and it is linear in whichever of $\tilde{X}$ or $P$ is unknown. Both the triangulation
and PnP linear solves use this one step.

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

The unit-norm constraint is necessary: without it $x = 0$ minimizes the cost and says nothing.

Take the SVD $A = U \Sigma V^\top$ with singular values $\sigma_1 \ge \dots \ge \sigma_n \ge 0$
and $V$ orthonormal. Since the columns of $V$ are an orthonormal basis, write
$x = \sum_j c_j v_j$ with $\sum_j c_j^2 = 1$. Then $\lVert A x \rVert^2 = \sum_j \sigma_j^2 c_j^2$,
a weighted average of the $\sigma_j^2$ with weights summing to one, and the way to minimize a
weighted average is to put all the weight on the smallest entry. So $c_n = 1$ and
$\hat{x} = v_n$, the last column of $V$. That is the entire justification for the instruction
"take the right singular vector with the smallest singular value", which appears three times
below. (If $A$ has fewer rows than columns, $\sigma_n = 0$ and $v_n$ is still the answer, but
only if the SVD returns all $n$ columns of $V$.)

Two properties of that solve matter later. The residual norm at the optimum is $\sigma_n$ (the squared cost is $\sigma_n^2$), so
$\sigma_n$ measures how nearly the constraints are consistent. And the sensitivity of
$\hat{x}$ to noise in $A$ is set by how well separated $\sigma_n$ is from $\sigma_{n-1}$:
perturbing $A$ by $\Delta$ rotates $v_n$ by roughly
$\lVert \Delta \rVert / (\sigma_{n-1} - \sigma_n)$. Measurement noise typically scales
$\lVert \Delta \rVert$ with the overall size of $A$, which is $\sigma_1$, and by construction
$\sigma_n \approx 0$, so the number that predicts accuracy is the ratio
$\sigma_1 / \sigma_{n-1}$. A large ratio means an ill-conditioned solve: the null direction is
barely distinguishable from its neighbors, and a little noise moves it a lot. The
normalization step in the eight-point algorithm exists to shrink that ratio.

### Algebraic error, geometric error, and Gauss-Newton

Every $A x = 0$ estimator minimizes $\lVert A x \rVert$, and the entries of $A x$ are the
residuals of whatever algebraic identity was stacked into the rows. They carry mixed units,
they weight correspondences by where in the image the points sit, and multiplying one row by a
constant would change the answer without changing what the row asserts. This is the algebraic
error, and minimizing it is a convenience that makes the problem linear, not a statement about
measurement noise.

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
repeat. This is the linearization the EKF uses for its measurement update, applied repeatedly
and with no prior term; the iterated EKF is the filter version of the same loop. $H$ is the
Gauss-Newton approximation to the Hessian of the cost: it drops the term containing second
derivatives of $h$, which is negligible when the residuals are small. It is symmetric positive
semidefinite, so the step is a Cholesky-type solve rather than a general matrix inverse. Near a
solution with zero residual the iteration converges quadratically, so on noise-free data it
reaches machine precision in a handful of steps.

Both refinements in this assignment are this loop with a different $\theta$: the three
coordinates of a point in triangulation, and a six-dimensional pose step in PnP.

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

The DLT does not minimize the error that matters, which is why a refinement follows. The
first row's residual is $x\,p_3 - p_1 = p_3\,(x - p_1/p_3)$, which is the reprojection offset
in one image coordinate multiplied by $p_3$, the projective depth of the point in that camera
(its depth along the optical axis when $P = K[\,R \mid t\,]$ with the usual $K$). So the DLT
minimizes reprojection error weighted by depth, which gives distant points more weight than
their measurements deserve. That is the gap between algebraic and geometric error in concrete
form.

The refinement is the Gauss-Newton loop above with $\theta = X \in \mathbb{R}^3$ and the
4-vector residual stacking both views' $(x - p_1/p_3,\ y - p_2/p_3)$. The Jacobian of the
prediction follows from the quotient rule,

$$\frac{\partial}{\partial X}\left(\frac{p_k}{p_3}\right)
= \frac{1}{p_3}\left(P^{(k)}_{1:3} - \frac{p_k}{p_3}\,P^{(3)}_{1:3}\right), \qquad k = 1, 2,$$

where the subscript $1{:}3$ takes the first three entries of the row (the fourth multiplies the
constant 1 of $\tilde{X}$ and contributes nothing to the derivative). On noise-free
correspondences the DLT already lands on the point to machine precision, so the refinement
only changes the answer once there is pixel noise and the depth weighting matters.

> **Exercise 1 (triangulating our sideways-step point).** In our sideways-step example, use
> normalized coordinates, so $P_1 = [\,I \mid 0\,]$ and $P_2 = [\,I \mid t\,]$ with
> $t = (1, 0, 0)$, and the image points are $x_1 = (0, 0)$ and $x_2 = (0.25, 0)$. Write the
> four DLT rows of $A$, find the null vector, and dehomogenize. Then redo it with
> $x_2 = (0.125, 0)$. What general formula for the depth do the two answers follow?

<details>
<summary>Answer</summary>

The rows of $P_1$ are $(1,0,0,0)$, $(0,1,0,0)$, $(0,0,1,0)$. With $x_1 = (0, 0)$ the two
view-1 rows $x\,P^{(3)} - P^{(1)}$ and $y\,P^{(3)} - P^{(2)}$ are $(-1, 0, 0, 0)$ and
$(0, -1, 0, 0)$. The rows of $P_2$ are $(1,0,0,1)$, $(0,1,0,0)$, $(0,0,1,0)$, so with
$x_2 = (0.25, 0)$ the view-2 rows are $0.25\,(0,0,1,0) - (1,0,0,1) = (-1, 0, 0.25, -1)$ and
$(0, -1, 0, 0)$:

$$A = \begin{bmatrix} -1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ -1 & 0 & 0.25 & -1 \\ 0 & -1 & 0 & 0 \end{bmatrix}.$$

The first two rows force $X = Y = 0$; the third then says $0.25\,Z - W = 0$. The null vector
is $\tilde{X} \simeq (0, 0, 4, 1)$, and dehomogenizing gives $X = (0, 0, 4)$, the true point.
The matrix has rank 3 (rows 2 and 4 are equal), so the null space is exactly one-dimensional.

With $x_2 = (0.125, 0)$ the third row becomes $0.125\,Z - W = 0$, so $Z = 8$. Halving the
disparity doubles the depth: $Z = b / d$ with baseline $b = 1$ and disparity $d$ in normalized
coordinates, which in pixels is the stereo formula $Z = f b / d_{\text{px}}$. On a rectified
pair the DLT reduces to depth from disparity.

</details>

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

So the essential matrix is the coplanarity condition written as a bilinear form. The condition
is linear in the entries of $E$, which is what the eight-point algorithm uses.

Fix $x_1$ and read the constraint as a condition on $x_2$. It says
$(E x_1)^\top x_2 = 0$, which by the line convention above says $x_2$ lies on the line
$\ell_2 = E x_1$. This is the epipolar line of $x_1$: the matching point in image 2 must lie
on it, so correspondence search drops from a 2D window to a 1D line, and a candidate match can
be scored by its distance from that line. Symmetrically $\ell_1 = E^\top x_2$ is a line in
image 1. In our sideways-step example the epipolar lines are the horizontal scanlines of the
figure, which is why stereo matchers search along the rows of a rectified pair.

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
direction (2): 5. Both are below 8, so the eight-point algorithm is not a minimal solver (one
that uses exactly as many correspondences as the model has degrees of freedom), and $E$
carries structure beyond rank 2 that the linear method throws away. Concretely, the two nonzero
singular values of a true essential matrix are equal, which follows from $E = [t]_\times R$:
the singular values of $[t]_\times$ are $(\lVert t \rVert, \lVert t \rVert, 0)$, and
right-multiplying by an orthogonal matrix does not change them.

> **Exercise 2 (the essential matrix of our sideways step).** For our sideways-step example
> ($R = I$, $t = (1, 0, 0)$), write out $E$, its singular values, and the epipoles $e_1$ and
> $e_2$ as homogeneous 3-vectors. Show that $x_2^\top E x_1 = 0$ says $y_1 = y_2$. Then form
> $F = K^{-\top} E K^{-1}$ with $f = 500$ and principal point $(320, 240)$, and compute the
> epipolar line in image 2 of the pixel $u_1 = (320, 240)$.

<details>
<summary>Answer</summary>

$$E = [t]_\times = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}.$$

$E^\top E = \operatorname{diag}(0, 1, 1)$, so the singular values are $(1, 1, 0)$: two equal
nonzero values of size $\lVert t \rVert = 1$, as the general argument says.

Both null spaces are spanned by $(1, 0, 0)$, so $e_1 = e_2 = (1, 0, 0)$. Its third coordinate
is 0, which in homogeneous coordinates is a point at infinity in the $x$ direction: each camera
center lies on the other camera's $x$ axis, so neither projects to a finite image point, and
the pencil of epipolar lines through a point at infinity is a family of parallel horizontal
lines.

With $x_1 = (x_1, y_1, 1)$, $E x_1 = (0, -1, y_1)$, and $x_2^\top E x_1 = -y_2 + y_1$. The
constraint is $y_1 = y_2$: a match must lie on the same row.

$K^{-1}$ maps $(u, v, 1)$ to $((u - 320)/500, (v - 240)/500, 1)$. Multiplying out,

$$F = K^{-\top} E K^{-1} = \frac{1}{500^2}\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & -500 \\ 0 & 500 & 0 \end{bmatrix},$$

and $F\,(320, 240, 1)^\top = (0, -0.002, 0.48) \simeq (0, 1, -240)$, the line $v = 240$. The
match $u_2 = (445, 240)$ lies on it.

</details>

### The eight-point algorithm

The constraint $x_2^\top F x_1 = 0$ is linear in the entries of $F$, so each correspondence
$((u, v), (u', v'))$ gives one row of a homogeneous system in the 9-vector
$f = \mathrm{vec}(F)$, stacking $F$ row by row:

$$[\,u'u,\ u'v,\ u',\ v'u,\ v'v,\ v',\ u,\ v,\ 1\,]\, f = 0.$$

With eight or more correspondences, $f$ is the smallest right singular vector of the stacked
$n \times 9$ matrix, by the argument in the SVD section - hence "eight-point". Two steps
separate the textbook formula from a usable estimator.

Rank-2 enforcement. The linear solve has no way to impose $\det F = 0$, so it returns a
full-rank matrix. Fix it afterwards: take the SVD $F = U \Sigma V^\top$, set the smallest
singular value to zero, and rebuild $U\,\mathrm{diag}(\sigma_1, \sigma_2, 0)\,V^\top$. Zeroing
the smallest singular value gives the closest rank-2 matrix in Frobenius norm, by the
Eckart-Young theorem, so this is a projection onto the set of rank-2 matrices. Skipping it
leaves $F$ invertible, so it has no null vector and therefore no epipole: the lines $F x_1$
for different points fan out across the image instead of meeting in a pencil.

Normalization and conditioning. Pixel coordinates on this scene's 640x480 image run to about
640, so the columns of the $n \times 9$ matrix have very different magnitudes - the $u'u$
column reaches roughly $4 \times 10^5$ while the last column is exactly 1. By the sensitivity
argument above, the accuracy of the null vector is governed by $\sigma_1 / \sigma_8$, and on
this scene's raw matrix (noise-free correspondences) that ratio comes out around $6 \times 10^4$.

The fix is to precondition the data, which the literature calls Hartley normalization (a
different operation from forming the normalized rays $K^{-1}u$, despite the shared word).
Replace each image's points by a similarity transform of them chosen to make all the numbers
order 1: translate so the centroid is at the origin, and scale so the mean distance from the
origin is $\sqrt{2}$, which puts a typical point near $(\pm 1, \pm 1)$ so that all three
homogeneous coordinates are comparable. Write the transform as the $3 \times 3$ matrix $T$
acting on homogeneous points, $\hat{x} = T x$. Solve on the normalized points for $\hat{F}$.
Then undo it: $\hat{x}_2^\top \hat{F} \hat{x}_1 = 0$ with $\hat{x} = Tx$ is
$x_2^\top (T_2^\top \hat{F} T_1) x_1 = 0$, so $F = T_2^\top \hat{F}\, T_1$. On the same noise-free
scene the $\sigma_1 / \sigma_8$ ratio after normalization is about 30, three orders of magnitude
better than the raw one.

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
$U \Sigma V^\top$ by $2 \sigma_3 u_3 v_3^\top$ (up to sign), and $\sigma_3 = 0$ for a rank-2 matrix, so it
changes nothing. With the fixed matrix

$$W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix},$$

the two factorizations of $E$ into a skew-symmetric matrix times a rotation (Hartley and
Zisserman derive that there are exactly two) give two rotation candidates and one translation
direction:

$$R_1 = U W V^\top, \qquad R_2 = U W^\top V^\top, \qquad t = U_{:,3}.$$

Both $R_1$ and $R_2$ are products of rotations, so both are proper rotations, and $t$ is a
column of an orthogonal matrix, so it already has unit length. $W$ is a rotation by $90^\circ$
about the third axis, and $R_1$ and $R_2$ differ by a half turn about the baseline direction
$t$; Hartley and Zisserman call the two resulting camera placements a twisted pair. In our
sideways-step example the two candidates are $I$ and the half turn about the $x$ axis, which
we call $R_{\text{flip}} = \operatorname{diag}(1, -1, -1)$.

The formulas use only $U$ and $V$ and discard $\Sigma$ entirely, which matters in practice. An
$E$ estimated by the eight-point algorithm has been forced to rank 2, but its two nonzero
singular values are not equal, so it is not exactly an essential matrix. Throwing $\Sigma$
away amounts to replacing it with $\mathrm{diag}(1, 1, 0)$, which is the projection onto the
set of essential matrices; the $(R_1, R_2, t)$ that come back define an exact essential
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

![The four pose candidates of our sideways-step example, top view. Each panel places camera 2 (blue; the wedge opens along its viewing direction) by one candidate and triangulates the point seen at x1 = (0, 0) and x2 = (0.25, 0). Only (I, +t) puts the point at positive depth in both cameras; the other three put it behind one camera or both](assets/cheirality.png)

The translation comes back only as a direction, never a length. Scale the entire scene and the
baseline by the same factor - every 3D point to $sX$, the baseline to $st$ - and every
projection is unchanged, because the pinhole divides by depth and the factor cancels top and
bottom. No image measurement can determine $s$. A direction in parameter space that the
measurements cannot see is called a gauge freedom, and monocular two-view geometry has exactly
one, this scale. Estimators deal with a gauge by fixing it arbitrarily; here it is fixed by
returning $\lVert t \rVert = 1$. So translations are compared by angle rather than by vector
difference, and a single moving camera cannot report metric distance without an outside
reference: a second camera at a known baseline, an object of known size, wheel odometry, or an
accelerometer. The EKF-SLAM notes met the same kind of freedom in the unobservable absolute
position of the map frame.

> **Exercise 3 (the four candidates by hand).** In our sideways-step example, first check that
> $[t]_\times R_{\text{flip}} = -E$, so $R_{\text{flip}}$ is a valid factorization of the same
> essential matrix. Then for each of the four candidates $(I, \pm t)$ and
> $(R_{\text{flip}}, \pm t)$, triangulate the correspondence $x_1 = (0, 0)$,
> $x_2 = (0.25, 0)$ with $P_1 = [\,I \mid 0\,]$ and $P_2 = [\,R \mid t\,]$, and report the
> point's depth in camera 1 and in camera 2. Which candidate passes the cheirality test?

<details>
<summary>Answer</summary>

$[t]_\times R_{\text{flip}}$ multiplies the columns of $[t]_\times$ by $(1, -1, -1)$:

$$\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}\operatorname{diag}(1, -1, -1) = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{bmatrix} = -E.$$

In every case the view-1 rows force $X = Y = 0$, so only the first view-2 row,
$0.25\,p_3 - p_1 = 0$, is left to fix $Z$ against $W = 1$.

- $(I, +t)$: $p = (X + 1, Y, Z)$, so $0.25 Z - 1 = 0$, $Z = 4$. Depth in camera 2 is
  $(R X + t)_3 = 4$. Both positive.
- $(I, -t)$: $p = (X - 1, Y, Z)$, so $0.25 Z + 1 = 0$, $Z = -4$. Depth in camera 2 is $-4$.
  Behind both cameras.
- $(R_{\text{flip}}, +t)$: $p = (X + 1, -Y, -Z)$, so $-0.25 Z - 1 = 0$, $Z = -4$. Depth in
  camera 2 is $-Z = 4$. Behind camera 1, in front of camera 2.
- $(R_{\text{flip}}, -t)$: $p = (X - 1, -Y, -Z)$, so $-0.25 Z + 1 = 0$, $Z = 4$. Depth in
  camera 2 is $-4$. In front of camera 1, behind camera 2.

Only $(I, +t)$ passes, and it is the true pose. All four reproduce the same two image points
exactly, which is why the epipolar constraint cannot choose between them and a depth sign has
to.

</details>

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
something that is no longer a rigid transform. The retraction from the Lie-groups notes
handles it: parameterize the step by a six-vector twist $\xi = [\rho; \theta]$ (translation
part first, rotation part second, the ordering used throughout this module), map it onto the
group with the exponential $\exp(\xi^\wedge)$ (written $\exp(\widehat{\xi})$ in the Lie-groups
notes), and apply it on the right,

$$T \leftarrow T\, \exp(\xi^\wedge),$$

which reads as "move by $\xi$ expressed in the current body frame", here the camera frame. This
is the right-perturbation convention that the Lie-groups notes fix for this module in their
section on left and right perturbations; the ICP assignment is the one place that composes on
the left. The optimization is then the Gauss-Newton loop from above with $\theta = \xi$,
re-parameterized around the current pose at every iteration.

The measurement Jacobian is a chain rule with two links. The first link is how the camera-frame
point moves with the twist. Substitute $\exp(\xi^\wedge) \approx I + \xi^\wedge$ into
$T \exp(\xi^\wedge) \tilde{X}_i$ and drop second-order terms:
$X_{\text{cam}} \approx (R X_i + t) + R\rho - R\,[X_i]_\times \theta$, where $X_i$ is the world
point and the last term uses $\theta \times X_i = -[X_i]_\times \theta$. So

$$\frac{\partial X_{\text{cam}}}{\partial \xi} = \big[\,R \ \big| \ -R\,[X_i]_\times\,\big] \in \mathbb{R}^{3 \times 6}.$$

The second link is the pinhole derivative $\partial \pi / \partial X_{\text{cam}}$, the
$2 \times 3$ Jacobian of the camera-geometry assignment's projection. Composing them,

$$A_i = \frac{\partial \pi}{\partial X_{\text{cam}}}\Big[\,R \ \big| \ -R\,[X_i]_\times\,\Big] \in \mathbb{R}^{2 \times 6}.$$

Accumulate $H = \sum_i A_i^\top A_i$ and $g = \sum_i A_i^\top e_i$ with the residual
$e_i = u_i - \pi(K(R X_i + t))$, solve $H\,\delta\xi = g$, retract, and iterate.

One Lie-group detail shows up in the next assignment. The right
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
produce confident matches between points that are not the same scene point. A wrong
correspondence is an outlier; a correct one, with only localization noise, is an inlier.

The breakdown point of an estimator is the largest fraction of arbitrarily corrupted
measurements it can absorb before its output can be pushed arbitrarily far from the truth. For
least squares that fraction is zero, and the reason is in the cost function: a residual enters
as $r^2$, so a match that is 300 pixels wrong contributes $10^4$ times what a 3-pixel error
does, and the fit will give up its agreement with every good measurement to reduce that one
term. A single gross outlier is enough in principle.

On this assignment's scene, with 35% of the 120 correspondences replaced by wrong matches, the
plain eight-point fit on all of them misses the rotation by more than the camera actually
rotated; the assignment section gives the numbers.

This is the data association problem of the EKF-SLAM notes in another form: decide which
measurements belong to the model before fitting the model. EKF-SLAM could gate each reading
against a predicted landmark with a Mahalanobis distance because it had a prior. Two-view
geometry starts with no model at all, so the model and the inlier set have to be found
together. That takes two things: a distance with units that says whether one correspondence
agrees with a candidate model, and a way to generate good candidates.

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
threshold has to be scaled accordingly (divided by the focal length: 1.5 px at $f = 500$ px is
0.003).

> **Exercise 4 (the Sampson distance of a slightly wrong match).** In our sideways-step example
> in pixels, with $F$ from Exercise 2, suppose the matcher reports $u_1 = (320, 240)$ and
> $u_2 = (445, 243)$, three pixels below the true match. Compute the algebraic residual
> $C = x_2^\top F x_1$, the Jacobian $J$, and the Sampson distance $d$. What are $C$ and $d$ if
> $F$ is multiplied by 1000? Is the match inside a 1.5 px inlier threshold?

<details>
<summary>Answer</summary>

From Exercise 2, $x_2^\top E x_1 = y_1 - y_2$ in normalized coordinates, and $y = (v - 240)/500$,
so in pixels $C = (v - v')/500 = (240 - 243)/500 = -0.006$.

From the formula, $F x_1 = (0, -0.002, 0.48)$ and $F^\top x_2 = (0, 0.002, -0.486)$, so

$$J = \big((F^\top x_2)_1,\ (F^\top x_2)_2,\ (F x_1)_1,\ (F x_1)_2\big) = (0,\ 0.002,\ 0,\ -0.002),$$

which is $(0, 1/500, 0, -1/500)$: $C$ depends only on $v$ and $v'$. Then
$\lVert J \rVert = \sqrt{2}/500$ and

$$d = \frac{0.006}{\sqrt{2}/500} = \frac{3}{\sqrt{2}} \approx 2.12 \text{ px}.$$

Multiplying $F$ by 1000 makes $C = -6$, and $\lVert J \rVert$ grows by the same factor, so
$d = 2.12$ px again. The algebraic residual changed by three orders of magnitude while the
geometry did not; the Sampson distance did not change either.

Here the constraint $v = v'$ is linear in the measurements, so the tangent-plane approximation is
exact: moving $v$ down by 1.5 px and $v'$ up by 1.5 px makes the pair consistent, a total
correction of $\sqrt{1.5^2 + 1.5^2} = 2.12$ px. That is above a 1.5 px threshold, so the match
is rejected. A 3 px vertical error is large compared with the 0.5 px localization noise of the
assignment's scene.

</details>

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

The method works because of an asymmetry between clean and contaminated samples. A sample that
happens to contain only inliers produces a model near the truth, and every other inlier then
agrees with it, so its consensus set is large. A sample containing even one outlier is
constrained by a correspondence with no geometric meaning, so the model it produces satisfies
few other points and its consensus set is small. Maximizing consensus therefore selects,
indirectly, for having drawn a clean sample.

The iteration budget comes from a probability calculation. With an inlier fraction $w$ and a
sample size $s$, one draw is all-inliers with probability $w^s$ (treating draws as independent,
which is close enough when the number of correspondences $n$ is much larger than $s$). Getting at least one clean sample with
probability $p$ takes

$$N = \frac{\log(1 - p)}{\log(1 - w^s)}$$

rounds. This assignment's scene is 35% outliers, so $w = 0.65$; at $s = 8$ and $p = 0.99$ the
formula asks for about 142 rounds. The exponent drives the cost: $w^s$ decays geometrically in
$s$, so a five-point solver at the same $w$ needs about 37 rounds for the same confidence. At
50% outliers the same comparison is roughly 1180 rounds against 145. That gap is the argument
for minimal solvers, and it widens as conditions get worse.

> **Exercise 5 (what a 500-round budget buys).** The assignment's front end runs a fixed 500
> rounds of eight-point RANSAC. At the scene's inlier fraction $w = 0.65$, compute the
> probability that one sample of 8 is clean, the expected number of clean samples in 500
> rounds, and the probability that all 500 rounds draw at least one outlier. Repeat the last
> number for $w = 0.5$.

<details>
<summary>Answer</summary>

$w^8 = 0.65^8 = 0.0319$, about one sample in 31. In 500 rounds the expected number of clean
samples is $500 \cdot 0.0319 = 15.9$.

All 500 rounds fail with probability
$(1 - 0.0319)^{500} = \exp(500 \ln 0.9681) = \exp(-16.19) \approx 9.3 \times 10^{-8}$.
The budget is more than three times the 142 rounds the formula asks for at $p = 0.99$, and the
failure probability falls exponentially in the number of rounds.

At $w = 0.5$, $w^8 = 1/256$ and $(1 - 1/256)^{500} \approx e^{-500/256} = e^{-1.95} \approx 0.14$.
The same budget now fails about one time in seven, because 500 is well below the roughly 1180
rounds needed for 99% confidence. A five-point solver at $w = 0.5$ has $w^5 = 1/32$ and would
fail with probability $(31/32)^{500} \approx 1.3 \times 10^{-7}$.

</details>

### The composed front end

The pieces assemble into the actual visual-odometry front end. RANSAC the fundamental matrix
from the pixel correspondences, convert it to the essential matrix with
$E = K_2^\top F K_1$ (the inverse of $F = K_2^{-\top} E K_1^{-1}$), recover $(R, t)$ by
cheirality on the inlier rays, and triangulate the inliers into 3D points expressed in frame 1.
The output is a relative pose up to the monocular scale plus a sparse point cloud: one camera's
worth of motion and structure. That is the unit a SLAM back end consumes. Front end here means
the per-frame-pair geometry that produces measurements; back end means the optimization that
takes many such measurements and solves for all the poses and points at once.

### Later developments

The eight-point algorithm is the teaching route, not the state of the art. The minimal
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

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the
declaration and comments in `multiview.hpp` give the signature and shapes.

| # | Function | File | Concept from the notes |
|---|---|---|---|
| 1 | [`triangulate_dlt()`](multiview.cpp) | `multiview.cpp` | triangulation: two DLT rows per view, smallest right singular vector, dehomogenize |
| 2 | [`triangulate_refine()`](multiview.cpp) | `multiview.cpp` | Gauss-Newton on the two-view reprojection residual over $X \in \mathbb{R}^3$ |
| 3 | [`eight_point()`](multiview.cpp) | `multiview.cpp` | Hartley normalization, the $n \times 9$ solve, rank-2 projection, denormalize |
| 4 | [`decompose_essential()`](multiview.cpp) | `multiview.cpp` | the SVD with proper $U, V$; $R_1 = UWV^\top$, $R_2 = UW^\top V^\top$, $t = U_{:,3}$ |
| 5 | [`recover_pose()`](multiview.cpp) | `multiview.cpp` | the cheirality test over the four candidates |
| 6 | [`pnp_dlt()`](multiview.cpp) | `multiview.cpp` | the PnP DLT, sign-preserving scale, Procrustes projection onto $SO(3)$ |
| 7 | [`pnp_refine()`](multiview.cpp) | `multiview.cpp` | Gauss-Newton on $SE(3)$ with $T \leftarrow T\exp(\xi^\wedge)$ |
| 8 | [`sampson_distance()`](multiview.cpp) | `multiview.cpp` | the Sampson distance, one value per correspondence |
| 9 | [`ransac_fundamental()`](multiview.cpp) | `multiview.cpp` | RANSAC with eight-point samples, Sampson scoring, refit on the consensus set |
| 10 | [`two_view_relative_pose()`](multiview.cpp) | `multiview.cpp` | the composed front end |

You may not include OpenCV, Ceres, GTSAM, g2o, Sophus, manif, or OpenGV; a test scans the
sources.

Notes on specific holes:
- The provided primitives in `models.cpp` are `hat3` (the skew matrix $[w]_\times$), `se3_exp`
  (the $SE(3)$ exponential, $\xi = [\rho; \theta]$), `pinhole_project`, and `pinhole_jacobian`
  (the $2 \times 3$ $\partial \pi / \partial X_{\text{cam}}$). Call these; do not reimplement
  them.
- For every null-vector solve, use `Eigen::JacobiSVD` with `Eigen::ComputeFullV` and take the
  last column of `matrixV()`. A RANSAC minimal sample gives an $8 \times 9$ matrix, and a thin
  $V$ would be $9 \times 8$ and miss the null vector.
- `eight_point`: `p1`, `p2` are $n \times 2$. Unpack $f$ into $F$ row by row, in the column
  order of the row equation in the notes.
- `recover_pose`: `x1`, `x2` are normalized image coordinates ($n \times 2$). Use
  `triangulate_dlt` with $P_1 = [\,I \mid 0\,]$, $P_2 = [\,R \mid t\,]$, and count a point only
  if its depth is positive in both cameras.
- `pnp_dlt`: `std::cbrt` is the sign-preserving real cube root the notes need.
- `pnp_refine`: the residual is measured minus predicted, `u_i - pinhole_project(K, Xc)`;
  solve with `H.ldlt().solve(g)`, update with `T = T * se3_exp(dxi)`, and stop after a fixed
  number of iterations or when the step is tiny (the reference uses 15 and $10^{-12}$).
- `sampson_distance`: return 0 when the denominator is 0 (the reference does).
- `ransac_fundamental`: seed `std::mt19937` with `seed`, draw 8 distinct indices per round,
  count a correspondence as an inlier if its distance is strictly below `threshold`, and
  replace the best set only when a new one is strictly larger. Refit with `eight_point` on the
  winning set (on all points if it has fewer than 8), and return the inlier indices sorted.
- `two_view_relative_pose`: $E = K^\top F K$ (here $K_1 = K_2 = K$); convert only the inlier
  pixels to normalized coordinates before `recover_pose` and triangulation.
- `config.py` holds the intrinsics, the scene (120 points at depths of 4 to 12 m,
  `REL_ROTVEC` and `REL_TRANS`), `NOISE_PX = 0.5`, `OUTLIER_FRAC = 0.35`,
  `RANSAC_THRESH_PX = 1.5`, and `RANSAC_ITERS = 500`, the budget of Exercise 5.

### Building and running

Same toolchain as the rest of the module: a C++17 compiler plus CMake, pybind11, and Eigen
(the `environment.yml` a14 block). You never call CMake by hand: importing `_impl.py` (which
the tests and `viz.py` do) configures and builds a pybind11 extension with CMake, from the
top-level `multiview.cpp` by default or from `solution/multiview.cpp` under
`NANOVISION_IMPL=solution`, compiled together with `models.cpp` and `bindings.cpp` in both
cases. The build is cached under `build/student/` or `build/solution/`, so only the first run
or a source change compiles.

Activate the environment first (`conda activate nanovision`); the Makefile calls the `python`
on the path. Then:

```
make verify A=a14_3_multiview   # build + run the reference (solution/); the green target
make test   A=a14_3_multiview   # build + run YOUR code; red until the holes are filled
make viz    A=a14_3_multiview        # render the two-view geometry (reference)
make viz-mine A=a14_3_multiview      # the same, from YOUR code, once the holes are filled
```

`make test` also accepts `K="sampson"` to run only matching tests and `SKIP_HOLES=1` to report
unfilled holes as skips instead of failures.

The tests favor implementation-independent statements:
- `test_triangulation.py`: the DLT reproduces every 3D point of a noise-free scene to within
  $10^{-6}$, and the refinement, started from the true point plus Gaussian noise of standard deviation 0.1 m per axis,
  converges back to it.
- `test_eight_point.py`: the recovered $F$ satisfies the epipolar constraint and is rank 2; the
  same solver on normalized rays gives an essential matrix satisfying the constraint on rays;
  with 0.5 px noise the median epipolar residual stays small.
- `test_recover_pose.py`: the decomposition yields proper rotations and a unit translation, and
  cheirality recovers the true pose (rotation and translation direction to better than
  $10^{-4}$ degrees, translation length not compared, since it is the free gauge) with all
  points in front of both cameras.
- `test_pnp.py`: the DLT recovers a known pose from noise-free data, the refinement drives the
  reprojection error below $10^{-8}$ pixels RMS, and it recovers the pose from a start whose
  translation is offset by $(0.3, -0.2, 0.4)$ m.
- `test_sampson.py`: the distance matches its closed form, and on the noisy scene with outliers
  the median over inliers is below the 1.5 px threshold while the median over outliers is
  more than five times the threshold.
- `test_ransac.py`: RANSAC recovers the inlier set with precision above 0.9 and recall above
  0.8; the front end recovers the rotation within 1 degree and the translation direction within
  10 degrees; and across five seeds the robust rotation error is under 1 degree and the naive
  least-squares rotation error is more than three times the robust one.
- `test_forbidden_imports.py`: no geometry or solver library is included. It passes with the
  holes still in place.

`make viz` writes `out/multiview.rrd`, a recording for the Rerun viewer. Add `SHOW=1` to open
it interactively instead: the triangulated points and the two recovered camera frusta (the
pyramid of space each camera sees) in 3D, the correspondences colored by RANSAC inlier versus
outlier in the two image views, and the epipolar lines of eight inliers drawn in the second
image, each passing through its matched point.

The figures in the notes are generated by `assets/make_figures.py` (run it from `assets/`),
which computes our sideways-step example in NumPy.

What you should see when you run this. On the noisy scene with outliers (120 correspondences,
35% of them wrong matches, 0.5 px noise on the rest), the front end recovers an inlier set with
essentially no false positives and a handful of true inliers missed: across the first eight
seeds, precision runs 0.99 to 1.00 and recall 0.87 to 0.99. The recovered rotation lands
within 0.06 to 0.23 degrees of the truth and the translation direction within 0.4 to 3.3
degrees, against the plain eight-point fit's 8 to 36 degrees of rotation error on the same
data, while the true relative rotation is only about 7 degrees: the naive error is larger than
the motion it is estimating. The tests assert the ordering rather than the numbers - robust
rotation error under 1 degree, and naive error more than three times robust - so the check
survives different seeds and different implementations. These are toy figures from a
synthetic scene: they show the mechanism working, not how a real front end behaves on real
features.

The scene is synthetic: 120 random points at depths of 4 to 12 m viewed by two 640x480 pinhole
cameras with $f = 500$ px, separated by a baseline of about 0.84 m and a relative rotation of
about 7 degrees, with the world frame chosen as camera 1 so the ground-truth relative pose is
exactly $T_{2 \leftarrow 1}$. The conditioning numbers quoted in the eight-point section come
from the noise-free version of this scene: $\sigma_1 / \sigma_8$ of the raw pixel matrix is
around $6 \times 10^4$ (between $5 \times 10^4$ and $8 \times 10^4$ over the first eight scene
seeds) and about 30 after Hartley normalization. With the 0.5 px noise and 35% outliers both
ratios are lower (raw about $5.5 \times 10^3$ to $10^4$, normalized about 3 to 4), because the
inconsistent rows raise $\sigma_8$. A real image pair would put a feature detector and descriptor
matcher in front of these estimators, but the geometry being estimated is identical, and
synthetic correspondences give the deterministic ground truth the tests need.

## In interviews

Multi-view geometry is a standard interview topic for perception, SLAM, and 3D-vision roles,
both as derivations on a whiteboard and as requests to assemble the two-view front end from
its parts.

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

## Where this goes next

The estimators here are the geometric layer, and modern systems mostly change the layers around
them rather than replacing them. Learned detectors and matchers produce the correspondences
that feed these solvers, and the pose is still recovered by the same essential-matrix
machinery. The pose-graph and bundle-adjustment assignment is the back end that takes many
two-view results and refines all the poses and points jointly, with the same Gauss-Newton loop
and right-perturbation retraction as the PnP refinement here. The geometry foundation-model
assignment goes the other way and predicts geometry directly from images, skipping the
correspondence-then-solve pipeline entirely.

## Further reading

- Hartley and Zisserman, *Multiple View Geometry in Computer Vision* (2004), chapters 9-12 -
  the standard reference for epipolar geometry, two-view reconstruction, and triangulation;
  chapter 7 for camera resection.
- Hartley, "In defense of the eight-point algorithm" (PAMI 1997) - normalization.
- Nister, "An efficient solution to the five-point relative pose problem" (PAMI 2004) - the
  modern minimal relative-pose solver.
- Lepetit, Moreno-Noguer, Fua, "EPnP: an accurate O(n) solution to the PnP problem" (IJCV
  2009), and the P3P literature - the production PnP solvers.
- Fischler and Bolles, "Random sample consensus" (1981) - RANSAC.
- Chum, Matas and Kittler, "Locally optimized RANSAC" (2003), and Barath, Matas and Noskova,
  "MAGSAC" (2019) - the refinements that production robust estimators actually use.
