# a14_5 - Pose-graph and bundle adjustment

The last assignment of the classical SLAM module, in C++17 with Eigen. It builds the
factor-graph back-end: the smoothing optimizer that replaced filtering as the standard SLAM
estimator. The pieces are the between-factor residual and its analytic Jacobians, the
Gauss-Newton loop that optimizes a pose graph, and the Schur complement that makes bundle
adjustment affordable at scale. The demo is loop closure: a trajectory that has drifted under
dead reckoning moves back onto the true loop once a single loop-closure edge is added.

It builds on the Lie-group material. Poses live on $SE(3)$, the residual is a matrix logarithm,
the Jacobians use the right perturbation and the inverse right Jacobian, and the updates retract
on the manifold rather than adding vectors. That $SE(3)$ machinery (the exponential and
logarithm, the adjoint, and the $6 \times 6$ inverse right Jacobian) is provided in `models.cpp`,
so the work here is the factors, the normal equations, and the marginalization built on top of
it. The notes restate the three facts about it that the edge Jacobians depend on, so the
derivation can be followed without going back to the
[Lie-groups assignment](../a14_0_lie_se3/README.md). The Python around the C++ (the build, the
simulator, the tests, and the Rerun visualization) is provided, as in the earlier assignments of
the module.

The bundle-adjustment hole is the linear-algebra step alone: it takes an already assembled
system with bundle-adjustment structure and eliminates its landmark block. Reprojection factors
are described in the notes but are not implemented here; the
[multi-view assignment](../a14_3_multiview/README.md) already covered the pinhole projection they
would use.

Required reading before starting:
- Grisetti, Kümmerle, Stachniss, Burgard, "A tutorial on graph-based SLAM" (2010) - the
  pose-graph formulation, the error function, and the Gauss-Newton solution.
- Dellaert and Kaess, "Factor graphs for robot perception" (2017) - the factor-graph view,
  smoothing, and the sparsity that makes it scale.
- Triggs et al., "Bundle adjustment - a modern synthesis" (2000) - bundle adjustment and the
  Schur complement.

## Lecture notes

### What pose-graph optimization is for

#### The task

The demo in this assignment is a robot driving once around a loop. The true trajectory is 24 poses
evenly spaced on a circle of radius 5 m, each facing along the circle. Between consecutive poses
the robot measures its own motion, the relative pose from one to the next, as wheel odometry or
the ICP of the [registration assignment](../a14_4_icp/README.md) would give it. Each of these 23
odometry measurements is the true relative pose followed by a small random rigid motion, with
standard deviation 0.02 rad on each rotation axis and 0.02 m on each translation axis. At the end
of the loop the robot recognizes the place where it started and measures the relative pose from
its last pose back to its first, with standard deviation 0.05 rad on each rotation axis and 0.05 m
on each translation axis. That one measurement is the loop closure.

The job is to estimate all 24 poses from these 24 relative measurements. No measurement says
where the trajectory is in the world, so the first pose defines the world frame.

#### Why the obvious approaches fail

Dead reckoning chains the odometry: start at the first pose and multiply the measured relative
poses one after another. Every chained measurement adds its error to all the poses after it, and
a small heading error turns into a sideways position error that grows with the distance driven
afterward. In the figure below, the dead-reckoned trajectory (red) ends about a meter from the
true last pose. The loop-closure measurement disagrees with the dead-reckoned trajectory by that
gap, but dead reckoning has no place to use it.

The EKF-SLAM filter of the [EKF-SLAM assignment](../a14_2_ekf_slam/README.md) can use it: a loop
closure there corrects the old part of the map through the cross-covariances. It pays for this in
two ways that EKF-SLAM's notes derived. Its covariance is dense, so each update costs time
quadratic in the size of the state. And it marginalizes out every past pose as soon as the robot
moves on (marginalizing a variable integrates it out of the joint Gaussian), so the
linearization at which each past pose was folded in can never be revisited, even when a loop
closure shows that the estimate at that time was wrong. Marginalizing also creates fill-in, new
direct couplings between every pair of variables the removed pose was connected to; the
EKF-SLAM notes worked one case, and the marginalization section below derives it.

#### The idea

Smoothing keeps every pose as an unknown. Each measurement becomes one term of a cost that
compares what the measurement says with what the current pose estimates predict, weighted by the
measurement's information (inverse covariance). The estimate is the set of poses that minimizes
the total cost, found by Gauss-Newton, which relinearizes every term at every iteration. The
loop-closure term is the only term that disagrees with the dead-reckoned trajectory, and
minimizing the total cost spreads that disagreement over all the odometry edges of the loop, in
proportion to how uncertain each one is.

The left panel of the figure shows the result. The optimized trajectory (blue) closes the loop.
The middle panel shows where the error went: dead reckoning's position error grows steadily
along the trajectory, while the optimized error stays bounded and returns near zero at both
ends of the loop. Mid-loop poses keep errors of up to about 0.33 m, larger than the 0.02 m noise
of a single odometry step, because the loop closure removes the accumulated misclosure, not the
individual odometry errors. The
right panel shows why this is cheap. Each measurement involves two poses, so the matrix the
solver works with has nonzero $6 \times 6$ blocks only on the diagonal, next to it (one per
odometry edge), and in the two corners (the loop closure).

![The 24-pose drift graph. Left, the ground-truth loop in gray, the dead-reckoned trajectory in red with its loop-closure edge in green, and the optimized trajectory in blue. Middle, the position error of each pose before and after optimization. Right, the nonzero 6x6 blocks of the system matrix H, a band from the odometry edges plus two corner blocks from the loop closure](assets/drift_graph.png)

#### Why it matters

Keeping every pose and solving the sparse least-squares problem removes both of EKF-SLAM's
problems: the solver works with a sparse matrix instead of rewriting a dense covariance whose size
grows with the square of the map, and each measurement is relinearized at the current estimate
rather than frozen at the estimate of the moment it arrived. This is why factor-graph smoothing
replaced filtering as the SLAM back-end, and why the production solvers g2o, GTSAM, and Ceres are
all built around it. Bundle adjustment, the joint refinement of camera poses and 3D points in
structure from motion and visual SLAM, is the same optimization with points added as variables.

### Our corridor loop (four poses on a line, one loop closure)

The 24-pose loop lives in $SE(3)$, with $144$ unknowns. To follow every number by hand we reduce
it to one dimension, the same corridor world as the corridor example of the EKF-SLAM notes, and
call it our corridor loop.

A robot moves along a line. Its pose $x_k$ is a single position. It starts at $x_0 = 0$, drives
$+1$ m, $+1$ m, and then $-2$ m, so its true positions are $0, 1, 2, 0$: it ends where it
started. Odometry reports the three moves as $+1.1$, $+1.1$, and $-2.0$, with standard deviation
$0.1$ m each, so each odometry edge has information $\Omega = 1/0.1^2 = 100\ \text{m}^{-2}$. At
the end the robot recognizes its starting point and measures $x_0 - x_3 = 0$ with information
$300\ \text{m}^{-2}$, three times that of one odometry edge. That is the loop closure.

Dead reckoning chains the odometry and gives $0, 1.1, 2.2, 0.2$. Every odometry measurement is
satisfied exactly by this estimate, since it was built from them, and the loop closure is off by
$0.2$ m. The figure shows the dead-reckoned estimate and the solution of the pose-graph problem,
$0, 1.04, 2.08, 0.02$, whose derivation is Exercise 4. The $0.2$ m misclosure has been spread over
all four edges. On the right is the matrix $H$ that the solver builds for this problem (derived
below, where it first appears as the information matrix $\Lambda$): a tridiagonal band from the
odometry chain, plus two corner entries from the loop closure.

![Our corridor loop. Left, the true positions in gray, the dead-reckoned estimate in red, and the estimate after one Gauss-Newton step in blue, with the loop-closure edge from x3 back to x0 in green. Right, the 4x4 system matrix H of the problem, with the loop-closure entries in green](assets/corridor.png)

Our corridor loop stands for the drift graph of the demo. Pose $x_0$ is the first pose, which
defines the world frame; the three odometry edges stand for the 23; the edge from $x_3$ back to
$x_0$ is the loop closure from the last pose to the first. It keeps the structure of the problem:
odometry that dead reckoning satisfies exactly, a single edge that disagrees, the sparsity
pattern, and the frame that no measurement fixes. It drops rotation, which makes the real
problem nonlinear. On a line, relative positions are differences, every residual is linear in
the unknowns, and Gauss-Newton converges in one step. On $SE(3)$ the residual is a logarithm of
a product of poses, and its Jacobians need the Lie-group tools of a later section.

### Overview of the rest of the notes

1. Filtering versus smoothing. What a filter discards and what a smoother keeps.
2. Factor graphs. The posterior as a product of small factors, and why the maximum a posteriori
   (MAP) estimate under Gaussian noise is a weighted least-squares problem. The weights are the
   information matrices of the Kalman notes' information form.
3. Marginalization and fill-in. Integrating variables out of a Gaussian is a Schur complement.
   This one formula explains EKF-SLAM's dense covariance and the bundle-adjustment solver.
4. The $SE(3)$ residual. The between-factor and its two Jacobians, derived from three facts about
   the exponential map, the adjoint, and the inverse right Jacobian.
5. Gauss-Newton. The normal equations, assembled edge by edge, with the same sparsity as the
   graph. Gauss-Newton is the iterated linearization behind the EKF, run to convergence over all
   poses at once.
6. Gauge freedom, damping, and loop closure. Why the system is singular until one pose is held
   fixed, when plain Gauss-Newton is enough, and why one edge corrects the whole loop.
7. Bundle adjustment and the Schur complement. Adding 3D points as variables, and eliminating
   them inside each iteration with the marginalization formula.
8. Incremental smoothing and the production solvers.

Our corridor loop makes each formula concrete. Five exercises, placed after the sections they
test, are meant to be worked by hand. Each has a worked answer that can be expanded.

### Filtering versus smoothing

The EKF-SLAM assignment kept a single Gaussian over the current robot pose and the map, and
integrated out every past pose as the robot moved. That is filtering, and it has the two costs
named above: a dense covariance that makes each update quadratic in the map size, and
linearization errors that stay in the estimate permanently, because a pose that has been
integrated out can never be re-linearized when later evidence arrives.

Smoothing keeps all the poses (and optionally the landmarks) as free variables and solves for all
of them at once, every time. The estimate is the configuration of all variables that best explains
all measurements, and it is recomputed from scratch (or updated incrementally) as measurements
arrive. Two properties make this practical despite keeping everything. The problem is sparse,
because each measurement constrains only two or three variables. And every variable can be
re-linearized at every iteration, so linearization error does not accumulate the way it does in a
filter.

The rest of these notes builds that claim up: what the objective is and where it comes from, what
integrating out a variable costs and why it is the same operation in both the filter and the
bundle-adjustment solver, and what the $SE(3)$ residual and its derivatives look like.

### Factor graphs and why the objective is a sum of squares

A factor graph is a bipartite graph with two kinds of node. Variable nodes are the unknowns:
here, one per robot pose, and in bundle adjustment also one per 3D point. Factor nodes are the
terms of a product, each connected by edges to the small subset of variables it depends on.
Drawing an estimation problem this way asserts one thing: the posterior over all variables $x$
given all measurements $z$ factorizes,

$$p(x \mid z) \;\propto\; \prod_k \phi_k(x_k),$$

where $x_k$ is the handful of variables that factor $k$ touches. A ten-thousand-pose problem is
a product of ten thousand small terms, not one ten-thousand-dimensional density that has to be
handled as a whole. Our corridor loop has four variable nodes and four factors, one per
measurement, each touching two poses.

Each measurement contributes one factor. Model measurement $k$ as a prediction $h_k(x_k)$
corrupted by zero-mean Gaussian noise of covariance $\Sigma_k$. Its factor is then

$$\phi_k(x_k) \;\propto\; \exp\!\Big(-\tfrac12\, r_k^\top\, \Omega_k\, r_k\Big), \qquad
\Omega_k = \Sigma_k^{-1},$$

where $r_k$ is the residual, the difference between what the current variable estimates predict
and what was actually measured. In our corridor loop the edge from pose $i$ to pose $j$ with
measurement $z_{ij}$ has residual $r_{ij} = (x_j - x_i) - z_{ij}$.

$\Omega_k$ is the information matrix, the inverse covariance, the same quantity the Kalman notes
used for the [information form](../a14_1_kalman/README.md). It is set by the sensor, not tuned:
a measurement that is precise along some direction has a large $\Omega$ entry there and pulls
the estimate hard along it, while a direction the sensor barely constrains contributes almost
nothing. The scalar $r^\top \Omega r$ is the squared Mahalanobis distance, the squared Euclidean
length of the residual after each direction has been rescaled by its own standard deviation, so
residuals in meters and residuals in radians become comparable numbers that can be added.

The maximum-a-posteriori estimate maximizes the product, and a product of exponentials is easiest
to maximize by minimizing the negative logarithm of it:

$$x^\star \;=\; \arg\min_x\; \sum_k r_k(x)^\top\, \Omega_k\, r_k(x).$$

Nonlinear least squares is the MAP estimate under Gaussian noise, and the weights are inverse
covariances because the Gaussian exponent contains inverse covariances. For our corridor loop the
cost is

$$100\,(x_1 - x_0 - 1.1)^2 + 100\,(x_2 - x_1 - 1.1)^2 + 100\,(x_3 - x_2 + 2.0)^2 + 300\,(x_0 - x_3)^2 .$$

At the dead-reckoned estimate $0, 1.1, 2.2, 0.2$ the first three terms are zero and the last is
$300 \cdot 0.2^2 = 12$.

The information form also gives the information matrix of the whole problem. In the Kalman notes a
linear measurement $z = Hx + v$ with noise covariance $R$ adds $H^\top R^{-1} H$ to the
information matrix. (That $H$ was a measurement Jacobian. From the Gauss-Newton section on, $H$
names the system matrix and Jacobians are written $J$.) A corridor edge from $i$ to $j$ measures
$x_j - x_i$, so it adds $\Omega$ to the diagonal entries $(i,i)$ and $(j,j)$ and $-\Omega$ to the
two entries $(i,j)$ and $(j,i)$, and nothing anywhere else. Adding the four edges gives the
information matrix of our corridor loop, which we write $\Lambda$. Because the corridor residuals
are linear, it is also exactly the Gauss-Newton system matrix $H$ defined below, which is why the
corridor figure labels it $H$:

$$\Lambda = \begin{bmatrix} 400 & -100 & 0 & -300 \\ -100 & 200 & -100 & 0 \\ 0 & -100 & 200 & -100 \\ -300 & 0 & -100 & 400 \end{bmatrix}.$$

Sparsity is a statement about which variables share a factor. In our corridor loop the entries
$(0,2)$ and $(1,3)$ are zero because no measurement involves both of those poses. A pose-graph
edge touches exactly two poses, so it can only ever contribute to four blocks of the system matrix
assembled later. The synthetic graph in this assignment has 24 poses and 24 edges (23 consecutive
odometry edges plus one loop closure), so its system matrix is $144 \times 144$ with 72 of its 576
six-by-six blocks nonzero: 24 on the diagonal and 2 per edge off it. Real SLAM graphs are sparser
still, because the number of edges stays proportional to the number of poses while the matrix
grows with its square.

### Marginalization and the fill-in it creates

Integrating a variable out of a joint Gaussian is the single operation behind the filter's dense
covariance, behind the elimination ordering in a sparse solver, and behind the bundle-adjustment
solver at the end of these notes.

Split the variables into two groups $a$ and $b$, and write the joint Gaussian in information form
with

$$\Omega = \begin{bmatrix} \Omega_{aa} & \Omega_{ab} \\ \Omega_{ba} & \Omega_{bb} \end{bmatrix}.$$

Marginalizing means integrating $b$ out to leave a distribution over $a$ alone. Completing the
square in the exponent, or equivalently taking one block step of Gaussian elimination, gives the
marginal's information matrix:

$$\Omega'_{aa} \;=\; \Omega_{aa} - \Omega_{ab}\,\Omega_{bb}^{-1}\,\Omega_{ba}.$$

The elimination step is the identity

$$\begin{bmatrix} I & -\Omega_{ab}\Omega_{bb}^{-1} \\ 0 & I \end{bmatrix}
\begin{bmatrix} \Omega_{aa} & \Omega_{ab} \\ \Omega_{ba} & \Omega_{bb} \end{bmatrix}
= \begin{bmatrix} \Omega_{aa} - \Omega_{ab}\Omega_{bb}^{-1}\Omega_{ba} & 0 \\ \Omega_{ba} & \Omega_{bb} \end{bmatrix},$$

which clears the $b$ coupling out of the first block row. The expression
$\Omega_{aa} - \Omega_{ab}\Omega_{bb}^{-1}\Omega_{ba}$ is called the Schur complement of
$\Omega_{bb}$ in $\Omega$. It is the same object whether the matrix is a Gaussian's information
matrix or the coefficient matrix of a linear system, and both readings are used below.

Two consequences recur throughout.

The correction term $\Omega_{ab}\Omega_{bb}^{-1}\Omega_{ba}$ is generally dense across every
variable that was coupled to $b$. In graph terms, eliminating a variable adds an edge between
every pair of its former neighbors, so eliminating a pose makes every landmark that pose saw
directly correlated with every other one. Those new edges are called fill-in. A filter drops the
previous pose at every timestep, so it performs this elimination once per timestep, and the
information matrix over the map fills in completely. That is where EKF-SLAM's dense covariance
comes from (Exercise 5 of the EKF-SLAM notes works one case), and why its update is quadratic in
the map size.

Marginalization itself is exact, not an approximation: the reduced system over $a$ has the same
solution for $a$ as the full system did. What is lost is the ability to revisit $b$. When the
problem is nonlinear, the correction term was computed at one linearization point, and once $b$
is gone that point is frozen. A smoother avoids the loss by never marginalizing across
iterations. The bundle-adjustment Schur complement at the end of these notes applies the same
formula, but to the already-linearized system inside a single Gauss-Newton iteration, so nothing
is frozen from one iteration to the next.

> **Exercise 1 (fill-in from eliminating one pose).** Marginalize $x_1$ out of the information
> matrix $\Lambda$ of our corridor loop, keeping $(x_0, x_2, x_3)$ in that order. Which zero entry
> becomes nonzero? Read the new $3 \times 3$ matrix as the information matrix of a pose graph
> on $x_0, x_2, x_3$: what edges does that graph have, and with what information?

<details>
<summary>Answer</summary>

Here $\Omega_{bb} = 200$ and $\Omega_{ab} = (-100, -100, 0)^\top$ for $a = (x_0, x_2, x_3)$. The
correction is

$$\frac{1}{200}\begin{bmatrix} -100 \\ -100 \\ 0 \end{bmatrix}\begin{bmatrix} -100 & -100 & 0 \end{bmatrix} = \begin{bmatrix} 50 & 50 & 0 \\ 50 & 50 & 0 \\ 0 & 0 & 0 \end{bmatrix},$$

and subtracting it from the $(x_0, x_2, x_3)$ block of $\Lambda$ gives

$$\Lambda' = \begin{bmatrix} 350 & -50 & -300 \\ -50 & 150 & -100 \\ -300 & -100 & 400 \end{bmatrix}.$$

The $(x_0, x_2)$ entry was $0$ and is now $-50$. That is fill-in: $x_0$ and $x_2$ were both
neighbors of $x_1$, and eliminating $x_1$ connected them.

Read as a pose graph, $\Lambda'$ has three edges: the loop closure $x_3$-$x_0$ with information
$300$ and the odometry edge $x_2$-$x_3$ with information $100$, both unchanged, and a new
$x_0$-$x_2$ edge with information $50$ that replaced the two odometry edges through $x_1$. The
diagonal entries check this: $350 = 300 + 50$ and $150 = 50 + 100$. The value $50$ is
$1/(0.01 + 0.01)$, the information of two measurements of variance $0.01$ added in series, so
eliminating a pose from a chain composes its two edges into one, as chaining two odometry
readings would. Removing one interior pose of a chain creates only one new edge. Removing a pose
that observed $m$ landmarks creates edges between all $m(m-1)/2$ pairs of them, which is the
filter's case.

</details>

### The SE(3) tools the residual needs

A pose is a 4x4 matrix $T = \begin{bmatrix} R & t \\ 0 & 1\end{bmatrix}$ with $R$ a rotation, so
it carries 12 numbers describing 6 degrees of freedom under the constraint $R^\top R = I$. An
optimizer cannot add a 12-vector correction to $T$ and stay on the manifold, so corrections live
in the 6-dimensional tangent space instead and are mapped onto the group. The Lie-groups
assignment derived everything in this section; what follows restates the parts the residual
uses, in that assignment's conventions.

The tangent vector is a twist $\xi = \begin{bmatrix}\rho \\ \theta\end{bmatrix} \in \mathbb{R}^6$,
translation part first and rotation part second. This module fixes that ordering, and the adjoint
below has its off-diagonal block in the upper right because of it. The exponential
$\mathrm{Exp}(\xi)$ maps a twist to a pose and the logarithm $\mathrm{Log}(T)$ inverts it.

Three facts about them are what the edge Jacobians are built from.

Retraction. The correction step is a right multiplication, $T \leftarrow T\,\mathrm{Exp}(\delta)$,
which the Lie-groups assignment wrote as box-plus. Reading it right to left: $\delta$ is a small
motion expressed in the body frame of $T$, applied before $T$ takes the result to the world frame.
The whole assignment uses this right convention, so every Jacobian below is a derivative with
respect to a body-frame perturbation.

The adjoint. $\mathrm{Ad}_T$ is the 6x6 matrix that converts a twist from the body frame to the
world frame, defined by

$$T\,\mathrm{Exp}(\xi) \;=\; \mathrm{Exp}(\mathrm{Ad}_T\,\xi)\,T .$$

For $T$ with rotation $R$ and translation $t$ it is
$\mathrm{Ad}_T = \begin{bmatrix} R & \widehat{t}\,R \\ 0 & R \end{bmatrix}$, where
$\widehat{t}$ is the skew-symmetric matrix with $\widehat{t}\,v = t \times v$. The same motion can
be applied on the right of $T$ as $\xi$ or on the left as $\mathrm{Ad}_T\,\xi$; the adjoint
translates between the two. Rearranging that identity gives the form used repeatedly below,
moving a left factor to the right past a pose $A$:

$$\mathrm{Exp}(\xi)\,A \;=\; A\,\mathrm{Exp}\big(\mathrm{Ad}_{A^{-1}}\,\xi\big).$$

The inverse right Jacobian. For an ordinary function one writes
$f(x + \delta) \approx f(x) + J\delta$. Here the input perturbation is a right multiplication on
the group and the output is a vector, so the statement needed is how the logarithm of a product
changes:

$$\mathrm{Log}\big(\mathrm{Exp}(\xi)\,\mathrm{Exp}(\delta)\big) \;=\;
\xi + \mathcal{J}_r^{-1}(\xi)\,\delta + O(\lVert\delta\rVert^2).$$

$\mathcal{J}_r^{-1}(\xi)$ is the 6x6 matrix that converts a right-multiplied group perturbation
into the change of the logarithm. It equals $I$ at $\xi = 0$ and departs from $I$ as $\xi$ grows,
which is the correction for the fact that $\mathrm{Log}$ is not linear and the group is not a
vector space. It is computed from Barfoot's closed form, via
$\mathcal{J}_r(\xi) = \mathcal{J}_l(-\xi)$ and an explicit 6x6 inverse.

### The pose graph and the between-factor

A pose graph is a factor graph whose only variables are robot poses $T_i \in SE(3)$ and whose
only factors are relative-pose measurements, one per edge. An edge between $i$ and $j$ carries a
measured relative pose $T_{\text{meas}} = T_{i \leftarrow j}$, meaning frame $j$ expressed in
frame $i$ under the module's source-to-target naming, and an information matrix $\Omega$. Such a
factor is called a between-factor. The current estimates predict the relative pose
$T_i^{-1} T_j$, and the residual is the difference between measurement and prediction taken on
the manifold, using box-minus:

$$r_{ij} = \mathrm{Log}\big(T_{\text{meas}}^{-1}\, (T_i^{-1} T_j)\big) \in \mathbb{R}^6.$$

It is zero exactly when the estimated relative pose equals the measurement, because then the
argument of the logarithm is the identity. Summed over edges with the weights from the previous
sections, the total cost is $\sum_{ij} r_{ij}^\top \Omega_{ij}\, r_{ij}$. On a line, poses are
positions, composition is addition, $T_i^{-1}T_j$ is $x_j - x_i$, and $\mathrm{Log}$ is the
identity, so this residual becomes the corridor residual $(x_j - x_i) - z_{ij}$.

The Jacobians of $r_{ij}$ with respect to the right perturbations $\delta_i, \delta_j$ follow from
the three $SE(3)$ facts above and nothing else. In the corridor they are $-1$ and $+1$; on
$SE(3)$ the $+1$ becomes an inverse right Jacobian and the $-1$ picks up an adjoint as well.

Start with $j$, which is the easier side. Perturbing gives $T_j \leftarrow
T_j\,\mathrm{Exp}(\delta_j)$, so the argument of the logarithm becomes

$$T_{\text{meas}}^{-1}\, T_i^{-1}\, T_j\, \mathrm{Exp}(\delta_j) \;=\; \mathrm{Exp}(r_{ij})\,\mathrm{Exp}(\delta_j),$$

where the second form uses the definition of $r_{ij}$ to name the unperturbed product. The
inverse-right-Jacobian identity reads off the result directly:

$$\frac{\partial r_{ij}}{\partial \delta_j} = \mathcal{J}_r^{-1}(r_{ij}).$$

Now $i$. Perturbing gives $T_i \leftarrow T_i\,\mathrm{Exp}(\delta_i)$, so
$T_i^{-1} \leftarrow \mathrm{Exp}(-\delta_i)\, T_i^{-1}$ and the argument becomes

$$T_{\text{meas}}^{-1}\, \mathrm{Exp}(-\delta_i)\, T_i^{-1} T_j .$$

The perturbation is now in the middle of the product, to the left of $T_i^{-1}T_j$, where the
inverse-right-Jacobian identity cannot be applied. Write $A = T_i^{-1} T_j$ and move the
perturbation past $A$ with the adjoint rearrangement,
$\mathrm{Exp}(-\delta_i)\,A = A\,\mathrm{Exp}(-\mathrm{Ad}_{A^{-1}}\delta_i)$:

$$T_{\text{meas}}^{-1}\, A\, \mathrm{Exp}\big(-\mathrm{Ad}_{A^{-1}}\,\delta_i\big) \;=\;
\mathrm{Exp}(r_{ij})\,\mathrm{Exp}\big(-\mathrm{Ad}_{A^{-1}}\,\delta_i\big).$$

That is the same shape as the $j$ case with $-\mathrm{Ad}_{A^{-1}}\delta_i$ in place of
$\delta_j$, and $A^{-1} = (T_i^{-1}T_j)^{-1} = T_j^{-1}T_i$, so

$$\frac{\partial r_{ij}}{\partial \delta_i} = -\,\mathcal{J}_r^{-1}(r_{ij})\, \mathrm{Ad}_{T_j^{-1} T_i}.$$

These are the standard published forms, matching Barfoot's derivation and the `BetweenFactor` in
GTSAM, the Georgia Tech factor-graph library. The minus sign on the $i$-side and the argument
$T_j^{-1}T_i$ of the adjoint are the two details a hand-rolled implementation usually gets wrong,
and comparing the analytic Jacobians against numerical differentiation catches both. Near
convergence $r_{ij} \to 0$ and $\mathcal{J}_r^{-1} \to I$, so a wrong or missing
$\mathcal{J}_r^{-1}$ still converges on easy problems and goes unnoticed; a check has to evaluate
the Jacobians at a nonzero residual to require the full form.

> **Exercise 2 (the lever arm in the i-side Jacobian).** Let $T_i = I$, let $T_j$ be a pure
> translation by $(2, 0, 0)$ m, and let $T_{\text{meas}} = T_j$, so the residual is zero and
> $\mathcal{J}_r^{-1}(r_{ij}) = I$. Pose $i$ is rotated by a small yaw $\varepsilon$ about its own
> origin, $\delta_i = (0, 0, 0, 0, 0, \varepsilon)$. (a) Compute
> $\frac{\partial r_{ij}}{\partial \delta_i}\,\delta_i$ from the formula. (b) Check it by
> composing the poses directly to first order in $\varepsilon$. (c) What does the answer become
> if the adjoint's argument is wrongly taken as $T_i^{-1}T_j$?

<details>
<summary>Answer</summary>

(a) $T_j^{-1}T_i$ is a translation by $t = (-2, 0, 0)$ with $R = I$, so
$\mathrm{Ad}_{T_j^{-1}T_i}\,\delta_i = (\widehat{t}\,\theta;\ \theta)$ with
$\theta = (0, 0, \varepsilon)$. The cross product is
$t \times \theta = (-2, 0, 0) \times (0, 0, \varepsilon) = (0, 2\varepsilon, 0)$. With the minus
sign,

$$\frac{\partial r_{ij}}{\partial \delta_i}\,\delta_i = -(0, 2\varepsilon, 0, 0, 0, \varepsilon) = (0, -2\varepsilon, 0, 0, 0, -\varepsilon).$$

(b) After the perturbation, $T_i = R_z(\varepsilon)$ and
$T_i^{-1}T_j = R_z(-\varepsilon)\,\mathrm{trans}(2, 0, 0)$, a rotation by $-\varepsilon$ with
translation $(2\cos\varepsilon, -2\sin\varepsilon, 0) \approx (2, -2\varepsilon, 0)$. Multiplying
by $T_{\text{meas}}^{-1} = \mathrm{trans}(-2, 0, 0)$ leaves a rotation by $-\varepsilon$ and a
translation $(0, -2\varepsilon, 0)$, whose logarithm to first order is
$(0, -2\varepsilon, 0, 0, 0, -\varepsilon)$. This matches (a). Turning pose $i$ left by
$\varepsilon$ makes pose $j$, 2 m ahead, appear $2\varepsilon$ to its right: the lever arm term
$\widehat{t}R$ in the adjoint is that 2 m distance.

(c) $T_i^{-1}T_j$ is a translation by $(+2, 0, 0)$, so the cross product changes sign and the
result is $(0, +2\varepsilon, 0, 0, 0, -\varepsilon)$. The rotation part is still right but the
sideways part has the wrong sign. When the two poses coincide, $t = 0$ and the two versions
agree, which is why a test with identity or coincident poses does not catch this mistake. The
reference implementation's numbers for this case are $(0, -2, 0, 0, 0, -1)$ per unit
$\varepsilon$, matching a finite-difference check.

</details>

### Gauss-Newton and the normal equations

Gauss-Newton minimizes $F(x) = \sum_k r_k(x)^\top \Omega_k\, r_k(x)$ by repeatedly solving a
linear approximation of it. It is the same linearize-then-solve step as the EKF update, but
repeated at the improved estimate until it converges, and applied to all poses at once. Around
the current estimate, replace each residual by its first-order expansion in the retracted
increment, $r_k(x \boxplus \delta) \approx r_k + J_k\,\delta$, and substitute:

$$F(x \boxplus \delta) \;\approx\; \underbrace{\sum_k r_k^\top \Omega_k r_k}_{\text{constant}}
\;+\; 2\,\delta^\top \underbrace{\sum_k J_k^\top \Omega_k\, r_k}_{g}
\;+\; \delta^\top \underbrace{\sum_k J_k^\top \Omega_k J_k}_{H}\, \delta .$$

This is a quadratic in $\delta$. Setting its gradient $2g + 2H\delta$ to zero gives the normal
equations

$$H\,\delta = -g,$$

the weighted version of the familiar linear-least-squares normal equations $A^\top A x = A^\top b$.
$H$ is not the exact Hessian of $F$: the exact one has an extra term built from second derivatives
of the residuals, weighted by the residuals themselves. Gauss-Newton drops it. The approximation
is accurate when residuals are small or the model is nearly linear, which is why the method
converges quickly near the solution and can diverge far from it.

$H$ is the information matrix of the linearized problem. Each term $J_k^\top \Omega_k J_k$ is the
$H^\top R^{-1} H$ of the information form, with the edge's Jacobian in place of the measurement
Jacobian. In our corridor loop the residuals are linear with Jacobians $-1$ and $+1$, so $H$ is
exactly the matrix $\Lambda$ above, and $g$ collects $J^\top \Omega\, r$ from each edge. At the
dead-reckoned estimate only the loop closure has a nonzero residual.

Assembly is per-edge and purely additive, and it has the sparsity of the graph. Edge $(i,j)$
writes $J_i^\top \Omega J_i$ into block $(i,i)$, $J_j^\top \Omega J_j$ into $(j,j)$,
$J_i^\top \Omega J_j$ into $(i,j)$ and its transpose into $(j,i)$, and adds
$J_i^\top \Omega\, r$ and $J_j^\top \Omega\, r$ to segments $i$ and $j$ of $g$. Blocks that no
edge touches stay zero. $H$ is symmetric and positive semi-definite by construction, being a sum
of terms $J^\top \Omega J$ with each $\Omega$ positive definite.

Solving a symmetric positive-definite system calls for Cholesky rather than a general solver.
Cholesky factors $H = L L^\top$ with $L$ lower triangular, after which the solve is two triangular
substitutions; it costs about half of an LU factorization and needs no pivoting for numerical
stability. The square-root-free variant $H = L D L^\top$ has $L$ unit-diagonal and $D$ diagonal;
with symmetric pivoting added (as in Eigen's `LDLT`) it also tolerates a system that is merely
semi-definite. The step is then retracted onto the manifold, $T_i \leftarrow
T_i\,\mathrm{Exp}(\delta_i)$, and the whole loop repeats: relinearize, reassemble, resolve.

Storing $H$ as a dense matrix and factoring it densely costs cubic time in the state size, which
is nothing at 24 poses (a $138 \times 138$ solve once the anchored pose is removed) and hopeless
at $10^5$ poses. A production solver stores $H$ sparse and factors it sparsely, and then the cost
is governed by fill-in: the nonzeros that appear in $L$ where $H$ had zeros. Fill-in is the same
phenomenon as in the marginalization section, since factorizing is eliminating variables one at
a time and each elimination connects the eliminated variable's neighbors to each other. How much
fill-in appears depends on the order the variables are eliminated in, so solvers run a
fill-reducing ordering first (approximate minimum degree or COLAMD, both cheap combinatorial
heuristics on the graph), which permutes the variables to keep that count small. The permuted
system has the same solution and can be an order of magnitude cheaper to factor. That machinery
is out of scope here.

### Gauge freedom

A pure pose graph has an unobservable degree of freedom. Replace every pose $T_i$ by $S\,T_i$ for
one fixed rigid transform $S$, and every relative pose $T_i^{-1}T_j$ is unchanged, so every
residual and the whole cost are unchanged. Nothing in relative measurements says where the
trajectory sits in the world.

The cost therefore has a flat six-dimensional valley through every point (three directions that
translate the whole trajectory and three that rotate it), and $H$, which is the curvature of that
cost, is singular with a six-dimensional null space spanned by exactly those directions. The
solve has no unique answer. The term "gauge" is borrowed from physics, where it names the same
situation: freedom in the description that no measurement can resolve. The EKF-SLAM notes met
this as the unobservable absolute frame.

Fixing the gauge means removing those six directions. The two standard options are to hold one
pose fixed, dropping its rows and columns from the system entirely and never retracting it, or to
add a prior factor on one pose with very large information, which is the same thing in the limit.
For our corridor loop, holding $x_0$ fixed means solving with the lower-right $3 \times 3$ block
of $\Lambda$.

> **Exercise 3 (the gauge direction of our corridor loop).** (a) Show that $\Lambda$ times the
> all-ones vector is zero, and say what that vector means physically. (b) Show that the
> lower-right $3 \times 3$ block of $\Lambda$, the system left after holding $x_0$ fixed, is
> invertible.

<details>
<summary>Answer</summary>

(a) Every row of $\Lambda$ sums to zero ($400 - 100 - 300$, $-100 + 200 - 100$, and so on),
because each edge adds $\Omega$ and $-\Omega$ to the same row. So
$\Lambda\,(1, 1, 1, 1)^\top = 0$. The all-ones vector shifts every pose by the same amount, which
changes no difference $x_j - x_i$, so no residual and no cost. On a line the gauge has one
dimension; on $SE(3)$ it has six. The eigenvalues of $\Lambda$ are $0$, $200$, $276.4$, and
$723.6$, so the null space is exactly this one direction.

(b) The block is

$$\begin{bmatrix} 200 & -100 & 0 \\ -100 & 200 & -100 \\ 0 & -100 & 400 \end{bmatrix},$$

with determinant $200\,(200 \cdot 400 - 100^2) + 100\,(-100 \cdot 400) = 14 \times 10^6 - 4 \times
10^6 = 10^7 \neq 0$. Holding $x_0$ fixed removes the shift direction, because a shift would have
to move $x_0$, and leaves a positive-definite system with a unique solution.

</details>

### Damping, and why Gauss-Newton is enough here

Gauss-Newton takes the exact minimizer of the linearized cost, with no bound on how far that is.
When the linearization is only valid nearby, which is the case with a poor initialization or a
large rotation error, that step can overshoot and raise the true cost. Levenberg-Marquardt adds a
damping term to the system:

$$\big(H + \lambda\,\mathrm{diag}(H)\big)\,\delta = -g .$$

Large $\lambda$ lets the added diagonal dominate, which shrinks $\delta$ and turns it toward
$-\mathrm{diag}(H)^{-1}g$, a steepest-descent direction with each coordinate scaled by its own
curvature, and a short enough step along it always lowers the cost. Small $\lambda$ recovers
Gauss-Newton and its fast final convergence. Levenberg-Marquardt picks $\lambda$ by trial:
compute the step, evaluate the true cost, and accept the step and lower $\lambda$ if the cost
fell, or reject it and raise $\lambda$ if it rose. Dogleg reaches the same end with an explicit
trust region of radius $\Delta$: it computes both the Gauss-Newton step and the steepest-descent
step, and follows a two-segment path: from the current estimate to the Cauchy point (the
minimizer of the quadratic model along $-g$), then on to the Gauss-Newton point. It stops where
that path crosses the trust-region boundary, takes the full Gauss-Newton step if it lies inside,
and takes the steepest-descent step truncated at the boundary if the Cauchy point itself lies
outside. It grows or shrinks $\Delta$ on the same accept-or-reject test.

Gauss-Newton is the undamped $\lambda = 0$ case, and this assignment implements it,
because both synthetic graphs start close to the answer. The exact-recovery graph perturbs each
pose by random noise with standard deviation 0.15 rad per rotation axis and 0.4 m per
translation axis, and the drift graph starts from integrated odometry, which is already
the right shape.

### Loop closure

Odometry drifts. Integrating noisy relative-pose measurements along a trajectory accumulates
error, so a robot that drives a loop comes back to a start pose that does not match where it
began.

Odometry edges alone cannot repair this. The drifted trajectory was constructed by chaining
exactly those odometry measurements, so every odometry residual is already zero at the
initialization. The gradient $g$ is zero, the Gauss-Newton step is zero, and the optimizer leaves
the trajectory untouched no matter how many iterations it runs. Drift is not a violated
constraint. It is the accumulation of many individually satisfied ones.

A loop-closure edge is a measurement that the robot has returned to a place it has already been,
produced by re-recognizing a landmark or matching a laser scan against an earlier one, and it ties
a late pose directly back to an early one. On the drifted trajectory that one edge has a large
residual, and because it connects two poses that are far apart along the graph, correcting it
requires the whole loop between them to move. Gauss-Newton distributes the correction across all
of those poses at once, weighted by their information, and the trajectory moves onto a
consistent shape. The correction goes where the measurements are least certain, not where the
error actually happened: the solver has no way to know which odometry reading was the bad one.

> **Exercise 4 (one Gauss-Newton step on our corridor loop).** Start from the dead-reckoned
> estimate $0, 1.1, 2.2, 0.2$. (a) Compute $g$ with and without the loop-closure edge. (b) With
> the loop closure, hold $x_0$ fixed, solve $H\delta = -g$ for $(\delta_1, \delta_2, \delta_3)$,
> and give the new estimate, every residual, and the cost before and after. (c) What does a
> second Gauss-Newton step do? (d) How is the $0.2$ m misclosure divided among the four edges?

<details>
<summary>Answer</summary>

(a) The odometry residuals are zero, so they contribute nothing to $g$. The loop closure is the
edge from $i = 3$ to $j = 0$ with $J_i = -1$, $J_j = +1$, and residual
$r = (x_0 - x_3) - 0 = -0.2$. It adds $J_j\,\Omega\,r = 300 \cdot (-0.2) = -60$ to $g_0$ and
$J_i\,\Omega\,r = +60$ to $g_3$, so $g = (-60, 0, 0, 60)$. Without the loop closure $g = 0$ and
the step is zero.

(b) Holding $x_0$ fixed leaves the $3 \times 3$ system from Exercise 3 with right-hand side
$-g = (0, 0, -60)$. The first row gives $\delta_2 = 2\delta_1$, the second
$\delta_3 = 4\delta_1 - \delta_1 = 3\delta_1$, and the third
$-100 \cdot 2\delta_1 + 400 \cdot 3\delta_1 = 1000\,\delta_1 = -60$. So
$\delta = (-0.06, -0.12, -0.18)$ and the new estimate is $0, 1.04, 2.08, 0.02$. Each odometry
residual is now $-0.06$ (for example $1.04 - 0 - 1.1$) and the loop-closure residual is
$0 - 0.02 = -0.02$. The cost falls from $12$ to
$3 \cdot 100 \cdot 0.06^2 + 300 \cdot 0.02^2 = 1.08 + 0.12 = 1.2$. The end-of-trajectory error
falls from $0.2$ m to $0.02$ m.

(c) Nothing. The corridor residuals are linear, so the quadratic model is the cost itself and one
step lands on its minimum; recomputing $g$ there gives zero. On $SE(3)$ the residuals are
nonlinear and Gauss-Newton needs several iterations.

(d) Around the loop the four measurements must add up to zero ($+1.1 + 1.1 - 2.0 + 0$ would have
to equal $0$), and they add to $0.2$. The solution removes $0.06$ from each odometry edge and
$0.02$ from the loop closure. Those shares are proportional to the edge variances
($0.01, 0.01, 0.01, 0.0033$, with total $0.0333$): $0.2 \cdot 0.01 / 0.0333 = 0.06$ and
$0.2 \cdot 0.0033 / 0.0333 = 0.02$. The true error was all in the first two moves, but the
optimizer spreads it evenly over the three equally uncertain odometry edges, so $x_1$ and $x_2$
end $0.04$ m and $0.08$ m from the truth.

</details>

### Bundle adjustment

Bundle adjustment is the same machinery with structure added to the state. The variables are
camera poses $T_c \in SE(3)$ and 3D points $p_\ell \in \mathbb{R}^3$, and each factor is one
observation: point $\ell$ was seen in camera $c$ at pixel $u_{c\ell}$. The residual is the
reprojection error, which transforms the world point into the camera frame and projects it with
the pinhole model from the multi-view assignment,

$$r_{c\ell} \;=\; \pi\big(K,\; T_c^{-1} p_\ell\big) \;-\; u_{c\ell} \;\in\; \mathbb{R}^2,
\qquad \pi\big(K, (X,Y,Z)\big) = \Big(f_x \tfrac{X}{Z} + c_x,\; f_y \tfrac{Y}{Z} + c_y\Big).$$

The name comes from the bundles of rays leaving each 3D point: adjusting the poses and points
until the bundles meet consistently at the observed pixels.

The structure that matters is which variables share a factor. Every reprojection factor touches
exactly one pose (6 variables) and one point (3 variables). No factor ever touches two points, so
in the assembled system the landmark-landmark block $H_{\ell\ell}$ has nothing off its diagonal:
it is block-diagonal with one 3x3 block per point. Poses do become coupled to each other, but
only indirectly, through points they both observe, so $H_{pp}$ is comparatively dense. Written
with the poses first, the whole matrix has what is called arrowhead structure: a dense square in
the top-left corner, a block-diagonal tail down the bottom-right, and dense cross bands joining
them.

Sizes are lopsided. A reconstruction has orders of magnitude more points than cameras, so almost
all of the system's dimension sits in the block-diagonal tail. Factoring the whole thing densely
is cubic in that total dimension and out of the question.

### The Schur complement in bundle adjustment

Partition the linear system solved inside one Gauss-Newton iteration, poses first:

$$\begin{bmatrix} H_{pp} & H_{p\ell} \\ H_{\ell p} & H_{\ell\ell} \end{bmatrix} \begin{bmatrix} \delta_p \\ \delta_\ell \end{bmatrix} = \begin{bmatrix} b_p \\ b_\ell \end{bmatrix}.$$

Here $b = -g$, the right-hand side of the normal equations. Eliminating the landmark block is the
marginalization formula from earlier, read this time as block Gaussian elimination on a linear
system rather than on a Gaussian. Clearing $H_{p\ell}$ out of the first block row leaves the
reduced camera system

$$\underbrace{\big(H_{pp} - H_{p\ell}\, H_{\ell\ell}^{-1}\, H_{\ell p}\big)}_{\text{reduced camera system}}\, \delta_p \;=\; b_p - H_{p\ell}\, H_{\ell\ell}^{-1}\, b_\ell,$$

which involves the pose variables only. Solve it for $\delta_p$, then recover the landmarks by
back-substitution into the second block row:

$$\delta_\ell = H_{\ell\ell}^{-1}\big(b_\ell - H_{\ell p}\, \delta_p\big).$$

The saving comes from $H_{\ell\ell}$ being block-diagonal. Inverting it is not one large inverse
but one independent 3x3 inverse per point, so that step is linear in the number of points rather
than cubic. The only cubic cost left is the factorization of the reduced camera system, whose
dimension is 6 per camera. The expensive part of the solve has been moved off the large variable
group and onto the small one.

This is exact, not an approximation: the elimination is invertible, so the pair
$(\delta_p, \delta_\ell)$ recovered this way solves the original system, up to floating point.
The reduced camera system is also where the fill-in of the marginalization section shows up: two
cameras that see a common point become directly coupled in it, even if no factor joins them.

> **Exercise 5 (a Schur solve by hand).** One scalar pose variable and two scalar landmark
> variables give the arrowhead system
> $$H = \begin{bmatrix} 4 & 1 & 1 \\ 1 & 2 & 0 \\ 1 & 0 & 1 \end{bmatrix}, \qquad b = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}.$$
> (a) Solve it by eliminating the two landmarks first, and check the answer against $H\delta = b$.
> (b) For 10 cameras and 10,000 points, compare the dimension of the full system with that of the
> reduced camera system, and the cost of a dense Cholesky factorization of each (about $n^3/3$
> floating-point operations for dimension $n$).

<details>
<summary>Answer</summary>

(a) $H_{\ell\ell} = \mathrm{diag}(2, 1)$, so its inverse is $\mathrm{diag}(1/2, 1)$, one small
inverse per landmark. The reduced system is
$(4 - 1^2/2 - 1^2/1)\,\delta_p = 1 - 1 \cdot 1/2 - 1 \cdot 0/1$, that is
$2.5\,\delta_p = 0.5$, so $\delta_p = 0.2$. Back-substitution gives
$\delta_{\ell 1} = (1 - 1 \cdot 0.2)/2 = 0.4$ and $\delta_{\ell 2} = (0 - 1 \cdot 0.2)/1 = -0.2$.
Check: $4(0.2) + 0.4 - 0.2 = 1$, $0.2 + 2(0.4) = 1$, and $0.2 - 0.2 = 0$. The pose variable is
solved from a $1 \times 1$ system although there are three unknowns.

(b) The full system has dimension $6 \cdot 10 + 3 \cdot 10{,}000 = 30{,}060$; stored densely
that is $9.0 \times 10^8$ entries (7.2 GB in double precision), and a dense Cholesky costs about
$30{,}060^3/3 \approx 9.1 \times 10^{12}$ operations. The reduced camera system has dimension
$60$ and costs about $60^3/3 = 72{,}000$ operations to factor, plus 10,000 independent
$3 \times 3$ inverses. Forming the reduced matrix costs, for each point, work proportional to the
square of the number of cameras that see it, which is linear in the number of points.

</details>

### Incremental smoothing and the production solvers

Batch Gauss-Newton reassembles and refactors the entire system on every iteration. A robot adds
one pose and a few factors per timestep, so refactoring from scratch each time repeats almost all
of the previous work. iSAM (Kaess, Ranganathan, Dellaert 2008) keeps the matrix factor between
timesteps and updates it in place: a new factor only invalidates the parts of the factor
downstream of the variables it touches in the elimination order, and the rest is still valid.
iSAM2 (Kaess et al. 2012) organizes the factorization as a Bayes tree, a tree whose nodes are the
cliques produced by the elimination, which turns "what does this new measurement invalidate" into
"which subtree must be re-eliminated". It also adds fluid relinearization: rather than
re-linearizing everything, it re-linearizes only the variables whose estimates have moved far
enough from their last linearization point to matter. Together those make a smoother that keeps
every pose run at sensor rate, which is why smoothing displaced filtering in practice and not
only in principle.

The production implementations are g2o (Kümmerle et al. 2011), GTSAM (Georgia Tech), and Ceres
(Google). All three do the same sequence: build the graph, assemble the sparse normal equations,
apply a fill-reducing ordering, factor, damp, iterate. GTSAM's `BetweenFactor` is the library
version of the residual and Jacobians derived above.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the
declaration and comments in `factor_graph.hpp` give the signature and shapes.

| # | Function | File | Concept from the notes |
|---|---|---|---|
| 1 | [`between_residual()`](factor_graph.cpp) | `factor_graph.cpp` | the between-factor residual $r_{ij} = \mathrm{Log}(T_{\text{meas}}^{-1} T_i^{-1} T_j)$ |
| 2 | [`between_jacobians()`](factor_graph.cpp) | `factor_graph.cpp` | $\partial r/\partial\delta_i = -\mathcal{J}_r^{-1}(r)\,\mathrm{Ad}_{T_j^{-1}T_i}$ and $\partial r/\partial\delta_j = \mathcal{J}_r^{-1}(r)$ |
| 3 | [`optimize_pose_graph()`](factor_graph.cpp) | `factor_graph.cpp` | Gauss-Newton: per-edge assembly of $H$ and $g$, gauge fixed by anchoring pose 0, retraction |
| 4 | [`schur_solve()`](factor_graph.cpp) | `factor_graph.cpp` | the reduced camera system and landmark back-substitution |

You may not include an existing SLAM or solver library (GTSAM, g2o, Ceres, Sophus, manif); a test
scans the sources.

Notes on specific holes:
- The provided primitives in `models.cpp` are `hat3` (the $SO(3)$ skew matrix), `se3_exp` and
  `se3_log`, `se3_adjoint` ($\mathrm{Ad}_T = [[R, \widehat{t}R], [0, R]]$), and
  `se3_right_jacobian_inv` ($\mathcal{J}_r^{-1}(\xi)$). Call them; do not reimplement them. A
  twist is a `Vector6d` in the $[\rho; \theta]$ ordering: `se3_exp` reads `xi.head<3>()` as
  $\rho$.
- `between_jacobians`: evaluate $\mathcal{J}_r^{-1}$ at the residual $r_{ij}$, not at zero.
  `test_between_factor.py` evaluates the Jacobians at a nonzero residual, so the full form is
  required.
- `optimize_pose_graph`: the edges arrive as parallel arrays (`edge_i[k]`, `edge_j[k]`,
  `meas[k]`, `info[k]`). Each iteration assembles $H$ ($6N \times 6N$) and $g$ from every edge,
  then fixes the gauge by holding pose 0: solve only the bottom-right
  $6(N-1) \times 6(N-1)$ block of $H$ against the matching tail of $-g$, and retract poses
  $1 \ldots N-1$, so pose 0 stays exactly where it started. Store $H$ as a dense
  `Eigen::MatrixXd` and solve with `ldlt()`. Return the poses and the total weighted squared
  residual evaluated at the returned poses, after the last retraction; with `num_iters = 0` that
  is the cost of the input, which `viz.py` uses for its first frame.
- `schur_solve` solves $H\,\delta = b$ with the right-hand side as given, so a caller assembling
  normal equations passes $b = -g$; `optimize_pose_graph` applies that sign internally. It assumes
  the pose variables occupy the leading `n_pose` columns and that every landmark block has the
  same size `lm_block`, which is how the test builds its system (6 variables per pose block, 3 per
  landmark). Invert $H_{\ell\ell}$ one `lm_block` x `lm_block` diagonal block at a time.
- `config.py` holds the tolerances, the loop (`N_POSES = 24`, `RADIUS = 5.0`), the initialization
  perturbation of the exact-recovery graph (`INIT_ROT_SIGMA = 0.15`, `INIT_TRANS_SIGMA = 0.4`), the
  measurement noise of the drift graph (`ODOM_SIGMA = 0.02`, `LOOP_SIGMA = 0.05`), and
  `GN_ITERS = 20`. `sim.py` builds both graphs: `pose_graph_exact` (noise-free measurements on the
  23 consecutive edges plus the wrap-around edge from pose 23 to pose 0, with a perturbed
  initialization) and `pose_graph_drift` (noisy odometry integrated into the initialization, plus
  one loop-closure edge from the last pose to the first).

### Building and running

Same toolchain as the rest of the module: a C++17 compiler plus CMake, pybind11, and Eigen (the
`environment.yml` a14 block). You never call CMake by hand: importing `_impl.py` (which the tests
and `viz.py` do) configures and builds a pybind11 extension with CMake, from the top-level
`factor_graph.cpp` by default or from `solution/factor_graph.cpp` under
`NANOVISION_IMPL=solution`, compiled together with `models.cpp` and `bindings.cpp` in both cases.
The build is cached under `build/student/` or `build/solution/`, so only the first run or a
source change compiles.

Activate the environment first (`conda activate nanovision`); the Makefile calls the `python` on
the path. Then:

```
make verify A=a14_5_factor_graph   # build + run the reference (solution/); the green target
make test   A=a14_5_factor_graph   # build + run YOUR code; red until the holes are filled
make viz    A=a14_5_factor_graph        # render the loop-closure correction (reference)
make viz-mine A=a14_5_factor_graph      # the same, from YOUR code, once the holes are filled
```

`make test` also accepts `K="schur"` to run only matching tests and `SKIP_HOLES=1` to report
unfilled holes as skips instead of failures.

The tests favor implementation-independent statements:
- `test_between_factor.py`: the residual is zero when the estimate matches the measurement, and
  the analytic edge Jacobians match numerical differentiation to ~1e-5 at ten random pose
  triples with nonzero residual (the test that catches the sign and adjoint-argument bug).
- `test_pose_graph.py`: Gauss-Newton recovers the ground-truth trajectory on the noise-free graph
  (mean position error below $10^{-6}$, up to the anchored gauge) and drives the cost below
  $10^{-12}$; the anchored pose does not move; adding the loop-closure edge reduces
  end-of-trajectory drift by at least a factor of five, while optimizing without it leaves the
  drift in place to within $10^{-6}$.
- `test_schur.py`: the Schur-complement solve equals `numpy.linalg.solve` of the same system to
  $10^{-9}$, and $H\delta - b$ has norm below $10^{-9}$. The test's $H$ is a synthetic
  positive-definite matrix with the landmark off-diagonal blocks zeroed rather than one assembled
  from real reprojection factors, which is enough to exercise the linear algebra.
- `test_oracle_gtsam.py`: if GTSAM is installed, cross-checks the optimized trajectory against
  GTSAM's Levenberg-Marquardt optimizer on the noise-free graph, with pose 0 anchored there by a
  prior of precision $10^8$; the two agree on the resulting trajectory to $10^{-3}$. GTSAM orders
  twists $[\theta; \rho]$, the reverse of this module, which does not affect the comparison on a
  graph with isotropic information. The test is skipped when GTSAM is absent.
- `test_forbidden_imports.py`: no SLAM or solver library is included.

`make viz` writes `out/pose_graph.rrd`. Add `SHOW=1` for the interactive viewer: the
ground-truth loop, the drifted trajectory moving onto it as the Gauss-Newton iterations replay
(frame $k$ is the result of $k$ iterations, for $k = 0 \ldots 12$), the loop-closure edge, a cost
panel, and the block-sparsity of $H$ (the band plus the off-corner the loop-closure edge
creates). Scrub the timeline to watch the correction propagate.

The figures in the notes are generated by `assets/make_figures.py` (run it from `assets/`). The
corridor figure is computed in closed form with NumPy. The drift-graph figure runs the reference
optimizer on `pose_graph_drift(seed=0)`, the graph `viz.py` shows, building `solution/` through
`_impl.py` on first use.

What you should see when you run this. On the drift graph the end-of-trajectory position error
drops by more than a factor of ten once the loop-closure edge is included: from 1.19 m to 0.073 m
on the seed `viz.py` uses, and from 1.02 m to 0.026 m on the seed the test uses. Removing that
single edge leaves the error exactly where it started, to numerical precision (a change of about
$10^{-15}$ m). Most of the correction lands on the first Gauss-Newton iteration, with later
iterations only polishing. In the viz the cost falls by roughly two orders of magnitude on the
first iteration (615 to 8.0 on that seed) and then flattens at a nonzero value (7.8) rather than
reaching zero, because that graph's measurements carry noise and cannot all be satisfied at once.
The poses in the middle of the loop keep errors of up to about 0.33 m: the loop closure removes
the misclosure, not the individual odometry errors, which is the corridor's Exercise 4(d) at
scale. Only the noise-free graph used by the recovery test drives the cost to zero; there the
cost falls from 20.9 to about $10^{-10}$ in three iterations and the error converges
quadratically, so `GN_ITERS = 20` is far more than needed.

The pose graph is implemented on $SE(3)$ (the trajectory here is planar, a loop with yaw, so it
reads as a 2D plot, but the code is the full 3D machinery and an $SE(2)$ graph is the same
construction in three dimensions instead of six). The canonical real datasets for this method are
the g2o pose-graph benchmarks: real logs and simulated graphs distributed in a common text
format, including the Intel Research Lab and MIT Killian Court indoor runs and the synthetic
sphere and torus graphs used to stress 3D optimizers. The tests use a synthetic loop instead
because it gives the deterministic ground truth they need.

## In interviews

Factor-graph SLAM and bundle adjustment are standard interview topics for SLAM, AR, and
3D-reconstruction roles, both as derivations and as system-design discussions.

Filter versus smoother. Expect "why did factor graphs replace the EKF for SLAM?" The answer has
two parts: smoothing keeps all poses and re-linearizes every iteration, so linearization error
does not stay in the estimate the way it does when a filter marginalizes a pose; and the problem
is sparse, so keeping everything is affordable. Incremental smoothing (iSAM, iSAM2) makes it
real-time by updating the existing factorization rather than refactoring from scratch.

The relative-pose residual and one of its Jacobians. A common on-the-spot ask is to write
$r_{ij} = \mathrm{Log}(T_{\text{meas}}^{-1} T_i^{-1} T_j)$ and one of its Jacobians. Knowing that
the Jacobian carries the inverse right Jacobian $\mathcal{J}_r^{-1}(r)$ (because the residual is
a logarithm) and the adjoint (because the $i$-side perturbation has to be moved past
$T_i^{-1}T_j$), with the minus sign on the $i$-side, is a strong signal.

The Schur complement and why BA exploits structure. Be ready to explain that the landmark block
is block-diagonal because no factor touches two landmarks, so inverting it is a per-point 3x3
inverse and the solve reduces to the camera poses, giving the same answer as the dense solve.
Being able to say that this is the same formula as Gaussian marginalization is a step further.

Gauge freedom and how to fix it. A pure pose graph (or a BA problem) has an unobservable global
transform, so the system matrix is singular; the fix is to anchor one pose or add a prior.
Not handling the gauge is a common bug: the solve fails or wanders. Related: Gauss-Newton
versus Levenberg-Marquardt versus dogleg (damping and trust regions, for tolerating a bad
initialization), and the production solvers g2o, GTSAM, and Ceres.

## Where this goes next

This is the last assignment of the classical SLAM module. A complete visual SLAM system is the
pieces of the module put together: two-view geometry, PnP, and ICP produce relative poses and
initial 3D points, place recognition proposes loop closures, and the factor-graph back-end here
refines everything jointly. The geometry foundation-model assignment takes the other route and
predicts camera poses and geometry directly from images with a network, skipping the
correspondence-then-solve pipeline.

## Further reading

- Grisetti et al., "A tutorial on graph-based SLAM" (2010) - the pose-graph formulation and
  Gauss-Newton solution.
- Dellaert and Kaess, "Factor graphs for robot perception" (2017) - the factor-graph view and
  incremental smoothing.
- Kaess, Ranganathan, Dellaert, "iSAM: incremental smoothing and mapping" (T-RO 2008) - the
  incremental update of the factorization.
- Kaess et al., "iSAM2: incremental smoothing and mapping using the Bayes tree" (IJRR 2012) -
  the incremental solver behind modern systems.
- Triggs et al., "Bundle adjustment - a modern synthesis" (2000) - BA and the Schur complement.
- Barfoot, *State Estimation for Robotics* (2017), chapter 7 - the SE(3) Jacobians used for the
  edge derivatives.
- Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics" (2018) - the
  right-perturbation conventions and the right-Jacobian identities used above.
- Kümmerle et al., "g2o: a general framework for graph optimization" (ICRA 2011) - a production
  pose-graph and BA solver, and the source of the benchmark graphs.
