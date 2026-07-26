# a14_5 - Pose-graph and bundle adjustment

The last assignment of the classical SLAM module, in C++17 with Eigen. You build the
factor-graph back-end: the smoothing optimizer that replaced filtering as the standard SLAM
estimator. You implement the between-factor residual and its analytic Jacobians, the
Gauss-Newton loop that optimizes a pose graph, and the Schur complement that makes bundle
adjustment affordable at scale. The centerpiece is loop closure - a drifted trajectory snapping
into shape the moment a single loop-closure edge is added.

It builds on the Lie-group material: poses live on SE(3), the residual is a matrix logarithm,
the Jacobians use the right perturbation and the inverse right Jacobian, and the updates retract
on the manifold rather than adding vectors. That SE(3) machinery - the exponential and
logarithm, the adjoint, and the 6x6 inverse right Jacobian - is provided in `models.cpp`, so the
work here is the factors, the normal equations, and the marginalization built on top of it. The
notes below re-derive the pieces of it that the edge Jacobians depend on, so the derivation is
self-contained.

The bundle-adjustment hole is the linear-algebra step alone: `schur_solve` takes an already
assembled system with bundle-adjustment structure and eliminates its landmark block. Reprojection
factors are described in the notes but are not implemented here; the multi-view assignment
already covered the pinhole projection they would use.

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
integrated out every past pose as the robot moved. That is filtering, and it has two costs that
assignment named: a dense covariance that makes each update quadratic in the map size, and
linearization errors that are baked in permanently, because a pose that has been integrated out
can never be re-linearized when later evidence arrives.

Smoothing keeps all the poses (and optionally the landmarks) as free variables and solves for all
of them at once, every time. The estimate is the configuration of all variables that best explains
all measurements, and it is recomputed from scratch (or updated incrementally) as measurements
arrive. Two properties make this practical despite keeping everything. The problem is sparse,
because each measurement constrains only two or three variables. And every variable can be
re-linearized at every iteration, so linearization error does not accumulate the way it does in a
filter.

The rest of these notes builds that claim up: what the objective is and where it comes from, what
"integrating out a variable" costs and why it is the same operation in both the filter and the
bundle-adjustment solver, and what the SE(3) residual and its derivatives look like.

### Factor graphs and why the objective is a sum of squares

A factor graph is a bipartite graph with two kinds of node. Variable nodes are the unknowns -
here, one per robot pose, and in bundle adjustment also one per 3D point. Factor nodes are the
terms of a product, each connected by edges to the small subset of variables it depends on.
Drawing an estimation problem this way asserts one thing: the posterior over all variables $x$
given all measurements $z$ factorizes,

$$p(x \mid z) \;\propto\; \prod_k \phi_k(x_k),$$

where $x_k$ is the handful of variables that factor $k$ touches. A ten-thousand-pose problem is
a product of ten thousand small terms, not one ten-thousand-dimensional blob.

Each measurement contributes one factor. Model measurement $k$ as a prediction $h_k(x_k)$
corrupted by zero-mean Gaussian noise of covariance $\Sigma_k$. Its factor is then

$$\phi_k(x_k) \;\propto\; \exp\!\Big(-\tfrac12\, r_k^\top\, \Omega_k\, r_k\Big), \qquad
\Omega_k = \Sigma_k^{-1},$$

where $r_k$ is the residual, the difference between what the current variable estimates predict
and what was actually measured. $\Omega_k$ is the information matrix, the inverse covariance, the
same dual coordinate the Kalman assignment used for the information form. It is the natural
weight rather than a tuning knob: a measurement that is precise along some direction has a large
$\Omega$ entry there and pulls the estimate hard along it, while a direction the sensor barely
constrains contributes almost nothing. The scalar $r^\top \Omega r$ is the squared Mahalanobis
distance, which is the squared Euclidean length of the residual after each direction has been
rescaled by its own standard deviation, so residuals in meters and residuals in radians become
comparable numbers that can be added.

The maximum-a-posteriori estimate maximizes the product, and a product of exponentials is easiest
to maximize by minimizing the negative logarithm of it:

$$x^\star \;=\; \arg\min_x\; \sum_k r_k(x)^\top\, \Omega_k\, r_k(x).$$

Nonlinear least squares is therefore not an arbitrary choice of objective. It is the MAP estimate
under Gaussian noise, and the weights are inverse covariances because that is what the Gaussian
exponent contains.

Sparsity is a statement about which variables share a factor. A pose-graph edge touches exactly
two poses, so it can only ever contribute to four blocks of the system matrix assembled later.
The synthetic graph in this assignment has 24 poses and 24 edges (23 consecutive odometry edges
plus one loop closure), so its system matrix is $144 \times 144$ with 72 of its 576 six-by-six
blocks nonzero: 24 on the diagonal and 2 per edge off it. Real SLAM graphs are sparser still,
because the number of edges stays proportional to the number of poses while the matrix grows with
its square.

### Marginalization and the fill-in it creates

Integrating a variable out of a joint Gaussian is the single operation behind the filter's dense
covariance, behind the elimination ordering in a sparse solver, and behind the bundle-adjustment
trick at the end of these notes. It is worth deriving once.

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
directly correlated with every other one. Those new edges are fill-in. Do this once per timestep,
which is exactly what a filter does when it drops the previous pose, and the information matrix
over the map fills in completely. That is where EKF-SLAM's dense covariance comes from, and why
its update is quadratic in the map size.

Marginalization itself is exact, not an approximation: the reduced system over $a$ has the same
solution for $a$ as the full system did. What is lost is the ability to revisit $b$. The
correction term was computed at one linearization point, and once $b$ is gone that point is
frozen. A smoother avoids the loss by never marginalizing across iterations. The
bundle-adjustment Schur complement at the end of these notes applies the same formula, but to the
already-linearized system inside a single Gauss-Newton iteration, so nothing is frozen from one
iteration to the next.

### The SE(3) tools the residual needs

A pose is a 4x4 matrix $T = \begin{bmatrix} R & t \\ 0 & 1\end{bmatrix}$ with $R$ a rotation, so
it carries 12 numbers describing 6 degrees of freedom under the constraint $R^\top R = I$. An
optimizer cannot add a 12-vector correction to $T$ and stay on the manifold, so corrections live
in the 6-dimensional tangent space instead and are mapped onto the group.

The tangent vector is a twist $\xi = \begin{bmatrix}\rho \\ \theta\end{bmatrix} \in \mathbb{R}^6$,
translation part first and rotation part second. This module fixes that ordering, and the code
follows it: `se3_exp` reads `xi.head<3>()` as $\rho$, and the adjoint below has its off-diagonal
block in the upper right because of it. The exponential $\mathrm{Exp}(\xi)$ maps a twist to a
pose and the logarithm $\mathrm{Log}(T)$ inverts it. Both are provided in `models.cpp`.

Three facts about them are what the edge Jacobians are built from.

Retraction. The correction step is a right multiplication, $T \leftarrow T\,\mathrm{Exp}(\delta)$,
which the Lie-group assignment wrote as box-plus. Reading it right to left: $\delta$ is a small
motion expressed in the body frame of $T$, applied before $T$ takes the result to the world frame.
The whole assignment uses this right convention, so every Jacobian below is a derivative with
respect to a body-frame perturbation.

The adjoint. $\mathrm{Ad}_T$ is the 6x6 matrix that converts a twist from the body frame to the
world frame, defined by

$$T\,\mathrm{Exp}(\xi) \;=\; \mathrm{Exp}(\mathrm{Ad}_T\,\xi)\,T .$$

The same motion can be applied on the right of $T$ as $\xi$ or on the left as
$\mathrm{Ad}_T\,\xi$; the adjoint translates between the two. Rearranging that identity gives the
form used repeatedly below, moving a left factor to the right past a pose $A$:

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
vector space. `models.cpp` computes it as `se3_right_jacobian_inv` from Barfoot's closed form,
via $\mathcal{J}_r(\xi) = \mathcal{J}_l(-\xi)$ and an explicit 6x6 inverse.

### The pose graph and the between-factor

A pose graph is a factor graph whose only variables are robot poses $T_i \in SE(3)$ and whose
only factors are relative-pose measurements, one per edge. An edge between $i$ and $j$ carries a
measured relative pose $T_{\text{meas}} = T_{i \leftarrow j}$, meaning frame $j$ expressed in
frame $i$ under the module's source-to-target naming, and an information matrix $\Omega$. The
current estimates predict the relative pose $T_i^{-1} T_j$, and the residual is the difference
between measurement and prediction taken on the manifold, using box-minus:

$$r_{ij} = \mathrm{Log}\big(T_{\text{meas}}^{-1}\, (T_i^{-1} T_j)\big) \in \mathbb{R}^6.$$

It is zero exactly when the estimated relative pose equals the measurement, because then the
argument of the logarithm is the identity. Summed over edges with the weights from the previous
sections, the total cost is $\sum_{ij} r_{ij}^\top \Omega_{ij}\, r_{ij}$.

The Jacobians of $r_{ij}$ with respect to the right perturbations $\delta_i, \delta_j$ follow from
the three SE(3) facts above and nothing else.

Start with $j$, which is the easier side. Perturbing gives $T_j \leftarrow T_j\,\mathrm{Exp}(\delta_j)$,
so the argument of the logarithm becomes

$$T_{\text{meas}}^{-1}\, T_i^{-1}\, T_j\, \mathrm{Exp}(\delta_j) \;=\; \mathrm{Exp}(r_{ij})\,\mathrm{Exp}(\delta_j),$$

where the second form uses the definition of $r_{ij}$ to name the unperturbed product. The
inverse-right-Jacobian identity reads off the result directly:

$$\frac{\partial r_{ij}}{\partial \delta_j} = \mathcal{J}_r^{-1}(r_{ij}).$$

Now $i$. Perturbing gives $T_i \leftarrow T_i\,\mathrm{Exp}(\delta_i)$, so
$T_i^{-1} \leftarrow \mathrm{Exp}(-\delta_i)\, T_i^{-1}$ and the argument becomes

$$T_{\text{meas}}^{-1}\, \mathrm{Exp}(-\delta_i)\, T_i^{-1} T_j .$$

The perturbation is now stuck in the middle, on the wrong side of $T_i^{-1}T_j$, where the
identity above cannot be applied. Write $A = T_i^{-1} T_j$ and move it past $A$ with the adjoint
rearrangement, $\mathrm{Exp}(-\delta_i)\,A = A\,\mathrm{Exp}(-\mathrm{Ad}_{A^{-1}}\delta_i)$:

$$T_{\text{meas}}^{-1}\, A\, \mathrm{Exp}\big(-\mathrm{Ad}_{A^{-1}}\,\delta_i\big) \;=\;
\mathrm{Exp}(r_{ij})\,\mathrm{Exp}\big(-\mathrm{Ad}_{A^{-1}}\,\delta_i\big).$$

That is the same shape as the $j$ case with $-\mathrm{Ad}_{A^{-1}}\delta_i$ in place of
$\delta_j$, and $A^{-1} = (T_i^{-1}T_j)^{-1} = T_j^{-1}T_i$, so

$$\frac{\partial r_{ij}}{\partial \delta_i} = -\,\mathcal{J}_r^{-1}(r_{ij})\, \mathrm{Ad}_{T_j^{-1} T_i}.$$

These are the standard published forms, matching Barfoot's derivation and the `BetweenFactor` in
GTSAM, the Georgia Tech factor-graph library used as the optional oracle here. The minus sign on
the $i$-side and the argument $T_j^{-1}T_i$ of the adjoint are the two details a hand-rolled
implementation usually gets wrong, and the numerical-versus-analytic test exists to catch exactly
them. Near convergence $r_{ij} \to 0$ and $\mathcal{J}_r^{-1} \to I$, so a wrong or missing
$\mathcal{J}_r^{-1}$ still converges on easy problems and hides; the test evaluates at a
deliberately nonzero residual so the full form is required.

### Gauss-Newton and the normal equations

Gauss-Newton minimizes $F(x) = \sum_k r_k(x)^\top \Omega_k\, r_k(x)$ by repeatedly solving a
linear approximation of it. Around the current estimate, replace each residual by its first-order
expansion in the retracted increment, $r_k(x \boxplus \delta) \approx r_k + J_k\,\delta$, and
substitute:

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

Assembly is per-edge and purely additive, which is the sparsity from earlier cashed out. Edge
$(i,j)$ writes $J_i^\top \Omega J_i$ into block $(i,i)$, $J_j^\top \Omega J_j$ into $(j,j)$,
$J_i^\top \Omega J_j$ into $(i,j)$ and its transpose into $(j,i)$, and adds
$J_i^\top \Omega\, r$ and $J_j^\top \Omega\, r$ to segments $i$ and $j$ of $g$. Blocks that no
edge touches stay zero. $H$ is symmetric and positive semi-definite by construction, being a sum
of terms $J^\top \Omega J$ with each $\Omega$ positive definite.

Solving a symmetric positive-definite system calls for Cholesky rather than a general solver.
Cholesky factors $H = L L^\top$ with $L$ lower triangular, after which the solve is two triangular
substitutions; it costs about half of an LU factorization and needs no pivoting for numerical
stability. Eigen's `LDLT`, used here, is the square-root-free variant $H = L D L^\top$ with $L$
unit-diagonal and $D$ diagonal, which also applies symmetric pivoting and so tolerates a system
that is merely semi-definite. The step is then retracted onto the manifold, $T_i \leftarrow T_i\,\mathrm{Exp}(\delta_i)$,
and the whole loop repeats: relinearize, reassemble, resolve.

This assignment stores $H$ as a dense `Eigen::MatrixXd` and factors it densely. That costs cubic
time in the state size, which is nothing at 24 poses (a $138 \times 138$ solve once the anchored
pose is removed) and hopeless at $10^5$ poses. A production solver stores $H$ sparse and factors
it sparsely, and then the cost is governed by fill-in: the nonzeros that appear in $L$ where $H$
had zeros. Fill-in is the same phenomenon as in the marginalization section, since factorizing is
eliminating variables one at a time and each elimination connects the eliminated variable's
neighbors to each other. How much fill-in appears depends on the order the variables are
eliminated in, so solvers run a fill-reducing ordering first - approximate minimum degree or
COLAMD, both cheap combinatorial heuristics on the graph - which permutes the variables to keep
that count small. The permuted system has the same solution and can be an order of magnitude
cheaper to factor. That machinery is out of scope here.

### Gauge freedom

A pure pose graph has an unobservable degree of freedom. Replace every pose $T_i$ by $S\,T_i$ for
one fixed rigid transform $S$, and every relative pose $T_i^{-1}T_j$ is unchanged, so every
residual and the whole cost are unchanged. Nothing in relative measurements says where the
trajectory sits in the world.

The cost therefore has a flat six-dimensional valley through every point - three directions that
translate the whole trajectory and three that rotate it - and $H$, which is the curvature of that
cost, is singular with a six-dimensional null space spanned by exactly those directions. The
solve has no unique answer. The term "gauge" is borrowed from physics, where it names this same
thing: freedom in the description that no measurement can resolve.

Fixing the gauge means removing those six directions. The two standard options are to hold one
pose fixed, dropping its rows and columns from the system entirely and never retracting it, or to
add a prior factor on one pose with very large information, which is the same thing in the limit.
`optimize_pose_graph` does the first: it assembles the full $6N \times 6N$ system, then solves
only the bottom-right $6(N-1) \times 6(N-1)$ block and retracts poses $1 \ldots N-1$, so pose 0
stays exactly where it started. A test checks that it has not moved. The optional GTSAM
cross-check does the second, anchoring pose 0 with a prior of precision $10^8$; the two agree on
the resulting trajectory.

### Damping, and why Gauss-Newton is enough here

Gauss-Newton takes the exact minimizer of the linearized cost, with no bound on how far that is.
When the linearization is only valid nearby, which is the case with a poor initialization or a
large rotation error, that step can overshoot and raise the true cost. Levenberg-Marquardt adds a
damping term to the system:

$$\big(H + \lambda\,\mathrm{diag}(H)\big)\,\delta = -g .$$

Large $\lambda$ lets the added diagonal dominate, which shrinks $\delta$ and turns it toward
$-g$, the steepest-descent direction, and a short enough steepest-descent step always lowers the
cost. Small $\lambda$ recovers Gauss-Newton and its fast final convergence. Levenberg-Marquardt
picks $\lambda$ by trial: compute the step, evaluate the true cost, and accept the step and lower
$\lambda$ if the cost fell, or reject it and raise $\lambda$ if it rose. Dogleg reaches the same
end with an explicit trust region of radius $\Delta$: it computes both the Gauss-Newton step and
the steepest-descent step, follows the two-segment path between them, and stops where that path
crosses the trust-region boundary, growing or shrinking $\Delta$ on the same accept-or-reject
test.

Gauss-Newton is the undamped $\lambda = 0$ case, and it is what this assignment implements,
because both synthetic graphs start close to the answer. The exact-recovery graph perturbs each
pose by 0.15 rad and 0.4 m, and the drift graph starts from integrated odometry, which is already
the right shape.

### Loop closure

Odometry drifts. Integrating noisy relative-pose measurements along a trajectory accumulates
error, so a robot that drives a loop comes back to a start pose that does not match where it
began.

Odometry edges alone cannot repair this, and the reason is worth stating precisely. The drifted
trajectory was constructed by chaining exactly those odometry measurements, so every odometry
residual is already zero at the initialization. The gradient $g$ is zero, the Gauss-Newton step is
zero, and the optimizer leaves the trajectory untouched no matter how many iterations it runs.
Drift is not a violated constraint. It is the accumulation of many individually satisfied ones.

A loop-closure edge is a measurement that the robot has returned to a place it has already been,
produced by re-recognizing a landmark or matching a laser scan against an earlier one, and it ties
a late pose directly back to an early one. On the drifted trajectory that one edge has a large
residual, and because it connects two poses that are far apart along the graph, correcting it
requires the whole loop between them to move. Gauss-Newton distributes the correction across all
of those poses at once, weighted by their information, and the trajectory snaps into a consistent
shape.

Running this assignment's drift graph, the end-of-trajectory position error drops by more than a
factor of ten once the loop-closure edge is included, and removing that single edge leaves the
error exactly where it started, to numerical precision. The test asserts both halves: at least a
factor-of-five reduction with the edge, and no change at all without it. Most of the correction
lands on the first Gauss-Newton iteration, with later iterations only polishing.

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
with the poses first, the whole matrix has what is called arrowhead structure - a dense square in
the top-left corner, a block-diagonal tail down the bottom-right, and dense cross bands joining
them.

Sizes are lopsided. A reconstruction has orders of magnitude more points than cameras, so almost
all of the system's dimension sits in the block-diagonal tail. Factoring the whole thing densely
is cubic in that total dimension and out of the question.

### The Schur complement in bundle adjustment

Partition the linear system solved inside one Gauss-Newton iteration, poses first:

$$\begin{bmatrix} H_{pp} & H_{p\ell} \\ H_{\ell p} & H_{\ell\ell} \end{bmatrix} \begin{bmatrix} \delta_p \\ \delta_\ell \end{bmatrix} = \begin{bmatrix} b_p \\ b_\ell \end{bmatrix}.$$

Eliminating the landmark block is the marginalization formula from earlier, read this time as
block Gaussian elimination on a linear system rather than on a Gaussian. Clearing $H_{p\ell}$ out
of the first block row leaves the reduced camera system

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
The test checks that directly, comparing against a dense `numpy.linalg.solve` of the same system
to $10^{-9}$, and separately checking that $H\delta - b$ has norm below $10^{-9}$.

Two conventions in the code differ from the pose-graph function.
`schur_solve` solves $H\,\delta = b$ with the right-hand side as given, so a caller assembling
normal equations passes $b = -g$; `optimize_pose_graph` applies that sign internally. And
`schur_solve` assumes the pose variables occupy the leading `n_pose` columns and that every
landmark block has the same size `lm_block`, which is how the test builds its system (6 variables
per pose block, 3 per landmark). The test's $H$ is a synthetic positive-definite matrix with the
landmark off-diagonal blocks zeroed rather than one assembled from real reprojection factors,
which is enough to exercise the linear algebra.

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
version of the residual and Jacobians implemented in this assignment, so the optional oracle
test is a meaningful comparison.

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
drift by at least a factor of five while removing it leaves the drift in place; and the
Schur-complement solve equals the dense solve on a BA-structured system. If GTSAM is installed,
`test_oracle_gtsam.py` cross-checks the optimized trajectory against GTSAM's optimizer; it is
skipped when GTSAM is absent.

`make viz` writes `out/pose_graph.rrd`. Add `SHOW=1` for the interactive viewer: the
ground-truth loop, the drifted trajectory snapping onto it as the Gauss-Newton iterations
replay, the loop-closure edge, a cost panel, and the block-sparsity of $H$ (the band plus the
off-corner the loop-closure edge creates). Scrub the timeline to watch the correction propagate.
The cost falls by roughly two orders of magnitude on the first iteration and then flattens at a
nonzero value rather than reaching zero, because that graph's measurements carry noise and cannot
all be satisfied at once; only the noise-free graph used by the recovery test drives the cost to
zero.

The pose graph is implemented on SE(3) (the trajectory here is planar, a loop with yaw, so it
reads as a 2D plot, but the code is the full 3D machinery and a SE(2) graph is the same
construction in three dimensions instead of six). The canonical real datasets for this method are
the g2o pose-graph benchmarks - real logs and simulated graphs distributed in a common text
format, including the Intel Research Lab and MIT Killian Court indoor runs and the synthetic
sphere and torus graphs used to stress 3D optimizers. The tests use a synthetic loop instead
because it gives the deterministic ground truth they need.

## In interviews

Factor-graph SLAM and bundle adjustment are standard interview topics for SLAM, AR, and
3D-reconstruction roles, both as derivations and as system-design discussions.

Filter versus smoother. Expect "why did factor graphs replace the EKF for SLAM?" The answer is
two things: smoothing keeps all poses and re-linearizes every iteration, so linearization error
does not get baked in the way it does when a filter marginalizes a pose; and the problem is
sparse, so keeping everything is affordable. Incremental smoothing (iSAM, iSAM2) makes it
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
transform, so the system matrix is singular; you fix it by anchoring one pose or adding a prior.
Not handling the gauge is a common bug - the solve fails or wanders. Related: Gauss-Newton
versus Levenberg-Marquardt versus dogleg (damping and trust regions, for tolerating a bad
initialization), and the production solvers g2o, GTSAM, and Ceres.

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
