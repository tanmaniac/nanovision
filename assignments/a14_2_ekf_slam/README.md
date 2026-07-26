# a14_2 - EKF-SLAM

The third assignment of the classical SLAM module, in C++17 with Eigen. It builds EKF-SLAM:
a single extended Kalman filter whose state is the robot pose and every landmark it has
mapped, estimated jointly. This is the historically first online SLAM method that worked,
and the one factor graphs later replaced. Building it shows both why a joint filter solves
the chicken-and-egg problem of mapping while localizing, and why its cost and its
linearization eventually made the field move on.

It builds directly on the extended Kalman filter from the Kalman-filter assignment: the
unicycle motion model, its Jacobian, and the range-bearing measurement model and its
Jacobians are carried over and provided here (in `models.cpp`), so the work here is the
SLAM-specific machinery on top of them.

Required reading before you start:
- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapter 10 - the EKF-SLAM
  derivation, the joint state, and the inverse measurement model for landmark
  initialization.
- Durrant-Whyte and Bailey, "Simultaneous localization and mapping (SLAM): part I" (2006),
  IEEE RAM - the problem statement and why the off-diagonal correlations are essential.

## Lecture notes

### Why map and pose must be estimated together

Localization assumes a known map and estimates the pose; mapping assumes a known pose and
estimates landmarks. SLAM has neither: the robot must build the map and locate itself in it
at the same time, and the two estimates are coupled. If the robot misjudges its own motion,
every landmark it places inherits that error, and when it later re-sees an early landmark,
that observation should correct the pose and, with it, the whole map built since.

Re-observing something seen long ago, after the pose estimate has drifted in between, is a
loop closure. It is the one event in a SLAM run that carries new information about the
accumulated drift: dead reckoning alone can only add uncertainty, and a measurement of a
landmark first seen five seconds ago says little about where the robot was thirty seconds
ago. A loop closure ties the current pose back to a part of the map built under a much
older, and much smaller, pose error. Making that correction reach backwards into the map
requires estimating the robot and the landmarks in one joint belief, together with the
correlations between them.

EKF-SLAM does this. It stacks the robot pose and all landmark positions into one state
vector and runs a single EKF over it. The state grows as new landmarks are seen:

$$\mu = \big[\,\underbrace{p_x,\,p_y,\,\theta}_{\text{robot}},\ \underbrace{\ell_{0x},\,\ell_{0y}}_{\text{lm }0},\ \underbrace{\ell_{1x},\,\ell_{1y}}_{\text{lm }1},\ \dots\,\big]^\top,$$

with the matching joint covariance $P$ of size $m \times m$, where $m = 3 + 2N$ for $N$
landmarks. Landmark $j$ is 0-indexed and lives at state indices $3+2j$ and $3+2j+1$, which
is exactly the layout `ekf_slam.hpp` documents and the code indexes with
`mu.segment<2>(3 + 2 * j)`.

### What the joint covariance stores

Write the joint covariance in blocks, robot first and map second:

$$P = \begin{bmatrix} P_{rr} & P_{rm} \\ P_{mr} & P_{mm} \end{bmatrix}.$$

$P_{rr}$ is $3\times3$ and holds the pose uncertainty, $P_{mm}$ is $2N \times 2N$ and holds
the landmark uncertainties, and $P_{rm} = P_{mr}^\top$ is the cross-covariance between them.
The operational meaning of a cross block comes straight from Gaussian conditioning. For a
jointly Gaussian pair $(a, b)$ with

$$\begin{bmatrix} a \\ b \end{bmatrix} \sim \mathcal{N}\!\left( \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}, \begin{bmatrix} P_{aa} & P_{ab} \\ P_{ba} & P_{bb} \end{bmatrix} \right),$$

learning the value of $b$ moves the estimate of $a$ by

$$\mathbb{E}[a \mid b] = \mu_a + P_{ab} P_{bb}^{-1} (b - \mu_b), \qquad \operatorname{Cov}(a \mid b) = P_{aa} - P_{ab} P_{bb}^{-1} P_{ba}.$$

Both expressions are gated by $P_{ab}$. If the cross block is zero, observing $b$ leaves the
mean of $a$ where it was and leaves its covariance at $P_{aa}$: the two variables cannot
inform each other at all. That is the entire argument for carrying the off-diagonal blocks.
Landmarks observed from the same uncertain pose share that pose's error, so their estimates
are correlated, and a later measurement that corrects one corrects the others through those
blocks. A block-diagonal approximation that drops the cross terms is not a cheaper
EKF-SLAM, it is a broken one: loop closure stops working, because the information from
re-seeing an old landmark has no path to the rest of the map.

Two names for pieces of that formula, both used later. The *marginal* covariance of a subset
of the state, meaning its covariance with everything else integrated out, is simply the
corresponding diagonal sub-block $P_{aa}$ - for a Gaussian, marginalizing is dropping rows
and columns. The *conditional* covariance $P_{aa} - P_{ab}P_{bb}^{-1}P_{ba}$ is the Schur
complement of $P_{bb}$ in $P$. Marginal and conditional are different matrices, and the
conditional is the smaller of the two.

Running this assignment's filter on the 300-step loop and reading off the final covariance,
the $2\times2$ landmark-landmark cross blocks have Frobenius norms between 0.06 and 0.16
against diagonal blocks between 0.07 and 0.17. The off-diagonal structure is the same
magnitude as the diagonal, so the map really is dense, not almost-block-diagonal.

### Propagating the belief through motion

A motion command moves the robot; the landmarks are static, so their mean does not change.
The mean update touches only the robot block, $\mu_r \leftarrow f(\mu_r, u)$, with $f$ the
unicycle model carried over from the Kalman-filter assignment.

The covariance is where SLAM differs from plain localization. The EKF rule is
$P \leftarrow F_{\text{full}} P F_{\text{full}}^\top + Q_{\text{full}}$, where
$F_{\text{full}}$ is the Jacobian of the motion model over the *whole* state. Since motion
maps the landmarks to themselves, that Jacobian is block diagonal,

$$F_{\text{full}} = \begin{bmatrix} F & 0 \\ 0 & I \end{bmatrix}, \qquad F = \frac{\partial f}{\partial \mu_r} \ \ (3\times3),$$

and $Q_{\text{full}}$ is zero outside the robot block, since a landmark that does not move
gains no process noise. Multiplying the block matrices out gives three statements:

$$P_{rr} \leftarrow F P_{rr} F^\top + Q, \qquad P_{rm} \leftarrow F P_{rm}, \qquad P_{mm}\ \text{unchanged.}$$

The cross block picks up one factor of $F$ and no transpose, because only one side of
$P_{rm}$ is a robot index. Nothing about $P_{rm}$ is optional: the robot's uncertainty grew,
and how that larger uncertainty lines up with each landmark's error has to be carried
forward or the map silently decorrelates from the robot and loop closure dies. This is one
`if (m > 3)` branch in `slam_predict`, and it is the easiest thing in the assignment to
leave out and not notice, because a filter without it still produces plausible-looking
trajectories.

### Initializing a landmark

The first time a landmark is seen it is not in the state, so it must be added. The forward
measurement model $h$ maps a state to a predicted reading; the *inverse* measurement model
goes the other way, taking the current pose and one reading and returning the world position
that would have produced it. For a range-bearing reading $z = [r, \phi]$ from pose
$(p_x, p_y, \theta)$, writing $\beta = \theta + \phi$ for the bearing in the world frame:

$$\ell = g(\mu_r, z) = \begin{bmatrix} p_x + r\cos\beta \\ p_y + r\sin\beta \end{bmatrix}.$$

Appending $\ell$ to the mean is easy. The covariance augmentation is the careful part, and
it is one application of the standard rule for pushing a Gaussian through a function. If
$\ell = g(\mu_r, z)$ with $\mu_r$ uncertain by $P_{rr}$ and $z$ uncertain by $R$, and the two
errors are independent, then to first order

$$P_{\ell\ell} = G_r P_{rr} G_r^\top + G_z R G_z^\top, \qquad G_r = \frac{\partial g}{\partial \mu_r}, \quad G_z = \frac{\partial g}{\partial z}.$$

A new landmark's uncertainty is the pose uncertainty pushed through the geometry plus the
sensor noise pushed through the geometry, and nothing else. For this $g$,

$$G_r = \begin{bmatrix} 1 & 0 & -r\sin\beta \\ 0 & 1 & r\cos\beta \end{bmatrix}, \qquad G_z = \begin{bmatrix} \cos\beta & -r\sin\beta \\ \sin\beta & r\cos\beta \end{bmatrix}.$$

The $r\sin\beta$ and $r\cos\beta$ terms say that a heading error at long range throws the
landmark sideways in proportion to the range, which is why distant landmarks come in with
long, thin, tangentially oriented uncertainty ellipses.

The new landmark also needs its cross-covariance with everything already in the state. Since
$\ell$ depends on the existing state only through $\mu_r$, the same first-order rule gives

$$P_{\ell x} = G_r P_{r,:},$$

where $P_{r,:}$ is the three robot rows of the current covariance ($2 \times m$ in the code,
`Gr * P.topRows<3>()`). The augmented covariance puts the old $P$ in the top-left,
$P_{\ell\ell}$ in the new bottom-right $2\times2$ block, and $P_{\ell x}$ and its transpose
in the new off-diagonal strips. A landmark is therefore born already correlated with the
robot and, through the robot's existing correlations, with every landmark already mapped.

### The measurement update over the joint state

To fold in an observation of a mapped landmark $j$, predict the reading
$h(\mu)$ - range and bearing from the robot to landmark $j$ - and form the innovation
$y = z - h(\mu)$, the difference between what the sensor said and what the filter expected.
The bearing component of $y$ is an angle difference and must be wrapped to $(-\pi, \pi]$, or
a reading at $+179^\circ$ against a prediction at $-179^\circ$ produces a $358^\circ$ error
and a violent spurious correction.

The measurement Jacobian $H$ is $2 \times m$ and almost entirely zero: a $2\times3$ block in
the robot columns and a $2\times2$ block in landmark $j$'s columns, both carried over from
the range-bearing model. Everything else is the standard EKF update, run over the full joint
state:

$$S = H P H^\top + R, \quad K = P H^\top S^{-1}, \quad \mu \leftarrow \mu + Ky, \quad P \leftarrow (I - KH)P(I-KH)^\top + KRK^\top.$$

The gain is the Gaussian conditioning coefficient from earlier with $b$ set to the predicted
measurement: $P H^\top$ is the cross-covariance between the state and that prediction, and
$S$ is the prediction's own covariance, so $K = P H^\top S^{-1}$ is exactly the
$P_{ab}P_{bb}^{-1}$ of the conditioning formula. $H$ is sparse but $P$ is dense, so $PH^\top$
and therefore $K$ are dense: every state component has a nonzero row in $K$, so a single
observation of one landmark corrects the robot and, through the correlations, every other
landmark. That mechanism is loop closure - re-seeing landmark 0 after a long drive tightens
the estimate of every landmark mapped in between.

The covariance line is the Joseph form rather than the shorter $P \leftarrow (I - KH)P$. The
two are algebraically equal at the optimal gain, but the short form subtracts two nearly
equal matrices and loses symmetry and positive-definiteness to round-off over a long run,
while the Joseph form is a sum of two symmetric positive semidefinite terms and stays
symmetric positive definite no matter how much cancellation happens inside it.

### A measurement adds information, in the Loewner order

The update has a monotonicity property worth stating exactly, because the test suite checks
it and because it explains what the suite deliberately does not check.

For symmetric matrices, $A \succeq B$ means $A - B$ is positive semidefinite, equivalently
that $v^\top A v \ge v^\top B v$ for every $v$. This is the Loewner order. It is a partial
order rather than a total one: two covariance matrices can easily be incomparable, each
larger than the other along some direction.

Carry the belief in the information matrix $\Omega = P^{-1}$, the dual coordinate from the
Kalman-filter assignment, where a measurement update is purely additive:

$$\Omega^+ = \Omega^- + H^\top R^{-1} H.$$

$H^\top R^{-1} H$ is positive semidefinite for any $H$, so $\Omega^+ \succeq \Omega^-$: a
measurement never removes information, and adds none in directions $H$ cannot see (those
directions get exactly zero). Matrix inversion reverses the Loewner order on positive
definite matrices, so $P^+ \preceq P^-$. Three consequences follow, and the first two are
what `tests/test_update_covariance.py` asserts. The determinant of the joint covariance
cannot increase. The information matrix cannot decrease in the Loewner order. And every
principal sub-block of $P$ - that is, every marginal, including each single landmark's
$2\times2$ block - cannot increase either, so no marginal determinant can grow.

What does *not* follow is that any particular marginal strictly shrinks, and the difference
matters for what the suite can assert. Take the suite's own two-landmark fixture: a robot at
a fixed pose initializes two landmarks, then re-observes landmark 0 without having moved.
The joint covariance determinant drops by a factor of four. Landmark 1's $2\times2$ block
comes back unchanged, and so does the robot's $3\times3$ block. The only change anywhere is
in landmark 0's own block, and it is exactly $\frac{1}{2} G_z R G_z^\top$ - the second
reading of the same relative offset averaged with the first, halving that half of
$P_{\ell\ell}$ while leaving the $G_r P_{rr} G_r^\top$ half alone.

That is not a numerical accident. Range and bearing measure the offset from the robot to a
landmark, never an absolute position, so no amount of measuring from a stationary pose says
anything about where the pair actually is. The absolute frame is unobservable, and the
unobservable directions are exactly the ones whose uncertainty refuses to shrink.

### Mahalanobis distance and the chi-square gate

The update above assumed the correct landmark index $j$ was known. Before dropping that
assumption, here is the statistic that decides it.

Start with the scalar case. A residual $y$ with variance $s$ is judged by its z-score
$y / \sqrt{s}$: a residual of one meter is unremarkable if $\sqrt{s}$ is a meter and absurd
if $\sqrt{s}$ is a centimeter. Squaring gives $y^2 / s$, the squared error divided by how
much error was expected.

In $k$ dimensions the residual has a covariance matrix $S$ rather than a variance, and
dividing by a standard deviation becomes multiplication by $S^{-1/2}$, any matrix satisfying
$S^{-1/2} S S^{-\top/2} = I$ (a Cholesky factor of $S^{-1}$ will do). Set $w = S^{-1/2} y$.
Then $\operatorname{Cov}(w) = I$: $w$ is $k$ independent standard normals, whatever
correlations and scale differences $S$ had. This is whitening, and it makes the components
of $w$ directly comparable, which the components of $y$ were not. The natural size of a
whitened residual is its squared length,

$$d^2 = w^\top w = y^\top S^{-1} y,$$

the squared Mahalanobis distance. Geometrically it is the number of standard deviations, but
measured along the direction the residual actually points, so a residual that lies along a
direction where $S$ is wide counts for little and the same residual across a narrow
direction counts for a lot.

Because $d^2$ is a sum of $k$ squared independent standard normals, it is by definition
chi-square distributed with $k$ degrees of freedom, which has mean $k$ and variance $2k$.
That fixes the threshold. A range-bearing innovation is 2-dimensional, so under a correct
association and a correctly calibrated filter $d^2 \sim \chi^2_2$, and the 95th percentile of
$\chi^2_2$ has the closed form $-2\ln(1 - 0.95) \approx 5.991$. `config.py` computes exactly
this as `chi2.ppf(0.95, df=2)` and calls it `GATE`.

The number carries a meaning that a hand-tuned threshold would not: about 5% of genuinely
correct associations fall outside the gate and get rejected. Choosing 0.99 instead
($\approx 9.21$) rejects only 1% of true matches at the cost of admitting more wrong ones.
Picking the percentile is picking a false-reject rate.

### Data association

In a real run the landmark index behind a measurement is unknown, and getting it wrong is
the dominant failure mode of SLAM: one bad association welds two parts of the map together
and the error never comes out. The nearest-neighbor test computes, for every mapped landmark
$j$, the innovation $y_j = z - h_j(\mu)$ (bearing wrapped), its covariance
$S_j = H_j P H_j^\top + R$, and then the squared Mahalanobis distance
$d_j^2 = y_j^\top S_j^{-1} y_j$.
The measurement goes to whichever landmark has the smallest $d_j^2$, provided that
value is strictly below the gate; otherwise `slam_associate` returns $-1$ and the caller
initializes a new landmark instead. An empty map returns $-1$ immediately, since there is
nothing to match against.

Note that $S_j$ contains $H_j P H_j^\top$, not just $R$. The gate therefore widens on its
own when the robot is unsure of its pose or of that landmark, which is precisely when a
prediction deserves less trust. A gate on raw Euclidean distance has no such behavior and
either rejects good matches after a long blind drive or accepts bad ones when the pose is
well known.

Nearest-neighbor gating tests each pairing on its own, and that is its weakness. Suppose two
measurements in the same scan both land within the gate of the same landmark, or a set of
pairings is individually plausible but geometrically impossible together because the implied
relative positions contradict each other. Joint compatibility branch and bound (JCBB, Neira
and Tardos 2001) fixes this by testing whole hypotheses: stack $n$ measurements into one
$2n$-vector, stack the corresponding predictions, build the full $2n \times 2n$ innovation
covariance including the cross terms between the pairings, and gate the resulting $d^2$
against $\chi^2_{2n}$. The cross terms are what individual gating discards, and they are
what rules out sets of pairings that each look fine alone. The search over hypotheses is
exponential, hence the branch and bound.

### Consistency and NEES

A filter is consistent when its reported covariance honestly describes its actual error - not
when the error is small, but when the error is as large as the filter claims it might be. An
optimistic (overconfident) filter reports a covariance smaller than its true error, and that
is dangerous in a way that a merely inaccurate filter is not: downstream consumers weight the
estimate by the reported covariance, so an overconfident SLAM back-end causes a planner to
trust a pose it should have questioned.

The statistic that measures it is the NEES, the normalized estimation error squared, which
is the Mahalanobis distance again with the state error in place of the innovation. Take the
robot part of the
state, the true pose $x_r$ (available in simulation, not on a real robot), and the error
$e = \mu_r - x_r$ with its heading component wrapped. Then

$$\text{NEES} = e^\top P_{rr}^{-1} e.$$

If the filter is telling the truth, $e \sim \mathcal{N}(0, P_{rr})$, so by the same whitening
argument NEES is $\chi^2_3$ for a 3-DOF pose: its expected value is 3. Above 3 on average
means the true error is larger than the reported covariance admits, which is optimism. Below
3 means conservatism, which wastes information but is not unsafe. The 95th percentile of
$\chi^2_3$ is $\approx 7.815$, and that is the bound `viz.py` plots alongside the NEES trace.
`robot_nees` in `_helpers.py` is those two lines of code.

EKF-SLAM is known to go optimistic, for two reasons that reinforce each other. The first is
generic to the EKF: $F$, $H$, $G_r$, and $G_z$ are first-order Taylor expansions taken at the
current estimate rather than at the truth, and the neglected curvature is never charged to
$P$ anywhere, so the filter under-reports its own approximation error and does so
cumulatively over thousands of steps. The second is specific to SLAM, and it is the
unobservable absolute frame from the update section. Because the true system cannot observe
global position and heading, a correct filter should never gain information in those three
directions. The linearized filter does gain some, because successive updates evaluate their
Jacobians at slightly different and slightly wrong estimates, and that inconsistency between
linearization points leaks information into directions where none exists. Julier and Uhlmann
(2001) gave the first explicit counterexample, and Huang and Dissanayake (2007) traced the
mechanism to exactly this observability mismatch.

What you will see running this assignment's 300-step loop with the provided seed: the robot
NEES starts far below 3 and climbs. Averaged over six equal segments of the run it goes
roughly $0.2 \to 0.9 \to 1.9 \to 3.0 \to 4.5 \to 1.7$. The early values below 3 are the
filter being conservative rather than optimistic: `runner.py` starts the filter at the exact
true pose but with `INIT_P_DIAG` variances of $(0.05, 0.05, 0.03)$, so it claims about 0.22 m
of position uncertainty while having none, and the ratio in the NEES starts near zero. That
start pose also defines the map frame, since nothing in the run measures absolute position,
so errors stay small nearby and grow with distance from it. The climb past 3 on the far side of the
loop is the optimism: at a segment mean of 4.5 the squared pose error is about 1.5 times what
$P_{rr}$ predicts, so the reported standard deviations are roughly 20% too small. Individual
steps cross the 7.815 bound only three times in 300. The final segment drops back to 1.7 as
the robot re-sees its first landmarks and the loop closure re-tightens the map. Over the same
run the mean mapped-landmark error, the second scalar panel in the viz, peaks near 0.72 m
about two-thirds of the way around and falls to about 0.25 m by the end. This is one seed on
a small simulated loop, not a measurement of how badly EKF-SLAM diverges at scale; the
literature above is the source for that.

### Why the cost killed it

Count the work in one update. $H$ is $2 \times m$ with only five nonzero columns, so $HP$
costs $O(m)$, $S$ is $2\times2$, and $K = PH^\top S^{-1}$ is $m \times 2$ and also costs
$O(m)$. The covariance correction is where the cost lives: it changes every one of the $m^2$
entries of $P$, because $K$ is dense. So EKF-SLAM is $\Theta(m^2) = \Theta(N^2)$ per update,
and no cleverness about the sparsity of $H$ removes that - the output itself is $N^2$ numbers.

Put numbers on it. At $N = 1000$ landmarks, $m = 2003$ and $P$ holds about 4.0 million
doubles, 32 MB. Ten observations per step at 10 steps per second means 100 updates per
second, so the filter rewrites 3.2 GB of covariance every second, before any arithmetic. That
is the wall EKF-SLAM hits, and it is why the method is remembered as a few-hundred-landmark
technique.

The reference implementation here is slower still: it forms the dense $m\times m$ matrix
$I - KH$ and multiplies it out twice, which is $\Theta(m^3)$. Written as a rank-2 correction
the same update is $\Theta(m^2)$. With $N = 10$ in this simulation, $m = 23$ and it makes no
difference, and the Joseph form written literally is much easier to read and to check against
the formula.

Cost is only half the story, and the structural half is more interesting. Consider the same
problem as a graph: one variable per robot pose and one per landmark, one factor per
measurement. Its information matrix has a nonzero block only where two variables appear in a
common measurement, so it is sparse - each pose touches a handful of landmarks. A filter
throws that away. Filtering keeps only the *current* pose and marginalizes out every past
one, and marginalizing a variable out of an information matrix is a Schur complement, the
same operation from the conditioning formula earlier. A Schur complement fills in: every pair
of variables the marginalized pose touched becomes directly connected. Marginalize away a
whole trajectory and the surviving landmarks end up connected to each other, which is the
dense covariance from the first section, seen from the other side.

Smoothing keeps all the poses as variables instead of marginalizing them. That keeps the
sparsity, and it also allows relinearizing every factor at the improved estimate on each
iteration rather than being stuck with the Jacobian computed once at whatever the estimate
happened to be at the time. Both of EKF-SLAM's problems are addressed by the same change,
which is the subject of the factor-graph and bundle-adjustment assignment later in this
module.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the declaration and comments in each file give the signature and shapes.

1. [`slam_predict()`](ekf_slam.cpp) in `ekf_slam.cpp`
2. [`slam_add_landmark()`](ekf_slam.cpp) in `ekf_slam.cpp`
3. [`slam_update()`](ekf_slam.cpp) in `ekf_slam.cpp`
4. [`slam_associate()`](ekf_slam.cpp) in `ekf_slam.cpp`

You may not include an existing SLAM or solver library; a test scans the sources.

### Building and running

Same toolchain as the rest of the module (C++17, CMake, pybind11, Eigen). You never call
CMake by hand.

```
make verify A=a14_2_ekf_slam   # build + run the reference (solution/); the green target
make test   A=a14_2_ekf_slam   # build + run YOUR code; red until the holes are filled
make viz    A=a14_2_ekf_slam        # render the EKF-SLAM loop (reference)
make viz-mine A=a14_2_ekf_slam      # the same, from YOUR code, once the holes are filled
```

The tests favor implementation-independent statements. Landmark initialization places a
landmark from a noise-free reading at its true position and leaves the existing robot block
untouched; the update never increases the joint covariance determinant and never decreases
the information matrix in the Loewner order, both from the "a measurement adds information"
argument above; the data-association gate matches a realistically noisy observation to the
right landmark and rejects a spurious one; after a run the landmark-landmark cross blocks
are non-negligible, so the map really did become correlated; on a short mild trajectory the
mean robot NEES stays inside a loose band around its expectation of 3 (the assertion is
$0 < \overline{\text{NEES}} < 6$, deliberately generous, since the long-loop optimism is a
demonstration and not something to gate on); and the mapped landmarks track the truth.

The per-landmark marginal determinant is checked by no test, even though the Loewner
argument says it cannot increase. The reason is that it need not *decrease*: on the suite's
own two-landmark fixture, re-observing landmark 0 from the pose it was initialized at leaves
landmark 1's marginal exactly where it was, for the unobservability reason given above. An
assertion of strict decrease would fail on correct code.

`make viz` writes `out/ekf_slam.rrd`. Add `SHOW=1` for the interactive viewer: ground truth,
the robot and landmark estimates with their 3-sigma ellipses, the active measurement rays,
and two scalar panels (robot NEES against its chi-square bound, and mean mapped-landmark
error). A 3-sigma ellipse is the level set $\{x : (x - \mu)^\top \Sigma^{-1} (x - \mu) = 9\}$
of a 2D marginal, so its axes point along the eigenvectors of $\Sigma$ with lengths
$3\sqrt{\lambda_i}$ - the same quadratic form as the Mahalanobis gate, drawn instead of
thresholded. Scrub to the far side of the loop to watch the NEES rise, and to the end to
watch loop closure pull it back down and tighten the whole map at once.

The simulated world is a point-landmark range-bearing loop, the standard EKF-SLAM teaching
environment. The canonical real dataset for this method is Victoria Park (a vehicle driving a
park, range-bearing to tree trunks); the tests do not gate on it because its ground truth is
sparse GPS and its data association is genuinely ambiguous, which is exactly what makes a
deterministic test fragile.

## In interviews

EKF-SLAM is a standard interview topic for perception and robotics roles, often as the lens
for "how does SLAM actually work."

What is in the state, and why the off-diagonal blocks matter. The answer is the robot pose
and all landmarks in one joint Gaussian, and the cross-covariance blocks are what let a loop
closure correct the whole map. Be ready for the follow-up: what happens if you drop the
correlations and keep only block diagonals? Loop closure breaks, because the Gaussian
conditioning update is proportional to the cross block, so a zero cross block means
re-observing an old landmark cannot move anything else.

Why EKF-SLAM is $O(N^2)$ and what that killed. The joint covariance is dense and every update
rewrites all of it, so it does not scale; this is the practical reason the field moved to
sparse, graph-based back-ends. Know that the information matrix of a pose graph is sparse
while its inverse, the covariance, is dense, and that filtering destroys the sparsity by
marginalizing out old poses, which fills in edges between everything they touched.

Consistency. Expect "is EKF-SLAM consistent?" The honest answer is no, not in general: it
linearizes at the current estimate and becomes optimistic as that error accumulates, which
is one of the two reasons (with cost) that factor-graph smoothing replaced it. Being able to
define NEES as $e^\top P^{-1} e$ and say its expected value is the state dimension is a
strong signal.

Data association as the real failure mode. The filter math is the easy part; deciding which
measurement is which landmark is what breaks real systems. Know the Mahalanobis gate, why its
threshold is a chi-square quantile at the innovation's dimension, and that harder settings
need joint compatibility (JCBB) or back-ends with outlier-rejecting cost functions. A common
on-the-spot question is simply
"what goes in the state vector and how does it grow," which is the augmentation step here.

## Further reading

- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapter 10 - the full EKF-SLAM
  derivation and the inverse measurement model.
- Durrant-Whyte and Bailey, "SLAM: part I" (2006) and Bailey and Durrant-Whyte, "SLAM: part
  II" (2006) - the problem, the correlations, and data association including JCBB.
- Neira and Tardos, "Data association in stochastic mapping using the joint compatibility
  test" (2001), IEEE Trans. Robotics and Automation - JCBB.
- Julier and Uhlmann, "A counter example to the theory of simultaneous localization and map
  building" (2001), ICRA - the first explicit demonstration of EKF-SLAM inconsistency.
- Huang and Dissanayake, "Convergence and consistency analysis for extended Kalman filter
  based SLAM" (2007), IEEE Trans. Robotics - the observability account of where the optimism
  comes from.
- Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation*
  (2001) - NEES and NIS as the standard consistency tests for a filter.
- Cadena et al., "Past, present, and future of SLAM: toward the robust-perception age"
  (2016) - where filtering sits relative to the smoothing methods that replaced it.
