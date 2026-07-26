# a14_1 - Kalman filtering: KF, EKF, UKF, and the information form

The second assignment of the classical SLAM module, again in C++17 with Eigen. It covers the
Gaussian filters at the center of classical state estimation: the linear Kalman filter (KF),
the extended Kalman filter (EKF) for nonlinear motion and measurement models, the unscented
Kalman filter (UKF), and the information (canonical) form. The unicycle motion model, the
range-bearing measurement model, and their Jacobians written here reappear in the EKF-SLAM
assignment, which carries them over as provided code and builds a joint robot-plus-map filter
on top.

The Python around the C++ (the build, the tests, the Rerun visualization) is provided, as in
the Lie-group assignment. The UKF's sigma-point loop is also provided in Python; the graded
C++ is the unscented-transform math it calls (see the assignment section).

Required reading before you start:
- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapter 3 - the Gaussian filters
  (KF, EKF, information filter) derived from the Bayes filter, and the cleanest statement
  of the recursion.
- Julier and Uhlmann, "Unscented filtering and nonlinear estimation," Proceedings of the
  IEEE 92(3), 2004 - the sigma-point transform and the $\alpha,\beta,\kappa$ weights.

## Lecture notes

### The Bayes filter

A filter tracks a hidden state $x$ (a robot pose, a velocity, a landmark position) that is
never observed directly. What it stores is a belief: the probability density over the state
given everything seen so far,

$$b_k(x) = p\big(x_k = x \mid z_{1:k},\, u_{1:k}\big),$$

with $u_k$ the control applied between step $k-1$ and step $k$ and $z_k$ the measurement at
step $k$. Two modeling assumptions make this belief updatable one step at a time. The state is
Markov: the next state depends on the previous state and the current control only, so
$p(x_k \mid x_{0:k-1}, u_{1:k}) = p(x_k \mid x_{k-1}, u_k)$. And a measurement depends only on
the state it was taken from, so $p(z_k \mid x_{0:k}, z_{1:k-1}) = p(z_k \mid x_k)$. Together
they mean the past enters only through $b_{k-1}$, never through the raw history.

Under those two assumptions the recursion is two lines. Prediction pushes the belief through
the motion model,

$$\bar b_k(x) = \int p(x \mid x', u_k)\, b_{k-1}(x')\, dx',$$

which says the chance of landing at $x$ is the chance of having been at $x'$ times the chance
of moving from $x'$ to $x$, summed over every $x'$. Correction folds in the measurement by
Bayes' rule,

$$b_k(x) = \frac{p(z_k \mid x)\, \bar b_k(x)}{\int p(z_k \mid x')\, \bar b_k(x')\, dx'}.$$

Nothing has been approximated: this is the exact posterior, for any models and any noise. The
problem is that both lines are operations on functions. Running them needs a representation of
an arbitrary density over the state space and an integral over that whole space at every step,
and in general neither is available in closed form.

Two cases escape. If the state takes finitely many values the integrals become sums and the
belief is a table of probabilities, which is the histogram filter. If the belief is Gaussian
and the models are linear, both lines stay Gaussian and reduce to matrix algebra, which is the
Kalman filter and the subject of this assignment. Particle filters keep the generality by
giving up exactness, representing $b_k$ by weighted samples.

### Why Gaussians close the recursion

A Gaussian belief is written $x \sim \mathcal{N}(\mu, P)$ with density

$$p(x) = \frac{1}{\sqrt{(2\pi)^n \det P}} \exp\!\Big(-\tfrac12 (x-\mu)^\top P^{-1} (x-\mu)\Big),$$

so the entire belief is carried by a mean vector $\mu$ and a covariance matrix $P$: $n$ numbers
plus $n(n+1)/2$ numbers, fixed forever, no matter how long the filter runs. Three standard
properties of the Gaussian family turn the two integral lines above into arithmetic on those
two objects.

The first is behavior under an affine map. If $x \sim \mathcal{N}(\mu, P)$ and $y = Ax + b$,
then $y$ is Gaussian with mean $A\mu + b$ and covariance

$$\operatorname{Cov}(Ax + b) = \mathbb{E}\big[A(x-\mu)(x-\mu)^\top A^\top\big] = A P A^\top,$$

straight from the definition of covariance. The second is that independent Gaussians add: the
sum of $\mathcal{N}(\mu_1, P_1)$ and an independent $\mathcal{N}(0, Q)$ is
$\mathcal{N}(\mu_1, P_1 + Q)$. Those two together are the prediction step.

The third is conditioning, and it is the whole update step. Stack the state and the
measurement into one jointly Gaussian vector,

$$\begin{bmatrix} x \\ z \end{bmatrix} \sim \mathcal{N}\!\left( \begin{bmatrix} \mu_x \\ \hat z \end{bmatrix},\ \begin{bmatrix} P_{xx} & P_{xz} \\ P_{xz}^\top & P_{zz} \end{bmatrix} \right).$$

Conditioning on an observed value of $z$ gives another Gaussian, with

$$\mu^+ = \mu_x + K\,(z - \hat z), \qquad P^+ = P_{xx} - K\,P_{zz}\,K^\top, \qquad K = P_{xz}\,P_{zz}^{-1}.$$

Read the gain $K = P_{xz}P_{zz}^{-1}$ as a regression coefficient: it is the least-squares
slope of $x$ on $z$, so $K(z-\hat z)$ is the best linear prediction of how far the state sits
from its prior mean given how far the measurement landed from its predicted value. The
correction is proportional to how strongly state and measurement covary and inversely
proportional to how uncertain the measurement is.

Every filter in this assignment computes exactly those three expressions. They differ only in
how they obtain the predicted measurement $\hat z$ and the two blocks $P_{xz}$ and $P_{zz}$:
the KF gets them from a matrix $H$, the EKF from a Jacobian of a nonlinear $h$, and the UKF
from sample points pushed through $h$. Keeping that in view makes the UKF's update, which
looks like a separate set of formulas, the same formula with different ingredients.

### Covariance, ellipses, and positive definiteness

A covariance $P$ is symmetric, and for any direction $v$ the scalar $v^\top P v$ is the
variance of the state component along $v$, so it cannot be negative. A matrix that is
symmetric with $v^\top P v > 0$ for every nonzero $v$ is called symmetric positive definite,
abbreviated SPD throughout this module. Positive semidefinite (PSD) is the same with $\ge 0$,
allowing directions of exactly zero variance.

The geometry is an ellipse. The level sets of the Gaussian density are the sets where the
quadratic form in the exponent is constant,

$$\{x : (x - \mu)^\top P^{-1} (x - \mu) = c^2\},$$

an ellipsoid centered at $\mu$ whose axes point along the eigenvectors of $P$ and whose
semi-axis lengths are $c\sqrt{s_i}$ for the corresponding eigenvalues $s_i$. The
visualization's 1-sigma ellipse is this set with $c = 1$ in the two position coordinates; in
two dimensions it contains $1 - e^{-1/2} \approx 39\%$ of the probability mass, not the $68\%$
that the one-dimensional interval contains.

All of that requires SPD, and a filter that lets it slip breaks in concrete ways. $P^{-1}$
stops existing, so the density and the information form are undefined. The Cholesky
factorization that the UKF uses to build its sample points fails outright. And a negative
eigenvalue means the filter is claiming negative variance along some direction, which usually
shows up as a gain that overcorrects and an estimate that runs away. Two things below turn on
this: the Joseph form is chosen because it keeps $P$ symmetric and PSD under round-off, and
the UKF cannot build its point set at all without a valid Cholesky factor.

### The linear Kalman filter

When the motion and measurement are linear with additive Gaussian noise,

$$x_{k} = F x_{k-1} + B u_k + w_k, \quad w_k \sim \mathcal{N}(0, Q), \qquad z_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R),$$

the belief stays exactly Gaussian, by the three properties above. Prediction is an affine map
plus independent noise:

$$\mu^- = F\mu + B u, \qquad P^- = F P F^\top + Q.$$

$F P F^\top$ is the affine-map rule, and $+Q$ is the process noise that makes the robot less
certain the longer it moves without a measurement.

The update is the conditioning formula. With $z = Hx + v$ and $v$ independent of $x$, the
three ingredients are immediate: the predicted measurement is $\hat z = H\mu^-$, the
cross-covariance is $P_{xz} = P^- H^\top$, and the measurement covariance is
$P_{zz} = H P^- H^\top + R$. Substituting into $K = P_{xz}P_{zz}^{-1}$ gives the familiar
names. The innovation $y = z - H\mu^-$ is what the measurement says minus what was predicted,
its covariance is $S = H P^- H^\top + R$, the gain is $K = P^- H^\top S^{-1}$, and

$$\mu^+ = \mu^- + K y, \qquad P^+ = (I - K H) P^- (I - K H)^\top + K R K^\top.$$

The scalar case makes the gain concrete. With a one-dimensional state observed directly
($H = 1$), $K = P/(P + R)$. If the prior variance is $P = 1.0$ and the measurement variance is
$R = 0.25$, then $K = 0.8$: the estimate moves $80\%$ of the way toward the measurement,
because the measurement is four times more certain than the prior. The posterior variance
$(1-K)P = 0.2$ is smaller than either input. As $R \to 0$ the gain goes to 1 and the filter
believes the sensor completely; as $R \to \infty$ the gain goes to 0 and the measurement is
ignored.

Chain that scalar update with a static state ($F = 1$, $Q = 0$) and the filter turns into
averaging. Each update adds $1/R$ to the inverse variance, so after $n$ measurements
$P_n = (1/P_0 + n/R)^{-1}$, which tends to $R/n$: the variance of the mean of $n$ independent
readings. The test `test_kf_static_scalar_converges` checks precisely this, running $400$
updates with $R = 0.25$ and asserting the variance lands within $10^{-4}$ of $R/n$.

In what sense is the KF optimal. For a linear-Gaussian system it is not an approximation at
all; it reproduces the exact Bayes posterior, and its mean is therefore the minimum
mean-squared-error estimate of the state. Drop the Gaussian assumption but keep the linear
models and the noise covariances, and the same equations still give the minimum-variance
estimate among all estimators that are linear in the measurements.

### Why the Joseph form

The covariance update above is written in the Joseph form, not the shorter
$P^+ = (I - KH)P^-$. The long form comes straight from tracking the estimation error. Write
$e^- = x - \mu^-$ and $e^+ = x - \mu^+$. Substituting $\mu^+ = \mu^- + K(Hx + v - H\mu^-)$
gives

$$e^+ = (I - KH)\,e^- - K v,$$

and since $e^-$ and $v$ are independent, the covariance of that expression is
$(I - KH)P^-(I - KH)^\top + K R K^\top$. This holds for any gain $K$, optimal or not. The
long form collapses to the short form after substituting the specific optimal
$K = P^-H^\top S^{-1}$, so it is correct only at that exact gain.

Both are algebraically equal at the optimal gain, but they behave differently in floating
point. The short form subtracts two nearly equal matrices, which loses significant digits and
leaves nothing enforcing symmetry, so over a long run round-off can drift $P$ into a
non-symmetric or indefinite matrix. The Joseph form is a sum of two terms of the shape
$MNM^\top$ with $N$ SPD, each of which is symmetric and PSD by construction, so the sum stays
symmetric and stays PSD no matter what the arithmetic does. It costs a couple of extra matrix
products, and both the KF and the EKF here use it.

### The extended Kalman filter

Real motion and measurement models are nonlinear: a robot turns, a range sensor computes a
square root. Push a Gaussian through a nonlinear function and the result is not Gaussian, so
the recursion no longer closes. The EKF forces it closed by replacing the function with its
first-order Taylor expansion at the current mean, which is affine, and affine maps do keep
Gaussians Gaussian.

For a nonlinear motion model $f(x, u)$ and measurement model $h(x)$, the predict and update
are the KF equations with the Jacobians

$$F_x = \left.\frac{\partial f}{\partial x}\right|_{\mu}, \qquad H = \left.\frac{\partial h}{\partial x}\right|_{\mu},$$

while the mean itself goes through the exact nonlinear model: $\mu^- = f(\mu, u)$ and
$y = z - h(\mu^-)$. The covariance still propagates as $F_x P F_x^\top + Q$, and the
measurement blocks are still $P_{xz} = P^- H^\top$ and $P_{zz} = H P^- H^\top + R$.

This assignment's demo is a planar robot with state $x = [p_x, p_y, \theta]$ and a unicycle
model: a vehicle that can drive forward and turn but cannot slide sideways, with control
$u = [v, \omega]$ (forward speed and turn rate). Integrated with a forward Euler step, which
holds the heading fixed at its value at the start of the interval,

$$p_x' = p_x + v\,\Delta t\cos\theta, \quad p_y' = p_y + v\,\Delta t\sin\theta, \quad \theta' = \theta + \omega\,\Delta t.$$

Exact integration of constant $v, \omega$ traces a circular arc; the forward Euler step
replaces that arc with a straight segment along the starting heading, so it carries a
discretization error that grows with $\omega\,\Delta t$. That error does not reach the filter
in this demo, because `simulate_unicycle` generates the ground truth with the same forward
Euler expressions; the only motion error the filter faces is the injected process noise.

The measurement is a range and bearing to a known landmark at $\ell = [\ell_x, \ell_y]$. With
$dx = \ell_x - p_x$, $dy = \ell_y - p_y$, and $q = dx^2 + dy^2$,

$$h(x) = \begin{bmatrix} r \\ \phi \end{bmatrix} = \begin{bmatrix} \sqrt{q} \\ \operatorname{atan2}(dy, dx) - \theta \end{bmatrix},$$

so $r$ is the distance to the landmark and $\phi$ is its direction measured relative to where
the robot is pointing, which is what a bearing sensor bolted to the vehicle reports. The
Jacobians are

$$F_x = \begin{bmatrix} 1 & 0 & -v\,\Delta t\sin\theta \\ 0 & 1 & v\,\Delta t\cos\theta \\ 0 & 0 & 1 \end{bmatrix}, \qquad H = \begin{bmatrix} -dx/r & -dy/r & 0 \\ dy/q & -dx/q & -1 \end{bmatrix}.$$

The bearing $\phi$ is an angle, so its residual must be wrapped to $(-\pi, \pi]$: a
measurement at $+179^\circ$ and a prediction at $-179^\circ$ are $2^\circ$ apart, not
$358^\circ$, and an unwrapped innovation injects a huge spurious correction. The heading state
is likewise kept wrapped. This is the most common EKF-on-a-pose bug; `wrap_angle` (provided)
handles it, and it belongs in the residual and after the mean update. The range-bearing model
is also singular at zero range: $H$ has $1/r$ and $1/r^2$ terms, so a landmark at the robot's
exact position has undefined bearing and an infinite Jacobian. Real sensors never report zero
range and the simulation keeps the landmark at least a couple of meters away, so the code does
not guard it, but it is the kind of degeneracy worth naming.

The EKF's weakness is the linearization. $F_x$ and $H$ describe the model only in a small
neighborhood of $\mu$, and the covariance says the state might be anywhere in an ellipse
around $\mu$. When the model curves appreciably over that ellipse, the linear map is a poor
stand-in and the propagated covariance is wrong, usually too small.

Too small is the dangerous direction, and the word for the failure is inconsistency. A filter
is consistent when its reported uncertainty matches its actual errors: the innovation
$y = z - h(\mu^-)$ should look like a zero-mean draw from $\mathcal{N}(0, S)$, which is
checkable because the squared innovation $y^\top S^{-1} y$ then averages to the dimension of
$z$ over a long run. An overconfident filter reports an $S$ that is too small, so it treats
each new measurement as less informative than the prediction it already has, discounts
corrections that would have pulled it back, and diverges while its covariance ellipse keeps
shrinking.

### What one landmark determines

Observability asks whether the measurements determine the state at all, before any question of
noise. Locally the test is the rank of the map from state to measurement, which is the rank of
$H$.

Here $H$ is $2 \times 3$: two measurement numbers, three state numbers. For any $r > 0$ its
two rows are independent, so its rank is 2 and its null space is exactly one dimensional.
Solving $Hn = 0$ gives

$$n \propto \begin{bmatrix} -dy \\ dx \\ -1 \end{bmatrix}.$$

That direction has a physical reading. The vector $(-dy, dx)$ is perpendicular to the line
from the robot to the landmark, so moving along it swings the robot around the landmark at
constant distance; the third component turns the robot by the matching amount. Range is
unchanged because the distance did not change, and relative bearing is unchanged because the
landmark moved in the robot's field of view by exactly as much as the robot turned. One
range-bearing reading of one landmark therefore pins down two combinations of the three pose
numbers and says nothing at all about the third.

Motion is what recovers the missing direction: as the robot drives, the geometry changes, and
the unobservable direction at one instant is not the unobservable direction at the next, so
successive measurements constrain different combinations. That recovery is slow when the
landmark is far away and the geometry changes little. Running the visualization in this
assignment shows the consequence directly: the position ellipse stays roughly an order of
magnitude wider across the landmark ray than along it, and the heading standard deviation
stays at a few degrees for the whole run. The along-ray direction is measured directly by the
range, while the across-ray direction is reached only through the bearing angle, whose
sub-degree noise turns into a lateral error proportional to the distance.

### The unscented transform

The UKF attacks the same nonlinearity without Jacobians. The idea is to stop approximating the
function and start approximating the distribution. Rather than linearize $f$, represent the
Gaussian by a small set of deterministically chosen points, push each one through the exact
$f$, and fit a Gaussian to where they land. This is called moment matching: the point set is
built so that its weighted mean and weighted covariance are exactly $\mu$ and $P$, and the
same weighted sums are read off after the transformation.

Building the point set needs a matrix square root of the covariance, a matrix $L$ with
$L L^\top = P$. It is the matrix version of $\sqrt{\sigma^2} = \sigma$: the columns of $L$
are $n$ directions whose outer products add up to $P$, so scattering points along them
reproduces the spread that $P$ describes. The one used here is the Cholesky factor, the unique
lower-triangular $L$ with positive diagonal satisfying $L L^\top = P$, which exists exactly
when $P$ is SPD and costs about a third of a matrix inversion to compute. The choice is not
canonical: any $L$ with $L L^\top = P$ works, including the symmetric eigenvector-based square
root, and all of them give the same recovered mean and covariance for a linear map. They place
the points differently, so for a nonlinear $f$ they give slightly different answers, and the
convention is to fix one and stay with it.

For an $n$-dimensional state the transform uses $2n+1$ points: the mean, and one symmetric
pair straddling the mean along each column of the square root. Writing $\lambda$ for a scaling
parameter defined below, and $L$ for the Cholesky factor of $(n+\lambda)P$ with columns
numbered from 1,

$$\mathcal{X}_0 = \mu, \qquad \mathcal{X}_i = \mu + L_{:,i}, \qquad \mathcal{X}_{i+n} = \mu - L_{:,i} \quad (i = 1\dots n).$$

After propagating the points through $f$ or $h$ to get $\mathcal{Y}_i$, the transformed mean
and covariance are weighted sums:

$$\mu' = \sum_i W^m_i \mathcal{Y}_i, \qquad P' = \sum_i W^c_i (\mathcal{Y}_i - \mu')(\mathcal{Y}_i - \mu')^\top + Q,$$

with weights

$$W^m_0 = \frac{\lambda}{n+\lambda}, \quad W^c_0 = \frac{\lambda}{n+\lambda} + (1 - \alpha^2 + \beta), \quad W^m_i = W^c_i = \frac{1}{2(n+\lambda)}.$$

The $(n+\lambda)$ inside the square root and the $1/(2(n+\lambda))$ in the weights are the
same constant, and they are chosen so that the two cancel. Feed the identity map in and check:
the $\pm$ pairs make the weighted mean collapse to $\mu$ because the mean weights sum to 1,
and the weighted covariance is
$\sum_{i=1}^{n} \frac{1}{2(n+\lambda)} \cdot 2\, L_{:,i} L_{:,i}^\top = \frac{1}{n+\lambda} L L^\top = P$.
The point set carries $\mu$ and $P$ exactly, which is what
`test_ukf_recovers_its_own_gaussian` asserts.

The same cancellation makes the transform exact for any affine $y = Ax + b$: the weighted sums
commute with the affine map, so they return $A\mu + b$ and $A P A^\top$ with no error beyond
floating point. That is why the UKF and the KF produce identical posteriors on a
linear-Gaussian system, which is what `test_ukf_matches_kf_linear_run` checks to $10^{-6}$.

Where the transform earns its cost is curvature. Take a scalar example: $x \sim
\mathcal{N}(0.3,\, 0.4^2)$ pushed through $f(x) = e^x$. The true mean of $f(x)$ is
$e^{\mu + \sigma^2/2} = 1.4623$. The EKF's answer is $f(\mu) = e^{0.3} = 1.3499$, off by
$0.11$, because linearization moves the mean through the function and drops the curvature term
entirely. The unscented transform with this assignment's parameters returns $1.4582$. Expand
$f$ in a Taylor series around $\mu$ and the reason is visible: the true mean is
$f(\mu) + \tfrac12 f''(\mu)\sigma^2 + \tfrac18 f''''(\mu)\sigma^4 + \dots$, the weighted sum
over the symmetric point set reproduces the $f(\mu)$ and $\tfrac12 f''(\mu)\sigma^2$ terms
exactly for any spread, and the first disagreement is in the $\sigma^4$ term. "Second-order
accurate" means exactly that: agreement with the true transformed mean through the $\sigma^2$
term, where the EKF agrees only through $f(\mu)$.

### The sigma-point parameters

Three numbers control the point set, through $\lambda = \alpha^2(n + \kappa) - n$. All of them
act through the single scale factor $n + \lambda = \alpha^2 (n + \kappa)$, which is what
multiplies $P$ before the Cholesky and what divides the weights.

$\alpha$ sets the spread. The columns of a square root of $P$ get multiplied by
$\sqrt{n+\lambda} = \alpha\sqrt{n+\kappa}$, so $\alpha$ is the fraction of the natural spread
$\sqrt{n+\kappa}$ at which the points sit. This assignment uses $\alpha = 0.5$, $\kappa = 0$
(see `config.py`), which for the 3-dimensional pose puts the points at
$0.5\sqrt{3} \approx 0.87$ times the square-root columns of $P$, just inside one standard
deviation. Small $\alpha$ keeps the points in a tight neighborhood where the local behavior of
$f$ dominates, which is the argument for the textbook $\alpha = 10^{-3}$, but it is paid for
in the weights: with $\kappa = 0$ the center weight is $W^m_0 = 1 - 1/\alpha^2$, so
$\alpha = 10^{-3}$ makes it about $-10^6$ while the other weights sum to $+10^6$, and the
weighted sums lose digits to cancellation. $\alpha = 0.5$ gives $W^m_0 = -3$ against $2n$
weights of $2/3$ each, which sums to 1 with no drama. A negative center weight is not a bug:
the weights must sum to 1, and drawing the points in closer than $\sqrt{n}$ makes the outer
weights overshoot, which the center weight has to absorb by going below zero.

$\kappa$ is a second spread knob, historically the only one. The standard choice $\kappa = 3-n$
comes from the scalar analysis above: with $\alpha = 1$ it makes $n + \lambda = 3$, and $3$ is
the value at which the $\sigma^4$ term of the weighted sum matches the fourth moment of a
Gaussian, so the transform picks up one more order of accuracy. It goes negative for $n > 3$,
which in the original unscaled transform, where $\kappa$ was the only knob, drives the center
weight negative and can leave the recovered covariance indefinite. Fixing $\kappa = 0$ and
tuning the spread with $\alpha$ is the common default, and the one used here.

$\beta$ appears in exactly one place, the extra $(1 - \alpha^2 + \beta)$ on the center
covariance weight $W^c_0$, and does not touch the mean weights at all. Its job is to correct
the covariance for the fourth-moment error that the mean-weight choice leaves behind;
$\beta = 2$ is the value that does this best when the underlying distribution really is
Gaussian. Dropping $\beta$ gives a filter whose mean is still right and whose covariance is
quietly wrong, but only when the transformed function is nonlinear: for the identity and for
any affine map the propagated center point lands exactly on the recovered mean, so
$\mathcal{Y}_0 - \mu' = 0$ and the weight multiplies a zero matrix. That is why
`test_ukf_weights` checks $W^c_0 - W^m_0 = 1 - \alpha^2 + \beta$ directly instead of relying on
the affine tests to catch it.

### The unscented Kalman filter

Wiring the transform into a filter is mechanical. Predict generates sigma points from
$(\mu, P)$, pushes them through $f$, and reads back the transformed Gaussian with $Q$ added.
Update generates sigma points again from the predicted $(\mu^-, P^-)$, pushes them through $h$
to get measurement points $\mathcal{Z}_i$, and applies the unscented transform with $R$ added
to obtain $\hat z$ and $P_{zz} = S$. Regenerating the points rather than reusing the
propagated ones costs one extra Cholesky and keeps the point set consistent with the covariance
it is supposed to represent.

That leaves the cross-covariance $P_{xz}$, which the transform does not produce because it
involves two different point sets:

$$P_{xz} = \sum_i W^c_i (\mathcal{X}_i - \mu^-)(\mathcal{Z}_i - \hat z)^\top.$$

With all three ingredients in hand the update is the conditioning formula from the start of
these notes, unchanged: $K = P_{xz} S^{-1}$, $\mu^+ = \mu^- + K(z - \hat z)$, and
$P^+ = P^- - K S K^\top$.

Against the EKF, the trade is derivation effort for evaluation cost. The UKF needs no
Jacobians at all, which matters when $h$ is a chunk of code rather than a formula, and it
tracks curvature to one order better. It pays $2n+1$ evaluations of $f$ and $h$ per step plus
a Cholesky, against one evaluation and one Jacobian for the EKF. Both collapse to the KF on a
linear system.

### The information form

The information (canonical) form carries the same Gaussian in a second parameterization, and
the reason it exists is visible by taking the logarithm of the density. Dropping constants,

$$\log p(x) = -\tfrac12 (x-\mu)^\top P^{-1} (x - \mu) + c = -\tfrac12 x^\top P^{-1} x + x^\top P^{-1}\mu + c'.$$

A Gaussian is a quadratic in the exponent, and a quadratic is determined by its second-order
coefficient and its first-order coefficient. Those two are the information matrix
$\Omega = P^{-1}$ and the information vector $\eta = \Omega\mu$. The name comes from the fact
that for a Gaussian with known covariance, $P^{-1}$ is the Fisher information the distribution
carries about its own mean: how sharply the log-density curves, which is how strongly the data
pins the parameter down.

Additivity follows immediately. Bayes' rule multiplies the prior by the likelihood, and
multiplying densities adds log-densities, so it adds quadratics coefficient by coefficient.
For a linear-Gaussian measurement $z = Hx + v$,

$$\log p(z \mid x) = -\tfrac12 (z - Hx)^\top R^{-1}(z - Hx) + c = -\tfrac12 x^\top H^\top R^{-1} H x + x^\top H^\top R^{-1} z + c'',$$

and adding the two quadratics gives the update this assignment implements:

$$\Omega^+ = \Omega^- + H^\top R^{-1} H, \qquad \eta^+ = \eta^- + H^\top R^{-1} z.$$

No inverse of a state-sized matrix appears, only the inverse of the measurement-sized $R$. Each
sensor contributes an independent additive term, so fusing many sensors is adding their
contributions in any order, which `test_information_fuses_two_sensors_additively` checks
against two sequential KF updates.

The term $H^\top R^{-1} H$ is worth reading on its own: it is the measurement's information
$R^{-1}$ pulled back into state space through $H$. It has the same null space as $H$, so a
direction the sensor cannot see receives exactly zero information, which is the observability
statement from earlier written as a matrix. The scalar averaging example fits here too: with
$H = 1$ and a static state, $n$ updates give $\Omega_n = \Omega_0 + n/R$, so $P_n \to R/n$ in
one line instead of a chained recursion.

The catch is the other half of the recursion. Prediction, which is cheap in the moment form,
becomes $\Omega^- = (F \Omega^{-1} F^\top + Q)^{-1}$, so it inverts a state-sized matrix twice
per step. The mean is also no longer sitting there to be read; recovering it means solving
$\Omega\mu = \eta$. The choice of form is therefore a trade over which step dominates. Filters
that predict constantly and measure rarely favor the moment form; multi-sensor fusion and
SLAM back-ends, which accumulate large numbers of constraints, favor the information form. The
other reason back-ends favor it is sparsity: the information matrix of a pose graph has a
nonzero block only where a constraint couples two states, which for a robot revisiting a few
places is almost all zeros, while its inverse, the covariance, is dense because every state is
correlated with every other.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the declaration and comments in each file give the signature and shapes.

1. [`kf_predict()`](kalman.cpp) in `kalman.cpp`
2. [`kf_update()`](kalman.cpp) in `kalman.cpp`
3. [`ekf_f()`](kalman.cpp) in `kalman.cpp`
4. [`ekf_h()`](kalman.cpp) in `kalman.cpp`
5. [`ekf_F_x()`](kalman.cpp) in `kalman.cpp`
6. [`ekf_H()`](kalman.cpp) in `kalman.cpp`
7. [`ekf_predict()`](kalman.cpp) in `kalman.cpp`
8. [`ekf_update()`](kalman.cpp) in `kalman.cpp`
9. [`ukf_sigma_points()`](kalman.cpp) in `kalman.cpp`
10. [`ukf_unscented_transform()`](kalman.cpp) in `kalman.cpp`
11. [`ukf_cross_covariance()`](kalman.cpp) in `kalman.cpp`
12. [`moments_to_information()`](kalman.cpp) in `kalman.cpp`
13. [`information_to_moments()`](kalman.cpp) in `kalman.cpp`
14. [`information_update()`](kalman.cpp) in `kalman.cpp`

You may not include an existing estimation or solver library; a test scans the sources.

### Building and running

Same toolchain as the Lie-group assignment: a C++17 compiler plus CMake, pybind11, and
Eigen (the `environment.yml` a14 block). You never call CMake by hand.

```
make verify A=a14_1_kalman   # build + run the reference (solution/); the green target
make test   A=a14_1_kalman   # build + run YOUR code; red until the holes are filled
make viz    A=a14_1_kalman        # render the EKF-vs-UKF tracking run (reference)
make viz-mine A=a14_1_kalman      # the same, from YOUR code, once the holes are filled
```

The test order is the intended workflow. The linear-KF tests come first: predict and the
Joseph update against an independent NumPy reference, then a check that an update shrinks the
covariance trace and leaves it symmetric and SPD, then variance convergence on a static
scalar. The EKF tests then check the analytic Jacobians against central differences, the gate
that catches a wrong derivative, before the bearing-wrap, SPD-preservation, and tracking runs.
The finite-difference comparison wraps the bearing difference before dividing, so the check
stays valid across the $\pm\pi$ seam. The UKF tests check the weights, recovery of the
original Gaussian through the identity map, exactness on an affine map, and agreement with the
KF on a linear-Gaussian run. The information-form tests check the round-trip, equivalence with
the KF update, and additive two-sensor fusion.

`make viz` writes a Rerun recording to `out/kalman_track.rrd`. Add `SHOW=1` for the
interactive viewer: a unicycle drives an arc observed only by range-bearing to one landmark,
and the EKF (red) and UKF (blue) estimates track it with their 1-sigma covariance ellipses,
on a timeline you can scrub. The scalar panels compare the two filters' position error and
covariance trace. Watch the ellipses against the observability argument above: they stay
stretched across the ray to the landmark, the direction a single reading cannot constrain, and
stay narrow along it, the direction the range measures directly.

## In interviews

This is bread-and-butter perception and robotics material, asked about constantly.

KF versus EKF versus UKF. Know the one-line distinction: the KF is exact for linear-Gaussian
systems; the EKF handles nonlinearity by linearizing the model with a Jacobian at the mean;
the UKF handles it by propagating sigma points through the exact model and refitting a
Gaussian. The EKF needs you to derive Jacobians and degrades when the model curves over the
covariance; the UKF needs no Jacobians and matches the true transformed mean one order
further, at a cost of $2n+1$ model evaluations per step. Both reduce to the KF on a linear
system.

Why the Joseph form. Expect "what's wrong with $P^+ = (I-KH)P$?" The answer: it is correct in
exact arithmetic at the optimal gain, but it subtracts nearly equal matrices and enforces
nothing, so round-off can drift the covariance non-symmetric or indefinite and the filter blows
up; the Joseph form is a sum of symmetric PSD terms, stays valid, and holds for any gain.

The information filter and where it wins. Be able to state the dual ($\Omega = P^{-1}$,
$\eta = \Omega\mu$), that it is the quadratic form of the log-density, and that Bayes therefore
turns into addition. That additivity makes multi-sensor fusion and sparse back-ends natural,
helped by the information matrix of a pose graph being sparse while the covariance is dense.
The trade-off: prediction is the expensive step in information form, and reading out the mean
needs a solve.

Observability. A favorite follow-up: with a single range-bearing landmark, is the planar pose
observable? A single reading gives two numbers for three unknowns, and the unconstrained
direction is orbiting the landmark while turning by the same angle, which leaves both range and
relative bearing unchanged. Motion recovers it as the geometry changes, slowly when the
landmark is distant, which the visualization shows as an ellipse stretched across the landmark
ray. Be ready to follow up on filter consistency: the covariance should match the actual error
spread, testable by checking that $y^\top S^{-1} y$ averages the measurement dimension.

A classic on-the-spot ask is to write the scalar (1D) Kalman update from scratch:
$K = P/(P+R)$, $\mu \leftarrow \mu + K(z-\mu)$, $P \leftarrow (1-K)P$. It is the whole
subject in three lines, and the matrix version is the same with the gain sandwiched by $H$.

## Further reading

- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapter 3 for the Gaussian filters
  and chapter 7 for EKF localization against known landmarks, which uses the same
  range-bearing model as here.
- Julier and Uhlmann, "Unscented filtering and nonlinear estimation," Proc. IEEE 92(3),
  2004 - the sigma-point transform and weight derivation.
- Barfoot, *State Estimation for Robotics* (2017), chapter 3 for linear-Gaussian estimation
  and the batch-versus-recursive view, chapter 4 for the nonlinear filters including the
  sigma-point family.
