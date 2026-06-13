# a14_1 - Kalman filtering: KF, EKF, UKF, and the information form

The second assignment of the classical SLAM module, again in C++17 with Eigen. You build
the Gaussian filters that every recursive state estimator is made of: the linear Kalman
filter (KF), the extended Kalman filter (EKF) for nonlinear motion and measurement models,
the unscented Kalman filter (UKF), and the information (canonical) form. The EKF motion
model and its Jacobian you write here are exactly what the EKF-SLAM assignment reuses for
its robot-motion update, so this is the estimation core the rest of the module stands on.

The Python around the C++ (the build, the tests, the Rerun visualization) is provided, as
in the Lie-group assignment. The UKF's sigma-point loop is also provided in Python; the
graded C++ is the unscented-transform math it calls (see the assignment section).

Required reading before you start:
- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapter 3 - the Gaussian filters
  (KF, EKF, information filter) derived from the Bayes filter, and the cleanest statement
  of the recursion.
- Julier and Uhlmann, "Unscented filtering and nonlinear estimation," Proceedings of the
  IEEE 92(3), 2004 - the sigma-point transform and the $\alpha,\beta,\kappa$ weights.

## Lecture notes

### The Gaussian belief recursion

A filter maintains a belief about a hidden state $x$ (a robot pose, a velocity, a landmark
position) and updates it as controls and measurements arrive. The Bayes filter does this in
full generality, but it is intractable unless the belief has a closed form. The Kalman
family makes one assumption that closes it: the belief is Gaussian, summarized by a mean
$\mu$ and a covariance $P$. Every step maps one Gaussian to another, so the entire history
collapses into those two numbers.

The recursion alternates two steps. Predict pushes the belief forward through the motion
model and grows the covariance (motion adds uncertainty). Update folds in a measurement and
shrinks the covariance (an observation removes uncertainty). The whole subject is four
matrix expressions for those two steps, plus the question of what to do when the models are
nonlinear.

### The linear Kalman filter

When the motion and measurement are linear with Gaussian noise,

$$x_{k} = F x_{k-1} + B u_k + w_k, \quad w_k \sim \mathcal{N}(0, Q), \qquad z_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R),$$

the belief stays exactly Gaussian and the KF is the optimal estimator. Predict is

$$\mu^- = F\mu + B u, \qquad P^- = F P F^\top + Q.$$

$F P F^\top$ is how a linear map transforms a covariance; $+Q$ is the process noise that
makes the robot less certain the longer it moves blind. Update computes the innovation
$y = z - H\mu^-$ (what the measurement says minus what we predicted), its covariance
$S = H P^- H^\top + R$, and the Kalman gain $K = P^- H^\top S^{-1}$, then corrects:

$$\mu^+ = \mu^- + K y, \qquad P^+ = (I - K H) P^- (I - K H)^\top + K R K^\top.$$

The gain $K$ interpolates between trusting the prediction and trusting the measurement, in
proportion to their relative certainties. The covariance update is written in the Joseph
form above, not the shorter $P^+ = (I - KH)P^-$. The two are algebraically equal, but the
short form subtracts two nearly equal matrices and loses symmetry and positive-definiteness
to round-off over a long run; the Joseph form is a sum of two symmetric positive
semidefinite terms and stays SPD. Use it.

### The extended Kalman filter

Real motion and measurement models are nonlinear: a robot turns, a range sensor computes a
square root. The EKF keeps the KF machinery and linearizes the models at the current mean.
For a nonlinear motion model $f(x, u)$ and measurement model $h(x)$, the predict and update
are the KF equations with the Jacobians

$$F_x = \left.\frac{\partial f}{\partial x}\right|_{\mu}, \qquad H = \left.\frac{\partial h}{\partial x}\right|_{\mu},$$

while the mean itself goes through the exact nonlinear model: $\mu^- = f(\mu, u)$ and
$y = z - h(\mu^-)$. The covariance still propagates as $F_x P F_x^\top + Q$ and
$H P^- H^\top + R$.

This assignment's demo is a planar robot with state $x = [p_x, p_y, \theta]$, a unicycle
motion model with control $u = [v, \omega]$ (forward speed and turn rate),

$$p_x' = p_x + v\,\Delta t\cos\theta, \quad p_y' = p_y + v\,\Delta t\sin\theta, \quad \theta' = \theta + \omega\,\Delta t,$$

and a range-bearing measurement of a known landmark at $\ell = [\ell_x, \ell_y]$, with
$dx = \ell_x - p_x$, $dy = \ell_y - p_y$, and $q = dx^2 + dy^2$:

$$h(x) = \begin{bmatrix} r \\ \phi \end{bmatrix} = \begin{bmatrix} \sqrt{q} \\ \operatorname{atan2}(dy, dx) - \theta \end{bmatrix}.$$

Their Jacobians are

$$F_x = \begin{bmatrix} 1 & 0 & -v\,\Delta t\sin\theta \\ 0 & 1 & v\,\Delta t\cos\theta \\ 0 & 0 & 1 \end{bmatrix}, \qquad H = \begin{bmatrix} -dx/r & -dy/r & 0 \\ dy/q & -dx/q & -1 \end{bmatrix}.$$

The bearing $\phi$ is an angle, so its residual must be wrapped to $(-\pi, \pi]$: a
measurement at $+179^\circ$ and a prediction at $-179^\circ$ are $2^\circ$ apart, not
$358^\circ$, and an unwrapped innovation injects a huge spurious correction. The heading
state is likewise kept wrapped. This is the most common EKF-on-a-pose bug; `wrap_angle`
(provided) handles it, and you call it in the residual and after the mean update.

The EKF's weakness is the linearization. $F_x$ and $H$ are a first-order Taylor expansion
at the mean, so when the model curves sharply over the spread of the covariance, the
predicted Gaussian is wrong and the filter can grow overconfident and diverge.

### The unscented Kalman filter

The UKF attacks the same nonlinearity without Jacobians. Instead of linearizing the
function, it represents the Gaussian by a small set of deterministic sample points (sigma
points), pushes each one through the exact nonlinear function, and fits a Gaussian to the
transformed points. For an $n$-dimensional state it uses $2n+1$ points: the mean, and one
pair straddling the mean along each principal axis of the covariance. With
$\lambda = \alpha^2(n + \kappa) - n$ and the matrix square root $L$ (a Cholesky factor) of
$(n+\lambda)P$,

$$\mathcal{X}_0 = \mu, \qquad \mathcal{X}_i = \mu + L_{:,i}, \qquad \mathcal{X}_{i+n} = \mu - L_{:,i} \quad (i = 1\dots n).$$

After propagating the points through $f$ or $h$ to get $\mathcal{Y}_i$, the transformed
mean and covariance are weighted sums (this is the unscented transform):

$$\mu' = \sum_i W^m_i \mathcal{Y}_i, \qquad P' = \sum_i W^c_i (\mathcal{Y}_i - \mu')(\mathcal{Y}_i - \mu')^\top + Q,$$

with weights

$$W^m_0 = \frac{\lambda}{n+\lambda}, \quad W^c_0 = \frac{\lambda}{n+\lambda} + (1 - \alpha^2 + \beta), \quad W^m_i = W^c_i = \frac{1}{2(n+\lambda)}.$$

The three parameters control the spread and the weighting: $\alpha$ sets how far the points
sit from the mean, $\kappa$ is a secondary scaling, and $\beta$ adds prior knowledge of the
distribution's shape. The $(1 - \alpha^2 + \beta)$ term appears only on the center
covariance weight $W^c_0$, and $\beta = 2$ is optimal for a Gaussian. Dropping it is a quiet
bug: the mean comes out right, the covariance comes out wrong. The unscented transform is
exact for any affine $f$ or $h$ (it reproduces $A\mu + b$ and $A P A^\top$ exactly), which
is why on a linear-Gaussian system the UKF and KF give the same posterior to floating point.
Where they differ is nonlinearity: the UKF captures curvature to second order because it
evaluates the true function at spread-out points, so it usually beats the EKF when the model
bends, without anyone deriving a Jacobian.

For the update, the gain needs the state-measurement cross-covariance
$P_{xz} = \sum_i W^c_i (\mathcal{X}_i - \mu)(\mathcal{Z}_i - \hat z)^\top$, and then
$K = P_{xz} S^{-1}$, $\mu^+ = \mu + K(z - \hat z)$, $P^+ = P - K S K^\top$.

### The information form

The information (canonical) form carries the same Gaussian in dual coordinates: the
information matrix $\Omega = P^{-1}$ and the information vector $\eta = \Omega\mu$. The
appeal is the measurement update, which is purely additive:

$$\Omega^+ = \Omega^- + H^\top R^{-1} H, \qquad \eta^+ = \eta^- + H^\top R^{-1} z.$$

No inverse of a state-sized matrix, and each sensor contributes an independent additive
term, so fusing many sensors is just adding their contributions in any order. That is why
multi-sensor fusion and sparse SLAM back-ends are natural in this form: the information
matrix of a pose graph is sparse (it has a nonzero block only where two states share a
constraint), while its inverse, the covariance, is dense. The catch is the other half of the
recursion: the prediction step, cheap in the moment form, is the expensive one in
information form (it requires inverting back), so the choice of form is a trade between which
step you do more of.

## The assignment

Implement the four filters in C++. The pybind11 bindings, the CMake build, the tests, and
the Rerun visualization are provided.

### Files to modify

`kalman.cpp` holds every hole, in four groups:

- Linear KF: `kf_predict` and `kf_update` (Joseph form).
- EKF: the models `ekf_f` and `ekf_h` and their analytic Jacobians `ekf_F_x` and `ekf_H`,
  then the filter steps `ekf_predict` and `ekf_update` that wire them together with the
  angle-wrapped residual.
- UKF: `ukf_sigma_points` (the $2n+1$ points and the weights, including the
  $(1-\alpha^2+\beta)$ term), `ukf_unscented_transform` (the weighted mean and covariance),
  and `ukf_cross_covariance`. The loop that pushes each sigma point through $f$ or $h$ is
  provided in `_helpers.py` (`ukf_predict` / `ukf_update`), so your C++ is the transform
  math, and the same helper drives the UKF with an arbitrary linear model to compare against
  the KF.
- Information form: `moments_to_information`, `information_to_moments`, and the additive
  `information_update`.

Each function's contract (inputs, outputs, formula) is in the comment at its hole; the math
is in the lecture notes above. The header `kalman.hpp` is shared with the reference and is
not edited; `wrap_angle` lives there, provided. The reference is in `solution/kalman.cpp`,
in plain sight. You may not include an existing estimation or solver library; a test scans
the sources.

### Building and running

Same toolchain as the Lie-group assignment: a C++17 compiler plus CMake, pybind11, and
Eigen (the `environment.yml` a14 block). You never call CMake by hand.

```
make verify A=a14_1_kalman   # build + run the reference (solution/); the green target
make test   A=a14_1_kalman   # build + run YOUR code; red until the holes are filled
make viz    A=a14_1_kalman        # render the EKF-vs-UKF tracking run (reference)
make viz-mine A=a14_1_kalman      # the same, from YOUR code, once the holes are filled
```

The test order is the intended workflow. The linear-KF tests come first (predict and the
Joseph update against an independent NumPy reference, then variance convergence on a static
scalar). The EKF tests then check the analytic Jacobians against central differences, the
gate that catches a wrong derivative, before the SPD-preservation and tracking runs. The UKF
tests check the weights, exactness on an affine map, and agreement with the KF on a
linear-Gaussian run. The information-form tests check the round-trip, equivalence with the
KF update, and additive two-sensor fusion.

`make viz` writes a Rerun recording to `out/kalman_track.rrd`. Add `SHOW=1` for the
interactive viewer: a unicycle drives an arc observed only by range-bearing to one landmark,
and the EKF (red) and UKF (blue) estimates track it with their 1-sigma covariance ellipses,
on a timeline you can scrub. The scalar panels compare the two filters' position error and
covariance trace. With one landmark the heading is only weakly observable, so the ellipses
stay elongated along the bearing direction, a concrete picture of partial observability.

## In interviews

This is bread-and-butter perception and robotics material, asked about constantly.

KF versus EKF versus UKF. Know the one-line distinction: the KF is exact for linear-Gaussian
systems; the EKF handles nonlinearity by linearizing the model with a Jacobian at the mean;
the UKF handles it by propagating sigma points through the exact model and refitting a
Gaussian. The EKF needs you to derive Jacobians and degrades when the model curves over the
covariance; the UKF needs no Jacobians and captures curvature to second order, at a modest
constant-factor cost. Both reduce to the KF on a linear system.

Why the Joseph form. Expect "what's wrong with $P^+ = (I-KH)P$?" The answer: it is correct
in exact arithmetic but loses symmetry and positive-definiteness to round-off, and a
covariance that drifts indefinite makes the filter blow up; the Joseph form is a sum of
symmetric PSD terms and stays valid.

The information filter and where it wins. Be able to state the dual ($\Omega = P^{-1}$,
$\eta = \Omega\mu$) and that the measurement update is additive, which makes multi-sensor
fusion and sparse back-ends natural because the information matrix of a pose graph is sparse
while the covariance is dense. The trade-off: prediction is the expensive step in
information form.

Observability. A favorite follow-up: with a single range-bearing landmark, is the planar
pose observable? Range and bearing to one point pin position but leave heading weakly
constrained until the robot moves and the geometry changes, which the visualization shows as
a covariance ellipse that stays stretched. Be ready to reason about when a filter is
consistent (its covariance honestly reflects its error) versus overconfident.

A classic on-the-spot ask is to write the scalar (1D) Kalman update from scratch:
$K = P/(P+R)$, $\mu \leftarrow \mu + K(z-\mu)$, $P \leftarrow (1-K)P$. It is the whole
subject in three lines, and the matrix version is the same with the gain sandwiched by $H$.

## Further reading

- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapters 3 and 7 - Gaussian filters
  and the velocity motion / range-bearing models used here.
- Julier and Uhlmann, "Unscented filtering and nonlinear estimation," Proc. IEEE 92(3),
  2004 - the sigma-point transform and weight derivation.
- Barfoot, *State Estimation for Robotics* (2017), chapter 4 - the linear-Gaussian
  estimation theory and the batch-versus-recursive view.
