# a14_2 - EKF-SLAM

The third assignment of the classical SLAM module, in C++17 with Eigen. You build EKF-SLAM:
a single extended Kalman filter whose state is the robot pose and every landmark it has
mapped, estimated jointly. This is the historically first online SLAM method that worked,
and the one factor graphs later replaced. Building it shows both why a joint filter solves
the chicken-and-egg problem of mapping while localizing, and why its cost and its
linearization eventually made the field move on.

It builds directly on the extended Kalman filter from the Kalman-filter assignment: the
unicycle motion model, its Jacobian, and the range-bearing measurement model and its
Jacobians are carried over and provided here (in `models.cpp`), so your work is the
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
that observation should correct the pose and, with it, the whole map built since. Capturing
that requires estimating the robot and the landmarks in one joint belief, with the
correlations between them.

EKF-SLAM does exactly this. It stacks the robot pose and all landmark positions into one
state vector and runs a single EKF over it. The state grows as new landmarks are seen:

$$\mu = \big[\,\underbrace{p_x,\,p_y,\,\theta}_{\text{robot}},\ \underbrace{\ell_{0x},\,\ell_{0y}}_{\text{lm }0},\ \underbrace{\ell_{1x},\,\ell_{1y}}_{\text{lm }1},\ \dots\,\big]^\top,$$

with the matching joint covariance $P$ of size $(3+2N)\times(3+2N)$ for $N$ landmarks.
Landmark $j$ lives at indices $3+2j$ and $3+2j+1$.

### The off-diagonal blocks are the point

In the joint covariance

$$P = \begin{bmatrix} P_{rr} & P_{rm} \\ P_{mr} & P_{mm} \end{bmatrix},$$

the robot block $P_{rr}$, the map block $P_{mm}$, and the robot-map cross block
$P_{rm}$ (and the landmark-landmark cross blocks inside $P_{mm}$) all matter. The
off-diagonal blocks encode that landmarks observed from the same uncertain pose are
correlated: if the robot pose was off when it placed them, their errors are shared, and a
later measurement that corrects one corrects the others through these blocks. A
block-diagonal approximation that drops the cross terms is not a cheaper EKF-SLAM, it is a
broken one: loop closure stops working, because the information from re-seeing an old
landmark has no path to the rest of the map. Running this assignment's filter and looking at
the final covariance, the off-diagonal landmark-landmark blocks are as large as the diagonal
ones - the map is genuinely dense.

### Predict: only the robot moves, but the correlations propagate

A motion command moves the robot; the landmarks are static, so their mean does not change.
The mean update touches only the robot block, $\mu_r \leftarrow f(\mu_r, u)$. The covariance
is where SLAM differs from plain localization. With $F$ the $3\times3$ motion Jacobian
$\partial f/\partial \mu_r$,

$$P_{rr} \leftarrow F P_{rr} F^\top + Q, \qquad P_{rm} \leftarrow F P_{rm}, \qquad P_{mm}\ \text{unchanged.}$$

The robot-map cross block $P_{rm}$ rotates through $F$ even though the landmarks did not
move: moving the robot changes how its (now larger) uncertainty correlates with the map.
Forgetting to propagate $P_{rm}$ silently decorrelates the robot from its map, and again
loop closure dies. Only the robot block takes the process noise $Q$.

### Initializing a landmark

The first time a landmark is seen it is not in the state, so it must be added. The inverse
measurement model places it in the world from the current pose and the range-bearing reading
$z = [r, \phi]$, writing $\beta = \theta + \phi$:

$$\ell = \begin{bmatrix} p_x + r\cos\beta \\ p_y + r\sin\beta \end{bmatrix}.$$

Appending $\ell$ to the mean is easy; the covariance augmentation is the careful part. With
the Jacobians of this inverse model with respect to the pose and the measurement,

$$G_r = \frac{\partial \ell}{\partial \mu_r} = \begin{bmatrix} 1 & 0 & -r\sin\beta \\ 0 & 1 & r\cos\beta \end{bmatrix}, \qquad G_z = \frac{\partial \ell}{\partial z} = \begin{bmatrix} \cos\beta & -r\sin\beta \\ \sin\beta & r\cos\beta \end{bmatrix},$$

the new landmark's own covariance is $P_{\ell\ell} = G_r P_{rr} G_r^\top + G_z R G_z^\top$
(its uncertainty comes from the pose uncertainty and the measurement noise), and its cross
covariance with the entire existing state is $P_{\ell x} = G_r P_{r,:}$, where $P_{r,:}$ is
the three robot rows of the current covariance. The augmented covariance puts the old $P$ in
the top-left, $P_{\ell\ell}$ in the new bottom-right $2\times2$ block, and $P_{\ell x}$ (and
its transpose) in the new off-diagonal strips. A new landmark is born already correlated
with the robot and, through it, with every other landmark.

### The measurement update spreads through the whole state

To update on an observation of a mapped landmark $j$, predict the measurement
$h(\mu) = $ range-bearing from the robot to landmark $j$, and form the innovation
$y = z - h(\mu)$ with the bearing wrapped to $(-\pi,\pi]$. The measurement Jacobian $H$ is
$2 \times (3+2N)$ and almost entirely zero: a $2\times3$ block in the robot columns and a
$2\times2$ block in landmark $j$'s columns, both carried over from the EKF measurement
model. Everything else is the standard EKF update, but run over the full joint state:

$$S = H P H^\top + R, \quad K = P H^\top S^{-1}, \quad \mu \leftarrow \mu + Ky, \quad P \leftarrow (I - KH)P(I-KH)^\top + KRK^\top.$$

Although $H$ touches only the robot and one landmark, the gain $K = P H^\top S^{-1}$ is dense
(because $P$ is dense), so a single observation corrects the robot and, through the
correlations, every other landmark. That is loop closure: re-seeing landmark 0 after a long
drive tightens the estimate of every landmark mapped in between. The covariance update is the
Joseph form, for the same numerical reason as in the Kalman assignment.

### Data association

The update above assumed you know which landmark a measurement came from. In reality you do
not, and getting it wrong is the dominant failure mode of real SLAM: one bad association
corrupts the map irreversibly. The standard test is the Mahalanobis distance of the
innovation. For each mapped landmark $j$, with innovation $y_j$ and innovation covariance
$S_j = H_j P H_j^\top + R$,

$$d_j^2 = y_j^\top S_j^{-1} y_j,$$

which is the squared error scaled by how uncertain that error is. The measurement is
associated with the nearest landmark by $d^2$ if that distance falls under a gate, and is
otherwise treated as a new landmark. Because $d^2$ on a 2-DOF range-bearing innovation is
chi-square distributed with 2 degrees of freedom, the gate is a chi-square quantile (the
95th percentile, $\approx 5.99$), which gives the gate a probabilistic meaning rather than a
hand-tuned threshold.

### Consistency and the cost that killed it

EKF-SLAM has two well-known problems. The first is cost: the joint covariance is dense and
every update is $O(n^2)$ in the number of landmarks, so the filter does not scale past a few
hundred landmarks. The second is consistency. A filter is consistent when its reported
covariance honestly matches its actual error; EKF-SLAM is known to become optimistic
(overconfident) because it linearizes the nonlinear models at the current, slightly-wrong
estimate, and the linearization error accumulates. Running this assignment's loop and
plotting the robot NEES (normalized estimation error squared, expectation 3 for a consistent
3-DOF estimate), you will see it climb as the robot drives away from its anchored start -
roughly $0.2 \to 0.9 \to 1.9 \to 3.0 \to 4.5$ across the loop in the provided run, brushing
the chi-square bound near the far side - and then drop back down when loop closure re-sees
the first landmarks and re-tightens the map. The early-loop values below 3 are the
absolute-frame uncertainty being conservative (the map is only known relative to the start);
the climb toward and across the bound is the optimism. Both problems are why the field moved
to factor-graph smoothing, the subject of the bundle-adjustment assignment: it keeps all
poses and relinearizes, and it exploits sparsity instead of fighting a dense covariance.

## The assignment

Implement the four EKF-SLAM operations in C++. The motion and measurement primitives, the
pybind11 bindings, the CMake build, the simulator, the tests, and the Rerun visualization
are provided.

### Files to modify

`ekf_slam.cpp` holds four holes:

- `slam_predict` - move the robot through the motion model and propagate the covariance,
  including the robot-map cross blocks.
- `slam_add_landmark` - the inverse measurement model and the covariance augmentation that
  grows the state by two when a landmark is first seen.
- `slam_update` - the EKF measurement update for a known landmark, assembling the sparse $H$
  and running the Joseph-form update over the full joint state.
- `slam_associate` - nearest-neighbor data association with the chi-square Mahalanobis gate.

`models.cpp` (the unicycle `robot_f`/`robot_F_x` and the range-bearing
`range_bearing`/`range_bearing_H_robot`/`range_bearing_H_land`, carried over from the Kalman
assignment) is provided and compiled in both builds; call those, do not reimplement them.
Each hole's contract is in the comment at its hole and in `ekf_slam.hpp`; the math is in the
lecture notes. The reference is in `solution/ekf_slam.cpp`. You may not include an existing
SLAM or solver library; a test scans the sources.

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
landmark from a noise-free reading at its true position; the update never increases the joint
covariance determinant and never decreases the information matrix in the Loewner order (a
measurement adds information); the data-association gate matches a real observation and
rejects a spurious one; on a short mild trajectory the filter is not optimistic (the robot
NEES stays below its chi-square bound); and the mapped landmarks track the truth. The strict
per-landmark marginal-determinant decrease is deliberately not asserted: a single landmark's
marginal is a Schur complement of the joint and is not monotone under a near-uninformative
update, so it can fail for correct code.

`make viz` writes `out/ekf_slam.rrd`. Add `SHOW=1` for the interactive viewer: ground truth,
the robot and landmark estimates with their 3-sigma ellipses, the active measurement rays,
and two scalar panels (robot NEES against its chi-square bound, and mean mapped-landmark
error). Scrub to the far side of the loop to watch the NEES rise, and to the end to watch
loop closure pull it back down and tighten the whole map at once.

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
correlations and keep only block diagonals? Loop closure breaks, because re-observing an old
landmark can no longer inform the rest of the map.

Why EKF-SLAM is $O(n^2)$ and what that killed. The joint covariance is dense and every update
touches all of it, so it does not scale; this is the practical reason the field moved to
sparse, graph-based back-ends. Know that the information form's matrix is sparse for a pose
graph while the covariance is dense, which is the structural insight the smoothers exploit.

Consistency. Expect "is EKF-SLAM consistent?" The honest answer is no, not in general: it
linearizes at the current estimate and becomes optimistic as that error accumulates, which
is one of the two reasons (with cost) that factor-graph smoothing replaced it. Being able to
say what NEES is and what its expected value should be is a strong signal.

Data association as the real failure mode. The filter math is the easy part; deciding which
measurement is which landmark is what breaks real systems. Know the Mahalanobis gate and why
it uses a chi-square threshold, and that harder settings need joint-compatibility (JCBB) or
robust back-ends. A common on-the-spot question is simply "what goes in the state vector and
how does it grow," which is the augmentation step here.

## Further reading

- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapter 10 - the full EKF-SLAM
  derivation and the inverse measurement model.
- Durrant-Whyte and Bailey, "SLAM: part I" (2006) and Bailey and Durrant-Whyte, "SLAM: part
  II" (2006) - the problem, the correlations, and data association including JCBB.
- Cadena et al., "Past, present, and future of SLAM: toward the robust-perception age"
  (2016) - where filtering sits relative to the smoothing methods that replaced it.
