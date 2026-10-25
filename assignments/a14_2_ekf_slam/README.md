# a14_2 - EKF-SLAM

The third assignment of the classical SLAM module, in C++17 with Eigen. It builds EKF-SLAM:
a single extended Kalman filter whose state is the robot pose and every landmark it has
mapped, estimated jointly. This is the historically first online SLAM method that worked,
and the one factor graphs later replaced. Building it shows both why a joint filter solves
the chicken-and-egg problem of mapping while localizing, and why its cost and its
linearization eventually made the field move on.

It builds directly on the extended Kalman filter (EKF) from the Kalman-filter assignment: the
unicycle motion model, its Jacobian, and the range-bearing measurement model and its
Jacobians are carried over and provided here (in `models.cpp`), so the work here is the
SLAM-specific machinery on top of them. The Python around the C++ (the build, the simulator,
the filter driver loop, the tests, and the Rerun visualization) is provided, as in the
Lie-group and Kalman-filter assignments.

Required reading before starting:
- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005), chapter 10 - the EKF-SLAM
  derivation, the joint state, and the inverse measurement model for landmark
  initialization.
- Durrant-Whyte and Bailey, "Simultaneous localization and mapping (SLAM): part I" (2006),
  IEEE RAM - the problem statement and why the off-diagonal correlations are essential.

## Lecture notes

### What EKF-SLAM is for

#### The task

The demo in this assignment is a planar robot driving a loop among ten landmarks whose
positions it does not know. The robot is a unicycle with pose $[p_x, p_y, \theta]$ (meters and
radians). It starts at $(5, 0)$ facing $+y$ and is commanded to drive at 1.2 m/s while turning
just fast enough to trace a circle of radius 5 m around the origin, one control every 0.1 s.
Commands are executed imperfectly: every step the true pose drifts from the commanded one by
random noise with standard deviations of 0.03 m on each position axis and 0.015 rad on the
heading. Each step the robot also receives a range and a bearing to every landmark within
6 m, with noise of standard deviation 0.12 m on the range and $2.5^\circ$ on the bearing. The
landmarks are point features scattered near the loop, one of them well inside it. Over 300 steps
the robot goes a little more than once around, so near the end it is back among the landmarks
it saw first.

The job, at every step, is to estimate both the robot pose and the position of every landmark
seen so far, each with a covariance that matches the size of its actual error. This joint
problem is simultaneous localization and mapping (SLAM).

#### Why the obvious approaches fail

Localization assumes a known map and estimates the pose; mapping assumes a known pose and
estimates landmarks. SLAM has neither: the robot must build the map and locate itself in it
at the same time, and the two estimates are coupled. If the robot misjudges its own motion,
every landmark it places inherits that error, and when it later re-sees an early landmark,
that observation should correct the pose and, with it, the whole map built since.

The obvious shortcut alternates the two known problems: place each new landmark using the
current pose estimate, then localize against the landmarks placed so far as if they were
known. This loses the coupling. A landmark placed from a drifted pose carries that drift, the
robot then localizes against it and treats the result as information, and nothing records
that the landmark's error and the robot's error came from the same source. Solving everything
at once as a batch least-squares problem over all poses and all measurements avoids that, but
its size grows with every step, and the robot needs an estimate now.

Re-observing something seen long ago, after the pose estimate has drifted in between, is a
loop closure. It is the one event in a SLAM run that carries new information about the
accumulated drift: dead reckoning alone can only add uncertainty, and a measurement of a
landmark first seen five seconds ago says little about where the robot was thirty seconds
ago. A loop closure ties the current pose back to a part of the map built under a much
older, and much smaller, pose error. Making that correction reach backwards into the map
requires estimating the robot and the landmarks in one joint belief, together with the
correlations between them.

#### The idea

EKF-SLAM does this. It stacks the robot pose and all landmark positions into one state
vector and runs a single EKF over it, with one covariance matrix over the whole stack. Motion
moves only the robot part of the mean, but it also updates how the robot's error lines up with
each landmark's error. A new landmark is appended to the state, already correlated with the
robot that placed it. An observation of one landmark is an ordinary EKF update over the whole
stack, and because the covariance couples everything, that update moves the robot and every
correlated landmark at once. That is how a loop closure reaches back into the map.

The figure shows the demo run at two moments. About two-thirds of the way around the loop
(left, step 204) the landmarks mapped most recently were placed from a drifted pose: their
3-sigma ellipses (the regions within three standard deviations of each estimate) are large and their estimates are visibly off. After the robot returns to its first landmarks
(right), the whole map has tightened, including the landmarks mapped late in the
loop, which the robot is no longer looking at.

![EKF-SLAM on the demo loop at step 204 (left), about two-thirds of the way around the loop, and at step 300 (right), after the robot has returned to the landmarks it saw first. True path in gray, estimated path in red, true landmarks as black crosses, estimated landmarks in blue with their 3-sigma (three standard deviation) ellipses, and the robot's 3-sigma position ellipse in red](assets/loop_run.png)

#### Why it matters

EKF-SLAM was the first method that solved this problem online, and the argument for why the
cross-covariances are needed (the point Durrant-Whyte and Bailey's tutorial stresses) carries
over to every later SLAM system. It was replaced for two reasons that these notes derive:
every update rewrites the whole covariance, which is quadratic in the number of landmarks, and
the linearization makes the filter overconfident over long runs. Factor-graph smoothing, the
subject of a later assignment in this module, fixes both.

### Our corridor example (one robot, two landmarks, one return trip)

The loop has ten landmarks and a three-number pose, so its covariance is $23 \times 23$, too
big to follow by hand. We use a one-dimensional version of it that keeps the same story and
call it our corridor example. A robot moves along a line. Its state is a single position $x$,
and each landmark is a single position $\ell_j$ on the same line. The sensor reports the
signed offset $z = \ell_j - x + v$ with noise variance $r = 0.01$ (standard deviation 0.1 m),
and each leg of driving adds variance $q = 0.04$ (standard deviation 0.2 m) to the robot
position. Measurements below are taken noise-free so that the arithmetic stays clean; the
filter does not know that.

1. The robot starts at $x_0 = 0$, which defines the map frame, so its variance there is zero.
   It sees landmark $\ell_0$ (truly at 3) at offset $3$ and places it at $3$ with variance
   $0.01$.
2. It is commanded to drive $+10$ m and truly drives $10.2$. Dead reckoning says $x = 10$ with
   variance $0.04$.
3. It sees landmark $\ell_1$ (truly at 13) at offset $2.8$ and places it at $10 + 2.8 = 12.8$.
   The 0.2 m drift of the first leg is now in both the robot estimate and $\ell_1$.
4. It is commanded to drive $-9$ m and truly drives $-8.9$, ending at $1.3$. Dead reckoning says
   $x = 1$ with variance $0.08$.
5. It re-sees $\ell_0$, at offset $1.7$, where the map and the dead-reckoned pose predict $2$.
   This is the loop closure.

The figure shows the belief before step 5 and two versions of it after: the full EKF-SLAM
update, and the same update with the robot-landmark cross-covariance set to zero just before
it. In this example that gives the same numbers as a filter that never kept the cross terms,
because no diagonal entry of the prediction or the landmark initialization depends on them.
Both correct the robot from $1.00$ to $1.24$. Only the full filter also moves $\ell_1$, from
$12.80$ to $12.92$, and shrinks its uncertainty, although $\ell_1$ was not observed in step 5.
The sections below derive each number.

![Our corridor example. Three rows show the robot x and the landmarks l0 and l1 with 1-sigma bars, before re-seeing l0, after the update with the full joint covariance, and after the same update with the cross-covariance dropped. Black ticks mark the true positions. Only the full update moves l1 toward its true position](assets/corridor.png)

In the demo loop, $\ell_0$ stands for a landmark near the start, such as the one at $(4, 0)$,
$\ell_1$ for a landmark on the far side, and the two legs for the two halves of the loop. The
corridor keeps the joint state, the way a landmark inherits the pose error at the moment it is
placed, and the way a loop closure spreads through the cross-covariance. It drops the heading,
which in the plane turns a small angular error into a large sideways error at long
range, and it drops the nonlinearity: the corridor's models are linear, so its filter is an
exact Kalman filter and cannot show the overconfidence discussed at the end.

### Overview of the rest of the notes

1. The joint state and what its covariance stores. The off-diagonal blocks, read through
   Gaussian conditioning, the same formula that is the Kalman update.
2. Prediction. The EKF prediction step applied to a state where only the robot moves.
3. Landmark initialization. Pushing a Gaussian through the inverse of the sensor model, the
   linear-map rule for covariances.
4. The measurement update over the joint state. The EKF update with a sparse Jacobian and a
   dense gain, which is where loop closure happens.
5. What a measurement can and cannot reduce. The update written in information form (storing
   the inverse covariance, the information matrix, instead of the covariance), and the
   unobservable absolute frame.
6. The Mahalanobis gate and data association. A chi-square test on the innovation, the same
   statistic as the normalized innovation squared of filter consistency.
7. Consistency and the normalized estimation error squared (NEES), a chi-square test of whether
   the reported covariance matches the actual error. Why EKF-SLAM becomes overconfident.
8. Why the cost limited it. The quadratic update cost and the fill-in of marginalization, which
   lead to factor graphs.

Five exercises, placed after the sections they test, are meant to be worked by hand. Each has
a worked answer that can be expanded.

Notation follows the Kalman-filter assignment: $\mu$ and $P$ are the mean and covariance,
$\mu^-, P^-$ the predicted belief and $\mu^+, P^+$ the updated one where the distinction
matters. *Probabilistic Robotics* writes $\Sigma$ for $P$.

### The joint state

The state grows as new landmarks are seen:

$$\mu = \big[\,\underbrace{p_x,\,p_y,\,\theta}_{\text{robot}},\ \underbrace{\ell_{0x},\,\ell_{0y}}_{\text{lm }0},\ \underbrace{\ell_{1x},\,\ell_{1y}}_{\text{lm }1},\ \dots\,\big]^\top,$$

with the matching joint covariance $P$ of size $m \times m$, where $m = 3 + 2N$ for $N$
landmarks. Landmark $j$ is 0-indexed and lives at state indices $3+2j$ and $3+2j+1$. With the
demo's ten landmarks, $m = 23$. In our corridor example the state is $[x, \ell_0, \ell_1]$ and
$m = 3$.

### What the joint covariance stores

Write the joint covariance in blocks, robot first and map second:

$$P = \begin{bmatrix} P_{rr} & P_{rm} \\ P_{mr} & P_{mm} \end{bmatrix}.$$

$P_{rr}$ is $3\times3$ and holds the pose uncertainty, $P_{mm}$ is $2N \times 2N$ and holds
the landmark uncertainties, and $P_{rm} = P_{mr}^\top$ is the cross-covariance between them.
What a cross block does follows from Gaussian conditioning. For a jointly Gaussian pair
$(a, b)$ with

$$\begin{bmatrix} a \\ b \end{bmatrix} \sim \mathcal{N}\!\left( \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}, \begin{bmatrix} P_{aa} & P_{ab} \\ P_{ba} & P_{bb} \end{bmatrix} \right),$$

learning the value of $b$ moves the estimate of $a$ by

$$\mathbb{E}[a \mid b] = \mu_a + P_{ab} P_{bb}^{-1} (b - \mu_b), \qquad \operatorname{Cov}(a \mid b) = P_{aa} - P_{ab} P_{bb}^{-1} P_{ba}.$$

Both expressions are scaled by $P_{ab}$. If the cross block is zero, observing $b$ leaves the
mean of $a$ where it was and leaves its covariance at $P_{aa}$: the two variables cannot
inform each other at all. That is the entire argument for carrying the off-diagonal blocks.
Landmarks observed from the same uncertain pose share that pose's error, so their estimates
are correlated, and a later measurement that corrects one corrects the others through those
blocks. Dropping the cross terms to get a block-diagonal covariance breaks EKF-SLAM rather
than approximating it. Loop closure stops working, because the information from
re-seeing an old landmark has no path to the rest of the map.

Two names for pieces of that formula, both used later. The *marginal* covariance of a subset
of the state, meaning its covariance with everything else integrated out, is simply the
corresponding diagonal sub-block $P_{aa}$ - for a Gaussian, marginalizing is dropping rows
and columns. The *conditional* covariance $P_{aa} - P_{ab}P_{bb}^{-1}P_{ba}$ is the Schur
complement of $P_{bb}$ in $P$. Marginal and conditional are different matrices, and the
conditional is the smaller of the two in the Loewner order, the ordering in which
$A \succeq B$ means $A - B$ is positive semidefinite (the section on what a measurement can
reduce says more).

In our corridor example, just before the loop closure,

$$P = \begin{bmatrix} 0.08 & 0 & 0.04 \\ 0 & 0.01 & 0 \\ 0.04 & 0 & 0.05 \end{bmatrix} \quad \text{over } [x, \ell_0, \ell_1].$$

The $0.04$ between $x$ and $\ell_1$ is the first leg's drift, which both share: $\ell_1$ was
placed from the pose at the end of that leg. The robot's $0.08$ is that same $0.04$ plus the
second leg's $0.04$, which $\ell_1$ does not share. The zeros in the $\ell_0$ row say that
$\ell_0$'s error (its sensor noise) is independent of everything else, because it was placed
from a pose known exactly. The figure's third row shows the filter with the $0.04$ entries
set to zero.

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
forward or the map silently decorrelates from the robot and loop closure stops correcting
the map. It is the
easiest thing in the assignment to leave out and not notice, because a filter without it
still produces plausible-looking trajectories.

In our corridor example the motion model is $x \leftarrow x + u$, so $F = 1$. The second leg
adds $q = 0.04$ to the robot's variance and leaves the cross-covariance $0.04$ with $\ell_1$
unchanged, which is how $P_{xx}$ went from $0.04$ to $0.08$ while $P_{x\ell_1}$ stayed at
$0.04$. In the plane $F$ is not the identity: it turns a heading error into a sideways
position error, and $F P_{rm}$ applies the same mixing to the robot rows of the cross blocks.

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

where $P_{r,:}$ is the three robot rows of the current covariance, a $3 \times m$ matrix, so
$P_{\ell x}$ is $2 \times m$. The augmented covariance puts the old $P$ in the top-left,
$P_{\ell\ell}$ in the new bottom-right $2\times2$ block, and $P_{\ell x}$ and its transpose
in the new off-diagonal strips. A landmark is therefore born already correlated with the
robot and, through the robot's existing correlations, with every landmark already mapped.

In our corridor example the inverse model is $\ell = x + z$, so $G_r = 1$ and $G_z = 1$.
Placing $\ell_1$ from the robot at $x = 10$ with variance $0.04$ gives
$P_{\ell_1\ell_1} = 0.04 + 0.01 = 0.05$, a cross-covariance $0.04$ with the robot (the robot's
own row, $G_r P_{x,:} = [0.04, 0]$), and $0$ with $\ell_0$, because the robot was uncorrelated
with $\ell_0$ at that moment. These are the entries of the matrix in the previous section.

> **Exercise 1 (a landmark born from an uncertain heading).** The demo robot starts with pose
> covariance $P_{rr} = \operatorname{diag}(0.05, 0.05, 0.03)$ (variances, in m$^2$, m$^2$, and
> rad$^2$). Suppose it stands at $(0, 0)$ facing $+x$ and sees a new landmark straight ahead,
> $z = [4, 0]$, with the demo's sensor noise
> $R = \operatorname{diag}(0.12^2, (2.5^\circ)^2)$, where $2.5^\circ = 0.04363$ rad. Compute
> $G_r$, $G_z$, $P_{\ell\ell}$ split into its pose part and its sensor part, the landmark's
> standard deviations along and across the ray, and the cross-covariance $G_r P_{rr}$. Which
> source dominates the across-ray uncertainty?

<details>
<summary>Answer</summary>

With $\beta = 0$, $\cos\beta = 1$ and $\sin\beta = 0$:

$$G_r = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 4 \end{bmatrix}, \qquad G_z = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}.$$

The pose part is $G_r P_{rr} G_r^\top = \operatorname{diag}(0.05,\ 0.05 + 16 \cdot 0.03) = \operatorname{diag}(0.05, 0.53)$.
The sensor part is $G_z R G_z^\top = \operatorname{diag}(0.0144,\ 16 \cdot 0.001904) = \operatorname{diag}(0.0144, 0.0305)$.
So $P_{\ell\ell} = \operatorname{diag}(0.0644, 0.5605)$: a standard deviation of $0.254$ m along
the ray and $0.749$ m across it.

Across the ray, the heading term $16 \cdot 0.03 = 0.48$ is $86\%$ of the total. A heading
standard deviation of $0.173$ rad ($9.9^\circ$) at 4 m throws the landmark sideways by about
$0.69$ m, far more than the $2.5^\circ$ bearing noise does ($0.17$ m). The new landmark
inherits the robot's uncertainty much more than the sensor's.

The cross-covariance is

$$G_r P_{rr} = \begin{bmatrix} 0.05 & 0 & 0 \\ 0 & 0.05 & 0.12 \end{bmatrix},$$

rows for $\ell_x, \ell_y$ and columns for $p_x, p_y, \theta$. The $0.12$ says the landmark's
sideways error is tied to the robot's heading error: if the robot really faces a little to the
left, the landmark really is a little to the left. A later correction of the heading moves the
landmark through this entry.

</details>

### The measurement update over the joint state

To fold in an observation of a mapped landmark $j$, predict the reading
$h(\mu)$ - range and bearing from the robot to landmark $j$ - and form the innovation
$y = z - h(\mu)$, the difference between what the sensor said and what the filter expected.
The bearing component of $y$ is an angle difference and must be wrapped to $(-\pi, \pi]$, or
a reading at $+179^\circ$ against a prediction at $-179^\circ$ produces a $358^\circ$ error
and a violent spurious correction.

The measurement Jacobian $H$ is $2 \times m$ and almost entirely zero: a $2\times3$ block in
the robot columns and a $2\times2$ block in landmark $j$'s columns, both carried over from
the range-bearing model. Range and bearing depend on the robot position and the landmark
only through the offset $\ell_j - p$, so the landmark block is the negative of the robot
block's two position columns. Everything else is the standard EKF update, run over the full
joint state:

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
symmetric positive definite no matter how much cancellation happens inside it. The
Kalman-filter assignment derives it.

Here is the loop closure of our corridor example, step 5. The measurement $z = \ell_0 - x$ has
$H = [-1, 1, 0]$ over $[x, \ell_0, \ell_1]$, so with the covariance from the previous sections

$$PH^\top = \begin{bmatrix} -0.08 \\ 0.01 \\ -0.04 \end{bmatrix}, \qquad S = 0.08 + 0.01 + 0.01 = 0.10, \qquad K = \begin{bmatrix} -0.8 \\ 0.1 \\ -0.4 \end{bmatrix}.$$

The innovation is $y = 1.7 - (3 - 1) = -0.3$: the landmark is 0.3 m closer than the map and the
dead-reckoned pose predict. The corrections are $Ky = [+0.24, -0.03, +0.12]$, which move the
robot to $1.24$, $\ell_0$ to $2.97$, and $\ell_1$ to $12.92$. The variances drop to $0.016$,
$0.009$, and $0.034$ (standard deviations $0.126$, $0.095$, $0.184$). The filter blames most
of the surprise on the robot, whose variance $0.08$ dominates $S$, and moves $\ell_1$ half as
far as the robot because half of the robot's variance, the first leg, is shared with
$\ell_1$. The $\ell_1$ entry of $K$ is $P_{\ell_1 x} \cdot (-1) / S = -0.04 / 0.10$, the
cross-covariance divided by the innovation variance. In the filter that dropped the
cross-covariance this entry is $0$, and $\ell_1$ stays at $12.80$ with its old variance
$0.05$, the third row of the corridor figure.

> **Exercise 2 (a loop closure with a perfect second leg).** Rerun the corridor loop closure,
> but suppose the second leg was driven without error, so the robot's variance before step 5
> is $0.04$ instead of $0.08$ (everything else as in the matrix above, and the same innovation
> $y = -0.3$). Compute $S$, $K$, the corrections, and the posterior variances of $x$ and
> $\ell_1$. Then compute the variance of the offset $\ell_1 - x$ before and after the update,
> and explain why it does not change.

<details>
<summary>Answer</summary>

Now $P_{xx} = 0.04$, $P_{x\ell_1} = 0.04$, $P_{\ell_0\ell_0} = 0.01$, $P_{\ell_1\ell_1} = 0.05$.
Then $PH^\top = [-0.04, 0.01, -0.04]^\top$, $S = 0.04 + 0.01 + 0.01 = 0.06$, and
$K = [-2/3, 1/6, -2/3]^\top$. The corrections are $Ky = [+0.2, -0.05, +0.2]$: the robot and
$\ell_1$ move by exactly the same amount.

The posterior variances are $P_{xx} = 0.04 - 0.04^2/0.06 = 0.0133$ and
$P_{\ell_1\ell_1} = 0.05 - 0.04^2/0.06 = 0.0233$, with $P_{x\ell_1} = 0.0133$.

The offset variance is $P_{\ell_1\ell_1} + P_{xx} - 2P_{x\ell_1}$: before,
$0.05 + 0.04 - 0.08 = 0.01$; after, $0.0233 + 0.0133 - 0.0267 = 0.01$. With a perfect second
leg, all of the robot's error is the first-leg drift that $\ell_1$ also inherited, so
$\ell_1 - x$ is just the measurement taken at step 3, with variance $r = 0.01$. Re-seeing
$\ell_0$ says where the robot is relative to $\ell_0$; it says nothing new about where $\ell_1$
is relative to the robot, so the filter moves the two rigidly together. The cross-covariance
routes the correction to the shared error and leaves the independent part alone.

</details>

### What a measurement can and cannot reduce

The update has a monotonicity property. The test suite checks it, and it also explains what
the suite deliberately does not check.

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
what the update tests assert. The determinant of the joint covariance
cannot increase. The information matrix cannot decrease in the Loewner order. And every
principal sub-block of $P$ - that is, every marginal, including each single landmark's
$2\times2$ block - cannot increase either, so no marginal determinant can grow.

It does *not* follow that any particular marginal strictly shrinks, so a test cannot assert
that. Take the suite's own two-landmark fixture: a robot at
a fixed pose initializes two landmarks, then re-observes landmark 0 without having moved.
The joint covariance determinant drops by a factor of four. Landmark 1's $2\times2$ block
comes back unchanged, and so does the robot's $3\times3$ block. The only change anywhere is
in landmark 0's own block, and it is exactly $\frac{1}{2} G_z R G_z^\top$ - the second
reading of the same relative offset averaged with the first, halving that half of
$P_{\ell\ell}$ while leaving the $G_r P_{rr} G_r^\top$ half alone.

That is not a numerical accident. Range and bearing measure the offset from the robot to a
landmark, never an absolute position, so no amount of measuring from a stationary pose says
anything about where the pair actually is. The absolute frame is unobservable, and the
unobservable directions are exactly the ones whose uncertainty refuses to shrink. In the
plane there are three such directions: translating the robot and every landmark together by
the same vector (two directions), and rotating them all together about a point (one), leave
every range and bearing unchanged. In our corridor example there is one, shifting $x$,
$\ell_0$, and $\ell_1$ by the same amount, and indeed $H = [-1, 1, 0]$ gives
$H\,[1, 1, 1]^\top = 0$. The corridor gets its map frame from the start pose, whose variance
is zero. The demo starts with a small nonzero pose covariance instead, which plays the same
role: it is uncertainty about where the map frame is, and range-bearing readings carry no
information about it.

> **Exercise 3 (re-observing from a standstill).** In the corridor, a robot with variance $p$
> places a landmark $\ell_0$ (so $P_{\ell_0\ell_0} = p + r$ and $P_{x\ell_0} = p$) and, without
> moving, observes $\ell_0$ again. Compute $S$, $K$, and the posterior covariance of
> $[x, \ell_0]$ in terms of $p$ and $r$. By what factor does the joint determinant drop? Compare
> with the two-dimensional fixture described above.

<details>
<summary>Answer</summary>

With $H = [-1, 1]$ and

$$P = \begin{bmatrix} p & p \\ p & p + r \end{bmatrix}, \qquad PH^\top = \begin{bmatrix} 0 \\ r \end{bmatrix}, \qquad S = p - 2p + (p + r) + r = 2r,$$

so $K = [0, \tfrac12]^\top$. The robot gets no correction at all, and

$$P^+ = P - K S K^\top = \begin{bmatrix} p & p \\ p & p + r/2 \end{bmatrix}.$$

The robot's variance stays $p$, the landmark's drops from $p + r$ to $p + r/2$: the sensor half
is averaged over two readings and the pose half is untouched. The determinant goes from
$p(p + r) - p^2 = pr$ to $pr/2$, a factor of $2$. The two-dimensional fixture has a
two-dimensional measurement, and halving the sensor part along both of its directions gives
the factor $2^2 = 4$ quoted above. In both cases the measurement is about the offset
$\ell_0 - x$, and $[1, 1]$, the shift of both together, is the direction it cannot see.

</details>

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
$\chi^2_2$ has the closed form $-2\ln(1 - 0.95) \approx 5.991$.

The number carries a meaning that a hand-tuned threshold would not: about 5% of correct
associations fall outside the gate and get rejected. Choosing 0.99 instead
($\approx 9.21$) rejects only 1% of true matches at the cost of admitting more wrong ones.
Picking the percentile is picking a false-reject rate.

### Data association

In a real run the landmark index behind a measurement is unknown, and getting it wrong is
the dominant failure mode of SLAM: one bad association welds two parts of the map together
and the error never comes out. The nearest-neighbor test computes, for every mapped landmark
$j$, the innovation $y_j = z - h_j(\mu)$ (bearing wrapped), its covariance
$S_j = H_j P H_j^\top + R$, and then the squared Mahalanobis distance
$d_j^2 = y_j^\top S_j^{-1} y_j$.
The measurement goes to whichever landmark has the smallest $d_j^2$, provided that value is
strictly below the gate; otherwise the measurement is treated as a new landmark and
initialized. An empty map has nothing to match against, so every reading starts a new
landmark.

Note that $S_j$ contains $H_j P H_j^\top$, not just $R$. The gate therefore widens on its
own when the robot is unsure of its pose or of that landmark, which is precisely when a
prediction deserves less trust. A gate on raw Euclidean distance has no such behavior and
either rejects good matches after a long blind drive or accepts bad ones when the pose is
well known.

> **Exercise 4 (the same innovation, two gates).** A reading of a landmark straight ahead at
> 4 m comes back with innovation $y = [0.25\ \text{m}, 4^\circ]$ ($4^\circ = 0.06981$ rad). The
> sensor noise is the demo's, $R = \operatorname{diag}(0.0144, 0.001904)$. (a) Compute $d^2$
> using $S = R$ alone, as if the robot pose and the landmark were known exactly. (b) Now the
> landmark is known exactly but the robot has $P_{rr} = \operatorname{diag}(0.05, 0.05, 0.03)$,
> and the robot block of $H$ at this geometry is
> $\begin{bmatrix} -1 & 0 & 0 \\ 0 & -0.25 & -1 \end{bmatrix}$. Compute $S$ and $d^2$. Which
> versions pass the 95% gate of $5.991$?

<details>
<summary>Answer</summary>

(a) $S = R$ is diagonal, so $d^2 = 0.25^2/0.0144 + 0.06981^2/0.001904 = 4.34 + 2.56 = 6.90$.
That exceeds $5.991$, so the reading is rejected and would start a duplicate landmark.

(b) $H P_{rr} H^\top = \operatorname{diag}(0.05,\ 0.0625 \cdot 0.05 + 0.03) = \operatorname{diag}(0.05, 0.0331)$,
so $S = \operatorname{diag}(0.0644, 0.0350)$ and
$d^2 = 0.0625/0.0644 + 0.004874/0.0350 = 0.97 + 0.14 = 1.11$, comfortably inside the gate.

The same innovation is an outlier for a filter that is sure of its pose and an ordinary reading
for one that is not. The bearing term changes most: a heading standard deviation of
$0.173$ rad ($9.9^\circ$) makes a $4^\circ$ bearing surprise unremarkable. A gate built on $R$
alone would reject correct matches exactly when the robot most needs them, after a stretch of
dead reckoning.

</details>

Nearest-neighbor gating tests each pairing on its own, and that is its weakness. Suppose two
measurements in the same scan both land within the gate of the same landmark, or a set of
pairings is individually plausible but geometrically impossible together because the implied
relative positions contradict each other. Joint compatibility branch and bound (JCBB, Neira
and Tardos 2001) fixes this by testing whole hypotheses: stack $n$ measurements into one
$2n$-vector, stack the corresponding predictions, build the full $2n \times 2n$ innovation
covariance including the cross terms between the pairings, and gate the resulting $d^2$
against $\chi^2_{2n}$. Individual gating discards the cross terms, and the cross terms
rule out sets of pairings that each look fine alone. The search over hypotheses is
exponential, hence the branch and bound. The demo in this assignment gives the filter the true
landmark identity of each reading, so the filter itself can be checked separately from
association; the association step is tested on its own.

### Consistency and NEES

A filter is consistent when its reported covariance correctly describes its actual error - not
when the error is small, but when the error is as large as the filter claims it might be. An
optimistic (overconfident) filter reports a covariance smaller than its true error, and that
is dangerous in a way that a merely inaccurate filter is not: downstream consumers weight the
estimate by the reported covariance, so an overconfident SLAM back-end causes a planner to
trust a pose it should have questioned.

The statistic that measures it is the NEES, the normalized estimation error squared, which
is the Mahalanobis distance again with the state error in place of the innovation. Take the
robot part of the state, the true pose $x_r$ (available in simulation, not on a real robot),
and the error $e = \mu_r - x_r$ with its heading component wrapped. Then

$$\text{NEES} = e^\top P_{rr}^{-1} e.$$

If the filter is telling the truth, $e \sim \mathcal{N}(0, P_{rr})$, so by the same whitening
argument NEES is $\chi^2_3$ for a 3-DOF pose: its expected value is 3. Above 3 on average
means the true error is larger than the reported covariance admits, which is optimism. Below
3 means conservatism, which wastes information but is not unsafe. The 95th percentile of
$\chi^2_3$ is $\approx 7.815$. The Kalman-filter assignment checks consistency through the
innovation instead (the normalized innovation squared, which averages the measurement
dimension); NEES needs the ground truth and tests the state directly.

EKF-SLAM is known to go optimistic, for two reasons that reinforce each other. The first is
generic to the EKF: $F$, $H$, $G_r$, and $G_z$ are first-order Taylor expansions taken at the
current estimate rather than at the truth, and the neglected curvature is never charged to
$P$ anywhere, so the filter under-reports its own approximation error and does so
cumulatively over thousands of steps. The second is specific to SLAM, and it is the
unobservable absolute frame from the previous sections. Because the true system cannot observe
global position and heading, a correct filter should never gain information in those three
directions. The linearized filter does gain some, because successive updates evaluate their
Jacobians at slightly different and slightly wrong estimates, and that inconsistency between
linearization points leaks information into directions where none exists. Julier and Uhlmann
(2001) gave the first explicit counterexample, and Huang and Dissanayake (2007) traced the
mechanism to exactly this observability mismatch. Our corridor example cannot show either
effect: its models are linear, so its Jacobians do not depend on the estimate, and its filter is
consistent in distribution: over many noisy runs its reported covariance equals the actual error
covariance. A single noise-free run like the one above is one realization, not a test of that.

### Why the cost limited it

Count the work in one update. $H$ is $2 \times m$ with only five nonzero columns, so $HP$
costs $O(m)$, $S$ is $2\times2$, and $K = PH^\top S^{-1}$ is $m \times 2$ and also costs
$O(m)$. The covariance correction is where the cost lives: it changes every one of the $m^2$
entries of $P$, because $K$ is dense. So EKF-SLAM is $\Theta(m^2) = \Theta(N^2)$ per update,
and no cleverness about the sparsity of $H$ removes that - the output itself is $N^2$ numbers.

Put numbers on it. At $N = 1000$ landmarks, $m = 2003$ and $P$ holds about 4.0 million
doubles, 32 MB. Ten observations per step at 10 steps per second means 100 updates per
second, so the filter rewrites 3.2 GB of covariance every second, before any arithmetic. That
cost limits EKF-SLAM in practice, and it is why the method is remembered as a few-hundred-landmark
technique.

The second problem is structural. Consider the same
problem as a graph: one variable per robot pose and one per landmark, one factor per
measurement. Its information matrix has a nonzero block only where two variables appear in a
common measurement, so it is sparse - each pose touches a handful of landmarks. A filter
throws that away. Filtering keeps only the *current* pose and marginalizes out every past
one, and marginalizing a variable out of an information matrix is a Schur complement, the
same operation from the conditioning formula earlier. A Schur complement fills in: every pair
of variables the marginalized pose touched becomes directly connected. Marginalize away a
whole trajectory and the surviving landmarks end up connected to each other, which is the
dense covariance from the first section, seen from the other side.

> **Exercise 5 (fill-in from marginalizing one pose).** In the corridor, a robot pose $x$ with
> a prior of information $1$ (variance $1$) observes two landmarks, each with $r = 1$, and the
> landmarks have no prior. The information matrix over $[x, \ell_0, \ell_1]$ is the prior plus
> $H^\top r^{-1} H$ for each measurement. Write it down, note which entries are zero, and
> marginalize out $x$ with the Schur complement
> $\Omega_{mm} - \Omega_{mx}\Omega_{xx}^{-1}\Omega_{xm}$. What appears that was not there?

<details>
<summary>Answer</summary>

The measurement of $\ell_j$ has $H = [-1, 1, 0]$ or $[-1, 0, 1]$, contributing $+1$ on the
$x$ and $\ell_j$ diagonals and $-1$ on their off-diagonal pair. With the prior,

$$\Omega = \begin{bmatrix} 3 & -1 & -1 \\ -1 & 1 & 0 \\ -1 & 0 & 1 \end{bmatrix}.$$

The $\ell_0$-$\ell_1$ entry is $0$: no measurement involves both landmarks. Marginalizing $x$,

$$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} - \frac{1}{3}\begin{bmatrix} -1 \\ -1 \end{bmatrix}\begin{bmatrix} -1 & -1 \end{bmatrix} = \begin{bmatrix} 2/3 & -1/3 \\ -1/3 & 2/3 \end{bmatrix}.$$

The new $-1/3$ links the two landmarks directly. The result equals the inverse of the landmark
block of $\Omega^{-1}$, as marginalization in information form must. Each pose a filter
marginalizes adds such links between every pair of landmarks that pose observed, and a filter
marginalizes every pose, so after a long run the landmark information matrix, like the
covariance, is dense.

</details>

Smoothing keeps all the poses as variables instead of marginalizing them. That keeps the
sparsity, and it also allows relinearizing every factor at the improved estimate on each
iteration rather than being stuck with the Jacobian computed once at whatever the estimate
happened to be at the time. Both of EKF-SLAM's problems are addressed by the same change,
which is the subject of the factor-graph and bundle-adjustment assignment later in this
module.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the
declaration and comments in `ekf_slam.hpp` give the signature and shapes.

| # | Function | File | Concept from the notes |
|---|---|---|---|
| 1 | [`slam_predict()`](ekf_slam.cpp) | `ekf_slam.cpp` | prediction: robot mean through $f$, $P_{rr} \leftarrow FP_{rr}F^\top + Q$, $P_{rm} \leftarrow FP_{rm}$ |
| 2 | [`slam_add_landmark()`](ekf_slam.cpp) | `ekf_slam.cpp` | the inverse measurement model and the augmentation with $P_{\ell\ell}$ and $P_{\ell x}$ |
| 3 | [`slam_update()`](ekf_slam.cpp) | `ekf_slam.cpp` | the joint EKF update: sparse $H$, wrapped bearing residual, Joseph form |
| 4 | [`slam_associate()`](ekf_slam.cpp) | `ekf_slam.cpp` | nearest neighbor by Mahalanobis distance, gated by the chi-square threshold |

You may not include an existing SLAM or solver library (Sophus, manif, Ceres, GTSAM, g2o); a
test scans the sources.

Notes on specific holes:
- The state is an `Eigen::VectorXd` that grows by two entries per landmark. Landmark $j$ is
  `mu.segment<2>(3 + 2 * j)`, and its covariance block starts at row and column `3 + 2 * j`.
- The provided primitives in `models.cpp` are `robot_f`, `robot_F_x`, `range_bearing`,
  `range_bearing_H_robot` (the $2\times3$ robot block of $H$), and `range_bearing_H_land` (the
  $2\times2$ landmark block). `wrap_angle` (wrap to $(-\pi, \pi]$) is in `ekf_slam.hpp`. Call
  these; do not reimplement them.
- `slam_predict`: the cross-block update $P_{rm} \leftarrow FP_{rm}$ only exists when the map
  is non-empty (`m > 3`); write both the top-right strip and its transpose.
- `slam_add_landmark`: $P_{\ell x}$ is `Gr * P.topRows<3>()`, a $2 \times m$ matrix. The new
  covariance is $(m+2) \times (m+2)$ with the old $P$ in its top-left corner.
- `slam_update`: wrap the bearing component of the residual, and wrap the heading after the
  mean update. The reference forms the dense $m\times m$ matrix $I - KH$ and multiplies it out
  twice, which is $\Theta(m^3)$; written as a rank-2 correction the same update is
  $\Theta(m^2)$. With $N = 10$ in this simulation, $m = 23$ and it makes no difference, and
  the Joseph form written literally is much easier to read and to check against the formula.
- `slam_associate`: return the index with the smallest $d_j^2$ only if it is strictly below
  `gate`, else $-1$; return $-1$ immediately on an empty map. `runner.run_slam` calls it only
  with `known_association=False`, and on $-1$ calls `slam_add_landmark`. The demo runs in
  `viz.py`, `assets/make_figures.py`, and `test_consistency.py` use the default
  `known_association=True`, which assigns each reading to its landmark by the simulator's true
  id, so `slam_associate` does not run there; `test_association.py` tests it directly.
- `config.py` holds the noise, the initial pose covariance `INIT_P_DIAG` (variances), the
  landmark positions, and `GATE`, computed as `chi2.ppf(0.95, df=2)`, the $5.991$ from the
  notes. `robot_nees` in `_helpers.py` is the two-line NEES formula.

### Building and running

Same toolchain as the rest of the module: a C++17 compiler plus CMake, pybind11, and Eigen
(the `environment.yml` a14 block). You never call CMake by hand: importing `_impl.py` (which
the tests and `viz.py` do) configures and builds a pybind11 extension with CMake, from the
top-level `ekf_slam.cpp` by default or from `solution/ekf_slam.cpp` under
`NANOVISION_IMPL=solution`, compiled together with `models.cpp` and `bindings.cpp` in both
cases. The build is cached under `build/student/` or `build/solution/`, so only the first run
or a source change compiles.

Activate the environment first (`conda activate nanovision`); the Makefile calls the `python`
on the path. Then:

```
make verify A=a14_2_ekf_slam   # build + run the reference (solution/); the green target
make test   A=a14_2_ekf_slam   # build + run YOUR code; red until the holes are filled
make viz    A=a14_2_ekf_slam        # render the EKF-SLAM loop (reference)
make viz-mine A=a14_2_ekf_slam      # the same, from YOUR code, once the holes are filled
```

`make test` also accepts `K="associate"` to run only matching tests and `SKIP_HOLES=1` to
report unfilled holes as skips instead of failures.

The tests favor implementation-independent statements:
- `test_landmark_init.py`: a landmark initialized from a noise-free reading lands at its true
  position, the augmented covariance is $5 \times 5$, symmetric, and PSD, and the existing
  robot block is carried over untouched.
- `test_update_covariance.py`: on the two-landmark fixture from the notes, an update never
  increases the joint covariance determinant and never decreases the information matrix in
  the Loewner order, both from the argument in "What a measurement can and cannot reduce", and
  it keeps the joint covariance symmetric and PSD.
- `test_association.py`: an empty map associates nothing, a realistically noisy observation of
  each of two landmarks associates to the right one, and a spurious reading is rejected.
- `test_consistency.py`: on a short mild trajectory (40 steps) the mean robot NEES stays inside
  a loose band around its expectation of 3 (the assertion is
  $0 < \overline{\text{NEES}} < 6$, deliberately generous, since the long-loop optimism is a
  demonstration and not something to gate on); after a 60-step run the landmark-landmark cross
  blocks are non-negligible (largest Frobenius norm above $0.02$), so the map really did become
  correlated; and the mapped landmarks track the truth (mean error below 0.6 m).
- `test_forbidden_imports.py`: no SLAM or solver library is included.

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
thresholded. Scrub through the second half of the run to watch the NEES rise, and to the end to
watch loop closure pull it back down and tighten the whole map at once.

The figures in the notes are generated by `assets/make_figures.py` (run it from `assets/`),
which reimplements the filter in NumPy with the same simulator, seed, and driver logic as
`viz.py` and `runner.py` and prints the run statistics quoted below. Its run matches the
reference C++ filter to within $10^{-9}$.

What you should see when you run this. On the 300-step loop with the provided seed and known
association (so these numbers do not exercise the gate), the
final covariance has $2\times2$ landmark-landmark cross blocks with Frobenius norms between
0.06 and 0.16 against diagonal blocks between 0.07 and 0.17. The off-diagonal structure is the
same magnitude as the diagonal, so the map really is dense, not almost-block-diagonal.

The robot NEES starts far below 3 and climbs. Averaged over six equal segments of the run it
goes roughly $0.2 \to 0.9 \to 1.9 \to 3.0 \to 4.5 \to 1.7$. The early values below 3 are the
filter being conservative rather than optimistic: `runner.py` starts the filter at the exact
true pose but with `INIT_P_DIAG` variances of $(0.05, 0.05, 0.03)$, so it claims about 0.22 m
of position uncertainty while having none, and the ratio in the NEES starts near zero. That
start pose also defines the map frame, since nothing in the run measures absolute position,
so errors stay small nearby and grow with distance from it. The climb past 3 in the fourth and fifth
segments is the optimism: at a segment mean of 4.5 the squared pose error is about 1.5
times what $P_{rr}$ predicts, so the reported standard deviations are roughly 20% too small.
Individual steps cross the 7.815 bound only three times in 300. The final segment drops back
to 1.7 as the robot re-sees its first landmarks and the loop closure re-tightens the map.
Over the same run the mean mapped-landmark error, the second scalar panel in the viz, peaks
near 0.72 m about two-thirds of the way around (step 204, the left panel of the loop figure)
and falls to about 0.25 m by the end. This is one seed on a small simulated loop, not a
measurement of how badly EKF-SLAM diverges at scale; the literature above is the source for
that.

The simulated world is a point-landmark range-bearing loop, the standard EKF-SLAM teaching
environment. The canonical real dataset for this method is Victoria Park (a vehicle driving a
park, range-bearing to tree trunks); the tests do not gate on it because its ground truth is
sparse GPS and its data association is truly ambiguous, and both make a deterministic
test fragile.

## In interviews

EKF-SLAM is a standard interview topic for perception and robotics roles, often as the lens
for "how does SLAM actually work."

What is in the state, and why the off-diagonal blocks matter. The answer is the robot pose
and all landmarks in one joint Gaussian, and the cross-covariance blocks let a loop
closure correct the whole map. Be ready for the follow-up: what happens if you drop the
correlations and keep only block diagonals? Loop closure breaks, because the Gaussian
conditioning update is proportional to the cross block, so a zero cross block means
re-observing an old landmark cannot move anything else.

Why EKF-SLAM is $O(N^2)$ and what that killed. The joint covariance is dense and every update
rewrites all of it, so it does not scale; this is the practical reason the field moved to
sparse, graph-based back-ends. Know that the information matrix of a pose graph is sparse
while its inverse, the covariance, is dense, and that filtering destroys the sparsity by
marginalizing out old poses, which fills in edges between everything they touched.

Consistency. Expect "is EKF-SLAM consistent?" The answer is no, not in general: it
linearizes at the current estimate and becomes optimistic as that error accumulates, which
is one of the two reasons (with cost) that factor-graph smoothing replaced it. Be able to
define NEES as $e^\top P^{-1} e$ and say its expected value is the state dimension.

Data association as the real failure mode. The filter math is the easy part; real systems
break on deciding which measurement belongs to which landmark. Know the Mahalanobis gate, why its
threshold is a chi-square quantile at the innovation's dimension, and that harder settings
need joint compatibility (JCBB) or back-ends with outlier-rejecting cost functions. A common
on-the-spot question is simply
"what goes in the state vector and how does it grow," which is the augmentation step here.

## Where this goes next

The factor-graph and bundle-adjustment assignment later in this module keeps every pose as a
variable instead of marginalizing it, solves the resulting sparse least-squares problem, and
relinearizes at each iteration, which removes both the quadratic per-update cost and the
fixed-linearization optimism described above. The information form from the Kalman-filter
assignment and the fill-in of Exercise 5 are the two facts behind that sparsity argument.

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
