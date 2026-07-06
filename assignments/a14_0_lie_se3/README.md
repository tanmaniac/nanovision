# a14_0 - Lie groups for state estimation

This is the first assignment of the classical SLAM module, and the only new language in
the course: the mechanism code here is C++17 with Eigen, not Python. SLAM and localization
ship in C++, and interviews for those roles ask you to write it, so the from-scratch parts
of this module are in C++ to build that fluency. The Python around them (the build, the
tests, the Rerun visualization) is provided.

You implement the Lie-group machinery that every later assignment in the module depends on:
the exponential and logarithm maps for rotations (SO(3)) and rigid transforms (SE(3)), the
left and right Jacobians, the adjoint, and the box-plus / box-minus retraction that lets an
estimator do calculus on a curved space. The Kalman filters, EKF-SLAM, PnP, ICP, and the
pose-graph optimizer all call back into this file.

Required reading before you start:
- Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics" (2018),
  https://arxiv.org/abs/1812.01537 - the single best short reference for everything here.
- Barfoot, *State Estimation for Robotics* (2017), chapter 7 - the definitive treatment of
  the SO(3)/SE(3) Jacobians and the adjoint.

## Lecture notes

### Why a rotation is not a vector

A position is a vector: you can add two of them, scale them, and interpolate between them,
and the result is still a position. A rotation is not. A 3D rotation is a $3\times3$ matrix
$R$ with $R^\top R = I$ and $\det R = +1$, the set of which is called $SO(3)$, the special
orthogonal group. Average two rotation matrices entrywise and the result is generally not a
rotation: its columns are no longer orthonormal and its determinant is no longer 1. The
visualization in this assignment shows exactly this - linearly blending two rotation
matrices makes the determinant dip from 1 toward 0 at the midpoint and the orthonormality
error swell, while the correct interpolation holds a valid rotation the whole way.

$SO(3)$ and $SE(3)$ (the rigid transforms) are Lie groups: smooth curved surfaces
(manifolds) that also have a group structure (you can compose two elements and invert one).
The curvature creates the difficulty. Estimation is built on linear algebra (gradients,
covariances, least-squares updates), and none of that works directly on a curved surface.
Lie theory gives the bridge: at any point on the manifold there is a flat tangent space (the
Lie algebra), and two maps move between the surface and the tangent space. Do the calculus
in the flat tangent space, then map back.

### The exponential and logarithm maps

For $SO(3)$ the tangent space at the identity, written $\mathfrak{so}(3)$, is the set of
$3\times3$ skew-symmetric matrices, which is just $\mathbb{R}^3$ in disguise. The hat
operator turns a 3-vector into its skew matrix and vee inverts it:

$$\widehat{w} = \begin{bmatrix} 0 & -w_z & w_y \\ w_z & 0 & -w_x \\ -w_y & w_x & 0 \end{bmatrix}, \qquad \widehat{w}\,v = w \times v.$$

The exponential map sends a tangent vector (a rotation vector $w$, whose direction is the
axis and whose norm $\theta = \lVert w \rVert$ is the angle) to a rotation. It has a closed
form, Rodrigues' formula, with $K = \widehat{w}$:

$$R = \exp(\widehat{w}) = I + \frac{\sin\theta}{\theta} K + \frac{1 - \cos\theta}{\theta^2} K^2.$$

As $\theta \to 0$ the coefficients $\sin\theta/\theta \to 1$ and
$(1-\cos\theta)/\theta^2 \to \tfrac12$, so the small-angle branch is $R \approx I + K + \tfrac12 K^2$.
That branch is not optional: dividing by $\theta$ or $\theta^2$ near zero is a numerical
blow-up, and these maps are evaluated at tiny increments on every optimizer step. The
logarithm inverts the exponential, recovering $w$ from $R$ via
$\theta = \arccos\!\big(\tfrac{\operatorname{tr}R - 1}{2}\big)$ and
$w = \frac{\theta}{2\sin\theta}\operatorname{vee}(R - R^\top)$, again with a small-angle
branch where $w = \tfrac12 \operatorname{vee}(R - R^\top)$, and a separate branch near
$\theta = \pi$ where $\sin\theta \to 0$ makes that form ill-conditioned.

$SE(3)$, the rigid transforms $T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$, works the
same way one dimension up. Its tangent vector is a six-dimensional twist
$\xi = \begin{bmatrix}\rho \\ \theta\end{bmatrix}$, translation part $\rho$ first and
rotation part $\theta$ second (this module fixes that ordering; a swapped ordering is the
usual source of a sign error later). The exponential sets the rotation block to
$\exp(\widehat{\theta})$ and the translation to $t = V\rho$, where $V = J_l(\theta)$ is the
SO(3) left Jacobian below. The translation is not just $\rho$: a screw motion couples
rotation into translation, and $V$ is that coupling.

### The Jacobians

When you perturb the input of the exponential map, how does its output move? The Jacobians
answer that, and it is the quantity an optimizer needs to take a step. The right
Jacobian $J_r(w)$ is defined by

$$\exp(\widehat{w + \delta}) \approx \exp(\widehat{w})\,\exp(\widehat{J_r(w)\,\delta}),$$

i.e. a small change $\delta$ in the tangent vector becomes a small right-multiplied rotation
$\exp(\widehat{J_r \delta})$ on the group. The left Jacobian $J_l$ is the same with the
increment on the left. Both have closed forms with $K = \widehat{w}$:

$$J_l(w) = I + \frac{1 - \cos\theta}{\theta^2} K + \frac{\theta - \sin\theta}{\theta^3} K^2, \qquad J_r(w) = J_l(-w) = J_l(w)^\top,$$

each with its own small-angle series. The inverses also have closed forms,

$$J_l^{-1}(w) = I - \tfrac{1}{2} K + \frac{1}{\theta^2}\Big(1 - \frac{\theta}{2}\cot\frac{\theta}{2}\Big) K^2, \qquad J_r^{-1}(w) = J_l^{-1}(-w),$$

with the small-angle series $J_l^{-1} \approx I - \tfrac{1}{2}K + \tfrac{1}{12}K^2$. The left-versus-right
choice is not cosmetic: it decides whether a perturbation (and the covariance built around
it) lives on the left or the right of the group element. This module uses the right
convention throughout, so $J_r$ and $J_r^{-1}$ are what the pose-graph Jacobians in the
factor-graph assignment call.

### The adjoint

The adjoint $\mathrm{Ad}_T$ is the tool that moves a twist from one frame to another. For
$\xi = [\rho;\theta]$ it is the $6\times6$ matrix

$$\mathrm{Ad}_T = \begin{bmatrix} R & \widehat{t}\,R \\ 0 & R \end{bmatrix},$$

and its defining identity is $T\,\exp(\widehat{\xi}) = \exp(\widehat{\mathrm{Ad}_T\,\xi})\,T$.
Read that left to right: a twist applied on the right of $T$ (a body-frame perturbation)
equals the same motion as a different twist applied on the left (a spatial-frame
perturbation), and $\mathrm{Ad}_T$ converts the first into the second. Getting that
direction right ($\mathrm{Ad}_T$ versus $\mathrm{Ad}_{T^{-1}}$) makes the
pose-graph edge Jacobians come out with the correct sign.

### Box-plus and box-minus

Optimization and filtering both need an "add a small correction to the current estimate"
operation, $x \leftarrow x + \delta$. On a manifold that addition is the retraction, written
box-plus. With the right convention this module uses

$$T \boxplus \xi = T\,\exp(\widehat{\xi}), \qquad T_2 \boxminus T_1 = \log\!\big(T_1^{-1} T_2\big).$$

Box-plus takes a group element and a tangent correction and returns the nearby group
element; box-minus takes two group elements and returns the tangent vector between them.
These two are the manifold's stand-in for $+$ and $-$. Every estimator downstream is the
familiar vector-space algorithm with $+$ replaced by $\boxplus$: an EKF whose state is a
pose updates the mean with $\boxplus$, a Gauss-Newton step retracts the solved increment
with $\boxplus$, and the geodesic interpolation in this assignment's visualization is
$T_0 \boxplus \big(s\,(T_1 \boxminus T_0)\big)$.

## The assignment

Implement the SO(3) and SE(3) Lie-group operations in C++. Everything else - the pybind11
bindings, the CMake build, the tests, and the Rerun visualization - is provided.

### Files to modify

`so3.cpp` holds the rotation group. Fill `hat3`/`vee3`, `so3_exp`/`so3_log`, and the four
Jacobians `so3_left_jacobian`, `so3_left_jacobian_inv`, `so3_right_jacobian`,
`so3_right_jacobian_inv`. Each function's contract (inputs, outputs, the formula, and which
small-angle branch it needs) is in the comment at its hole; the math is in the lecture notes
above. The header `so3.hpp` is shared with the reference and is not edited.

`se3.cpp` holds the rigid transforms: `hat6`/`vee6`, `se3_exp`/`se3_log`, `se3_adjoint`, and
the retraction `se3_boxplus`/`se3_boxminus`. These call the SO(3) functions, so finish
`so3.cpp` first.

The reference implementation for both files is in `solution/`, in plain sight - write your
own and read it when stuck. You may not include an existing Lie-group or solver library
(Sophus, manif, Ceres, GTSAM, g2o); a test enforces this by scanning the sources.

### Building and running

This module needs a C++17 compiler, plus CMake, pybind11, and Eigen (all in the conda env;
see the repo's `environment.yml` a14 block). You never call CMake by hand - the test and viz
targets build the C++ on demand and cache it under `build/`, so the first run compiles (a
few seconds) and later runs are incremental.

```
make verify A=a14_0_lie_se3   # build + run the reference (solution/); the green target
make test   A=a14_0_lie_se3   # build + run YOUR code; red until the holes are filled
make viz    A=a14_0_lie_se3        # render the manifold-vs-naive interpolation (reference)
make viz-mine A=a14_0_lie_se3      # the same, from YOUR code, once the holes are filled
```

`make verify` runs the suite against `solution/` and is green from the start, so it shows
the target before you write anything. `make test` runs the same suite against your
`so3.cpp`/`se3.cpp`: the holes throw `NOT_IMPLEMENTED` and surface as failing tests, and the
suite goes green as you fill them. The test order is the intended workflow: the hat/vee and
exp/log round-trips first, then the Jacobian inverses and the numerical-versus-analytic
right-Jacobian check (the gate that catches a wrong derivation), then the adjoint identity.

`make viz` writes a Rerun recording to `out/lie_interp.rrd` (headless, for CI). Add `SHOW=1`
(`make viz A=a14_0_lie_se3 SHOW=1`) to open the interactive viewer instead: orbit the two
coordinate frames as they interpolate, and scrub the timeline. The manifold frame sweeps a
clean screw motion with its determinant pinned at 1 and orthonormality error at 0; the naive
componentwise-blended frame shears and its determinant dips to near zero at the midpoint.
`make viz-mine` runs the same against your code, the way to eyeball a finished
implementation.

## In interviews

This material is asked about directly in perception, robotics, and AV interviews, so the
depth here pays off.

Why rotations need a manifold, not a vector space. The short answer, which the
visualization makes concrete: rotations do not add or interpolate linearly and stay
rotations. The follow-up is usually about parameterizations and their failure modes. Euler
angles suffer gimbal lock (a degenerate configuration where two axes align and a degree of
freedom is lost) and have discontinuities. A quaternion is a clean four-number
representation with no gimbal lock, but it is a double cover of $SO(3)$ ($q$ and $-q$ are
the same rotation) and it must be renormalized. The rotation-vector / matrix-plus-Lie-
algebra view makes calculus clean, which is why estimators use it.

Exp / log and where they show up. Be able to write Rodrigues' formula and its small-angle
limit on the spot, and explain why the small-angle branch exists (the $1/\theta$ and
$1/\theta^2$ terms blow up near zero, and you evaluate exp at tiny increments constantly).
Know that log is multivalued (a rotation by $\theta$ and by $\theta + 2\pi$ share a matrix)
and that the implementation picks the principal value, with a special case near $\theta=\pi$.

Left versus right Jacobian. This probes whether you understand that
a perturbation can sit on either side of a group element, and that the covariance you
propagate is tied to that choice. Be able to state the defining property
$\exp(w+\delta) \approx \exp(w)\exp(J_r(w)\delta)$ and that $J_r(w) = J_l(-w) = J_l(w)^\top$.

The adjoint. Expect "what does the adjoint do" - it changes the frame a twist is expressed
in, body to spatial, via $T\exp(\xi)T^{-1} = \exp(\mathrm{Ad}_T\,\xi)$. It makes the chain
rule work when you differentiate a product of transforms, which is exactly
the pose-graph setting in the factor-graph assignment.

Box-plus and on-manifold optimization. How do you run Gauss-Newton or
an EKF when the state is a pose? You linearize in the tangent space, solve for a tangent-
space increment $\delta$, and retract with $x \boxplus \delta = x \exp(\widehat\delta)$,
repeating. Every later assignment in this module is an instance of that pattern.

A classic on-the-spot coding ask for this topic is to implement `hat`, Rodrigues, and its
small-angle limit, which is precisely the first hole here.

## Further reading

- Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics" (2018),
  https://arxiv.org/abs/1812.01537 - exp/log, Jacobians, adjoint, box-plus, with the
  conventions used here.
- Barfoot, *State Estimation for Robotics* (2017), ch. 7 - the SO(3)/SE(3) Jacobians and
  uncertainty on Lie groups in depth.
- Blanco, "A tutorial on SE(3) transformation parameterizations and on-manifold
  optimization" (2010) - the derivations of the SE(3) Jacobians and retraction.
