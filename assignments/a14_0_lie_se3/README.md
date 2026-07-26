# a14_0 - Lie groups for state estimation

This is the first assignment of the classical SLAM module, and the only new language in
the course: the mechanism code here is C++17 with Eigen, not Python. SLAM and localization
ship in C++, and interviews for those roles ask you to write it, so the from-scratch parts
of this module are in C++ to build that fluency. The Python around them (the build, the
tests, the Rerun visualization) is provided.

This assignment builds the Lie-group machinery the rest of the module rests on: the
exponential and logarithm maps for rotations ($SO(3)$) and rigid transforms ($SE(3)$), the
left and right Jacobians, the adjoint, and the box-plus / box-minus retraction that lets an
estimator do calculus on a curved space. The later assignments that carry a 3D pose use
exactly this machinery and these sign conventions: the pose refinement inside PnP, the
point-to-plane ICP step, and the pose-graph optimizer. Each of those ships its own provided
copy in its `models.cpp` so that it builds standalone, so they do not literally link against
this file, but the derivations and the conventions below are the ones they assume.

Required reading before you start:
- Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics" (2018),
  https://arxiv.org/abs/1812.01537 - the single best short reference for everything here.
- Barfoot, *State Estimation for Robotics* (2017), chapter 7 - the definitive treatment of
  the SO(3)/SE(3) Jacobians and the adjoint.

## Lecture notes

### Why a rotation is not a vector

A position is a vector: two positions add, they scale, and a linear blend of two of them is
another position. A rotation is not. A 3D rotation is a $3\times3$ matrix $R$ with
$R^\top R = I$ and $\det R = +1$, and the set of all of them is $SO(3)$, the special
orthogonal group. Averaging two rotation matrices entrywise generally does not produce a
rotation: the columns stop being orthonormal and the determinant stops being 1.

The visualization in this assignment measures exactly that. It interpolates from the
identity to a rotation of about $2.75$ radians ($158^\circ$) two ways. Blending the matrices
entrywise, $R(s) = (1-s)R_0 + sR_1$, drives $\det R$ down to roughly $0.04$ at the midpoint
and the orthonormality error $\lVert R^\top R - I\rVert_F$ up to roughly $1.4$; the manifold
interpolation built later in these notes holds $\det R = 1$ and zero orthonormality error at
every step. A determinant near zero means the frame has been squashed nearly flat: the
"rotation" has collapsed the space it acts on.

The rest of these notes are about the machinery that replaces $+$, $-$, and derivatives for
objects like this.

### Groups, manifolds, and tangent spaces

Three words carry most of the weight in this module, and each is a short definition rather
than a body of theory.

A group is a set with a composition rule that is closed (composing two members gives a
member), associative, has an identity element, and gives every member an inverse that is
also a member. Rotation matrices under matrix multiplication satisfy all four: a product of
rotations is a rotation, matrix multiplication is associative, $I$ is the identity, and
$R^{-1} = R^\top$ is again a rotation. Nothing in that list mentions addition, and $SO(3)$
has none: $R_1 + R_2$ is a matrix, not a rotation.

A manifold is a set that is curved globally but looks flat locally. The sphere is the
standard picture: it sits in $\mathbb{R}^3$, but every small patch of it looks like a piece
of $\mathbb{R}^2$, so a point on it needs two numbers, not three. $SO(3)$ is the same kind of
object with different numbers. A rotation matrix has 9 entries, and $R^\top R = I$ is an
equation between symmetric matrices, so it imposes 6 independent scalar constraints (3 on the
diagonal, 3 off it). That leaves $9 - 6 = 3$, so $SO(3)$ is a 3-dimensional curved surface
inside the 9-dimensional space of $3\times3$ matrices, matching the familiar count of three
rotational degrees of freedom. The condition $\det R = +1$ selects one of the two
disconnected pieces the constraint leaves, the one containing the identity. $SE(3)$ has 6
dimensions the same way, 3 of rotation and 3 of translation.

A Lie group is both at once: a group whose elements form a manifold, with composition and
inversion smooth functions of their arguments. $SO(3)$ and $SE(3)$ are the two this module
uses.

The tangent space at a point of a manifold is the flat space of velocities that curves
through that point can have. On the sphere it is the plane touching the sphere at that point,
and it is an honest vector space: tangent vectors add, they scale, and the result is another
tangent vector. That is the property $SO(3)$ itself lacks, and it is why the whole
construction is worth the trouble. Estimation runs on linear algebra. A covariance is the
matrix of second moments of a vector-valued error, and a Gauss-Newton step solves a linear
system for a vector-valued increment; neither has a meaning on a curved surface, and both
have one in the tangent space. Two maps move between the group and the tangent space at its
identity, and everything below is those two maps and their derivatives.

### The Lie algebra of SO(3)

Which matrices are tangent to $SO(3)$? Take any smooth curve of rotations $R(t)$ and
differentiate the constraint it satisfies. From $R(t)^\top R(t) = I$,

$$\dot R^\top R + R^\top \dot R = 0 \quad\Longrightarrow\quad \big(R^\top \dot R\big)^\top = -\,R^\top \dot R .$$

So $R^\top \dot R$ is skew-symmetric for every $t$, and at a point where $R = I$ the velocity
$\dot R$ is itself skew-symmetric. The tangent space of $SO(3)$ at the identity is therefore
the set of $3\times3$ skew-symmetric matrices. That set is written $\mathfrak{so}(3)$ and
called the Lie algebra of the group; the fraktur letters and the lowercase name are pure
convention, and the object is this vector space plus one extra operation defined at the end
of this section. A skew matrix has 3 free entries, which agrees with the dimension count of
the previous section.

Three free entries means $\mathfrak{so}(3)$ is $\mathbb{R}^3$ in disguise, and the hat
operator makes the disguise explicit. Hat turns a 3-vector into its skew matrix, and vee
inverts hat:

$$\widehat{w} = \begin{bmatrix} 0 & -w_z & w_y \\ w_z & 0 & -w_x \\ -w_y & w_x & 0 \end{bmatrix}, \qquad \widehat{w}\,v = w \times v .$$

The cross-product identity on the right fixes the sign pattern, and a test checks it
directly. In basis terms, the three matrices $G_1 = \widehat{e_1}$, $G_2 = \widehat{e_2}$,
$G_3 = \widehat{e_3}$ obtained by hatting the coordinate axes span $\mathfrak{so}(3)$, and
$\widehat{w} = w_x G_1 + w_y G_2 + w_z G_3$. Those three are called the generators of the
group. A generator is nothing more than a basis vector of the Lie algebra, read as an
infinitesimal motion: $\exp(t\,G_i)$ traces out the family of rotations about axis $i$ as $t$
runs over the reals, so $G_i$ generates that family.

The extra operation is the Lie bracket, which for matrix groups is the commutator
$[A, B] = AB - BA$. It measures how far two elements are from commuting, and
$\mathfrak{so}(3)$ is closed under it: $[\widehat{a}, \widehat{b}] = \widehat{a \times b}$, so
the bracket on $\mathfrak{so}(3)$ is the cross product on $\mathbb{R}^3$ written in matrix
form. It comes back below as the reason composition on a group is not addition.

A physical reading anchors all of this. If $R(t)$ is the orientation of a rigid body, then
$R^\top \dot R = \widehat{\omega}$ holds the body-frame angular velocity $\omega$. Angular
velocities add and scale like vectors; orientations do not. The Lie algebra is where the
angular velocities live.

### The exponential map

If a constant angular velocity is a tangent vector, then holding that angular velocity for
one unit of time ought to produce a rotation, and it does. Solving $\dot R = R\,\widehat{w}$
from $R(0) = I$ gives $R(1) = \exp(\widehat{w})$, the ordinary matrix exponential

$$\exp(X) = I + X + \tfrac{1}{2!}X^2 + \tfrac{1}{3!}X^3 + \cdots .$$

This is the exponential map of $SO(3)$: it sends a tangent vector to a group element. Its
argument $w$ is a rotation vector, read as an axis $u = w/\theta$ and an angle
$\theta = \lVert w \rVert$ in radians.

The infinite series collapses to a closed form because powers of a skew matrix cycle. For a
unit axis $u$, writing $K_u = \widehat{u}$, direct computation gives $K_u^2 = uu^\top - I$
and $K_u^3 = K_u(uu^\top - I) = -K_u$, since $u \times u = 0$. Every higher power is
therefore $\pm K_u$ or $\pm K_u^2$. Substituting $\widehat{w} = \theta K_u$ and grouping the
odd and even terms:

$$\exp(\widehat{w}) = I + \Big(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\Big) K_u + \Big(\frac{\theta^2}{2!} - \frac{\theta^4}{4!} + \cdots\Big) K_u^2 = I + \sin\theta\,K_u + (1 - \cos\theta)\,K_u^2 .$$

That is Rodrigues' formula. Rewriting it with $K = \widehat{w}$, so that the argument is the
rotation vector the code actually receives rather than the unit axis:

$$R = \exp(\widehat{w}) = I + \frac{\sin\theta}{\theta} K + \frac{1 - \cos\theta}{\theta^2} K^2 .$$

As $\theta \to 0$ the coefficients tend to $\sin\theta/\theta \to 1$ and
$(1-\cos\theta)/\theta^2 \to \tfrac12$, so the small-angle branch is
$R \approx I + K + \tfrac12 K^2$. That branch is not optional. Dividing by $\theta$ or
$\theta^2$ near zero loses all the significant digits, and an optimizer evaluates $\exp$ at
tiny increments on every single step, which is precisely the regime where the closed form
misbehaves. The reference switches branches at $\theta < 10^{-8}$.

### The logarithm map

The logarithm inverts the exponential: given $R$, recover the rotation vector $w$. Both
pieces come out of Rodrigues' formula by taking its symmetric and antisymmetric parts.

For the angle, take the trace. $K_u$ is skew so $\operatorname{tr}K_u = 0$, and
$\operatorname{tr}K_u^2 = \operatorname{tr}(uu^\top - I) = 1 - 3 = -2$, so Rodrigues gives
$\operatorname{tr}R = 3 - 2(1 - \cos\theta) = 1 + 2\cos\theta$, hence

$$\theta = \arccos\!\Big(\frac{\operatorname{tr}R - 1}{2}\Big).$$

For the axis, subtract the transpose. $K_u^2 = uu^\top - I$ is symmetric and $K_u$ is
antisymmetric, so $R - R^\top = 2\sin\theta\,K_u$ and therefore
$\operatorname{vee}(R - R^\top) = 2\sin\theta\,u$, giving

$$w = \theta u = \frac{\theta}{2\sin\theta}\operatorname{vee}(R - R^\top).$$

Three situations break that expression, and each needs its own branch in the code.

Round-off in the trace. Floating-point error can push $(\operatorname{tr}R - 1)/2$ a hair
outside $[-1, 1]$, and $\arccos$ of $1 + 10^{-16}$ is NaN. Clamping the argument to $[-1, 1]$
before the $\arccos$ costs nothing and removes the failure.

Small angles. As $\theta \to 0$ the factor $\theta/(2\sin\theta)$ is a ratio of two vanishing
quantities. Use $R \approx I + \widehat{w}$ instead, which gives
$w = \tfrac12 \operatorname{vee}(R - R^\top)$ with no division.

Angles near $\pi$. Here $\sin\theta \to 0$ while $\theta$ does not, so the coefficient
diverges and the antisymmetric part it multiplies vanishes; the axis is being read from a
quantity that is going to zero. Use the symmetric part instead. At $\theta = \pi$, Rodrigues
gives $R = I + 2K_u^2 = 2uu^\top - I$, so $R + I = 2uu^\top$, a rank-1 matrix all of whose
columns are multiples of $u$. Normalizing its largest column recovers the axis. The sign is
not determined by that construction, so the reference resolves it against
$\operatorname{vee}(R - R^\top)$, which is small near $\pi$ but still points along the axis.
Exactly at $\theta = \pi$ that tiebreaker is zero and either sign is a correct answer, since
$\pi u$ and $-\pi u$ are the same rotation.

The logarithm is many-valued in one more way worth naming: rotating by $\theta$ and by
$\theta + 2\pi$ about the same axis produce the same matrix, so $\exp$ is not injective and
$\log$ has to choose. Because $\arccos$ returns a value in $[0, \pi]$, the implementation
returns the principal value, the rotation vector of smallest norm, with
$\lVert w \rVert \le \pi$.

### Why composition is not addition

For scalars $\exp(a)\exp(b) = \exp(a+b)$. For matrices that holds only when $A$ and $B$
commute, and rotations do not: a quarter turn about $x$ followed by a quarter turn about $z$
is a different rotation from the same two in the other order. So in general
$\exp(\widehat{a})\exp(\widehat{b}) \ne \exp(\widehat{a + b})$, and there is no way to
compose two rotations by adding their rotation vectors.

The Baker-Campbell-Hausdorff formula (BCH) says exactly what the discrepancy is. It is the
series for the tangent vector whose exponential equals the product:

$$\log\big(\exp(A)\exp(B)\big) = A + B + \tfrac{1}{2}[A,B] + \tfrac{1}{12}\big[A,[A,B]\big] - \tfrac{1}{12}\big[B,[A,B]\big] + \cdots,$$

where every term past the first two is built from nested Lie brackets. Two readings of that
matter here. First, the entire correction is made of brackets, so if $A$ and $B$ commute all
of it vanishes and the scalar rule returns; the brackets are a direct measure of the cost of
curvature. Second, when one of the two arguments is small the series is useful rather than
merely true, because the nested terms are higher order in the small argument. Keeping only
terms of first order in a small $\delta$:

$$\log\big(\exp(\widehat{w})\exp(\widehat{\delta})\big) \approx w + J_r^{-1}(w)\,\delta, \qquad
\log\big(\exp(\widehat{\delta})\exp(\widehat{w})\big) \approx w + J_l^{-1}(w)\,\delta .$$

The two matrices $J_r^{-1}$ and $J_l^{-1}$ collect the whole infinite tail of bracket terms
into one $3\times3$ matrix each. They are the inverse right and left Jacobians of the section
after next, and this is where they come from: composing a small increment onto a group
element, instead of adding it to a vector, costs one matrix factor on the increment. The
factor depends on which side the increment goes, which is the subject of the next section.

### Left and right perturbations

On a vector space, "the estimate plus a small error" means one thing. On a group it means two
things, because multiplication has two sides:

$$R = \bar R\,\exp(\widehat{\delta_r}) \quad \text{(right)}, \qquad R = \exp(\widehat{\delta_l})\,\bar R \quad \text{(left)} .$$

Both describe the same true rotation $R$ relative to the same estimate $\bar R$; they differ
in the coordinates the error is written in. The right form applies the correction in the body
frame, in the axes the body already carries. The left form applies it in the world frame. A
gyro scale error or a wheel-slip error is generated in the body and belongs on the right; an
error in where the map's origin sits is a world-frame quantity and belongs on the left.

The two are related by one identity. For any rotation and any 3-vector,
$R\,\widehat{a}\,R^\top = \widehat{Ra}$, so conjugating the exponential gives
$\bar R \exp(\widehat{\delta_r})\bar R^\top = \exp(\widehat{\bar R \delta_r})$, hence

$$\delta_l = \bar R\,\delta_r, \qquad \Sigma_l = \bar R\,\Sigma_r\,\bar R^\top .$$

A covariance stated in the wrong convention is therefore wrong by a rotation. The two agree
exactly when $\bar R = I$ or when $\Sigma_r$ is isotropic, which is to say the mistake hides
in exactly the test cases that are easiest to write, and shows up as a slowly rotating
inconsistency in a filter that has been running for a while. $SE(3)$ behaves the same way,
with the adjoint below in place of $\bar R$.

This module fixes the right convention everywhere: $T \boxplus \xi = T\exp(\widehat{\xi})$,
perturbations in the body frame. So $J_r$ and $J_r^{-1}$ are the Jacobians the downstream
assignments call, and $J_l$ is present because the $SE(3)$ exponential needs it internally.

### The Jacobians

Perturb the input of the exponential map by a small $\delta$; how does the output move? An
optimizer needs the answer because it works entirely in the tangent space: it solves for a
tangent-space increment and needs to know how much the group element, and therefore the
residual, moves per unit of that increment. Because motion on a group is expressed by
multiplication and multiplication has two sides, there are two answers.

The right Jacobian $J_r(w)$ is defined by

$$\exp(\widehat{w + \delta}) \approx \exp(\widehat{w})\,\exp\big(\widehat{J_r(w)\,\delta}\big),$$

to first order in $\delta$: adding $\delta$ to the tangent vector has the same effect as
multiplying on the right by the small rotation $\exp(\widehat{J_r\delta})$. The left Jacobian
$J_l$ is the identical statement with the small rotation on the left. Rearranging the right
definition gives the form that makes it computable without any derivation,

$$J_r(w)\,\delta = \log\!\big(\exp(\widehat{w})^{-1}\,\exp(\widehat{w + \delta})\big),$$

so column $i$ of $J_r(w)$ is that expression evaluated at $\delta = \varepsilon e_i$ and
divided by $\varepsilon$. A central difference in each of the three coordinates produces the
whole matrix numerically. `numerical_right_jacobian` in `_helpers.py` does exactly that, and
the analytic closed form is checked against it.

Both Jacobians have closed forms. With $K = \widehat{w}$ and $\theta = \lVert w \rVert$,

$$J_l(w) = I + \frac{1 - \cos\theta}{\theta^2} K + \frac{\theta - \sin\theta}{\theta^3} K^2, \qquad J_r(w) = J_l(-w) = J_l(w)^\top .$$

The relation between $J_r$ and $J_l$ is one line of algebra. Negating $w$ flips the sign of
$K$ and leaves $K^2$ unchanged; transposing does the same thing, since $K^\top = -K$ makes
$(K^2)^\top = K^2$. The small-angle limits are $(1-\cos\theta)/\theta^2 \to \tfrac12$ and
$(\theta - \sin\theta)/\theta^3 \to \tfrac16$, so $J_l \approx I + \tfrac12 K + \tfrac16 K^2$
near zero.

The inverses have closed forms too, which is why the assignment asks for them as separate
functions rather than as a $3\times3$ matrix solve:

$$J_l^{-1}(w) = I - \tfrac{1}{2} K + \frac{1}{\theta^2}\Big(1 - \frac{\theta}{2}\cot\frac{\theta}{2}\Big) K^2, \qquad J_r^{-1}(w) = J_l^{-1}(-w),$$

with the small-angle series $J_l^{-1} \approx I - \tfrac12 K + \tfrac{1}{12}K^2$, which
follows from $\tfrac{\theta}{2}\cot\tfrac{\theta}{2} = 1 - \theta^2/12 - O(\theta^4)$. $J_l$
divides by $\theta^3$ and $J_l^{-1}$ evaluates a cotangent at $\theta/2$, so both need a
small-angle branch for the same reason $\exp$ does. $J_l$ is also singular at
$\theta = 2\pi$, which is where the $\cot(\theta/2)$ in $J_l^{-1}$ diverges, but the
logarithm never returns an angle above $\pi$, so this module never reaches it.

The two halves of the story now meet. The $J_r^{-1}$ that came out of the BCH truncation
above and the $J_r$ defined here are inverse matrices, so one pair of functions answers both
"how much does the group element move per unit of tangent step" and "how much does the
tangent vector move per unit of group multiplication".

### Twists and the SE(3) exponential

$SE(3)$ is the group of rigid transforms, stored as the $4\times4$ matrix

$$T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}, \qquad T^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1\end{bmatrix},$$

acting on homogeneous points, with composition by matrix multiplication. Its tangent space
follows from the same differentiate-the-constraint argument as before. A curve $T(t)$ through the
identity has a velocity whose top-left block is skew (that is the $SO(3)$ argument applied to
the rotation block), whose top-right block is an unconstrained 3-vector, and whose bottom row
is zero because the bottom row of every $T$ is the constant $[0,0,0,1]$. That is
$\mathfrak{se}(3)$: six free numbers. The module packs them into a 6-vector with the
translation part first,

$$\xi = \begin{bmatrix}\rho \\ \theta\end{bmatrix}, \qquad \widehat{\xi} = \begin{bmatrix} \widehat{\theta} & \rho \\ 0 & 0\end{bmatrix} .$$

Here $\theta$ is the rotation part, itself a rotation vector, and $\lVert \theta \rVert$ is
written out whenever the angle is meant. The $[\rho; \theta]$ ordering is fixed across the
whole module; the opposite ordering is equally common in the literature and is the usual
source of a sign or argument error later.

A tangent vector of $SE(3)$ is called a twist: the instantaneous velocity of a rigid body,
six numbers holding an angular part $\theta$ and a linear part $\rho$. Holding a constant
twist for one unit of time sweeps out a screw motion, a rotation about a fixed axis combined
with a translation along that same axis, the way a bolt advances into a nut. The screw is not
a special case. Chasles' theorem says every rigid displacement can be written as one screw
motion, which is why a single 6-vector describes the motion from any pose to any other.

The screw picture explains why the translation part of $\exp(\widehat{\xi})$ is not simply
$\rho$. The body travels with body-frame linear velocity $\rho$ while it is rotating, so the
displacement in world coordinates is the integral of that velocity rotated by however much
rotation has accumulated at each instant:

$$t = \int_0^1 \exp(s\,\widehat{\theta})\,\rho\;ds = \Big(\int_0^1 \exp(s\,\widehat{\theta})\,ds\Big)\rho .$$

Evaluate that integral term by term with Rodrigues, writing $u = \theta/\lVert\theta\rVert$:
$\int_0^1 \sin(s\lVert\theta\rVert)\,ds = (1 - \cos\lVert\theta\rVert)/\lVert\theta\rVert$ and
$\int_0^1 (1 - \cos(s\lVert\theta\rVert))\,ds = 1 - \sin\lVert\theta\rVert/\lVert\theta\rVert$,
so

$$\int_0^1 \exp(s\,\widehat{\theta})\,ds = I + \frac{1 - \cos\lVert\theta\rVert}{\lVert\theta\rVert}\,\widehat{u} + \Big(1 - \frac{\sin\lVert\theta\rVert}{\lVert\theta\rVert}\Big)\widehat{u}^2 = J_l(\theta),$$

which is the $SO(3)$ left Jacobian written in the unit-axis form. So the coupling matrix that
turns linear velocity into displacement is the left Jacobian, and no new function is needed:

$$\exp(\widehat{\xi}) = \begin{bmatrix} \exp(\widehat{\theta}) & J_l(\theta)\,\rho \\ 0 & 1\end{bmatrix} .$$

$\rho$ is therefore not the translation but the linear velocity that produces it; the two
coincide only when $\theta = 0$, where $J_l = I$. Inverting is immediate. Read
$\theta = \log(R)$ off the rotation block, then $\rho = J_l(\theta)^{-1}\,t$. That is where
$J_l^{-1}$ earns its place in the hole list.

### The adjoint

The adjoint answers a frame question: given a twist expressed on one side of a transform,
what is the equivalent twist on the other side?

Start from conjugation. For any invertible $T$, the series definition gives
$T\exp(X)T^{-1} = \exp(TXT^{-1})$, because every term $X^k$ becomes $TX^kT^{-1}$ once the
inner $T^{-1}T$ pairs cancel. When $X = \widehat{\xi}$ is in $\mathfrak{se}(3)$ and $T$ is in
$SE(3)$, the conjugate $T\widehat{\xi}T^{-1}$ is again in $\mathfrak{se}(3)$, so it is the hat
of some 6-vector. That 6-vector depends linearly on $\xi$, and the $6\times6$ matrix
implementing the map is the adjoint:

$$\widehat{\mathrm{Ad}_T\,\xi} = T\,\widehat{\xi}\,T^{-1}, \qquad\text{equivalently}\qquad T\exp(\widehat{\xi})\,T^{-1} = \exp\big(\widehat{\mathrm{Ad}_T\,\xi}\big).$$

Working the block algebra out for $T = \begin{bmatrix} R & t \\ 0 & 1\end{bmatrix}$ and
$\xi = [\rho;\theta]$ gives

$$\mathrm{Ad}_T = \begin{bmatrix} R & \widehat{t}\,R \\ 0 & R \end{bmatrix} .$$

The angular part transforms by $R$ alone, as any 3-vector does. The linear part picks up the
extra term $\widehat{t}R\,\theta = t \times (R\theta)$, which is the lever arm: an angular
velocity about a point offset by $t$ shows up as a linear velocity at the origin, with
magnitude proportional to the offset.

Multiplying the identity on the right by $T$ puts it in the form the tests check and the form
worth memorizing:

$$T\exp(\widehat{\xi}) = \exp\big(\widehat{\mathrm{Ad}_T\,\xi}\big)\,T .$$

Read left to right: a twist applied on the right of $T$, which is a body-frame perturbation,
produces the same transform as a different twist applied on the left, which is a
spatial-frame perturbation, and $\mathrm{Ad}_T$ converts body to spatial. The reverse
direction is $\mathrm{Ad}_{T^{-1}} = (\mathrm{Ad}_T)^{-1}$. This is the $SE(3)$ version of
$\delta_l = \bar R \delta_r$ from the perturbations section, and covariances move with it the
same way: $\Sigma_l = \mathrm{Ad}_T\,\Sigma_r\,\mathrm{Ad}_T^\top$.

The adjoint is also how the chain rule survives a product of transforms. Perturb the left
factor of $T_a T_b$ on the right by $\delta$ and push the perturbation out to the end of the
product by inserting $T_b^{-1}T_b$:

$$T_a\exp(\widehat{\delta})\,T_b = T_a T_b \exp\big(\widehat{\mathrm{Ad}_{T_b^{-1}}\,\delta}\big).$$

A perturbation of an inner factor reappears at the output as the same perturbation carried
through one adjoint. The pose-graph residual in the factor-graph assignment is
$\log\!\big(T_{\text{meas}}^{-1} T_i^{-1} T_j\big)$, and its derivatives with respect to right
perturbations of $T_i$ and $T_j$ come out as an inverse right Jacobian, an adjoint of exactly
this kind, and a minus sign on the $i$ side. Using $\mathrm{Ad}_T$ where $\mathrm{Ad}_{T^{-1}}$
belongs produces a Jacobian of the right shape that still converges on easy graphs, which is
what makes the mistake hard to catch.

### Box-plus, box-minus, and geodesic interpolation

Every estimator has an update of the form "estimate plus a correction",
$x \leftarrow x + \delta$, and a residual of the form "measured minus predicted". Neither $+$
nor $-$ exists on a group. A retraction supplies the first: a map that takes a group element
and a tangent vector and returns a nearby group element, agreeing with ordinary addition to
first order at $\delta = 0$. The exponential is the retraction used throughout this module.
Written box-plus, with box-minus as its inverse and the right convention fixed above:

$$T \boxplus \xi = T\,\exp(\widehat{\xi}), \qquad T_2 \boxminus T_1 = \log\!\big(T_1^{-1} T_2\big).$$

Box-plus takes a group element and a tangent correction and returns the nearby group element.
Box-minus takes two group elements and returns the tangent vector from the first to the
second. They invert each other in both directions, $(T \boxplus \xi) \boxminus T = \xi$ and
$T_1 \boxplus (T_2 \boxminus T_1) = T_2$, which is what the box-plus test asserts.

Every estimator downstream is then the familiar vector-space algorithm with $+$ replaced by
$\boxplus$ and $-$ by $\boxminus$. An extended Kalman filter whose state is a pose carries the
mean as a group element and the covariance as the covariance of the tangent-space error, and
updates the mean with $\boxplus$. A Gauss-Newton step solves the normal equations for a
tangent increment and retracts it with $\boxplus$. A residual between a measured and a
predicted pose is a $\boxminus$.

Interpolation is the same two operations. $T_1 \boxminus T_0$ is the twist that carries $T_0$
to $T_1$ in one unit of time, so scaling it and retracting traces the path:

$$T(s) = T_0 \boxplus \big(s\,(T_1 \boxminus T_0)\big), \qquad s \in [0,1].$$

This is the constant-twist path, the screw motion from one pose to the other, and it stays on
the manifold at every $s$ by construction, because $\exp$ of any twist is a valid transform.
For the rotation part alone this is the geodesic, the shortest path on $SO(3)$ under its
natural bi-invariant metric, and it is the same curve traced by slerp, the spherical linear
interpolation used on unit quaternions. For full $SE(3)$, whether the screw path is also the
shortest one depends on how translation and rotation are weighed against each other in the
metric, so the property to rely on is the one the visualization measures: the path never
leaves the manifold, while the entrywise blend $R(s) = (1-s)R_0 + sR_1$ leaves it
immediately.

## The assignment

Fill these holes, in order. Each is one `NOT_IMPLEMENTED` throw with a matching test; the declaration and comments in each file give the signature and shapes.

1. [`hat3()`](so3.cpp) in `so3.cpp`
2. [`vee3()`](so3.cpp) in `so3.cpp`
3. [`so3_exp()`](so3.cpp) in `so3.cpp`
4. [`so3_log()`](so3.cpp) in `so3.cpp`
5. [`so3_left_jacobian()`](so3.cpp) in `so3.cpp`
6. [`so3_left_jacobian_inv()`](so3.cpp) in `so3.cpp`
7. [`so3_right_jacobian()`](so3.cpp) in `so3.cpp`
8. [`so3_right_jacobian_inv()`](so3.cpp) in `so3.cpp`
9. [`hat6()`](se3.cpp) in `se3.cpp`
10. [`vee6()`](se3.cpp) in `se3.cpp`
11. [`se3_exp()`](se3.cpp) in `se3.cpp`
12. [`se3_log()`](se3.cpp) in `se3.cpp`
13. [`se3_adjoint()`](se3.cpp) in `se3.cpp`
14. [`se3_boxplus()`](se3.cpp) in `se3.cpp`
15. [`se3_boxminus()`](se3.cpp) in `se3.cpp`

You may not include an existing Lie-group or solver library (Sophus, manif, Ceres, GTSAM, g2o); a test enforces this by scanning the sources.

### Building and running

This module needs a C++17 compiler, plus CMake, pybind11, and Eigen. Those three are in the
conda env; the compiler is your system `g++` or `clang++` (see the repo's `environment.yml`
a14 block). You never call CMake by hand - the test and viz targets build the C++ on demand
and cache it under `build/`, so the first run compiles (a few seconds) and later runs are
incremental.

```
make verify A=a14_0_lie_se3   # build + run the reference (solution/); the green target
make test   A=a14_0_lie_se3   # build + run YOUR code; red until the holes are filled
make viz    A=a14_0_lie_se3        # render the manifold-vs-naive interpolation (reference)
make viz-mine A=a14_0_lie_se3      # the same, from YOUR code, once the holes are filled
```

`make verify` runs the suite against `solution/` and is green from the start, so it shows
the target before you write anything. `make test` runs the same suite against your
`so3.cpp`/`se3.cpp`: the holes throw `NOT_IMPLEMENTED` and surface as failing tests, and the
suite goes green as you fill them. Filling the holes in the listed order brings the suite
green in groups: the hat/vee and exp/log round-trips first, then the Jacobian inverses and
the numerical-versus-analytic right-Jacobian check (central differences at step $10^{-6}$,
the gate that catches a wrong derivation), then the SE(3) round-trip, the adjoint identity,
and the box-plus / box-minus pair. Pytest collects the test files alphabetically rather than
in that order, so the adjoint failures appear at the top of the output even though they are
the last thing to fix.

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
angles hit gimbal lock, the configuration where the middle rotation aligns two of the three
axes so that two of the angles turn about the same direction and one degree of freedom
disappears from the parameterization, and they are also discontinuous at the wrap. A
quaternion is a clean four-number representation with no gimbal lock, but it is a double
cover of $SO(3)$, meaning $q$ and $-q$ name the same rotation, and it has to be renormalized
as errors accumulate. Carrying the state as a matrix and the correction as a rotation vector
gives a 3-parameter local coordinate with no constraint to maintain and no singularity
anywhere in a ball of radius $\pi$, which is why estimators use it.

Exp / log and where they show up. Be able to write Rodrigues' formula and its small-angle
limit on the spot, and explain why the small-angle branch exists (the $1/\theta$ and
$1/\theta^2$ terms blow up near zero, and you evaluate exp at tiny increments constantly).
Know that log is multivalued (a rotation by $\theta$ and by $\theta + 2\pi$ share a matrix)
and that the implementation picks the principal value, with a special case near $\theta=\pi$.

Left versus right Jacobian. This probes whether you understand that a perturbation can sit on
either side of a group element, and that the covariance you propagate is tied to that choice:
the same uncertainty written on the other side is $\Sigma_l = \bar R \Sigma_r \bar R^\top$.
Be able to state the defining property
$\exp(\widehat{w+\delta}) \approx \exp(\widehat{w})\exp(\widehat{J_r(w)\delta})$ and that
$J_r(w) = J_l(-w) = J_l(w)^\top$.

The adjoint. Expect "what does the adjoint do" - it changes the frame a twist is expressed
in, body to spatial, via $T\exp(\widehat{\xi})T^{-1} = \exp(\widehat{\mathrm{Ad}_T\,\xi})$.
The follow-up is what it is good for: a perturbation of an inner factor of a product of
transforms comes out at the end of the product multiplied by an adjoint,
$T_a\exp(\widehat{\delta})T_b = T_aT_b\exp(\widehat{\mathrm{Ad}_{T_b^{-1}}\delta})$, which is
how you differentiate a chain of poses, and exactly the pose-graph setting in the
factor-graph assignment.

Box-plus and on-manifold optimization. How do you run Gauss-Newton or an EKF when the state
is a pose? You linearize in the tangent space, solve for a tangent-space increment $\delta$,
and retract with $x \boxplus \delta = x \exp(\widehat\delta)$, repeating. Every later
assignment in this module is an instance of that pattern.

A classic on-the-spot coding ask for this topic is to implement `hat`, Rodrigues, and its
small-angle limit, which is precisely the first hole here.

## Further reading

- Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics" (2018),
  https://arxiv.org/abs/1812.01537 - exp/log, Jacobians, adjoint, box-plus, with the
  conventions used here.
- Barfoot, *State Estimation for Robotics* (2017), ch. 7 - the SO(3)/SE(3) Jacobians and
  uncertainty on Lie groups in depth.
- Murray, Li, Sastry, *A Mathematical Introduction to Robotic Manipulation* (1994), ch. 2 -
  twists, screws, and the adjoint from the kinematics side.
- Blanco, "A tutorial on SE(3) transformation parameterizations and on-manifold
  optimization" (2010) - the derivations of the SE(3) Jacobians and retraction.
