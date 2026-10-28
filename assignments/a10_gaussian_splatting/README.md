# A10 - 3D Gaussian splatting

3D Gaussian splatting stores a scene as an explicit cloud of small translucent ellipsoids, each a
3D Gaussian with a position, an anisotropic shape, an opacity, and a color. Rendering projects
each 3D Gaussian to a 2D Gaussian in the image and alpha-composites the projected splats front to
back, with no network in the render path. The cloud is fit by gradient descent on photometric
error: render the current Gaussians into each training camera, compare to the captured image, and
push the pixel error back to every Gaussian's parameters by the chain rule.

This assignment builds the differentiable forward rasterizer in pure PyTorch, with gradients from
autograd rather than a hand-written CUDA backward pass, and fits a toy scene through it. The
holes are the scale-plus-rotation factorization that turns unconstrained parameters into a valid
3D covariance, the EWA projection that linearizes perspective to turn a 3D Gaussian into a 2D
screen-space Gaussian, and the depth-sorted alpha-compositing rasterizer. Densification, the
heuristic that grows and prunes the cloud, is read-and-understand here; only an opacity prune is
implemented.

The last step of the render is the same front-to-back "over" chain the ray-marched volume
renderer of the NeRF assignment ended with, $C = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j)$.
What changes is where $\alpha$ comes from and where the ordering comes from: a projected 2D
Gaussian evaluated at the pixel instead of $1 - \exp(-\sigma\delta)$ on a ray segment, and a depth
sort over primitives instead of a march along $t$. Splatting keeps the compositing and drops the
per-ray network.

Required reading before starting:
- Kerbl et al. 2023, "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
  [arXiv:2308.04079](https://arxiv.org/abs/2308.04079).
- Zwicker et al. 2001, "EWA Volume Splatting", IEEE Visualization,
  [DOI:10.1109/VISUAL.2001.964490](https://doi.org/10.1109/VISUAL.2001.964490) (and the 2002
  TVCG "EWA Splatting" follow-up): the covariance projection.

## Lecture notes

### What Gaussian splatting is for

#### The task

The task is the one the NeRF assignment solved, novel-view synthesis. Given images of a scene from
a set of cameras with known poses and intrinsics, render the image a camera at a new pose would
see. Our toy sphere is the smallest instance. A sphere of radius 1 sits at the world origin, with a
smooth surface color that varies with position on the surface. Six cameras on a horizontal ring of
radius 4 look at it. Five views are for training and the sixth is held out: the output asked for is
that sixth image, rendered without ever seeing it.

![Our toy sphere seen from the six ring cameras at 32x32; the last view is held out](assets/toy_views.png)

At full scale the same task is rendering a captured room, a street, or a garden from a camera path
nobody recorded, at video resolution and at the frame rate of an interactive viewer or a driving
simulator.

#### Why a NeRF is slow

For three years after the original NeRF, view synthesis meant training a coordinate MLP and
querying it hundreds of times per ray. The quality was high and the cost was severe: a single
scene took hours to days on a GPU to train, and rendering one frame meant millions of MLP
evaluations, far from interactive.

A line of work attacked the speed by moving part of the scene out of the network weights and into
an explicit data structure the network could read. A dense voxel grid stores a feature vector at
each grid corner and interpolates between them, so the MLP only has to decode a local feature
instead of memorizing the whole scene. A multiresolution hash grid does the same at several
resolutions, with each grid's entries held in a fixed-size hash table so memory stays bounded even
at fine resolution. A tensor factorization stores the volume as a low-rank sum of outer products of
per-axis vectors and matrices, trading exactness for a much smaller parameter count. Instant-NGP's
hash grid (Müller et al. 2022, [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)) cut training
from hours to seconds, but rendering still marched samples along rays and queried a network at
every sample.

#### Why plain points are not enough

The other classical option is an explicit point cloud, drawn by projecting each point to a pixel.
That is fast, but it fits badly into gradient descent. A point that lands on one pixel leaves holes
between its neighbors, and a point drawn as a hard-edged disk changes the image only where its edge
crosses a pixel center, so moving it slightly either changes nothing or flips a whole pixel. The
derivative of the image with respect to the point's position is zero almost everywhere, which gives
the optimizer nothing to follow.

#### The idea, soft ellipsoids rendered by sorting and compositing

3D Gaussian splatting (Kerbl et al. 2023) changed the representation instead of accelerating the
query. A scene is a cloud of explicit 3D Gaussians: soft, translucent ellipsoids whose material
fades smoothly away from the center. Rendering one camera view takes three steps.
1. Project each 3D Gaussian into the image. Its footprint there is approximately a 2D Gaussian, an
   elliptical patch of pixels. Graphics calls such a footprint a splat, and these notes use "splat"
   for the 2D footprint and "Gaussian" for the 3D primitive.
2. Sort the Gaussians by depth, nearest first.
3. For each pixel, read off how opaque each splat is at that pixel and composite the splats front
   to back with the "over" chain from the NeRF assignment.

There is no network in the render path. Every step is a smooth function of each Gaussian's
position, shape, opacity, and color, so a pixel error can be backpropagated to all of them, and
fitting is gradient descent on the difference between rendered and captured images. For our toy
sphere, fitting moves a few hundred Gaussians, which start scattered through a cube around the
origin, onto the sphere and gives each the local surface color; rendering them from the held-out
camera produces the missing image.

#### Why it matters

A tile-based GPU rasterizer sorts the Gaussians by depth once per frame and blends them, so a
trained scene renders in real time at 1080p, which the paper summarizes as at least 30 frames per
second and reports above 100 on several scenes, while matching or beating the strongest NeRF
variants on image quality and fitting in tens of minutes. The method spread through view synthesis
within months of publication because it renders in real time without giving up NeRF-level quality.

### Our two-Gaussian example (two blobs in front of one toy camera)

A real scene has millions of Gaussians and our toy fit a few hundred, too many to follow by hand.
The mechanism is visible with two. Take the first camera of our toy sphere, at world position
$(4, 0, 0)$ looking at the origin, rendering a 16x16 image with focal length $f = 16$ pixels and the
image center at pixel $(7.5, 7.5)$. In front of it place two Gaussians, both inside the toy sphere:
- the near Gaussian, red, opacity $0.8$, centered at world $(0.8, 0, -0.25)$, a ball with standard
  deviation $0.2$ in every direction, at depth $3.2$ from the camera;
- the far Gaussian, blue, opacity $0.9$, centered at world $(0, 0, 0.25)$, with standard deviations
  $0.1$, $0.1$, and $0.5$ along world $x$, $y$, and $z$, a thin rod lying across the camera's view,
  at depth $4$.

![Our two-Gaussian example. (a) The camera and the two Gaussians seen from above. (b) The 16x16 render with each splat's one-sigma ellipse, solid after the dilation filter (a small blur added to every splat, defined below) and dotted before. (c) Along pixel row 7, each Gaussian's alpha, its compositing weight, and the leftover transmittance that lets the white background through; the red alpha and weight curves coincide because nothing is in front of the red Gaussian](assets/two_gaussians.png)

Panel (b) is the render over a white background. The near red ball becomes a round splat centered
at pixel $u = 6.25$; the far blue rod becomes a horizontal elliptical splat centered at $u = 8.5$
(both centers are derived in the projection section below). Where they overlap, the red one is in front and hides part of the blue one. Panel (c) shows how,
along one row of pixels: each splat's alpha, how opaque it is at that pixel (dashed), the share of the pixel's color
it actually contributes after the splats in front have taken theirs (solid), and the fraction of
white background that shows through (black). The blue splat's contribution dips where the red splat
sits in front of it.

Our two-Gaussian example stands for every pixel of every training view: each of them is composited
from the Gaussians whose splats cover it, and the training loss is a sum of one such pixel error per
pixel. It keeps what the renderer does: the projection of a 3D shape into an ellipse, the effect of
depth and of distance from the image center on that ellipse, the depth sort, and the compositing.
It drops what fitting has to discover (where the Gaussians go and what shape and color they take),
the interaction of many overlapping Gaussians, and the engineering that makes the real renderer
fast. Every number in it can be checked by hand, and the exercises do that.

### Overview of the rest of the notes

The sections below build the renderer in the order data flows through it, and most steps are a
familiar idea used in a new place.

1. What a 3D Gaussian is. The multivariate normal shape, used as a blob of material rather than a
   probability density.
2. Keeping the parameters valid. The covariance is stored as a factor $M$ with $\Sigma = MM^\top$, so
   it stays positive semi-definite, for the same reason square-root Kalman filters propagate a
   factor of the state covariance instead of the covariance itself.
3. Color and viewing direction. Spherical harmonics, the Fourier series of the sphere.
4. Projecting a Gaussian into the image. The covariance propagation $A\Sigma A^\top$ of a Kalman
   filter, with perspective linearized at the mean as an extended Kalman filter linearizes a
   measurement model.
5. The dilation filter. A small blur that keeps sub-pixel splats from aliasing, the sampling
   theorem applied to splats.
6. The rasterizer. Sort, evaluate, and composite with the "over" chain from the NeRF assignment.
7. Fitting the cloud. Gradient descent on photometric error, the dense relative of bundle
   adjustment.
8. Densification. A non-differentiable outer loop that adds and removes Gaussians.

Our two-Gaussian example makes each formula concrete. Five exercises, placed after the sections
they test, are meant to be worked by hand with a calculator. Each has a worked answer that can be
expanded.

### What a 3D Gaussian is

The primitive is the multivariate normal shape, used as a blob of material rather than as a
probability. (In these notes "blob" always means this 3D shape, never its 2D image.) Recall the density of a normal with mean $\mu$ and covariance $\Sigma$ in
$\mathbb{R}^3$:

$$\mathcal{N}(x;\mu,\Sigma) = \frac{1}{(2\pi)^{3/2}\lvert\Sigma\rvert^{1/2}}
\exp\!\left(-\tfrac12 (x-\mu)^\top \Sigma^{-1} (x-\mu)\right).$$

The quantity in the exponent, $(x-\mu)^\top \Sigma^{-1} (x-\mu)$, is the squared Mahalanobis
distance: the displacement from the mean measured in units of the spread in each direction, so a
point one standard deviation out along any axis has squared Mahalanobis distance 1. Its level sets
are ellipsoids. Writing the eigendecomposition $\Sigma = U \Lambda U^\top$, the columns of $U$ are
the ellipsoid's axis directions and $\sqrt{\lambda_i}$ is the standard deviation along axis $i$,
which is the semi-axis length of the one-sigma ellipsoid. Isotropic means all three eigenvalues are
equal, so the ellipsoid is a ball; anisotropic means they differ, so the blob can be long and thin
or flat like a disk. In our two-Gaussian example the near Gaussian is isotropic and the far one is
anisotropic, a rod five times longer than it is wide.

A 3D Gaussian in this method is that shape without the normalization. Drop the
$(2\pi)^{-3/2}\lvert\Sigma\rvert^{-1/2}$ factor and multiply by a stored opacity $o \in [0,1]$, so
the value at the center is exactly $o$ and it falls off with Mahalanobis distance. The result is not
a probability density and does not integrate to one; it is a soft ellipsoidal lump of colored,
partly transparent material, with $o$ its opacity at the densest point.

Three properties make this shape convenient for rendering. It stays Gaussian under a linear map,
which the projection step below uses directly. It is smooth everywhere, so every pixel it touches
returns a gradient to its position, shape, opacity, and color, with no hard boundary where the
gradient jumps or vanishes (the problem with hard-edged points above). And its anisotropy lets one
primitive cover a flat surface patch that would take many spheres.

A scene is $N$ of these. Each carries a world-space mean $\mu_w \in \mathbb{R}^3$, a 3D
covariance $\Sigma_{3D} \in \mathbb{R}^{3\times3}$ stored in factored form, an opacity
$o \in [0,1]$, and a color $c \in [0,1]^3$.

### Keeping the parameters valid

Only the mean is a free vector. The covariance has to be a valid covariance, and the opacity and
the color have to stay in $[0,1]$, and plain gradient descent respects neither requirement. The fix
in every case is the same: store an unconstrained number and map it into the valid set inside the
forward pass.

Start with the covariance. It must be symmetric positive semi-definite, because otherwise the
shape itself breaks. If $\Sigma$ picks up a negative eigenvalue, then $\Sigma^{-1}$ has one too, the
quadratic form $d^\top \Sigma^{-1} d$ goes negative along that direction, and
$\exp(-\tfrac12 d^\top \Sigma^{-1} d)$ grows without bound as $d$ moves away from the center. The
ellipsoid becomes a hyperbolic ridge that brightens toward infinity. The
symmetric positive semi-definite matrices form a convex cone inside the six-dimensional space of
symmetric $3\times3$ matrices, with its boundary where the smallest eigenvalue reaches zero, and an
unconstrained step on those six entries can walk straight out of the cone with nothing to pull it
back.

The factorization avoids the constraint by construction. Store a rotation $R$ (built from a
quaternion $q$) and a per-axis scale $S = \mathrm{diag}(s_1, s_2, s_3)$, where each standard
deviation $s_k$ is stored as its logarithm $\log s_k$. Set $M = RS$ and define

$$\Sigma_{3D} = M M^\top = R S S^\top R^\top.$$

For any vector $v$, $v^\top M M^\top v = \lVert M^\top v \rVert^2 \ge 0$, so the result is positive
semi-definite for every parameter value, and $MM^\top$ is symmetric for the same reason. There is
no constraint, no projection back onto the cone, and no eigendecomposition in the loop. A
square-root Kalman filter uses the same idea: it carries a factor of the state covariance so that
rounding errors cannot make the covariance indefinite. The factorization also reads geometrically:
if $z$ is a standard normal 3-vector then $Mz + \mu$ has covariance $MM^\top$, so $M$ is the map
that takes the unit ball to the ellipsoid, $S$ setting the three semi-axis lengths and $R$
orienting them.

The eigenvalues of $\Sigma_{3D}$ are exactly the squared scales. $R\,\mathrm{diag}(s^2)\,R^\top$ is
an orthogonal similarity transform of a diagonal matrix, and a similarity transform does not change
eigenvalues, so the spectrum is $\{s_1^2, s_2^2, s_3^2\}$ whatever the rotation. Comparing the
sorted eigenvalues with the sorted squared scales is therefore a check that catches a transposed
$R$ or a scale applied to rows instead of columns.

Scales are stored as logs because $s = e^{\log s}$ maps the whole real line onto the positive
reals, so an unconstrained update can never drive a standard deviation to zero or negative. A fixed
step in $\log s$ is a fixed ratio change in $s$, which suits a quantity that ranges over orders of
magnitude within one scene.

Opacity and color get the same treatment through the sigmoid, $\mathrm{sig}(x) = 1/(1+e^{-x})$,
which maps the real line onto the open interval $(0,1)$. The stored parameter is the pre-sigmoid
number, conventionally called a logit because the inverse map $\log\big(p/(1-p)\big)$ is the
log-odds. Gradient descent runs on the logits, unconstrained, while the rendered opacities and
colors stay in range.

The rotation is stored as a quaternion $q = (w, x, y, z)$, real part first, and normalized to unit
norm inside the forward pass, $\hat q = q / \lVert q \rVert$. The stored $q$ can drift in magnitude
and the rotation it produces stays valid, which is the same reason the log and sigmoid maps are
there. Normalization is singular at $q = 0$, where the direction is undefined and the derivative
blows up, so order-one quaternions are safe to differentiate and near-zero ones are not; gradient
checks should seed quaternions near the identity for that reason. Antipodal quaternions $q$ and
$-q$ give the same rotation, so the parameterization is two-to-one, which is harmless here since
the optimizer only needs some path to a good rotation.

> **Exercise 1 (a covariance from scale and rotation, and one from a raw step).** A Gaussian has
> scales $(s_1, s_2, s_3) = (0.4, 0.1, 0.1)$ and the unit quaternion
> $q = (\cos 22.5^\circ, 0, 0, \sin 22.5^\circ)$, a rotation by $45^\circ$ about the $z$ axis, so
> $R$ has rows $(c, -c, 0)$, $(c, c, 0)$, $(0, 0, 1)$ with $c = 1/\sqrt2$. Compute
> $\Sigma_{3D} = R S S^\top R^\top$ and its eigenvalues. Then suppose the optimizer stored the six
> covariance entries directly and a gradient step added $0.02$ to the $xy$ entry (and the $yx$
> entry). Is the result still a valid covariance? Evaluate $v^\top \Sigma v$ for
> $v = (1, -1, 0)/\sqrt2$.

<details>
<summary>Answer</summary>

$SS^\top = \mathrm{diag}(0.16, 0.01, 0.01)$. With $c^2 = 1/2$, the upper-left block of
$R\,\mathrm{diag}(0.16, 0.01, 0.01)\,R^\top$ has diagonal entries $(0.16 + 0.01)/2 = 0.085$ and
off-diagonal entries $(0.16 - 0.01)/2 = 0.075$, and the $zz$ entry is $0.01$:

$$\Sigma_{3D} = \begin{pmatrix} 0.085 & 0.075 & 0 \\ 0.075 & 0.085 & 0 \\ 0 & 0 & 0.01 \end{pmatrix}.$$

The eigenvalues of the upper-left block are $0.085 \pm 0.075$, that is $0.16$ and $0.01$, and the
$z$ block adds another $0.01$: the squared scales, as the similarity argument says. The long axis
points along $(1, 1, 0)/\sqrt2$, the $x$ axis rotated by $45^\circ$.

After the raw step the upper-left block is
$\begin{pmatrix} 0.085 & 0.095 \\ 0.095 & 0.085\end{pmatrix}$, with eigenvalues $0.085 \pm 0.095$,
that is $0.18$ and $-0.01$. Along $v = (1,-1,0)/\sqrt2$, $v^\top\Sigma v = 0.085 - 0.095 = -0.01 < 0$.
A step of $0.02$ on one entry left the cone, and the blob would brighten without bound along $v$.
Under the factorization the same intent (a longer, more tilted ellipsoid) is reached by changing
$\log s_1$ and $q$, and no value of those can produce a negative eigenvalue.

</details>

### Color and viewing direction

Real surfaces change color with the viewing angle. A specular highlight, the mirror-like bright
spot on a glossy surface, slides across the object as the camera moves, while diffuse color stays
put. Reproducing that means each Gaussian's color is a function on the sphere of viewing
directions, $c(\mathbf{d})$, not a single triple.

Representing a function on the sphere compactly is the same problem as representing a periodic
function on a circle, where the answer is a Fourier series: expand in
$1, \cos\theta, \sin\theta, \cos 2\theta, \dots$, ordered by frequency, and truncate once the terms
are finer than anything worth keeping. Spherical harmonics are the analogue on the sphere. They are
real-valued functions $Y_{\ell m}(\mathbf{d})$ indexed by a degree $\ell \ge 0$ and an order $m$
running from $-\ell$ to $\ell$, so degree $\ell$ contributes $2\ell+1$ functions and everything up
to degree $\ell$ is $(\ell+1)^2$ functions. They are orthonormal over the sphere, and higher degrees
oscillate faster across it, exactly as higher Fourier terms do around the circle. A per-Gaussian
color is then a set of coefficients $k_{\ell m}$ per channel with

$$c(\mathbf{d}) = \sum_{\ell} \sum_{m=-\ell}^{\ell} k_{\ell m}\, Y_{\ell m}(\mathbf{d}).$$

At degree 0 there is a single basis function, the constant $Y_{00} = 1/(2\sqrt{\pi})$, so a
degree-0 color does not vary with direction at all. Degree 1 adds three functions linear in the
direction components, enough for a broad brightening on one side. The original 3D Gaussian
splatting goes to degree 3, which is 16 coefficients per channel and 48 numbers of appearance per
Gaussian: enough for soft, broad specular variation, not enough for a sharp mirror reflection,
which would need a high-degree expansion.

Our toy sphere is a diffuse surface: its color depends only on the surface point, not on the
direction it is seen from, so a view-independent color per Gaussian (degree 0) is all it needs. Our
two-Gaussian example uses fixed colors for the same reason.

### Projecting a Gaussian into the image

The rasterizer needs each Gaussian's footprint in pixels: a 2D mean and a 2D covariance. Getting
there takes one exact fact and one approximation.

The exact fact is how a covariance moves through a linear map. If $x$ has mean $\mu$ and covariance
$\Sigma$ and $y = Ax$, then

$$\mathrm{Cov}(y) = \mathbb{E}\big[A(x-\mu)(x-\mu)^\top A^\top\big] = A\,\Sigma\,A^\top,$$

and when $x$ is Gaussian, $y$ is Gaussian too. This is the covariance propagation step of a Kalman
filter, the same $A \Sigma A^\top$ that pushes a state covariance through a linear motion model.

The approximation is needed because perspective projection is not linear: it divides by depth. The
exact image of a 3D Gaussian under a divide-by-$z$ map is not a Gaussian and has no closed form
worth rendering with. Elliptical weighted average (EWA) splatting linearizes the projection at each
Gaussian's mean and pushes the covariance through that linear map, which is the same first-order
move an extended Kalman filter makes when the measurement model is nonlinear:
$\pi(\mu_c + \delta) \approx \pi(\mu_c) + J\delta$, then treat the spread as if $\pi$ really were
that affine map. The error grows with how much $J$ varies across the blob's extent, so it grows
with a Gaussian's angular size and with its distance from the optical axis. Small blobs, which the
fit produces anyway, stay accurate.

The figure below checks this by sampling. Each gray dot is a point drawn from a 3D Gaussian and
projected exactly, with the divide by depth; the orange ellipses are the EWA approximation's one-
and two-sigma ellipses, the orange cross is the EWA mean, the black plus is the median of the exact projections, and the
dashed square is the 16x16 image of our toy camera. On the left is
the far Gaussian of our two-Gaussian example: the exact projections fill the EWA ellipses closely.
On the right is a Gaussian with a standard deviation of half a unit sideways and one unit in depth,
three units from the camera and off to the side of the optical axis. Its near half projects larger
and farther from the image center than its far half, so the exact footprint is lopsided, with a
long tail away from the axis that the symmetric EWA ellipse cannot represent.

![The EWA approximation checked by sampling. Gray, points drawn from a 3D Gaussian and projected exactly. Orange, the EWA one- and two-sigma ellipses. Left, the small far Gaussian of our two-Gaussian example; right, a large Gaussian off the optical axis, whose exact footprint is lopsided](assets/ewa_check.png)

The name comes from texture filtering. Greene and Heckbert (1986) observed that when a textured
surface is viewed obliquely, the preimage of one square pixel in texture space is an ellipse, and
they filtered the texture by averaging the texels under an elliptical Gaussian weight over that
ellipse instead of point-sampling. Zwicker et al. (2001) ran the same geometry in the other
direction for volume rendering: each primitive is a 3D Gaussian reconstruction kernel, its image
footprint after the affine approximation is an elliptical Gaussian, and the renderer accumulates
those footprints into the image. An accumulated footprint is the "splat".

The camera convention is OpenCV, $+x$ right, $+y$ down, $+z$ forward into the scene. The mean goes
to camera space as $\mu_c = R_{c \leftarrow w}\,\mu_w + t$, where $R_{c \leftarrow w}$ and $t$ are
the rotation and translation of the world-to-camera transform, the inverse of the camera pose. The
pinhole projection is

$$\pi(x,y,z) = \left(f_x \tfrac{x}{z} + c_x,\; f_y \tfrac{y}{z} + c_y\right),$$

the same pinhole projection the NeRF assignment built. (Here $c_x, c_y$ are the principal point,
not a color.) Its Jacobian at the camera-space mean is

$$J = \frac{\partial \pi}{\partial (x,y,z)} = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix} \in \mathbb{R}^{2\times3},$$

the two rows being the derivatives of $u$ and $v$, and the third column carrying the $-f_x x/z^2$
and $-f_y y/z^2$ terms that the quotient rule produces when differentiating the divide by depth.
The first two columns say that a sideways displacement of one unit moves the pixel by $f/z$ pixels,
so the same blob looks smaller farther away. The third column says that moving a point toward or
away from the camera also moves its pixel, radially, by an amount proportional to how far off-axis
it is; on the optical axis ($x = y = 0$) it is zero.

Write $W = R_{c \leftarrow w}$ for the world-to-camera rotation, following Zwicker's notation. The
linear map from world space to image space is then $J W$, and the covariance transforms as

$$\Sigma_{2D} = J\, W\, \Sigma_{3D}\, W^\top J^\top \in \mathbb{R}^{2\times2}.$$

(The letter $W$ also names the image width in the rasterizer section below. The two never appear in
the same expression.)

Only the rotation $W$ enters, because the translation of a rigid transform shifts the mean and
leaves the spread alone. The result is $2\times2$ rather than $3\times3$ because $J$ is $2\times3$:
the depth row and column drop out, so the operation projects onto the image plane rather than
merely changing basis.

In our two-Gaussian example the camera's axes are camera $x$ along world $z$, camera $y$ along
world $y$, and camera $z$ along world $-x$, with the world origin at depth 4. So the far Gaussian's
mean, world $(0, 0, 0.25)$, is at camera $(0.25, 0, 4)$, and it projects to pixel
$(16 \cdot 0.25/4 + 7.5,\ 7.5) = (8.5, 7.5)$. The near Gaussian at world $(0.8, 0, -0.25)$ is at
camera $(-0.25, 0, 3.2)$ and projects to $(16 \cdot (-0.25)/3.2 + 7.5,\ 7.5) = (6.25, 7.5)$. At its
depth $f/z = 5$, so its standard deviation of $0.2$ becomes about $5 \cdot 0.2 = 1$ pixel.

This Jacobian is the same $\partial\pi/\partial X$ that linearizes reprojection error in bundle
adjustment. The use differs: bundle adjustment linearizes a residual, the projected point minus the
observed feature, to take a Gauss-Newton step, while EWA propagates a covariance through the same
linear map as $J \Sigma J^\top$. The derivative is the same; what it transports differs.

> **Exercise 2 (a rod pointing at the camera).** With our toy camera ($f_x = f_y = 16$), take a
> Gaussian whose camera-space covariance is $\mathrm{diag}(0.01, 0.01, 0.25)$: standard deviation
> $0.1$ sideways and $0.5$ along the viewing direction, a rod pointing at the camera. Compute its
> $\Sigma_{2D}$ (without any dilation) when its mean is at camera $(0, 0, 4)$, and again at
> $(1, 0, 4)$. What does the off-axis rod look like in the image, and which entry of $J$ is
> responsible?

<details>
<summary>Answer</summary>

Here the covariance is already in camera space, so $\Sigma_{2D} = J\Sigma J^\top$. On the axis,
$J = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 4 & 0\end{pmatrix}$ and
$\Sigma_{2D} = \mathrm{diag}(16 \cdot 0.01, 16 \cdot 0.01) = \mathrm{diag}(0.16, 0.16)$: a round
splat with standard deviation $0.4$ pixels. The rod is seen end-on and its length is invisible.

At $(1, 0, 4)$ the third-column entry of the first row is $-16 \cdot 1/16 = -1$, so
$J = \begin{pmatrix} 4 & 0 & -1 \\ 0 & 4 & 0\end{pmatrix}$ and

$$\Sigma_{2D} = \begin{pmatrix} 16 \cdot 0.01 + 1 \cdot 0.25 & 0 \\ 0 & 0.16\end{pmatrix}
= \begin{pmatrix} 0.41 & 0 \\ 0 & 0.16\end{pmatrix}.$$

The splat is stretched horizontally, standard deviation $\sqrt{0.41} \approx 0.64$ pixels across
and $0.4$ down: the near end of the rod projects farther from the image center than the far end, so
an off-axis rod pointing at the camera is seen slightly from the side. The $-f_x x/z^2$ entry does
this. (Projecting samples exactly gives a horizontal standard deviation of about $0.67$ pixels,
slightly more than the linearization, because the near half of the rod spreads more than the far
half shrinks.)

</details>

### The dilation filter and aliasing

A small amount is added to the diagonal of every projected covariance:

$$\Sigma_{2D} \mathrel{+}= \lambda I, \qquad \lambda = 0.3 \text{ px}^2.$$

Adding $\lambda I$ is a blur with a precise meaning. Convolving two Gaussians gives a Gaussian whose
covariance is the sum of the two covariances, so $\Sigma_{2D} + \lambda I$ is exactly the splat
convolved with an isotropic Gaussian of standard deviation $\sqrt{0.3} \approx 0.55$ px. That is a
low-pass filter: it removes image detail finer than about half a pixel.

The filter is there because the image is sampled at one point per pixel. A splat much narrower
than the pixel spacing violates the sampling condition that a signal must be sampled above twice
its highest frequency, so moving the camera by a fraction of a pixel makes such a Gaussian wink in
and out as its peak crosses between sample points. Blurring it to at least pixel width removes the
frequencies the pixel grid cannot resolve. The same addition also bounds the determinant away from
zero, so the closed-form $2\times2$ inverse the rasterizer takes next is numerically safe. The
reference implementation of 3D Gaussian splatting by Kerbl et al. uses the same $\lambda = 0.3$
(the value is in their code, not in the paper's text).

In our two-Gaussian example the dilation matters most for the thin direction of the far rod, as the
dotted and solid ellipses in panel (b) of the figure show and as the next exercise computes.

A fixed $\lambda$ is the wrong filter at every scale but one, because the required filter width
depends on the sampling rate and the sampling rate changes as the camera moves. A Gaussian that
should shrink below a pixel when the camera pulls back instead stays pixel-sized, so it is too wide
at some distances and detail flickers as the view changes. Mip-Splatting (Yu et al. 2023,
[arXiv:2311.16493](https://arxiv.org/abs/2311.16493)) replaces the fixed dilation with a 3D
smoothing filter that band-limits each Gaussian according to the highest sampling rate any training
view demands of it, plus a 2D filter that approximates integrating over the pixel's area at the
current scale. The name refers to mipmaps, the prefiltered image pyramids graphics has used against
the same problem since the 1980s, where the pyramid level is picked from the local sampling rate.
The course uses the fixed dilation.

> **Exercise 3 (projecting our far Gaussian).** In our two-Gaussian example the far Gaussian has
> world covariance $\mathrm{diag}(0.1^2, 0.1^2, 0.5^2)$ and camera-space mean $(0.25, 0, 4)$, and the
> world-to-camera rotation has rows $(0, 0, 1)$, $(0, 1, 0)$, $(-1, 0, 0)$. Compute
> $W\Sigma_{3D}W^\top$, $J$, and $\Sigma_{2D}$ before and after the dilation. Give the standard
> deviations of the splat in pixels along $u$ and $v$ in both cases.

<details>
<summary>Answer</summary>

$W$ permutes the axes (camera $x$ is world $z$, camera $y$ is world $y$, camera $z$ is world $-x$),
and for a diagonal covariance the sign does not matter, so
$W\Sigma_{3D}W^\top = \mathrm{diag}(0.25, 0.01, 0.01)$: the rod lies along camera $x$, across the
view. At $(0.25, 0, 4)$ the Jacobian is
$J = \begin{pmatrix} 4 & 0 & -0.25 \\ 0 & 4 & 0 \end{pmatrix}$ (the corner entry is
$-16 \cdot 0.25/16$). Then

$$\Sigma_{2D} = \begin{pmatrix} 16 \cdot 0.25 + 0.0625 \cdot 0.01 & 0 \\ 0 & 16 \cdot 0.01\end{pmatrix}
= \begin{pmatrix} 4.000625 & 0 \\ 0 & 0.16\end{pmatrix},$$

with standard deviations $2.00$ px along $u$ and $0.40$ px along $v$. After the dilation it is
$\mathrm{diag}(4.300625, 0.46)$, standard deviations $2.07$ px and $0.68$ px. The dilation barely
changes the long axis and widens the thin one by 70%, from well under a pixel to about two thirds of
a pixel: only splats near or below the pixel spacing are affected.

</details>

### The rasterizer

Given projected means $\mu_i \in \mathbb{R}^2$, screen covariances $\Sigma_{2D,i}$, colors $c_i$,
opacities $o_i$, and depths $z_i$, the render is a depth sort, a per-pixel Gaussian evaluation, and
a compositing pass.

The sort is by camera-space $z$ ascending, nearest first, and every per-Gaussian quantity is put
into that order so the compositing pass can walk front to back. In our two-Gaussian example the red
Gaussian at depth $3.2$ comes before the blue one at depth $4$.

Each covariance is then inverted once. The rasterizer never needs $\Sigma_{2D}$ itself, only
$\Sigma_{2D}^{-1}$, since only the inverse appears in the Mahalanobis exponent. The level sets of
$d^\top \Sigma_{2D}^{-1} d = \text{const}$ are ellipses, and a quadratic curve in the plane is a
conic section, so graphics calls the inverse covariance the conic of the splat. For a $2\times2$
matrix the inverse is closed form (the letters $a, b, c, d$ here are just the four entries),

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix},$$

with the dilation keeping $ad-bc$ away from zero. The inverse is computed once per Gaussian and
reused at every pixel, not recomputed per pixel.

For pixel $x$ and Gaussian $i$, with $d = x - \mu_i$ the offset from the splat center,

$$\alpha_i(x) = o_i \exp\!\left(-\tfrac12\, d^\top \Sigma_{2D,i}^{-1} d\right), \qquad \alpha_i \le 0.99.$$

The opacity $o_i$ is the peak value at the center and the exponential is the falloff in Mahalanobis
distance. The two words are kept apart in these notes: the opacity $o_i$ is a stored property of
the Gaussian, and the alpha $\alpha_i(x)$ is how opaque its splat is at one particular pixel. The
clamp keeps $1 - \alpha_i \ge 0.01$. Kerbl et al. list it among their numerical-stability measures,
alongside a backward pass that recovers intermediate values by repeatedly dividing the accumulated
opacity by each Gaussian's alpha; the paper states no further reason, and their CUDA code divides
the transmittance by $1 - \alpha$, which the clamp keeps away from zero. A second reason is our
own reading: with the clamp, the transmittance behind a single Gaussian never collapses to exactly
zero, so the Gaussians behind an opaque one still receive a nonzero gradient. Two limits apply. The
reference rasterizer stops blending at a pixel once the accumulated opacity would exceed $0.9999$,
and behind two layers clamped at $0.99$ the transmittance is already $10^{-4}$, so the gradient to
deeper Gaussians ends there. And where the clamp is active its derivative is zero, so at that pixel
the front Gaussian's own opacity gets no gradient.

Compositing is the front-to-back Porter-Duff "over" chain developed for the volume renderer of the
NeRF assignment, applied here with no change. Each Gaussian is a layer, and putting a layer of
color $c_i$ and alpha $\alpha_i$ over what is behind it contributes $\alpha_i c_i$ and passes
through a fraction $1-\alpha_i$ of everything further back. Chaining that over the sorted list gives

$$C(x) = \sum_i c_i\, \alpha_i(x)\, T_i(x) + T_{\mathrm{final}}(x)\, \mathrm{bg},
\qquad T_i = \prod_{j<i}(1-\alpha_j), \qquad T_0 = 1.$$

$T_i$ is called the exclusive transmittance because the product runs over layers strictly in front
of $i$ and excludes $i$ itself; the empty product $T_0 = 1$ says nothing occludes the nearest
Gaussian. $T_{\mathrm{final}}$ is the transmittance surviving all $N$ layers, and it lets the
background color $\mathrm{bg}$ through. An off-by-one here (an inclusive product, or $T$ shifted by
one) still produces a plausible-looking image, which is why the checks for it should be exact
rather than visual. The weight $T_i\alpha_i$ is the solid curve in panel (c) of the two-Gaussian
figure, and $T_{\mathrm{final}}$ is the black one.

Three things differ from the ray-marched version of the same chain. The $\alpha$ has no
Beer-Lambert step behind it: there is no density and no segment length, only a learned peak opacity
times a Gaussian falloff. The ordering comes from sorting primitives by the depth of their centers
once per view rather than from marching $t$ along each ray, which is an approximation whenever two
Gaussians interpenetrate, since one global order cannot be right at every pixel, and the error
shows up as splats popping in and out as the camera moves. And the same list of layers is shared by
every pixel, with most Gaussians contributing $\alpha \approx 0$ at most of them.

```mermaid
flowchart TB
  subgraph nerf [ray-marched volume render]
    A1[density sigma_i, segment delta_i] --> A2["alpha_i = 1 - exp(-sigma_i delta_i)"]
  end
  subgraph splat [Gaussian splat]
    B1[opacity o_i, conic, pixel offset d] --> B2["alpha_i = o_i exp(-0.5 d^T Sigma_2D^-1 d)"]
  end
  A2 --> C["T_i = prod_{j<i}(1 - alpha_j)"]
  B2 --> C
  C --> D["C = sum_i c_i alpha_i T_i + T_final bg"]
```

The depth sort is not differentiable: an argsort is piecewise-constant in the parameters, so its
derivative is zero almost everywhere and undefined at the ties. That costs nothing here, because the
sort only reorders the per-Gaussian quantities. Gradients flow through the reordered values (means,
covariances, colors, opacities), so automatic differentiation is correct as long as the loss does
not depend on the depths through the sort key alone. A gradient check of the full render should
therefore differentiate with respect to the means and opacities, not with respect to the depths as
a separate input.

A straightforward render is vectorized over pixels: all $N$ Gaussians are evaluated against the
$H \times W$ pixel grid at once, with no loops over screen tiles. The real CUDA implementation is
tile-based instead: it buckets Gaussians into 16x16 screen tiles by their projected bounding box and
blends only the Gaussians touching a tile, which brings it to real time at 1080p. The vectorized
render is simpler to read and correct to differentiate, but it touches every Gaussian at every
pixel, so it is slower than the tiled kernel and, at a tiny resolution, not necessarily faster than
a batched NeRF MLP.

> **Exercise 4 (one pixel of our two-Gaussian example).** After the dilation, the near (red)
> Gaussian has $\mu = (6.25, 7.5)$, $\Sigma_{2D} = \mathrm{diag}(1.3061, 1.3)$, $o = 0.8$, and the far
> (blue) one has $\mu = (8.5, 7.5)$, $\Sigma_{2D} = \mathrm{diag}(4.3006, 0.46)$, $o = 0.9$. The
> background is white, $\mathrm{bg} = (1, 1, 1)$. For the pixel at $(u, v) = (7, 7)$, compute each
> alpha, the transmittances, and the composited RGB color. Then swap the depths so the blue one is
> in front and recompute. What stays the same?

<details>
<summary>Answer</summary>

Red: $d = (0.75, -0.5)$, $d^\top\Sigma^{-1}d = 0.5625/1.3061 + 0.25/1.3 = 0.4307 + 0.1923 = 0.6230$,
$\exp(-0.3115) = 0.7324$, so $\alpha_{\text{red}} = 0.8 \cdot 0.7324 = 0.586$.
Blue: $d = (-1.5, -0.5)$, $d^\top\Sigma^{-1}d = 2.25/4.3006 + 0.25/0.46 = 0.5232 + 0.5435 = 1.0667$,
$\exp(-0.5333) = 0.5866$, so $\alpha_{\text{blue}} = 0.9 \cdot 0.5866 = 0.528$.

Red in front: $T_{\text{red}} = 1$, $T_{\text{blue}} = 1 - 0.586 = 0.414$, and
$T_{\mathrm{final}} = 0.414 \cdot (1 - 0.528) = 0.195$. The weights are $0.586$ for red and
$0.414 \cdot 0.528 = 0.219$ for blue, and the color is
$0.586\,(1,0,0) + 0.219\,(0,0,1) + 0.195\,(1,1,1) = (0.781, 0.195, 0.414)$, a pinkish red.

Blue in front: the weights become $0.528$ for blue and $0.472 \cdot 0.586 = 0.277$ for red, and the
color is $(0.472, 0.195, 0.723)$, a blue-leaning purple.

$T_{\mathrm{final}} = 0.195$, and so the amount of white, is the same in both orders, because it is
a product of the $(1-\alpha_i)$ and products commute. The split between red and blue depends on the
order. This is why a wrong sort key or a reversed sort produces images that look plausible but have
the wrong color wherever splats overlap.

</details>

> **Exercise 5 (why the 0.99 clamp).** At some pixel, a front Gaussian has alpha $\alpha_1$ and a
> Gaussian behind it has alpha $\alpha_2 = 0.5$ and color $c_2$; nothing else covers the pixel and
> the background is $\mathrm{bg}$. Write the pixel color $C$ and compute $\partial C/\partial c_2$
> and $\partial C/\partial \alpha_2$ for $\alpha_1 = 1$ (no clamp) and for $\alpha_1 = 0.99$. Use
> $c_2 = 0$ and $\mathrm{bg} = 1$ for the second derivative.

<details>
<summary>Answer</summary>

$C = \alpha_1 c_1 + (1-\alpha_1)\big(\alpha_2 c_2 + (1-\alpha_2)\,\mathrm{bg}\big)$, so
$\partial C/\partial c_2 = (1-\alpha_1)\alpha_2$ and
$\partial C/\partial \alpha_2 = (1-\alpha_1)(c_2 - \mathrm{bg})$.

With $\alpha_1 = 1$ both are exactly $0$: the hidden Gaussian gets no gradient at this pixel at
all, whatever the error is. With $\alpha_1 = 0.99$ they are $0.01 \cdot 0.5 = 0.005$ and
$0.01 \cdot (0 - 1) = -0.01$: small but nonzero. The same factor $1 - \alpha_1$ is what a backward
pass that recovers transmittances by division would divide by, and the clamp keeps it at $0.01$ or
more.

</details>

### Fitting the cloud

With the render differentiable, fitting is ordinary gradient descent on the difference between the
render and the captured image. The original method uses a weighted mix of an L1 loss, the mean
absolute difference over pixels and channels, and a structural similarity term,
$1 - \mathrm{SSIM}$, where SSIM compares small windows of the two images by their local means,
variances, and cross-covariance and so scores structure rather than per-pixel error. Kerbl et al.
weight the structural term by $0.2$ and the L1 term by $0.8$.

The optimizer is Adam, a first-order method that keeps a running average of each parameter's
gradient and of its squared gradient and steps along the first divided by the square root of the
second, so every parameter gets a step scaled to its own gradient history rather than one global
step size. On top of that, each parameter group (means, scales, rotations, opacities, colors) gets
its own learning rate. The ordering follows how disruptive a change is. Rotating or resizing a
Gaussian changes which pixels it covers and how it interacts with every other Gaussian composited
alongside it, while raising its opacity or shifting its color only adjusts the contribution it
already makes where it already is, so shape moves cautiously and appearance does not have to. The
original method goes further and decays the position learning rate over training.

The whole fit sits next to bundle adjustment in classical structure-from-motion. Bundle adjustment
minimizes reprojection error, the pixel distance between a projected 3D point and its detected
image feature, over 3D point positions and camera poses. Gaussian splatting minimizes photometric
error, the color difference between a rendered pixel and the captured pixel, over 3D Gaussian
parameters. Both fit a 3D scene seen through cameras by minimizing an image-space residual through
its local derivatives; bundle adjustment uses squared error and Gauss-Newton steps, splatting the
L1 and SSIM loss and Adam. The splatting version replaces sparse feature correspondences with a
dense rendered image and adds appearance to the unknowns. Methods like InstantSplat (Fan et al.
2024, [arXiv:2403.20309](https://arxiv.org/abs/2403.20309)) put the camera poses back among the
unknowns and optimize them jointly with the Gaussians, which is bundle adjustment with a
photometric residual.

### Densification

Gradient descent moves and reshapes the Gaussians it is given; it cannot create or delete any. The
real method therefore wraps the differentiable fit in adaptive density control (ADC), a
non-differentiable outer loop that Kerbl et al. run every 100 iterations (from iteration 500 to
15,000 in their reference implementation).

Its trigger is the positional gradient: the magnitude of the loss gradient with respect to a
Gaussian's projected 2D position, accumulated over recent iterations and over views. A large value
means the optimizer is being pulled hard, and in inconsistent directions, to move that Gaussian,
which is the signature of one blob trying to explain image content it cannot fit. What to do about
it depends on the Gaussian's size. If it is small relative to the scene, the region is
under-reconstructed (there are not enough primitives there), so ADC clones it and displaces the
copy along the gradient. If it is large, the region is over-reconstructed (one broad blob is
covering detail that needs several), so ADC splits it into two smaller children with centers
sampled from the parent's own Gaussian and their scales divided by $1.6$.

Two removal rules balance the growth. Gaussians whose opacity has fallen below a threshold get
pruned, as do ones whose splat has grown implausibly large. Every 3000 iterations all
opacities are reset close to zero, which forces each Gaussian to re-earn its opacity from the
photometric fit; the ones that were only covering up error, such as floaters (stray semi-transparent Gaussians hanging in empty space, typically just in front
of a training camera, that reduce error in that one view and look wrong from every other), never recover and are pruned on the next pass.

Together these rules grow the sparse point cloud that structure-from-motion already produced while
solving for the camera poses into millions of Gaussians, at the local density each part of the
scene needs.

## Where this goes next

The 2D Gaussian splatting follow-up (Huang et al. 2024,
[arXiv:2403.17888](https://arxiv.org/abs/2403.17888)) replaces the 3D ellipsoids with oriented 2D
disks. Those are surfels, surface elements: a center, a normal, and a radius, describing a surface
as a cloud of small oriented patches with none of a mesh's connectivity, the same primitive
surfel-based SLAM mapping uses. A flat disk has an unambiguous normal, where a 3D ellipsoid does not
(its shortest axis is only a proxy), so the disk representation gives clean surface normals and
clean mesh extraction, the form most useful for mapping and SLAM in autonomous driving.

The reference implementation most work builds on is the gsplat library (Ye et al. 2024,
[arXiv:2409.06765](https://arxiv.org/abs/2409.06765)). Feed-forward splatting predicts the
Gaussians in a single network pass instead of optimizing per scene: pixelSplat (Charatan et al.
2023, [arXiv:2312.12337](https://arxiv.org/abs/2312.12337)) and MVSplat (Chen et al. 2024,
[arXiv:2403.14627](https://arxiv.org/abs/2403.14627)) regress a Gaussian cloud from a few input
images, which matters for camera-rig reconstruction where a per-scene fit is too slow. 3D Gaussian
splatting is also now a common scene representation in driving simulators and SLAM back-ends.

## The assignment

Fill these holes, in order. Each is one `NotImplementedError` with a matching test; the docstring
in each file gives the signature, shapes, and constraints.

| # | Function | File | Concept from the notes |
|---|---|---|---|
| 1 | [`quat_to_rotmat()`](gaussian.py) | `gaussian.py` | the unit-quaternion rotation, real part first, normalized in the forward pass |
| 2 | [`build_covariance_3d()`](gaussian.py) | `gaussian.py` | $\Sigma_{3D} = RSS^\top R^\top$ from quaternions and log-scales |
| 3 | [`perspective_jacobian()`](project.py) | `project.py` | the $2\times3$ pinhole Jacobian $J$ |
| 4 | [`project_cov_to_2d()`](project.py) | `project.py` | the EWA step $J W \Sigma_{3D} W^\top J^\top$ plus the dilation |
| 5 | [`splat_render()`](render.py) | `render.py` | depth sort, conic, per-pixel alpha with the 0.99 clamp, front-to-back compositing |

The rasterizer is assignment-local and not reused downstream, so these files are imported by bare
name rather than through a `nanovision` shim.

Notes on specific holes and the provided code:
- `GaussianModel` in `gaussian.py` holds `opacity_logits` and `color_logits` (with `means`,
  `log_scales`, and `quats`) and exposes `opacities` and `colors` as their sigmoids.
- The toy stores a raw RGB triple squashed by a sigmoid into $[0,1]$, with `sh_degree = 0` in
  `config.py`. That is view-independent color, matching what degree-0 spherical harmonics would
  give, but it is not the same arithmetic: degree-0 spherical harmonics would store one coefficient
  per channel, multiply by $1/(2\sqrt{\pi})$, and still need clamping to reach a displayable range,
  while the sigmoid lands in $[0,1]$ directly.
- `project_gaussians()` in `project.py` is provided wiring around holes 3 and 4. It moves the means
  to camera space with the world-to-camera transform `w2c` (the inverse of the camera pose), passes
  the rotation block `w2c[:3, :3]` as the argument `W`, and passes the `dilation`, whose default of
  0.3 matches the original method.
- `splat_render` takes the depths only as a sort key. Sort with `torch.argsort`, gather every
  per-Gaussian tensor by the result, and evaluate all Gaussians against the whole pixel grid with
  broadcasting, no Python tile loops. Pixel centers are at integer coordinates, $u$ along the width
  and $v$ along the height, matching the $(u, v)$ order of the projected means.
- `densify.py` implements only `prune_low_opacity()`, the opacity prune of adaptive density
  control; clone, split, and opacity reset are read-and-understand.
- The fits use an L1 loss only. On a 16x16 toy image the SSIM window statistics are degenerate, so
  the toy drops that term.
- `config.py` sets per-group Adam learning rates for the viz fit: 1e-3 for quaternions and 5e-3 for
  log-scales, 1e-2 for means and color logits, and 5e-2 for opacity logits.

### Running and validating

Activate the environment (`conda activate nanovision`), then:

```
make test     A=a10_gaussian_splatting   # run the tests against your top-level files (red until the holes are filled)
make verify   A=a10_gaussian_splatting   # run the same tests against the reference solution/ (green from the start)
make viz      A=a10_gaussian_splatting   # render the figures from the reference solution
make viz-mine A=a10_gaussian_splatting   # render the figures from your own code (once the holes are filled)
```

`make test` is the command to run while working. It runs the suite in `tests/` against the
top-level files (the ones with the holes) and goes from red (the holes raise
`NotImplementedError`) to green as the holes are filled. `make verify` runs the identical suite
against the reference in `solution/`: it sets `NANOVISION_IMPL=solution`, so the tests import the
reference implementation instead of the top-level files. `make verify` is green from the start, so
it shows the target and confirms the tests and the environment work before anything changes. The
goal is to bring `make test` to the same green as `make verify`.

Several of the checks below run `torch.autograd.gradcheck`, which compares the analytic backward
pass against finite differences of the forward pass and needs double precision for the comparison
to mean anything.

The suite checks, in workflow order:
- `test_covariance.py`: the covariance is symmetric PSD and its sorted eigenvalues equal the sorted
  squared scales; the rotation matrices are orthogonal with determinant 1; a float64 gradcheck of
  `build_covariance_3d` and `quat_to_rotmat`, using order-one quaternions, since the normalization
  is singular at $q=0$.
- `test_projection.py`: the perspective Jacobian matches a finite-difference Jacobian of the pinhole
  projection, and the EWA covariance is symmetric with positive eigenvalues, with a float64
  gradcheck.
- `test_render_forward.py`: one centered Gaussian renders as a splat brightest at the center, the output is
  in $[0,1]$, two overlapping opaque Gaussians let the nearer color dominate and swapping depths
  swaps the winner, and all-zero opacity gives the background.
- `test_render_grad.py`: a float64 gradcheck through the full forward with respect to the means and
  opacity logits only; the depths are excluded because they reach the output only through the sort.
- `test_overfit_image.py`: an end-to-end fit of 64 Gaussians to one 16x16 view.
- `test_forbidden_imports.py`: a static scan blocking prebuilt rasterizers (gsplat,
  diff_gaussian_rasterization, nerfstudio, simple_knn). The forbidden-imports scan passes with the
  holes in place.

`make viz` renders from the reference solution, so it works on a fresh checkout before any holes
are filled and shows the target figures. `make viz-mine` runs the same script against the
top-level code, which is the way to check by eye whether a finished implementation behaves; it
needs the holes filled, since it trains a model with them. Both fit 200 Gaussians to the five
training views of our toy sphere at 32x32 and write `splat_fit.png` (target vs render for a
training view and a held-out view), `psnr_curve.png` (held-out PSNR over training), and `speed.png`
(the measured per-view render time of the splat against the ray-marched NeRF) to `out/`, using
matplotlib's headless Agg backend so the commands behave the same over SSH, in WSL, and in CI with
no display. The viz fit runs on the GPU when one is present. Add `SHOW=1` (for example
`make viz-mine A=a10_gaussian_splatting SHOW=1`) to also open the figures in interactive windows
when a display is available.

The figures in the notes are generated by `assets/make_figures.py`: the toy views from the scene
generator, and our two-Gaussian example and the EWA sampling check computed directly from the
formulas in the notes with numpy. None of them uses the assignment code.

What you should see when you run this. The overfit test drives the L1 loss, the mean absolute
difference between the rendered and target pixels, from a roughly 0.10-0.14 start down to about
0.0048-0.0053 in 400 steps, so the 0.02 threshold sits comfortably above the floor; the floor was
measured at build time. A correct fit shows a monotone L1 drop in the first few dozen steps. A flat
loss means a projection sign error, a wrong transmittance, or a broken sort. The viz fit reaches
roughly 25 dB held-out PSNR in a few seconds on a 32x32 scene. PSNR is peak signal-to-noise ratio,
$-10\log_{10}(\mathrm{MSE})$ for pixel values in $[0,1]$, so higher is better and every 10 dB is a
tenfold drop in mean squared error. The printed speed ratio is whatever the run measures: at this
toy scale the untiled pure-PyTorch render is about the same speed as the NeRF (around 0.9x on an
RTX 4080 at 32x32), because without tile culling it evaluates every Gaussian at every pixel. These
are toy artifacts. The real-time advantage shows up at high resolution with the tiled CUDA kernel,
not at this scale.

## Stretch goals

- Add a tile-based culling pass: bucket Gaussians into screen tiles by their projected bounding
  box and blend only the Gaussians that touch each tile. Measure how the speedup ratio changes with
  image size.
- Implement degree-1 spherical harmonics for view-dependent color and check whether the held-out
  PSNR improves on a scene with a specular highlight.
- Add the full adaptive density control loop (clone, split, prune, opacity reset) around the fit
  and watch the Gaussian count adapt to the scene.

## References

- Kerbl et al. 2023, "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
  [arXiv:2308.04079](https://arxiv.org/abs/2308.04079): the original method, the tile rasterizer,
  ADC, and degree-3 SH.
- Zwicker et al. 2001, "EWA Volume Splatting", IEEE Visualization,
  [DOI:10.1109/VISUAL.2001.964490](https://doi.org/10.1109/VISUAL.2001.964490), and the 2002 TVCG
  "EWA Splatting" follow-up: the covariance projection this assignment implements.
- Greene and Heckbert 1986, "Creating Raster Omnimax Images from Multiple Perspective Views Using
  the Elliptical Weighted Average Filter", IEEE Computer Graphics and Applications: the original
  EWA texture filter the splatting work is named after.
- Porter and Duff 1984, "Compositing Digital Images", SIGGRAPH,
  [doi:10.1145/800031.808606](https://doi.org/10.1145/800031.808606): the "over" operator the
  rasterizer chains.
- Müller et al. 2022, Instant-NGP, [arXiv:2201.05989](https://arxiv.org/abs/2201.05989).
- Yu et al. 2023, "Mip-Splatting", [arXiv:2311.16493](https://arxiv.org/abs/2311.16493): the
  scale-aware filter that replaces the fixed dilation and removes zoom aliasing.
- Huang et al. 2024, "2D Gaussian Splatting",
  [arXiv:2403.17888](https://arxiv.org/abs/2403.17888): surfels with normals and clean meshing.
- Fan et al. 2024, "InstantSplat", [arXiv:2403.20309](https://arxiv.org/abs/2403.20309): joint
  Gaussian and pose optimization.
- Ye et al. 2024, "gsplat", [arXiv:2409.06765](https://arxiv.org/abs/2409.06765): the reference
  rasterizer the field builds on (blocked by the forbidden-imports test here).
- Charatan et al. 2023, "pixelSplat",
  [arXiv:2312.12337](https://arxiv.org/abs/2312.12337), and Chen et al. 2024, "MVSplat",
  [arXiv:2403.14627](https://arxiv.org/abs/2403.14627): feed-forward Gaussian prediction from
  sparse views, no per-scene optimization.
