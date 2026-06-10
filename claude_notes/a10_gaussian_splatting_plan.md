# A10 - 3D Gaussian splatting: build plan

Status: draft for expert review. Build subagent reads this whole file plus
`agent_build_guide.md` and mirrors the exemplar `assignments/a06_0_flow_matching` for layout.

## What this assignment teaches

3D Gaussian splatting represents a scene as a cloud of 3D Gaussians (each with a position,
covariance, opacity, color) and renders by projecting each to a 2D Gaussian in image space and
alpha-compositing them front to back. A10 builds the differentiable forward rasterizer and fits a
toy scene by gradient descent through it (autograd, no custom CUDA). The frame to stress: this is
differentiable optimization over an explicit 3D representation, close to bundle adjustment - you
minimize photometric error over a rendered field instead of reprojection error over point
correspondences.

The contrast with NeRF (A9) is the point: same front-to-back alpha compositing
$C = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j)$, but the $\alpha$ comes from a depth-sorted
projected 2D Gaussian, not from $1-\exp(-\sigma\delta)$ along a ray, and rendering evaluates a few
2D Gaussians per pixel instead of an MLP hundreds of times. Expect inference to be faster (a
handful of 2D Gaussians per pixel vs an MLP at every ray sample); MEASURE the wall-clock ratio in
viz and report whatever it is - do not assert a fixed multiplier (the ratio depends on the NeRF's
width/samples and the splat count).

Three core mechanisms the student implements: the scale+rotation -> 3D covariance factorization,
the EWA Jacobian projection of a 3D covariance to a 2D screen-space covariance, and the sorted
alpha-compositing rasterizer. Densification is read-and-understand (a simple opacity prune only).

## Reused, already in place (do NOT rebuild)

- `from nanovision.geometry import project_points, apply_transform, invert_transform`: the A11.5a
  camera primitives (OpenCV +z convention). `project_points(pts_cam, K)` is the perspective
  projection whose Jacobian the EWA step linearizes - the EWA Jacobian must match this convention.
- `nanovision.data.toy.nerf_synthetic_scene(...)`: the A9 posed colored-sphere scene (images
  (V,H,W,3), poses (V,4,4) c2w, K, near, far). A10 fits the SAME posed images. Reuse; do not add
  toy data.
- `nanovision.determinism.default_device` for the viz/training demo (GPU); tests stay CPU.

A10's renderer is assignment-LOCAL (2D-Gaussian splatting is not reused downstream - A11.5d
occupancy uses A9's ray-marched volume_render, not splatting). No new nanovision shim.

## Convention and the EWA projection (the main trap)

Everything is OpenCV +z-forward, matching nanovision.geometry. The 3D Gaussian mean in camera
space is $\mu_c = R_{wc}\mu_w + t$ (world-to-camera). Perspective projection
$\pi(x,y,z) = (f_x x/z + c_x,\ f_y y/z + c_y)$ has Jacobian at $\mu_c$:
$$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$$
(2x3, evaluated at the camera-space mean). With $W = R_{wc}$ the world-to-camera rotation, the
2D screen-space covariance is $\Sigma_{2D} = J W \Sigma_{3D} W^\top J^\top$ (2x2). Add a small
dilation $\Sigma_{2D} \mathrel{+}= \lambda I$ (the original uses $\lambda \approx 0.3$ px$^2$) so
$\Sigma_{2D}$ stays invertible and Gaussians cover at least a pixel. State this and the OpenCV
convention; getting the Jacobian sign or the W transpose wrong silently distorts every splat.

## Shapes (fix these numbers)

- N Gaussians: means_w (N,3), log_scales (N,3), quats (N,4), opacity_logits (N,), colors (N,3)
  (degree-0 SH = constant per-Gaussian RGB; view-dependent SH is README context, not implemented).
- Camera: K (3,3), c2w (4,4); world->camera via invert.
- 3D covariance (N,3,3); camera-space means (N,3); 2D means (N,2); 2D covariance (N,2,2); depths
  (N,).
- Render target: small image, H=W=32 for viz, 16 for the fast graded tests. Output (H,W,3) in
  [0,1].

## Files (mirror the exemplar layout)

### `gaussian.py` (holed; solution copy) - assignment-local

Holes:
- `quat_to_rotmat(q)`: normalize the quaternion (so the map is well-defined and differentiable),
  build the 3x3 rotation. (N,4) -> (N,3,3).
- `build_covariance_3d(quats, log_scales)`: $\Sigma_{3D} = R S S^\top R^\top$ with $S =
  \mathrm{diag}(\exp(\text{log\_scales}))$ and $R$ from `quat_to_rotmat`. (N,4),(N,3) -> (N,3,3).
  PSD by construction. Docstring: the factorization is what keeps the covariance positive
  semi-definite and differentiable, the standard pure-PyTorch pattern.

Provided: the `GaussianModel(nn.Module)` holding the parameters (means, log_scales, quats,
opacity_logits, colors), with `opacities = sigmoid(opacity_logits)`, `colors = sigmoid(colors)`,
and a random initializer in front of a camera. Note in the docstring: storing an RGB color and
squashing with sigmoid is a toy stand-in for VIEW-INDEPENDENT color, i.e. the degree-0 case. Do
NOT call `sigmoid(raw)` "degree-0 SH" as an equality - true degree-0 SH stores one coefficient per
channel scaled by the $Y_0 = 1/(2\sqrt\pi)$ basis constant; the sigmoid is just a convenient
[0,1] map for the toy. Higher SH degrees (view-dependent specular) are README context only.

### `project.py` (holed; solution copy) - assignment-local

Holes:
- `perspective_jacobian(means_cam, K)`: the 2x3 $J$ above, per Gaussian. (N,3),(3,3) -> (N,2,3).
- `project_cov_to_2d(cov3d, W, J)`: $\Sigma_{2D} = J W \Sigma_{3D} W^\top J^\top + \lambda I$,
  $\lambda = 0.3$ added to the two DIAGONAL entries (units px$^2$). $W$ is the world-to-camera
  ROTATION only (the 3x3 block of `invert_transform(c2w)`); translation shifts the mean, not the
  covariance. $J$ is evaluated at the camera-space mean. (N,3,3),(3,3),(N,2,3) -> (N,2,2). The
  dilation guards the determinant so the 2x2 inverse is safe.

Provided: `project_gaussians(model, K, w2c)` wiring - transform means to camera space (reuse
apply_transform), `project_points` for the 2D means and depths, build cov3d, call the holes -
returns means2d (N,2), cov2d (N,2,2), depths (N,).

### `render.py` (holed; solution copy) - assignment-local

Hole:
- `splat_render(means2d, cov2d, colors, opacities, depths, H, W, *, bg=0.0)`: the rasterizer.
  - sort Gaussians by depth (front to back); reorder all per-Gaussian tensors by the sort (gather
    - gradients flow through the gathered VALUES; the sort order itself is non-differentiable,
    which is fine).
  - for each pixel x, each Gaussian's weight is $\alpha_i = o_i \cdot \exp(-\tfrac12 (x-\mu_i)^\top
    \Sigma_{2D,i}^{-1}(x-\mu_i))$, clamped to <= 0.99.
  - composite: $C(x) = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j) + T_{\text{final}}\cdot bg$,
    the SAME exclusive-transmittance structure as A9's volume_render (write the parallel in the
    docstring).
  - Vectorize over pixels (evaluate all N Gaussians against the HxW pixel grid with broadcasting);
    do NOT write Python tile loops - they are too slow to test. Compute the conic
    $\Sigma_{2D}^{-1}$ ONCE per Gaussian (closed-form 2x2 inverse, determinant guarded by the 0.3
    dilation) and broadcast it over pixels - do not invert per pixel.
  Returns the image (H,W,3).

### `densify.py` (provided only) - assignment-local, read-and-understand

- `prune_low_opacity(model, threshold)`: return a new GaussianModel keeping only Gaussians with
  opacity above threshold. The simplified stand-in for adaptive density control. Provided with a
  docstring describing the full ADC (clone under-reconstructed, split over-reconstructed, prune
  low-opacity, periodic opacity reset) as read-and-understand, non-differentiable, out of graded
  scope.

### `config.py`, `viz.py` (provided)

- config: N, H, W, sh_degree=0, dilation lambda=0.3, lrs, n_steps.
- viz: fit the toy scene on the GPU (`default_device()`), show target vs render for a training
  view and a held-out view, the PSNR curve, and a wall-clock inference-time comparison against the
  A9 NeRF on the same scene (the 5-20x speedup). Save to out/. Not graded. The PSNR>=20 dB / full
  multi-view fit is a viz/demo on GPU, NOT a CPU test.

### `conftest.py`, `__init__.py`, `solution/`

Mirror the exemplar conftest. `solution/` holds the holed files: `gaussian.py`, `project.py`,
`render.py` (plus `__init__.py`). `densify.py`, `config.py`, `viz.py` top-level only.

## Tests (env python, CPU, seconds each; training-free exact checks preferred)

1. `test_covariance.py`: `build_covariance_3d` output is symmetric PSD (eigenvalues >= 0); for a
   unit quaternion and DISTINCT scales s, the sorted eigenvalues equal sorted $s^2$; gradcheck
   (float64) on `build_covariance_3d` w.r.t. quats and log_scales, and on `quat_to_rotmat`
   (rotation is orthogonal: $R R^\top = I$, det 1). HARD constraint: use O(1)-magnitude quaternions
   (e.g. [1,0,0,0] + small noise) - gradcheck FAILS at near-zero quaternions (the normalization is
   singular there), so do not seed tiny quats.
2. `test_projection.py`: `perspective_jacobian` matches a finite-difference Jacobian of
   `project_points` at a few camera-space means (atol ~1e-4); `project_cov_to_2d` returns
   symmetric 2x2 with positive eigenvalues (dilation keeps it invertible); gradcheck (float64) on
   `project_cov_to_2d`.
3. `test_render_forward.py` (training-free exact checks):
   - one Gaussian at the image center -> a visible blob: the brightest pixel is at the center,
     intensity falls off with radius.
   - output shape (H,W,3), values in [0,1].
   - two Gaussians, near one fully opaque -> the near color dominates at the overlap (front-to-back
     ordering correct); swap depths -> the other dominates.
   - all opacities 0 -> image equals bg.
4. `test_render_grad.py`: gradcheck (float64) through the FULL forward (means_w + pose ->
   project -> splat_render) w.r.t. opacity_logits and means_w ONLY, tiny case (N=4, 4x4 image).
   Confirms autograd flows through the sort (gathered values) and the 2D Gaussian eval. Do NOT
   include `depths` as a gradcheck input: the sort key reaches the output only through `argsort`,
   which carries NO gradient, so a depths-only loss would error. Note this in the docstring (the
   discrete sort order is piecewise-constant in the parameters; only gathered values are
   differentiable).
5. `test_overfit_image.py`: fit N Gaussians (N~32-64) to ONE toy view (16x16, a few hundred steps,
   Adam, L1 loss - no D-SSIM, it is degenerate on a 16x16) - L1 falls below a MEASURED threshold.
   N~32-64 against 256 pixels is enough to require the Gaussians to place/shape themselves but
   small enough that a projection/sort bug shows as a non-decreasing loss; avoid N in the thousands
   (per-step cost is O(N*H*W) on CPU). This exercises the whole forward end to end in seconds.
   Report the floor and set the threshold from it (no-thrash); do NOT hardcode a number here. Do
   NOT assert PSNR>=20 (GPU/longer budget - viz only).
6. `test_forbidden_imports.py`: static scan over holed files + solution. Forbid importing a
   ready-made splatter (gsplat, diff_gaussian_rasterization, nerfstudio); mirror the exemplar.

Solution mode all green; default mode fails only at the holes (NotImplementedError), except
`test_forbidden_imports`. Run with `/home/tanmay/miniconda3/envs/nanovision/bin/python`.

## README (comprehensive lecture notes, per the skill, real LaTeX)

Fixed section order. Cover:
- The representation: 3D Gaussians, covariance factored as $\Sigma = RSS^\top R^\top$ (why: PSD by
  construction, differentiable), opacity, color (degree-0 SH = constant color; degree-3 SH for
  view-dependent specular as context). Frame as differentiable optimization over an explicit 3D
  field, the bundle-adjustment parallel (photometric error vs reprojection error, jointly
  optimizable with poses - BAGS/InstantSplat as context).
- The EWA projection (Zwicker et al. 2001/2002): a 3D Gaussian projects to a 2D Gaussian; linearize
  perspective projection by its Jacobian $J$ at the mean; $\Sigma_{2D} = J W \Sigma_{3D} W^\top
  J^\top$, drop the third row/col. The EWA step reuses the perspective-projection Jacobian - the
  same $\partial\pi/\partial X$ that linearizes reprojection error in bundle adjustment - but here
  it PROPAGATES a covariance ($J\Sigma J^\top$) rather than linearizing a residual. State it that
  way; do not claim it is "the same Jacobian as BA" without the distinction. The dilation filter
  and the aliasing it leaves (Mip-Splatting, CVPR 2024, as the fix).
- The rasterizer: depth sort, $\alpha_i = o_i G_i(x)$, front-to-back compositing with the SAME
  exclusive transmittance $T_i = \prod_{j<i}(1-\alpha_j)$ as A9 - write the two renderers side by
  side, identical compositing, different $\alpha$ source (sorted 2D Gaussian vs ray quadrature).
- Why it is fast: a few 2D Gaussians per pixel vs an MLP per ray sample. Tile-based rasterization
  in the real CUDA implementation; the course uses a vectorized pure-PyTorch render and is slower,
  state the expectation honestly.
- Densification (read-and-understand): clone/split/prune + opacity reset, a non-differentiable
  heuristic outer loop; the course prunes low-opacity only.
- Where this goes: 2DGS (surfels, normals, meshing - better for AV mapping/SLAM), gsplat (the
  reference library, do not use its rasterizer), feed-forward splatting (pixelSplat/MVSplat/
  AnySplat - predict Gaussians in one pass, no per-scene optimization, relevant to camera-rig
  reconstruction), 3DGS in AV simulation and SLAM. Mention, do not implement.

Verify every arXiv id by fetching `https://arxiv.org/abs/<id>`: 3DGS 2308.04079, 2DGS 2403.17888,
gsplat 2409.06765, Mip-Splatting 2311.16493, pixelSplat 2312.12337, MVSplat 2403.14627. (Zwicker
EWA 2001/2002 are IEEE, not arXiv - cite by venue.) Run the mandatory context-less style review on
the README.

## ASSIGNMENT.md

Concise builder contract in `TEMPLATE.md` format: holes (quat_to_rotmat, build_covariance_3d,
perspective_jacobian, project_cov_to_2d, splat_render), what is provided, the verify command, the
measured thresholds. Do not echo the README prose.

## Decisions resolved by the expert review (build to these)

The expert numerically verified the core math: the EWA Jacobian matches nanovision.geometry's
projection to machine precision, the 2D-covariance projection matches a 2M-sample Monte-Carlo to
0.6%, and gradcheck passes through the quaternion normalization and the depth-sort. Resolutions:

1. EWA Jacobian + $\Sigma_{2D}=JW\Sigma_{3D}W^\top J^\top$, $W$=world-to-camera rotation, OpenCV
   +z: confirmed, no sign/transpose error. Dilation $\lambda=0.3$ px$^2$ on the diagonal is right.
2. Covariance factorization is gradcheck-clean. HARD test constraint: O(1) quaternions only
   (gradcheck fails at near-zero). Eigenvalue identity confirmed with distinct scales.
3. Compositing confirmed structurally identical to A9. autograd through the sort is correct
   (gather of values differentiable; sort key carries NO gradient - keep `depths` out of the
   gradcheck inputs). L1 without D-SSIM for the 16x16 test is acceptable.
4. CPU fit of N~32-64 Gaussians to one 16x16 view, a few hundred steps, is a sound fast gate;
   measure the L1 floor at build time (rough 0.01-0.05 band, but measure - do not hardcode).
   PSNR>=20 / novel-view stays GPU-viz-only.
5. Degree-0 (constant per-Gaussian RGB via sigmoid) is the right implemented scope. Do NOT call it
   "degree-0 SH" as an equality (true degree-0 SH scales one coeff/channel by $1/(2\sqrt\pi)$);
   present sigmoid(raw) as a toy view-independent color. Do not implement degree-1.
6. Reworded: BA shares $\partial\pi/\partial X$ but EWA propagates a covariance vs BA linearizing a
   residual (not "the same Jacobian"); the fixed "5-20x" speedup is now "measure and report"; the
   "same compositing as A9" claim is verified true; densification read-only and vectorized-not-
   tiled are correct.
