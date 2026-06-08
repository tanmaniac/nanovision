# assignments/a10_gaussian_splatting/ASSIGNMENT.md

```yaml
id: a10_gaussian_splatting
title: 3D Gaussian splatting
module: 3
type: Core
estimated_learner_hours: 8
depends_on: [a00_harness, a09_nerf, a11_5a_camera_geometry_bev]
builds_into_shared_lib: []          # the splat rasterizer is assignment-local, not reused downstream
forbidden_imports:
  - import gsplat
  - from gsplat
  - import diff_gaussian_rasterization
  - from diff_gaussian_rasterization
  - import nerfstudio
  - from nerfstudio
  - import simple_knn
  - from simple_knn
fits_12gb: true
external_data: "none (the A9 posed colored-sphere toy via nanovision.data.toy.nerf_synthetic_scene)"
```

## motivation
3D Gaussian splatting represents a scene as an explicit cloud of 3D Gaussians and renders by
projecting each to a 2D Gaussian and alpha-compositing front to back, with no MLP in the
render path. It matches NeRF quality while rendering in real time. A10 builds the
differentiable forward rasterizer in pure PyTorch and fits a toy scene through it by gradient
descent on photometric error, the differentiable-optimization-over-3D frame next to bundle
adjustment. See the README for the full treatment.

## background
Scene is N Gaussians: means_w $(N,3)$, log_scales $(N,3)$, quats $(N,4)$, opacity_logits
$(N,)$, color_logits $(N,3)$. Covariance factored as $\Sigma_{3D} = R S S^\top R^\top$,
$S = \mathrm{diag}(\exp(\text{log\_scales}))$, $R$ from the unit-normalized quaternion (PSD by
construction). EWA projection (OpenCV $+z$, matching `nanovision.geometry`): with camera-space
mean $(x,y,z)$,

$$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix},\qquad
\Sigma_{2D} = J W \Sigma_{3D} W^\top J^\top + 0.3 I,$$

$W$ = world-to-camera rotation (translation does not enter). Render: sort by depth front to
back, $\alpha_i = o_i \exp(-\tfrac12 d^\top \Sigma_{2D,i}^{-1} d)$ clamped to $\le 0.99$,
composite $C = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j) + T_\text{final}\,\text{bg}$ (same
exclusive transmittance as the NeRF volume renderer). Conic inverted once per Gaussian,
broadcast over pixels.

## what_you_implement
- `quat_to_rotmat`, `build_covariance_3d`: the quaternion rotation and the scale+rotation 3D covariance.
- `perspective_jacobian`, `project_cov_to_2d`: the 2x3 projection Jacobian and the EWA covariance projection.
- `splat_render`: the depth-sorted alpha-compositing rasterizer, vectorized over pixels.

The `GaussianModel` container, `project_gaussians` wiring, `prune_low_opacity`, config, and viz are provided.

## tasks
1. **Task 1 - quat_to_rotmat** (file: `gaussian.py`, symbol: `quat_to_rotmat`): $(N,4) \to (N,3,3)$. Normalize $q \to q/\lVert q\rVert$, build the standard rotation from $(w,x,y,z)$. Teaches the well-defined, differentiable quaternion-to-rotation map (singular only at $q=0$).
2. **Task 2 - build_covariance_3d** (file: `gaussian.py`, symbol: `build_covariance_3d`): $(N,4),(N,3) \to (N,3,3)$. $\Sigma = (RS)(RS)^\top$ with $S=\mathrm{diag}(\exp(\text{log\_scales}))$. Teaches the factorization that keeps the covariance PSD and differentiable.
3. **Task 3 - perspective_jacobian** (file: `project.py`, symbol: `perspective_jacobian`): $(N,3),(3,3) \to (N,2,3)$. The Jacobian of `project_points` at the camera-space mean. Teaches the linearization the EWA step propagates.
4. **Task 4 - project_cov_to_2d** (file: `project.py`, symbol: `project_cov_to_2d`): $(N,3,3),(3,3),(N,2,3) \to (N,2,2)$. $J W \Sigma W^\top J^\top + 0.3 I$, $W$ world-to-camera rotation. Teaches covariance propagation through a linear map and the dilation that guards the inverse.
5. **Task 5 - splat_render** (file: `render.py`, symbol: `splat_render`): $\to (H,W,3)$. Depth sort (gather values), one closed-form 2x2 conic inverse per Gaussian, per-pixel $\alpha$ clamped to 0.99, front-to-back compositing. Teaches the rasterizer and that the sort carries no gradient while the gathered values do.

## tests
- `tests/test_covariance.py` - PSD + symmetry, eigenvalues = sorted $\exp(\text{log\_scales})^2$ (distinct scales), float64 gradcheck of `build_covariance_3d` and `quat_to_rotmat` (reference-value + gradcheck). Order-one quaternions only.
- `tests/test_projection.py` - `perspective_jacobian` vs finite-difference of `project_points` (atol 1e-4); `project_cov_to_2d` symmetric PD; float64 gradcheck (reference-value + gradcheck).
- `tests/test_render_forward.py` - centered blob brightest at center; shape/range; front-to-back ordering and the depth swap; zero-opacity equals bg (training-free exact checks).
- `tests/test_render_grad.py` - float64 gradcheck through the full forward w.r.t. `means_w` and `opacity_logits` only (gradcheck).
- `tests/test_overfit_image.py` - fit N=64 to one 16x16 view, 400 Adam steps, L1 < 0.02 and < 0.5*initial (overfit / end-to-end).
- `tests/test_forbidden_imports.py` - static scan, no gsplat / diff_gaussian_rasterization / nerfstudio / simple_knn. Passes with holes in place.

Run order: covariance -> projection -> render_forward -> render_grad -> overfit_image -> forbidden_imports.

## provided_boilerplate
`GaussianModel` (sigmoid opacity/color, `random_init` centered on the world origin),
`project_gaussians` (means to camera space, `project_points`, build cov3d, call the holes),
`prune_low_opacity` (the ADC prune stand-in; full ADC described, not implemented), `config.py`
`SplatConfig`, `viz.py` (GPU fit of the 32x32 multi-view scene, target-vs-render panels, the
held-out PSNR curve, and the measured NeRF-vs-splat inference-time ratio). Reused:
`nanovision.geometry` (`project_points`, `apply_transform`, `invert_transform`),
`nanovision.data.toy.nerf_synthetic_scene`, `nanovision.determinism.default_device`.

## compute_notes
All tests CPU, seconds each. Graded renders 16x16 with a few dozen Gaussians; overfit is 400
Adam steps. Measured L1 floor ~0.005-0.007 from a ~0.10-0.14 start, threshold 0.02. A healthy
fit shows a monotone L1 drop in the first few dozen steps; a flat loss means a projection sign
error, a wrong transmittance, or a broken sort. Viz fits 200 Gaussians at 32x32 for 1500 steps
on the GPU, ~25 dB held-out PSNR in a few seconds; the printed speed ratio is whatever the run
measures (about 0.9x on an RTX 4080 at 32x32, since the untiled pure-PyTorch render touches
every Gaussian at every pixel).

## stretch_goals
1. Tile-based culling: bucket Gaussians into screen tiles by projected bounding box, blend only the touching ones; measure how the speed ratio scales with image size.
2. Degree-1 spherical harmonics for view-dependent color; check held-out PSNR on a specular scene.
3. Full adaptive density control (clone, split, prune, opacity reset) around the fit.

## further_reading
- Kerbl et al. 2023, "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (https://arxiv.org/abs/2308.04079): the original method.
- Zwicker et al., "EWA volume splatting", IEEE Visualization 2001 (https://doi.org/10.1109/VISUAL.2001.964490) and "EWA splatting", IEEE TVCG 2002: the covariance projection.
- Yu et al. 2023, "Mip-Splatting" (https://arxiv.org/abs/2311.16493): the scale-aware filter that replaces the fixed dilation.
- Huang et al. 2024, "2D Gaussian Splatting" (https://arxiv.org/abs/2403.17888): surfels with normals and meshing.
- Ye et al. 2024, "gsplat" (https://arxiv.org/abs/2409.06765): the reference library (blocked by the forbidden-imports test).
- Charatan et al. 2023, "pixelSplat" (https://arxiv.org/abs/2312.12337) and Chen et al. 2024, "MVSplat" (https://arxiv.org/abs/2403.14627): feed-forward Gaussian prediction.

## solution_notes
Quaternion convention is $(w,x,y,z)$, real part first. `quat_to_rotmat` normalizes in-forward;
gradcheck is singular at $q=0$, so the covariance and projection tests seed quats as
$(1,0,0,0)$ + 0.3*noise (order-one magnitude) - do NOT seed tiny quats. Eigenvalue identity
uses distinct scales [0.3, 0.7, 1.4] so the three eigenvalues are identifiable. EWA dilation
0.3 px^2 added to the 2D-cov diagonal; $W$ is the 3x3 rotation block of `invert_transform(c2w)`,
translation excluded. The render's depth sort uses `torch.argsort` (no gradient); gather the
per-Gaussian tensors by it so gradients flow through values. `test_render_grad` passes `w2c =
identity` directly (not via `invert_transform`) and excludes `depths` from gradcheck inputs,
because the sort key reaches the output only through `argsort` and a depths-only loss errors.
Overfit seed 0, `GaussianModel.random_init` centered on the world origin (the sphere sits at
the origin; a cube placed at depth `cam_dist` does not overlap the object and the loss stays
flat - this was the build-time bug, fixed by centering the init at the origin). Verify:
`NANOVISION_IMPL=solution /home/tanmay/miniconda3/envs/nanovision/bin/python -m pytest
assignments/a10_gaussian_splatting/tests`.
