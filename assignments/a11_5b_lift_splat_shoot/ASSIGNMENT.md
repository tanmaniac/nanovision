# assignments/a11_5b_lift_splat_shoot/ASSIGNMENT.md

```yaml
id: a11_5b_lift_splat_shoot
title: Lift-Splat-Shoot
module: 3.5
type: Core
estimated_learner_hours: 5
depends_on: [a00_harness, a11_5a_camera_geometry_bev]
builds_into_shared_lib:
  - nanovision.lift_splat.DepthLift
  - nanovision.lift_splat.frustum_points
  - nanovision.lift_splat.pillar_index
  - nanovision.lift_splat.cumsum_pool
  - nanovision.lift_splat.LiftSplatShoot
  - nanovision.lift_splat.bevdepth_depth_loss
forbidden_imports:
  - scatter_add        # the splat is the sort+cumsum trick, not a scatter
  - index_add
  - cv2.projectPoints
  - cv2.solvePnP
  - kornia
  - bev_pool
  - voxel_pooling
fits_12gb: true
external_data: none (tests run on the synthetic bev_toy_scene)
camera_axis_convention: OpenCV (+x right, +y down, +z forward); ego x forward, y left, z up
```

## motivation
Lift-Splat-Shoot (Philion & Fidler, ECCV 2020) pushes per-pixel image features out into 3D
using a learned categorical depth distribution, then splats them into an ego BEV grid. Depth is
a latent variable trained from the task loss alone ("implicit" depth). This is the push-out
half of the AV BEV module; BEVFormer is the pull-in half. The depth-lift and the sort+cumsum
splat are reused without change by occupancy (A11.5d). The README has the full treatment.

## background
A feature cell predicts D depth logits and a C-channel context. Softmax the logits to a depth
distribution $\alpha \in \mathbb{R}^D$; the outer product $\alpha \otimes c$ gives a $(D, C)$
volume per cell. The frustum places each $(d, i, j)$ entry at a 3D point: back-project the pixel
center to camera frame with $X=(u-c_x)d/f_x,\ Y=(v-c_y)d/f_y,\ Z=d$, then map to ego with
$T_{ego\_cam}=E^{-1}$ ($E$ is $T_{cam\_ego}$). Each point falls in a flat BEV pillar
$ix\cdot ny + iy$ (out-of-bounds $\to -1$). The splat sums all points per pillar via a stable
sort by pillar index then a cumsum and run-boundary differences (no scatter_add). BEVDepth adds
a cross-entropy on the GT depth bin (nearest bin center) at labeled cells.

Shapes: feature map $(B, C_{bb}, H_f, W_f)$; depth logits $(B, D, H_f, W_f)$; lift volume
$(B, D, C, H_f, W_f)$; frustum $(D, H_f, W_f, 3)$; pooled BEV $(n_x n_y, C)\to(C, n_x, n_y)$;
seg logit $(B, 1, n_x, n_y)$. Toy: $D=8$ bins (arange(1, 9, 1), EXCLUSIVE of d_max), 8x16 grid,
one 32x32 camera.

## what_you_implement
- `DepthLift.forward` / `DepthLift.lift` (the two conv heads and the outer-product lift).
- `frustum_points` (pixel-center frustum to ego-frame 3D, reusing `nanovision.geometry`).
- `pillar_index` (ego (x, y) to flat BEV index, -1 out of bounds).
- `cumsum_pool` (the sort+cumsum splat, differentiable, no scatter_add).
- `LiftSplatShoot.forward` (assemble backbone -> lift -> frustum -> splat -> BEV head).
- `bevdepth_depth_loss` (BEVDepth depth-supervision cross-entropy).

`LSSConfig`, the conv backbone / BEV encoder / seg head modules, the precomputed pixel-center
grid, and `bev_toy_scene` are provided.

## tasks
- **Task 1 - depth lift** (file: `lift_splat.py`, symbols: `DepthLift.forward`,
  `DepthLift.lift`): two 1x1 conv heads, softmax over depth bins, outer product with context.
  Teaches: the categorical-depth lift, the differentiable core of LSS.
- **Task 2 - frustum** (file: `lift_splat.py`, symbol: `frustum_points`): back-project the
  feature-cell frustum at each depth and transform to ego. Teaches: reuse of the camera
  unprojection and the extrinsic inverse on the OpenCV/ego axes.
- **Task 3 - splat** (file: `lift_splat.py`, symbols: `pillar_index`, `cumsum_pool`): map
  points to BEV pillars and pool with sort+cumsum. Teaches: the cumsum pooling trick and why it
  avoids a per-point scatter.
- **Task 4 - assemble** (file: `lift_splat.py`, symbol: `LiftSplatShoot.forward`): wire the
  full pipeline. Teaches: how the pieces compose into a differentiable view transform.
- **Task 5 - depth supervision** (file: `lift_splat.py`, symbol: `bevdepth_depth_loss`):
  cross-entropy on the GT depth bin at masked cells. Teaches: BEVDepth's explicit depth signal.

## tests
1. `tests/test_depth_lift.py` - shapes, one-hot depth picks the context, float64 gradcheck on
   the outer product. (shape + reference + gradcheck)
2. `tests/test_frustum.py` - center pixel at depth d maps to ego (d, 0, 0); a right pixel maps
   to ego -y; project the ego points back and recover the pixel centers (round-trip < 1e-3).
3. `tests/test_splat.py` - same-pillar sum, out-of-bounds dropped, match a scatter_add oracle,
   float64 gradcheck, `pillar_index` center cell and out-of-bounds. (reference + gradcheck)
4. `tests/test_bev_seg.py` - overfit `LiftSplatShoot` on one toy scene, <= 1500 steps; BCE <
   0.05 and BEV IoU(logit > 0, gt) > 0.6. (overfit; the logit, not the prob, is thresholded)
5. `tests/test_depth_supervision.py` - no-label loss is 0, single-cell gradient points at the
   bin, overfit the depth head; CE < 0.05 and argmax accuracy > 0.95. (overfit)
6. `tests/test_forbidden_imports.py` - tokenize scan over `lift_splat.py` top + solution + the
   `nanovision.lift_splat` shim for scatter_add / index_add / cv2 / kornia / bev_pool.

## provided_boilerplate
`LSSConfig` (depth bins, BEV grid, pixel-center grid), the `LiftSplatShoot` conv backbone / BEV
encoder / seg head, `nanovision.data.toy.bev_toy_scene`, and the whole `nanovision.geometry`
toolkit (`unproject`, `invert_transform`, `apply_transform`, `BEVGrid`, `CameraRig`) from the
camera-geometry assignment.

## compute_notes
CPU only for the graded tests, seconds each. The overfit tests run <= 1500 steps (BEV seg) and
<= 800 steps (depth) at Adam lr 1e-2 and reach ~0 loss. float32 throughout with float64
gradchecks on the outer-product lift and the cumsum pool. `viz.py` uses the GPU when present.

## stretch_goals
1. Multi-camera pooling: extend `LiftSplatShoot.forward` past B=1 / one camera by summing each
   camera's pooled BEV before the encoder.
2. Replace the discrete depth bins with a continuous Gaussian depth head (the GaussianLSS idea).
3. The frustum-tensor memory cost: measure $[N_{cam}, D, H_f, W_f, C]$ as you scale D and grid.

## further_reading
- Philion & Fidler, "Lift, Splat, Shoot" (ECCV 2020), arXiv:2008.05711.
- Li et al., "BEVDepth" (AAAI 2023), arXiv:2206.10092.
- Huang et al., "BEVDet" (2021), arXiv:2112.11790; "BEVPoolv2" (2022), arXiv:2211.17111.
- Li et al., "BEVFormer" (ECCV 2022), arXiv:2203.17270.
- Lu et al., "Toward Real-world BEV Perception" / GaussianLSS (CVPR 2025), arXiv:2504.01957.

## solution_notes
`E` is `T_cam_ego`, so `frustum_points` maps camera->ego with `invert_transform(E)`. The flat
pillar index is `ix * ny + iy` and the pooled buffer reshapes `(nx*ny, C) -> (C, nx, ny)`
row-major, matching `bev_gt` of shape `(nx, ny)`. `cumsum_pool` sorts stably by index, cumsums,
takes the cumsum at each equal-index run's last row, and differences successive run-ends; it
uses `index_copy` (not `index_add`) to place run sums. The toy is one camera and one fixed
scene, so depth is identifiable only trivially - the overfit test shows the mechanism composes
and is differentiable, not that implicit depth is resolved (see the README's limitation note).
```
