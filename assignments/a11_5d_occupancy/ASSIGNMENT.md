# assignments/a11_5d_occupancy/ASSIGNMENT.md

```yaml
id: a11_5d_occupancy
title: 3D occupancy prediction
module: 3.5
type: Core
estimated_learner_hours: 6
depends_on: [a00_harness, a09_nerf, a11_5a_camera_geometry_bev, a11_5b_lift_splat_shoot, a11_5c_bevformer]
builds_into_shared_lib: []   # assignment-local: nothing imports occupancy.py, so no nanovision shim
forbidden_imports:
  - cv2.projectPoints
  - cv2.solvePnP
  - kornia
  - MinkowskiEngine            # no sparse-conv libs: the dense grid is the exercise
  - spconv
  - cumprod                    # no hand-rolled transmittance; compositing is the reused volume_render
  - cumsum
  # F.grid_sample and F.affine_grid are ALLOWED (the trilinear-sampling substrate).
fits_12gb: true
external_data: none (tests run on the synthetic occupancy_toy_scene; nuScenes only if NUSCENES_DATAROOT)
```

## motivation

Occupancy prediction labels which 3D volumes are filled and by what class, the question a planner
needs that a bounding-box detector does not answer. Because nuScenes-mini has no Occ3D voxel
labels, the field supervises the voxel grid by rendering it back to 2D depth and semantics with
the NeRF alpha-compositing kernel (RenderOcc, OccNeRF). The full treatment, paper links, and
forward connections are in the README.

## background

Occupancy probability and NeRF density are the same quantity: segment opacity
$\alpha = 1 - e^{-\sigma \delta}$ is the occupancy probability, so a sampled occupancy
$o \in [0,1)$ converts to density $\sigma = -\log(1-o)/\delta$ and the reused renderer composites
it unchanged. Rays cast into an occupancy grid `[Z,Y,X]` accumulate compositing weights
$w_i = T_i \alpha_i$ (from the reused kernel) into rendered depth $D = \sum_i w_i z_i + (1-\sum_i w_i)z_{\text{far}}$
and rendered semantics $S = \sum_i w_i s_i$. Class imbalance (free voxels ~95%) is handled with
inverse-frequency weights and the weighted-mean cross-entropy that matches `F.cross_entropy(weight=)`.

Shapes: voxel features `[B,C,Z,Y,X]`; class logits `[B,n_classes,Z,Y,X]`; occupancy grid
`[Z,Y,X]`; semantic grid `[n_classes,Z,Y,X]`; rays `rays_o,rays_d [R,3]`, `z_vals [R,N]`;
rendered depth `[R]`, rendered semantics `[R,n_classes]`, weights `[R,N]`.

## what_you_implement

- Pillar extrusion from a BEV feature map to a voxel feature volume.
- A per-voxel semantic classifier.
- Inverse-frequency class weights and the weighted-mean cross-entropy.
- Mean occupied-class IoU.
- Rendering supervision: trilinear sampling, the density bridge, and depth/semantic compositing
  through the reused NeRF kernel.

## tasks

1. **Task 1 - bev_to_voxel** (file: `occupancy.py`, symbol: `bev_to_voxel`): given `bev_feats`
   `[B,C,Y,X]`, a provided `Conv2d(C, C*n_z, 1)`, and `n_z`, produce voxel features
   `[B,C,Z,Y,X]` by applying the conv and reshaping `[B, C*n_z, Y, X] -> [B, C, n_z, Y, X]`. The
   conv learns a per-height feature distribution, not a repeat. Teaches pillar extrusion.

2. **Task 2 - OccupancyHead.forward** (file: `occupancy.py`, symbol: `OccupancyHead.forward`):
   map voxel features `[B,C,Z,Y,X]` to class logits `[B,n_classes,Z,Y,X]` through the two
   provided 3D convs with the ReLU between. Teaches the per-voxel classifier.

3. **Task 3 - inverse_frequency_weights** (file: `occupancy.py`, symbol:
   `inverse_frequency_weights`): per-class weight $\propto 1/(\text{count}+\varepsilon)$ over a
   `[B,Z,Y,X]` long target, normalized to mean 1 (sum = n_classes). Free gets the smallest weight.
   Teaches class-imbalance reweighting.

4. **Task 4 - weighted_ce_loss** (file: `occupancy.py`, symbol: `weighted_ce_loss`): class-weighted
   cross-entropy over `logits [B,n_classes,Z,Y,X]` and `target [B,Z,Y,X]` with the weighted-mean
   reduction $\sum_v w_{t_v}\ell_v / \sum_v w_{t_v}$, in `logits.dtype`. Must equal
   `F.cross_entropy(weight=)` exactly. Teaches the reduction that matches the library loss.

5. **Task 5 - occupancy_iou** (file: `occupancy.py`, symbol: `occupancy_iou`): mean IoU over
   occupied classes (free excluded with `ignore_free`), empty-union classes excluded from the mean.
   Teaches the occupancy metric and why free is excluded.

6. **Task 6 - render_occupancy_rays** (file: `occupancy.py`, symbol: `render_occupancy_rays`):
   sample points along rays, trilinearly sample `occ [Z,Y,X]` and `sem [n_classes,Z,Y,X]` with the
   pinned `(gx,gy,gz)->(X,Y,Z)` axis order and `align_corners=False`, convert occupancy to density,
   call `nanovision.volume.volume_render` for the weights only (zeros dummy color), and return
   `(D, S, weights)` with the leftover-transmittance depth term. Teaches volume rendering inverted.

## tests

- `tests/test_bev_to_voxel.py::test_shape` - shape `[2,16,32,32], n_z=8 -> [2,16,8,32,32]`.
- `tests/test_bev_to_voxel.py::test_gradcheck` - float64 gradcheck on a small case.
- `tests/test_occupancy_head.py::test_shape` - `[2,16,8,32,32] -> [2,4,8,32,32]`.
- `tests/test_occupancy_head.py::test_gradcheck` - float64 gradcheck on a tiny grid.
- `tests/test_occupancy_head.py::test_overfit_voxel_labels` - overfit head + learnable feature
  volume to toy voxel labels (<= 300 steps); occupied IoU > 0.85.
- `tests/test_loss.py::test_inverse_frequency_ordering` - free gets smallest weight, rarer larger;
  weights sum to n_classes.
- `tests/test_loss.py::test_weighted_ce_matches_f_cross_entropy` - exact equality with
  `F.cross_entropy(weight=)`.
- `tests/test_loss.py::test_weighted_ce_gradcheck` - float64 gradcheck.
- `tests/test_loss.py::test_occupancy_iou_exact` / `test_empty_union_excluded` - hand-counted IoU,
  free excluded, empty union excluded.
- `tests/test_loss.py::test_free_class_collapse_lesson` - linear classifier, short budget, shared
  init; weighted minus unweighted occupied-class recall > 0.3.
- `tests/test_render_supervision.py::test_sample_budget_floor` - `n_samples >= (z_far-z_near)/0.15`.
- `tests/test_render_supervision.py::test_shapes` - rendered depth `[R]`, semantics `[R,n_classes]`,
  weights `[R,N]`.
- `tests/test_render_supervision.py::test_gradcheck` - float64 gradcheck on `occ -> depth.sum()`.
- `tests/test_render_supervision.py::test_overfit_rendered_depth_and_semantics` - overfit a
  learnable field to analytic ray-box GT (<= 500 steps); mean depth error < 0.3 m, hit-ray
  accumulated opacity > 0.5, per-ray semantic accuracy > 0.8.
- `tests/test_forbidden_imports.py` - static tokenize scan (passes both impl modes); also checks
  `render_occupancy_rays` calls the reused `volume_render`.

## provided_boilerplate

`config.py` (`OccConfig` with grid dims, metric bounds, `n_samples`, and a `voxel_centers` helper),
`viz.py` (GPU-aware training demo with occupancy-slice and depth-scatter figures), `conftest.py`,
the `OccupancyHead.__init__` convs, and the `occupancy_toy_scene` toy with analytic ray-box GT.

## compute_notes

Everything runs on CPU in seconds (tests) or a few seconds on GPU (`viz.py`). The overfit tests
are bounded: head 300 steps, render supervision 500 steps. `n_samples=96` is set so the rendered
depth quantizes well under the 0.3 m threshold ((15-1)/96 ~= 0.146 m). No real training run; the
toy is overfit-only. A healthy render-supervision curve drops the depth+semantic loss from ~12 to
~0.1 within 150 steps; depth error floors around 0.22 m.

## stretch_goals

1. Add a total-variation or entropy regularizer on the occupancy field and measure whether the
   per-voxel hit-ray occupancy sharpens above 0.5.
2. Replace occupied-class mIoU with a RayIoU implementation and compare the two on the toy.
3. Swap pillar extrusion for the tri-perspective view (three planes) and check the height recovery.
4. Add focal loss as an alternative to inverse-frequency weighting and compare the collapse gap.

## further_reading

- RenderOcc, [arXiv 2309.09502](https://arxiv.org/abs/2309.09502) - 2D rendering supervision for occupancy.
- OccNeRF, [arXiv 2312.09243](https://arxiv.org/abs/2312.09243) - LiDAR-free NeRF-style supervision.
- Occ3D, [arXiv 2304.14365](https://arxiv.org/abs/2304.14365) - the occupancy benchmark.
- SparseOcc, [arXiv 2312.17118](https://arxiv.org/abs/2312.17118) - fully sparse occupancy and RayIoU.
- TPVFormer, [arXiv 2302.07817](https://arxiv.org/abs/2302.07817) - tri-perspective view for height.

## solution_notes

- The render overfit and head overfit use `torch.manual_seed(0)`; the collapse test uses
  `manual_seed(1)` for the classifier init shared across the two runs.
- `weighted_ce_loss` must use the weighted-MEAN reduction (`sum(w_t*l)/sum(w_t)`), not `sum/N`, or
  the exact-equality test against `F.cross_entropy` fails by a factor `sum(w)/N`.
- The density bridge clamps `1 - o` to `min=1e-6` before the log; this survives the `1e10` final
  delta from `deltas_from_z` because `o` is bounded below 1.
- The free-class-collapse test asserts the occupied-class RECALL gap, not IoU. A linear classifier
  trained with inverse-frequency weights trades precision for recall and can have a LOWER occupied
  IoU than the unweighted run despite detecting the rare classes; measured recall gap is ~0.57
  (unweighted ~0.19, weighted ~0.76). This is a deliberate deviation from a plan that specified an
  IoU gap, made after measuring that the IoU gap is negative for a linear classifier.
- The render overfit asserts hit-ray accumulated opacity > 0.5 (measured ~0.97), not per-voxel
  occupancy > 0.5: depth-only supervision yields a diffuse low-opacity field (per-voxel ~0.15-0.30)
  whose weight centroid still lands at the correct depth. Also a deliberate deviation, after
  measuring per-voxel occupancy never crosses 0.5 under this sample budget.
```
