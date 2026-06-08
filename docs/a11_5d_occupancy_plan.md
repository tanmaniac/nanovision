# A11.5d - 3D occupancy prediction: build plan

Assignment dir: `assignments/a11_5d_occupancy`. Module name (owned, but assignment-LOCAL - nothing
imports it, so NO `nanovision` shim): `occupancy`. Exemplars: `assignments/a09_nerf` (the alpha-
compositing kernel this REUSES via `nanovision.volume`, and the closed-form-GT test pattern) and
`assignments/a11_5c_bevformer` (same module, BEV-feature input convention, conftest, forbidden-
imports test, toy-scene style).

Source: `docs/research/a115d_occupancy.md`. Read it. The spine the note settles on (sections 1,
2, 3): occupancy prediction is NeRF volume rendering run backward - build a voxel occupancy field
and supervise it by rendering back to 2D with the SAME alpha-compositing kernel from the NeRF
assignment (RenderOcc/OccNeRF), because nuScenes-mini has NO Occ3D 3D-occupancy labels (a real
constraint, note section 3 + 6). The duality: per-voxel opacity `alpha_i = 1 - exp(-sigma_i
delta_i)` IS the occupancy probability; NeRF integrates a known field to a pixel, occupancy
inverts pixels to estimate the field.

Deps: A9 (`nanovision.volume.volume_render` - the front-to-back alpha compositing, REUSED
verbatim for the rendering-supervision weights; also `sample_along_rays`, `deltas_from_z`),
A11.5b/c (the BEV feature tensor `[B, C, H, W]` that gets lifted to voxels - but tests feed RANDOM
BEV features, so there is no code import of LSS/BEVFormer, only the documented connector). The
toy keeps everything synthetic and CPU-cheap; no nuScenes download in graded tests.

## Conventions and the memory constraint

- TINY grid (note section 2 "memory warning"): `Z=8, Y=32, X=32`, `n_classes=4` (free + 3
  occupied). A full Occ3D 200x200x16x18 grid is ~46 MB/sample; the toy is ~25 K voxels. State the
  real cost in the README and frame the tiny grid as the mechanism isolator (TOY DOES NOT
  OVERRIDE SCALE: sparse methods discard >90% empty voxels at real resolution; the dense grid is
  the conceptual baseline, not 2026 production).
- Voxel tensor layout `[B, C, Z, Y, X]` (channels-first, depth/height axis Z first among spatial).
  Class logits `[B, n_classes, Z, Y, X]`. Be consistent everywhere.
- Grid bounds: ego x in [-X_m, X_m], y in [-Y_m, Y_m], z in [z_min, z_max]; pick small metric
  bounds (e.g. x,y in [-4, 4] m, z in [-1, 2] m) so the synthetic cameras and rays bracket the
  occupied boxes. Provide a `voxel_centers(grid)` helper (PROVIDED) returning ego coords per voxel.
- Occupancy probability <-> density: to REUSE `volume_render(sigmas, colors, deltas)` verbatim,
  convert a sampled occupancy probability `o in [0,1)` to a density `sigma = -log(1 - o) / delta`
  (so `1 - exp(-sigma*delta) = o` exactly). Clamp `o` to `[0, 1 - 1e-6]` before the log. This is
  the bridge that makes the A9 kernel the occupancy renderer; put it in the rendering hole.

## Files

Holed (top-level + identical-shell `solution/`):
- `occupancy.py` - the four mechanisms. Holes below.

Provided (top-level only):
- `config.py` - `OccConfig`: grid dims `(Z, Y, X)`, metric bounds, `n_classes`, channel widths,
  `n_samples` along a ray (set `n_samples >= (z_far - z_near)/0.15` so the rendered-depth
  quantization is well under the 0.3 m test threshold - see test D). A `voxel_centers()` helper
  (cell centers, `align_corners=False` convention) and a `grid_bounds` tuple.
- `viz.py` - GPU-aware (`default_device`). Figures: (1) a slice of the predicted vs GT occupancy
  grid (a few Z layers as heatmaps); (2) rendered depth vs GT depth for a camera. Synthetic
  `occupancy_toy_scene` (nuScenes path only if `NUSCENES_DATAROOT`, else synthetic). matplotlib
  tensors `.cpu()`.
- `conftest.py` - mirror A11.5c.

Tests (`tests/`):
- `test_bev_to_voxel.py` (pillar extrusion: shape + gradcheck)
- `test_occupancy_head.py` (head shape + gradcheck + overfit synthetic voxels, IoU)
- `test_loss.py` (weighted CE + inverse-frequency weights + mIoU + the free-class-collapse lesson)
- `test_render_supervision.py` (the spine: ray-march accumulation gradcheck + overfit rendered
  depth to analytic box GT)
- `test_forbidden_imports.py` (static scan; passes both modes)

New toy data: `occupancy_toy_scene(...)` added to `nanovision/data/toy.py` by the orchestrator
before the build (analytic ray-box GT - see "Toy scene").

## The holes (in `occupancy.py`)

1. `bev_to_voxel(bev_feats, n_z, conv)` (HOLE) - the pillar-extrusion connector (note Mechanism 3).
   - `bev_feats`: `[B, C, Y, X]`; a PROVIDED `Conv2d(C, C*n_z, 1)` produces `[B, C*n_z, Y, X]`;
     reshape to `[B, C, n_z, Y, X]` (the learnable height distribution, not a bare repeat).
   - Return `[B, C, Z, Y, X]`. Shape + gradcheck.

2. `OccupancyHead.forward(self, voxel_feats)` (HOLE) - the per-voxel classifier (note Mechanism 1).
   - `__init__` (PROVIDED): two `Conv3d` layers (C -> C -> n_classes), a nonlinearity between.
   - `forward` (HOLE): `[B, C, Z, Y, X] -> [B, n_classes, Z, Y, X]` logits. Shape + gradcheck.

3. Loss + metric (note Mechanism 1, the class-imbalance core):
   - `inverse_frequency_weights(target, n_classes)` (HOLE): per-class weight `~ 1 / (freq + eps)`,
     normalized (e.g. so weights sum to `n_classes` or mean 1); CAST to a float dtype derived from
     where it is used (the loss casts it to `logits.dtype` so float64 gradcheck holds). `target`
     is `[B, Z, Y, X]` long. The test asserts only the ORDERING (class 0 = free, the 90% class,
     gets the SMALLEST weight; rare classes larger) - ordering is normalization-invariant.
   - `weighted_ce_loss(logits, target, weights)` (HOLE): cross-entropy over the class axis with
     the per-class weight vector, using the WEIGHTED-MEAN reduction that matches `F.cross_entropy`
     with `weight=`: `sum(w_target * loss) / sum(w_target)`, NOT `sum / N` (expert blocker - the
     `/N` reduction differs from `F.cross_entropy` by a factor `sum(w)/N` and the exact-equality
     test would fail). Derive dtype from `logits` (the float64-gradcheck dtype lesson from A11).
   - `occupancy_iou(pred_labels, target, n_classes, ignore_free=True)` (HOLE): mean IoU over the
     OCCUPIED classes (exclude free=class 0 by default, so the metric is not dominated by the
     ~95% free voxels - the note's mIoU-vs-RayIoU point). A class absent from BOTH pred and target
     (empty union) is EXCLUDED from the mean (the standard mIoU convention); document this.
     Return mean IoU.

4. `render_occupancy_rays(occ, sem, rays_o, rays_d, z_vals, grid_bounds)` (HOLE) - the rendering-
   supervision spine (note Mechanism 2). REUSE the A9 kernel.
   - `occ`: `[Z, Y, X]` occupancy probability in `[0, 1]`; `sem`: `[n_classes, Z, Y, X]` semantic
     logits or probs; `rays_o, rays_d`: `[R, 3]`; `z_vals`: `[R, N]` sample depths along rays
     (from `nanovision.volume.sample_along_rays` / a provided ray setup).
   - Sample points `p = rays_o[:,None] + z_vals[...,None] * rays_d[:,None]` -> `[R, N, 3]`.
   - TRILINEAR-sample `occ` (fed as `[1, 1, Z, Y, X]`) and `sem` (fed as `[1, n_classes, Z, Y, X]`,
     ONE shared grid, all classes in one call) at `p` with `F.grid_sample`. EXACT axis mapping
     (expert blocker - the single highest-risk line; the volume is `[N,C,Z,Y,X]=[N,C,D,H,W]` so
     `D=Z, H=Y, W=X`, and grid_sample's grid last dim is `(gx, gy, gz)` mapping to `(X, Y, Z)`):
     ```
     gx = 2*(px - x0)/(x1 - x0) - 1     # X axis (W)
     gy = 2*(py - y0)/(y1 - y0) - 1     # Y axis (H)
     gz = 2*(pz - z0)/(z1 - z0) - 1     # Z axis (D)
     grid = stack([gx, gy, gz], dim=-1) # order MUST be (gx, gy, gz), NOT (gz, gy, gx)
     ```
     Use `align_corners=False` to match the PROVIDED `voxel_centers` cell-center convention
     (centers at `(i+0.5)/S`); state this. A wrong stack order silently transposes the field (Z=8
     vs Y=X=32 still broadcasts to a valid-but-garbage sample), so pin it. Get `o_i [R, N]` and
     `s_i [R, N, n_classes]`.
   - `deltas = deltas_from_z(z_vals)`; `sigma = -log(clamp(1 - o_i, min=1e-6)) / deltas` (this
     makes `alpha_i = 1 - exp(-sigma_i*delta_i) = o_i` exactly - expert-verified, survives the
     `1e10` last delta).
   - Call `nanovision.volume.volume_render(sigma, colors, deltas)` ONLY for its `weights [R, N]`
     return (`weights = T_i*alpha_i`, the A9 kernel - not re-implemented). Do NOT pass `s_i` as
     `colors` (the color path is RGB-shaped and compositing logits there is meaningless); pass a
     zeros/dummy color and use the second return.
   - Rendered depth WITH the leftover-transmittance term so MISS rays reach the far plane (expert
     blocker): `acc = weights.sum(-1)`; `D = (weights * z_vals).sum(-1) + (1 - acc) * z_far` `[R]`.
     Rendered semantics composite SEPARATELY from the returned weights:
     `S = (weights[...,None] * s_i).sum(1)` `[R, n_classes]`. Return `(D, S, weights)`.
   - gradcheck (float64) on `occ -> render_occupancy_rays(...).D.sum()` with fixed rays.

NOTE: `F.grid_sample` and `F.affine_grid` are ALLOWED (trilinear sampling substrate). The A9
alpha compositing MUST come from `nanovision.volume.volume_render`, not a re-implementation
(forbidden-imports / a code check enforces the reuse).

## Tests and exact pass conditions

A. `test_bev_to_voxel.py`
   - Shape: random `[2, 16, 32, 32]`, `n_z=8` -> `[2, 16, 8, 32, 32]`.
   - gradcheck (float64) on a small case (`[1, 2, 4, 4]`, `n_z=2`).

B. `test_occupancy_head.py`
   - Shape: random voxel feats `[2, 16, 8, 32, 32]` -> logits `[2, 4, 8, 32, 32]`.
   - gradcheck (float64) on a tiny grid (`[1, 2, 2, 2, 2]` -> `[1, 4, 2, 2, 2]`).
   - Overfit synthetic voxels: optimize the head (input = a fixed random feature volume) to match
     `occupancy_toy_scene`'s `sem_gt` voxel labels with `weighted_ce_loss` for <= 300 steps; pass:
     occupied-class IoU > 0.85. The toy boxes must be >= 2 voxels thick per side so occupied count
     greatly exceeds the boundary count (else a few boundary misclassifications swing the IoU -
     expert). CPU, deterministic. Pre-measure; report if it floors above.

C. `test_loss.py`
   - `inverse_frequency_weights`: on a target where class 0 is 90% and classes 1-3 share 10%,
     assert the weight for class 0 is the SMALLEST and rare classes get larger weights; assert the
     normalization convention (document it).
   - `weighted_ce_loss`: gradcheck (float64) on tiny logits/target; equals `F.cross_entropy` with
     `weight=` on a known case (exact).
   - `occupancy_iou`: a hand-built pred/target where the IoU is known by counting -> exact match;
     confirm free class is excluded with `ignore_free=True`.
   - Free-class-collapse lesson (the note's central imbalance point). Make the contrast robust, not
     seed-flaky (expert): use a LINEAR classifier (a single `Conv3d(C, n_classes, 1)`, no deep
     head, so capacity cannot trivially memorize), a SHORT budget (50-100 steps, where unweighted
     CE is still in the predict-majority basin), and a genuinely small occupied fraction (<= 5%).
     Train once with UNWEIGHTED CE and once with WEIGHTED CE from the same init. Pass:
     `weighted_occupied_IoU - unweighted_occupied_IoU > 0.3`; report both. Do NOT assert unweighted
     IoU ~ 0 as a hard floor (a longer budget lets it partially recover) - assert the GAP. This is
     the teaching point.

D. `test_render_supervision.py` (the spine)
   - Shape: rays `[R, N]` alphas -> accumulated depth `[R]`, semantics `[R, n_classes]`.
   - gradcheck (float64) on `occ -> render_occupancy_rays(...)` depth-sum, fixed rays.
   - Non-circular overfit (mirror A9's closed-form GT): `occupancy_toy_scene` gives synthetic
     cameras/rays with GT depth + GT semantic class per ray computed by ANALYTIC ray-box first-hit
     (NOT by the renderer); MISS rays have GT depth exactly `z_far`. SAMPLE BUDGET (expert blocker
     - rendered depth quantizes to sample spacing `Dz = (z_far - z_near)/n_samples`): the config
     must set `n_samples >= (z_far - z_near)/0.15` (a factor-2 margin under the 0.3 m threshold);
     assert this in the test. Optimize a learnable occupancy field (init random) so the rendered
     depth matches GT depth (L1/Huber) for <= 500 steps; pass: mean depth error < 0.3 m. The
     occupancy assertion is "occupancy at the FIRST-HIT voxels (the entry shell seen by >=1 camera)
     exceeds 0.5" - do NOT assert the box interiors fill in (depth-only supervision leaves occluded
     interior voxels unconstrained). Also supervise rendered semantics with CE -> per-ray class
     accuracy > 0.8 on hit rays. Pre-measure; report the floor.
   - Reuse check: a test (or the forbidden-imports scan) confirms `render_occupancy_rays` calls
     `nanovision.volume.volume_render` (the A9 kernel), not a hand-rolled cumprod.

E. `test_forbidden_imports.py` (mirror A11.5c's tokenize scan over `occupancy.py` top + solution).
   Forbidden: `cv2.projectPoints`, `cv2.solvePnP`, `import kornia`, `from kornia`,
   `MinkowskiEngine`, `spconv` (no sparse-conv libs - the dense grid is the exercise), and a
   hand-rolled `cumprod`/`cumsum` of `(1 - alpha)` inside `occupancy.py` (the compositing MUST be
   the reused `nanovision.volume.volume_render`). `F.grid_sample` / `F.affine_grid` ALLOWED. State
   this in the test.

## Toy scene (orchestrator adds to `nanovision/data/toy.py` before the build)

`occupancy_toy_scene(grid=(8,32,32), bounds=((-4,4),(-4,4),(-1,2)), n_classes=4, n_boxes=2,
n_cams=3, img=24, n_rays=None, seed=0, device="cpu")` returning a dict:
- `sem_gt`: `(Z, Y, X)` long voxel class in `[0, n_classes)`, 0 = free. A few axis-aligned solid
  boxes, each a distinct occupied class, placed inside the grid; each box is >= 2 voxels thick per
  side (so occupied count >> boundary count) and occupied voxels are <= ~5% of the grid (for the
  imbalance lesson).
- `occ_gt`: `(Z, Y, X)` float in {0, 1} (sem_gt > 0).
- Cameras: `n_cams` pinhole cameras ringing the scene looking at the center (reuse the OpenCV
  look-at from `nerf_synthetic_scene`), with `K`, `E` per camera.
- `rays_o, rays_d`: `(R, 3)` ego-frame ray origins + unit directions for a subset of pixels across
  the cameras (R = n_cams * sampled pixels).
- `gt_depth`: `(R,)` analytic first-hit depth = nearest ray-AABB entry `t` over the boxes,
  computed by the slab method (`t_min = max over axes of slab entry`, `t_max = min of slab exit`,
  hit iff `t_min < t_max` and `t_max > 0`) - INDEPENDENT of the renderer (the non-circularity
  guarantee, mirroring `nerf_synthetic_scene`'s ray-sphere chord). MISS rays get GT depth exactly
  `z_far` (matching the renderer's leftover-transmittance term so the depth loss is consistent on
  misses).
- `gt_sem`: `(R,)` long = class of the first-hit box (0/free for misses).
- `z_near, z_far`: ray sampling bracket; `grid_bounds`: the metric bounds tuple.
- Deterministic per seed. The orchestrator verifies standalone (boxes inside grid; analytic depth
  matches a brute-force voxel ray-march within a voxel; some rays hit, some miss) before the build.

## README (lecture notes per the skill)

Cover, with arXiv links VERIFIED by fetching `https://arxiv.org/abs/<id>`:
- The NeRF<->occupancy duality as the spine: occupancy is volume rendering inverted; `alpha_i =
  1 - exp(-sigma_i delta_i)` is the occupancy probability; the A9 alpha-compositing kernel is the
  occupancy renderer. RenderOcc (arXiv 2309.09502, ICRA 2024) and OccNeRF (arXiv 2312.09243)
  supervise a 3D field with only 2D depth/semantics. Name the reuse of the NeRF renderer.
- The voxel grid, semantic occupancy (C+1 logits per voxel), and the dense volumetric loss with
  class imbalance: free voxels are ~95% of the grid, rare classes ~10000x less frequent;
  inverse-frequency weights (built here), focal loss / Lovász (named, not built) as the standard
  mitigations. The free-class collapse without weighting is the measured lesson.
- The label-availability constraint: nuScenes-mini has NO Occ3D labels (Occ3D, arXiv 2304.14365,
  targets the full split), which is WHY rendering supervision (2D labels only) is the path; the
  toy uses synthetic boxes with analytic GT.
- BEV-to-voxel lifting (pillar extrusion from the BEV features of lift-splat-shoot / BEVFormer),
  and TPVFormer (arXiv 2302.07817) as the "BEV plus two perpendicular planes" extension that
  resolves flat-BEV height ambiguity (described, not built). MonoScene (arXiv 2112.00726) as the
  single-image SSC origin; SurroundOcc (arXiv 2303.09551) for where occupancy labels come from
  (derived, imperfect).
- 2024-2026 context (TOY DOES NOT OVERRIDE SCALE): sparse occupancy (the fully-sparse SparseOcc,
  arXiv 2312.17118, ECCV 2024, which introduced RayIoU) discards >90% empty voxels and is 2026
  production; RayIoU replaces mIoU because
  mIoU is dominated by free voxels; Gaussian occupancy (GaussianOcc, the 2025 frontier) swaps the
  NeRF density renderer for 3D Gaussian splatting; 4D occupancy world models (Drive-OccWorld) as
  the destination. The dense voxel grid + NeRF-density renderer here is the foundational
  mechanism, not the SOTA.
- Refer to prior concepts by NAME (the NeRF volume renderer / alpha compositing, the BEV features
  from lift-splat-shoot and BEVFormer, the camera projection), never by assignment number. Define
  jargon at first use: voxel occupancy grid, semantic occupancy, transmittance, rendered depth,
  pillar extrusion, inverse-frequency weighting, mIoU vs RayIoU.

Then the mandatory context-less style review (spawn a general-purpose subagent given ONLY the
README path + `~/.claude/CLAUDE.md`) and apply its edits.

## Verify (orchestrator, on disk, both modes)

- `NANOVISION_IMPL=solution .../python -m pytest assignments/a11_5d_occupancy/tests -q` fully green.
- Default mode fails ONLY at the holes; `test_forbidden_imports` passes both modes.
- Re-run A9 (the reused renderer) and A11.5c in their own sessions to confirm the new `toy.py`
  symbol did not regress the shared library.
- Record overfit IoU (head), the weighted-vs-unweighted occupied IoU gap, and the rendering-
  supervision depth error + per-ray semantic accuracy.
