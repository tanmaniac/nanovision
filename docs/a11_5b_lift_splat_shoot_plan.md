# A11.5b - Lift-Splat-Shoot: build plan

Assignment dir: `assignments/a11_5b_lift_splat_shoot`. Module name (owned mechanism):
`lift_splat`. Exemplars to mirror: `assignments/a11_5a_camera_geometry_bev` (same module,
geometry + nuScenes substrate, `conftest.py`, forbidden-imports test) and
`assignments/a09_nerf` (rendering-style mechanism + gradcheck tests + GPU-aware viz).

Source: `docs/research/a115b_lift_splat_shoot.md` (the expert research note). Read it; the four
mechanisms (A depth+lift, B frustum, C cumsum splat, D BEV seg head) and the BEVDepth depth
supervision come straight from it.

Deps: A11.5a (`nanovision.geometry`: `unproject`, `invert_transform`, `apply_transform`,
`BEVGrid`, `CameraRig`), A2 (image backbone - here a tiny provided conv stem, not the full ViT,
to keep the toy CPU-cheap). The note's headline: LSS pushes image features OUT into 3D via a
per-pixel categorical depth distribution and an outer-product lift, then splats them into a BEV
grid with a sort+cumsum pool. Everything is differentiable end to end; depth is a latent variable
trained from the task loss alone ("implicit" depth). BEVDepth adds explicit lidar depth
supervision, which is the standard production extension, not optional.

## What lands in the shared library

A11.5d (occupancy) reuses the depth-lift + frustum + pooling without modification (the note,
section 4 "Feeds A11.5d"). So expose a `nanovision.lift_splat` shim (orchestrator owns it) over
the assignment's owned `lift_splat.py`:

- `DepthLift` (the depth+context heads + outer-product lift)
- `frustum_points` (frustum pixel grid -> ego-frame 3D points)
- `cumsum_pool` (the generic sort+cumsum pooling by integer bin index - occupancy passes a 3D
  voxel index, LSS passes a 2D pillar index)
- `pillar_index` (ego (x,y) -> flat BEV pillar index, out-of-bounds -> -1)
- `LiftSplatShoot` (the assembled model: backbone -> lift -> frustum -> pool -> BEV encoder ->
  seg head)
- `bevdepth_depth_loss` (the BEVDepth CE on labeled depth bins)

Import rule reminder: `lift_splat.py` is the OWNED file, imported by tests and siblings ONLY via
`nanovision.lift_splat`, never bare. The assignment's own `viz.py`/`config.py` import it via the
shim too. The shim does `load("a11_5b_lift_splat_shoot", "lift_splat")`.

## Conventions fixed by the substrate (do not reinvent)

- Ego frame: x forward, y left, z up (right-handed). Camera frames OpenCV: x right, y down, z
  forward. Extrinsic `E` stored by `CameraRig` is `T_cam_ego` (ego/world -> camera).
- Depth bins: the note's nuScenes default is D=41 from 4.0 to 45.0 m in 1.0 m steps. The graded
  toy uses a SMALL set, D=8 with 1.0 m steps. `bins()` MUST be `torch.arange(d_min, d_max,
  d_step)` (exclusive of `d_max`): with `d_min=1, d_max=9, d_step=1` that is exactly 8 centers
  `[1, 2, ..., 8]`, deepest reachable point 8 m forward. Pin this convention in the config
  docstring - `arange(1, 10, 1)` would give D=9 and break every shape.
- DEPTH RANGE MUST COVER THE BEV X-EXTENT (expert blocker). For a forward camera (OpenCV z = ego
  +x) the camera-frame depth equals the ego forward distance, so a frustum reaches only as far
  forward as `d_max - d_step` = 8 m. The BEV grid forward extent MUST match: x in [0, 8] m, NOT
  [0, 16]. Use x in [0, 8], y in [-8, 8], 1.0 m resolution -> an 8x16 grid. Any vehicle past 8 m
  forward is unreachable by every frustum point and would make the overfit test unpassable (or
  falsely passable only when the seed places all vehicles within 8 m). The toy generator clamps
  vehicle x to the reachable, in-FOV range (below).
- A feature cell at feature-grid location (i, j) corresponds to image pixel
  `((j + 0.5) * stride, (i + 0.5) * stride)` for a backbone downsample `stride`. `frustum_points`
  takes the pixel-center grid as a provided input so the unproject math is the hole, not the
  bookkeeping.

## Files

Holed (top-level + identical-shell `solution/`):
- `lift_splat.py` - the four mechanisms + the assembled model. Holes listed below.

Provided (top-level only, no solution copy):
- `config.py` - `LSSConfig` dataclass: `d_min=1`, `d_max=9`, `d_step=1` (-> D=8), image size,
  feature stride, channel widths, focal length, and the BEV grid (a `BEVGrid` with the SMALL toy
  extent x in [0, 8] m, y in [-8, 8] m, 1.0 m resolution -> 8x16; see the depth-range constraint
  above). Provide `bins()` returning `torch.arange(d_min, d_max, d_step)` (the depth-bin centers,
  D=8). The focal is chosen so the camera sees the BEV region of interest; the toy generator
  rejection-samples vehicle cells that are both in-frame and at depth <= the deepest bin, so the
  build pre-measures a focal (start near `f = image_width`, reduce if too few cells are visible).
- `viz.py` - GPU-aware (`from nanovision.determinism import default_device`). Two figures:
  (1) the depth distribution for a few pixels as a bar chart over bins (shows the softmax
  concentrating at plausible depths after overfit); (2) predicted vs GT BEV occupancy heatmap.
  Use nuScenes-mini if `NUSCENES_DATAROOT` is set, else fall back to the synthetic toy scene.
  Keep matplotlib tensors on CPU (`.cpu()`).
- `conftest.py` - mirror A11.5a: insert assignment dir then impl dir on `sys.path`.

Tests (`tests/`):
- `test_depth_lift.py` (mechanism A)
- `test_frustum.py` (mechanism B)
- `test_splat.py` (mechanism C)
- `test_bev_seg.py` (mechanism D, overfit)
- `test_depth_supervision.py` (BEVDepth)
- `test_forbidden_imports.py` (static scan; passes both modes)

New toy data: add `bev_toy_scene(...)` to `nanovision/data/toy.py` (orchestrator adds it before
the build; see "Toy scene" below) so the build only consumes it.

## The holes (in `lift_splat.py`)

1. `DepthLift.forward(self, feat)` and `DepthLift.lift(self, feat)`.
   - `__init__` (PROVIDED): two 1x1 convs, `depth_head: Conv2d(c_in, D, 1)` and
     `ctx_head: Conv2d(c_in, C_ctx, 1)`.
   - `forward` (HOLE): return `depth_logits (B, D, Hf, Wf)`, `context (B, C_ctx, Hf, Wf)`.
   - `lift` (HOLE): `alpha = softmax(depth_logits, dim=1)`; outer product
     `volume = alpha[:, :, None] * context[:, None]` -> `(B, D, C_ctx, Hf, Wf)`. One einsum or
     broadcast multiply. This is the central differentiable op.

2. `frustum_points(pixel_xy, depths, K, E)` (HOLE).
   - `pixel_xy`: `(Hf, Wf, 2)` image-pixel centers per feature cell (PROVIDED by caller).
   - `depths`: `(D,)` bin-center depths.
   - `K`: `(3, 3)`, `E`: `(4, 4)` `T_cam_ego`.
   - For each `(d, i, j)`: camera-frame point `= unproject(pixel, depth=d, K)`; then ego point
     `= apply_transform(invert_transform(E), p_cam)`. Reuse `nanovision.geometry.unproject`
     (it already does `z * K^{-1} [u, v, 1]`), `invert_transform`, `apply_transform`.
   - Return `(D, Hf, Wf, 3)` ego-frame points.

3. `pillar_index(pts_ego_xy, bev_grid)` (HOLE).
   - `pts_ego_xy`: `(N, 2)` ego (x, y). Map to integer cell `(ix, iy)` via the grid bounds and
     resolution; flat index `ix * ny + iy`. Points outside `[x_min, x_max) x [y_min, y_max)` ->
     `-1` (dropped by the pool). Return `(N,)` long.

4. `cumsum_pool(feats, idx, n_bins)` (HOLE) - the splat.
   - `feats`: `(N, C)`, `idx`: `(N,)` long in `[-1, n_bins)`. Drop `idx < 0`. Sort the kept
     points by `idx` (stable). Cumsum the sorted features along N. The pooled sum for a bin is
     `cumsum[last index of segment] - cumsum[last index of previous segment]`; equivalently keep
     the last row of each equal-idx run and take successive differences. Scatter each segment sum
     into `out[bin]`. Return `(n_bins, C)`. Differentiable wrt `feats` (the sort is a fixed
     permutation given `idx`; gradient flows through the gather/scatter). No `scatter_add`, no
     custom CUDA - that is the point of the trick. (You MAY use `scatter_add` ONLY in a hidden
     reference oracle inside the test, not in `lift_splat.py`; see test C.)

5. `LiftSplatShoot.forward(self, images, K, E)` (HOLE) - assemble.
   - `__init__` (PROVIDED): a tiny conv `backbone` (2-3 conv layers, stride to the feature grid),
     a `DepthLift`, a `bev_encoder` (2 conv layers), a `seg_head: Conv2d(C, 1, 1)`. Stores the
     `LSSConfig`, depth bins, and precomputed `pixel_xy`.
   - `forward` (HOLE): backbone each camera image -> `feat`; `DepthLift.lift` -> volume
     `(B, D, C, Hf, Wf)`; build `frustum_points` per camera -> ego points `(D, Hf, Wf, 3)`;
     flatten volume to `(N, C)` and ego xy to `(N, 2)` over `(D, Hf, Wf)` and over cameras;
     `pillar_index` -> idx; `cumsum_pool` -> `(nx*ny, C)`; reshape to `(C, nx, ny)`;
     `bev_encoder` -> `seg_head` -> `(B, 1, nx, ny)` logit. Sum contributions across cameras.
   - Keep it batched B=1 in the toy if multi-batch pooling indexing is awkward; the note's toy is
     one scene. Document the B handling you choose.

6. `bevdepth_depth_loss(depth_logits, depth_bin_labels, mask)` (HOLE) - BEVDepth.
   - `depth_logits`: `(B, D, Hf, Wf)`; `depth_bin_labels`: `(B, Hf, Wf)` long (the GT bin from
     projected lidar/scene depth); `mask`: `(B, Hf, Wf)` bool of labeled cells.
   - Cross-entropy over the D bins at masked cells only (mean over labeled cells). This is the
     one-hot depth classification loss from BEVDepth (arXiv 2206.10092). Derive dtype/weights
     from `depth_logits` (the A11 float64-gradcheck dtype lesson).

## Tests and exact pass conditions

A. `test_depth_lift.py`
   - Shape: `DepthLift(c_in=4, D=4, C_ctx=8)` on `(1, 4, 2, 2)` -> logits `(1,4,2,2)`,
     context `(1,8,2,2)`, `lift` volume `(1,4,8,2,2)`.
   - One-hot: set depth logits so softmax is ~one-hot at bin d (large positive at d); assert the
     volume is ~zero at all other bins (`< 1e-4`) and equals context at bin d.
   - `gradcheck` (float64, `c_in=2, D=3, C_ctx=2, Hf=Wf=2`) on `lift` wrt a float64 feature input
     through float64-cast conv weights, or wrap `alpha,context -> volume` as a pure function and
     gradcheck that (cleaner: gradcheck the outer-product map `(alpha, context) -> volume`).

B. `test_frustum.py` (expert-verified: round-trip error is exactly 0, signs confirmed)
   - Forward camera AT THE ORIGIN for this test (so the center pixel lands at z=0, a clean
     assertion): `E = make_transform(R, t=0)` with `R` rows `[0,-1,0],[0,0,-1],[1,0,0]` (i.e.
     `x_cam=-y_ego, y_cam=-z_ego, z_cam=+x_ego`; det R = +1). The center ray points unambiguously
     along ego +x (forward). Center pixel at depth d -> ego point `(d, 0, ~0)`, `|y| < 1e-4`;
     assert this for each bin. (Scope this "(d,0,~0)" claim to the origin-camera `E` built here;
     the toy scene uses a camera at z=1.5 where the center lands at `(d,0,1.5)`, harmless because
     pillaring uses only x,y.)
   - Off-center pixel: a pixel right of center (`u > cx`) maps to ego -y. Assert sign.
   - Consistency vs `CameraRig.world_to_pixel`: take the produced ego points, project them back
     with the same K, E, and recover the original pixel centers (round-trip < 1e-3).

C. `test_splat.py`
   - Same-pillar sum: two points with idx=5, feats `a`, `b` -> `out[5] == a + b` (exact).
   - Drop out-of-bounds: a point with idx=-1 contributes nothing.
   - Match a `scatter_add` oracle on a random `(N=50, C=3)`, `idx` in `[-1, 10)` (exact equality).
   - `gradcheck` (float64) on `feats -> cumsum_pool(feats, idx, n_bins).sum()` with a fixed
     random `idx`.
   - `pillar_index`: a point at the grid center maps to the expected cell; a point past `x_max`
     -> -1.

D. `test_bev_seg.py` (overfit, bounded)
   - One synthetic scene from `bev_toy_scene` (1 camera, 8x16 BEV, M=3 vehicles). Train
     `LiftSplatShoot` with BCE-with-logits on the BEV occupancy for <= 1500 steps, Adam lr ~1e-2.
     Pass: final BCE < 0.05 AND BEV IoU(pred > 0, gt) > 0.6, where `pred > 0` thresholds the LOGIT
     at 0 (prob 0.5) - note that in the test so a builder does not threshold the probability
     elsewhere. CPU, deterministic seed.
   - HONEST LIMITATION (state in the test docstring AND README): with one camera and one fixed
     scene, no two objects share a pixel at different depths, so depth is identifiable but only
     trivially - the network memorizes one depth distribution per pixel and never has to RESOLVE
     depth ambiguity. The frustum geometry is still exercised (each depth bin along a ray maps to
     a DISTINCT pillar, so a wrong depth lands in a wrong pillar and the BCE penalizes it - the
     expert confirmed `ix` increases monotonically with depth), so this is a real test that the
     LSS mechanism composes and is differentiable, NOT evidence that implicit depth is learned in
     any non-trivial sense. Do not let the README imply otherwise.
   - Pre-measure the floor in the build; if it floors above threshold, REPORT the number and
     loosen to the measured value rather than thrashing (per the build guide). This is a
     mechanism-composes test, not a benchmark.

E. `test_depth_supervision.py` (BEVDepth)
   - `bev_toy_scene` returns sparse GT depth bins + mask. The label at a vehicle pixel is the
     NEAREST bin center, `label = argmin_k |z_cam - bins()[k]|`, computed with the SAME `bins()`
     tensor the model uses (not `round(z - d_min)`, which silently assumes step=1). This is the
     BEVDepth one-hot-at-projected-depth-bin label; it is only valid when `z_cam` falls within
     `[d_min, d_max]`, which the toy generator guarantees by clamping vehicle depth to the
     reachable range. Overfit the depth head alone (or the full `depth_logits` path) with
     `bevdepth_depth_loss` for <= 800 steps; pass: CE < 0.05 and argmax bin == label at masked
     cells (accuracy > 0.95).
   - Sanity: with `mask` all-False the loss is 0 (or a documented no-label convention); with a
     single labeled cell the gradient points the logit at the right bin.

F. `test_forbidden_imports.py` (mirror A11.5a, static tokenize scan over `lift_splat.py` top +
   solution + the shim). Forbidden: `cv2.projectPoints`, `cv2.solvePnP`, `import kornia`,
   `from kornia`, `scatter_add` and `index_add` (the splat must be the cumsum trick, not a
   scatter), `torch.nn.functional.grid_sample` is ALLOWED (viz IPM may use it) - scope the scan
   to `lift_splat.py` so viz is not penalized. Also forbid `bev_pool` / `voxel_pooling` from any
   external lib (there is none installed; this documents intent).

## Toy scene (orchestrator adds to `nanovision/data/toy.py` before the build)

`bev_toy_scene(n_vehicles=3, bev_x=(0,8), bev_y=(-8,8), res=1.0, img=32, focal=None, seed=0,
device="cpu")` returning a dict. Vehicle cells are rejection-sampled to be BOTH in-frame
(`CameraRig.world_to_pixel` valid) AND at camera-frame depth in `[d_min, d_max]` (reachable by
the frustum); raise if `n_vehicles` cannot be placed (signals a too-narrow focal). Returns:
- `image`: `(1, 3, img, img)` one forward camera. Each vehicle is a colored gaussian blob painted
  at the pixel where its 3D centroid (ego x, y, z=0.75 m) projects through the camera; blob
  intensity/scale ties to depth so the network has a learnable depth cue. Background gray.
- `K`: `(3, 3)`, `E`: `(4, 4)` `T_cam_ego` for a camera at ego (0, 0, 1.5) looking +x (OpenCV).
- `bev_gt`: `(nx, ny)` float in {0,1}, 1 in a 1-2 cell footprint at each vehicle's BEV cell.
- `depth_bin_labels`: `(Hf, Wf)` long, the GT depth bin at the feature cell each vehicle projects
  to (else 0), and `depth_mask`: `(Hf, Wf)` bool marking those cells. Hf, Wf follow the config
  stride.
- Deterministic per seed; vehicle (x, y) drawn on grid cells inside the BEV bounds and inside the
  camera FOV. The mapping image -> bev_gt is consistent (a vehicle's blob position is exactly its
  projected centroid), so an LSS pipeline with correct geometry can overfit it; a pipeline that
  ignores the frustum geometry cannot route the blob to the right pillar.

The orchestrator writes `bev_toy_scene` and verifies it standalone (projects, blobs land in
frame, bev_gt cells inside bounds) before spawning the build. Mirror the style of
`nerf_synthetic_scene` / `detection_batch` already in `toy.py`.

## README (lecture notes per the skill)

Cover, with verified arXiv links (fetch `https://arxiv.org/abs/<id>` and confirm the title):
- Lift-Splat-Shoot, Philion & Fidler, ECCV 2020, arXiv 2008.05711 - the primary source; the
  outer-product lift (sec 3.1) and the cumsum splat (sec 3.2).
- The lineage: BEVDet (arXiv 2112.11790) applies LSS to 3D detection; BEVDepth (arXiv 2206.10092,
  AAAI 2023) adds explicit lidar depth supervision - the main practical driver of detection
  accuracy; BEVPoolv2 (arXiv 2211.17111) eliminates the frustum tensor with a CUDA kernel (15x),
  a deployment pointer not a concept.
- The push-out vs pull-in framing: LSS pushes features out into 3D (depth lift); BEVFormer
  (arXiv 2203.17270, the next assignment) pulls features in via attention at projected reference
  points. Name this as the organizing contrast for the AV BEV module. Forward-point to occupancy
  (drop the BEV collapse, keep the 3D voxel grid) as the next reuse of this exact lift.
- GaussianLSS (arXiv 2504.01957, CVPR 2025) as the 2025 frontier: continuous Gaussian depth
  instead of discrete bins.
- Define jargon at first use: "implicit" depth (latent, trained from task loss only, no depth
  labels in vanilla LSS), frustum, pillar, the outer product, the cumsum trick. Refer to prior
  concepts by NAME (the camera extrinsics and BEV grid from the camera-geometry assignment, the
  conv backbone), never by assignment number. The center ray of a forward camera points along ego
  +x (forward) - state +x unambiguously, not the research note's hedged "+x or +y."
- State the frustum-tensor cost honestly: `[N_cam, D, Hf, Wf, C]` is the memory bottleneck (the
  note gives ~440 MB for a small real config); the toy is 1 camera x D=8 x small grid, so it is
  tiny - say so, and frame the real cost against it (TOY DOES NOT OVERRIDE SCALE: the cumsum
  trick's 2x and BEVPoolv2's 15x are real wins at real resolution; the toy will not show them).
- One sentence each on temporal fusion (BEVDet4D warps a previous BEV frame) and on why implicit
  depth suffices for segmentation but underperforms for precise 3D detection (the BEVDepth point).

Then the mandatory context-less style review (spawn a general-purpose subagent given ONLY the
README path + `~/.claude/CLAUDE.md`), and apply its edits.

## Verify (orchestrator, on disk, both modes)

- `NANOVISION_IMPL=solution .../python -m pytest assignments/a11_5b_lift_splat_shoot/tests -q`
  fully green.
- Default mode (`NANOVISION_IMPL` unset) fails ONLY at the holes (`NotImplementedError`), not
  collection/import errors; `test_forbidden_imports` passes in both modes.
- Run A9/A10/A11.5a again in their own sessions to confirm the new `toy.py` symbol and the
  `nanovision.lift_splat` shim did not regress the shared library (per-assignment runner, not one
  combined pytest session - the bare-import collision across dirs is pre-existing).
- Record the measured overfit BCE/IoU and depth-supervision CE/accuracy.
