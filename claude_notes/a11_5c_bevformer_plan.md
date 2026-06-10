# A11.5c - BEVFormer-style attention: build plan

Assignment dir: `assignments/a11_5c_bevformer`. Module name (owned mechanism): `bevformer`.
Exemplars: `assignments/a11_5b_lift_splat_shoot` (same module, the BEV/toy/multi-cam substrate,
the LSS contrast this assignment is written against) and `assignments/a01_transformer` (the A1
attention primitives this reuses - read how `nanovision.attention` / `nanovision.transformer`
are imported).

Source: `research/a115c_bevformer.md`. Read it. The core is the query-PULL view transform:
BEV cells reach back into image space and bilinear-sample the features they need, the opposite of
LSS's depth-PUSH. The teaching order the note settles on (section 3 "Suggested reordering"):
(1) build 3D pillar reference points, (2) project them and bilinear-sample - this alone already
gives reasonable BEV features (the simplified spatial cross-attention, no learned offsets),
(3) add learned deformable offsets as a refinement, (4) add temporal self-attention with
ego-motion BEV warping.

Deps: A11.5a (`nanovision.geometry`: `CameraRig`, `BEVGrid`, `project_points`, `apply_transform`,
`invert_transform`), A1 (`nanovision.attention` for multi-head attention; the deformable and
temporal attention are specializations of it), A11.5b (the LSS contrast - written takeaway, no
code dependency), A3.5 (temporal intuition - tube/temporal modeling, no code dependency).

## What lands in the shared library

A11.5d (occupancy) and A11.5e (map/prediction) consume dense BEV feature grids; the note
(section 4 "Feeds") says BEVFormer's BEV feature map is the canonical intermediate. So expose a
`nanovision.bevformer` shim (orchestrator owns it) over the assignment's owned `bevformer.py`:

- `bev_reference_points` (BEV grid + anchor heights -> ego-frame 3D pillar points)
- `project_reference_points` (ego 3D points + `CameraRig` -> normalized grid_sample coords + a
  hit/valid mask per camera)
- `SpatialCrossAttention` (the bilinear-sample-at-reference-points SCA; an `offsets=True` flag
  turns on the learned-deformable-offset path)
- `warp_bev` (ego-motion affine warp of the previous BEV feature grid)
- `TemporalSelfAttention` (attention over current queries + the warped history)
- `BEVFormerEncoder` (the assembled encoder: SCA -> TSA -> feed-forward, stacked) and a
  `BEVFormerSeg` wrapper (encoder + a BEV segmentation head)

Import rule: `bevformer.py` is the OWNED file, imported by tests and siblings ONLY via
`nanovision.bevformer`, never bare. `viz.py`/`config.py` import it via the shim. The shim does
`load("a11_5c_bevformer", "bevformer")`.

## Conventions fixed by the substrate (do not reinvent)

- Ego frame x forward, y left, z up; camera OpenCV x right, y down, z forward. Extrinsic stored
  by `CameraRig` is `E = T_cam_ego`. Reuse `CameraRig.world_to_pixel` / `project_points` for the
  projection - do NOT hand-roll a second projection path.
- Anchor heights: the paper uses N_ref=4 heights from -5 m to 3 m in ego z. The toy keeps N_ref=4
  with the same range (the vehicles sit near z in [0, 1.5], inside it).
- `grid_sample` normalization: pixel (u, v) -> `(2u/(W-1) - 1, 2v/(H-1) - 1)` with
  `align_corners=True`, OR the `align_corners=False` convention `(2(u+0.5)/W - 1, ...)`. PICK ONE
  and use it consistently in `project_reference_points` and any test oracle; document which. The
  note's step 4 uses `align_corners=False`; match that.
- Toy BEV grid: SMALL, x in [-8, 8], y in [-8, 8], 1.0 m resolution -> 16x16 (BEVFormer's grid is
  centered, unlike LSS's forward-only grid, because cameras ring the ego). Use a centered
  `BEVGrid`.

## Files

Holed (top-level + identical-shell `solution/`):
- `bevformer.py` - the mechanisms + the assembled encoder. Holes below.

Provided (top-level only):
- `config.py` - `BEVFormerConfig`: BEV grid (centered 16x16), `n_heights=4`, `z_min=-5`,
  `z_max=3`, channel width, n_heads, n_layers, image size, feature stride, n_cams.
- `viz.py` - GPU-aware (`default_device`). Figures: (1) for a few BEV cells, the projected
  reference points overlaid on each camera image (shows the geometry-as-attention-prior);
  (2) predicted vs GT BEV occupancy; (3) a PETR contrast panel (see README) is optional. nuScenes
  if `NUSCENES_DATAROOT` set else the synthetic toy. matplotlib tensors on `.cpu()`.
- `conftest.py` - mirror A11.5b.

Tests (`tests/`):
- `test_reference_points.py` (geometry: pillar construction + projection)
- `test_spatial_cross_attention.py` (SCA bilinear, shape + gradcheck + hit-mask)
- `test_deformable.py` (learned offsets receive gradients; offsets=0 reduces to bilinear SCA)
- `test_temporal.py` (warp correctness + the moving-object two-frame test)
- `test_bev_seg.py` (overfit the assembled encoder on the multi-cam toy)
- `test_forbidden_imports.py` (static scan; passes both modes)

New toy data: `bev_multicam_scene(...)` added to `nanovision/data/toy.py` by the orchestrator
before the build (see "Toy scene").

## The holes (in `bevformer.py`)

1. `bev_reference_points(bev_grid, n_heights, z_min, z_max)` (HOLE).
   - For each BEV cell center `(x, y)` (from `BEVGrid.cell_centers`, ego frame) build a vertical
     pillar of `n_heights` points at z uniformly in `[z_min, z_max]`. Return
     `(nx, ny, n_heights, 3)` ego-frame points.

2. `project_reference_points(ref3d, rig, image_hw)` (HOLE).
   - Project the `(nx, ny, n_heights, 3)` ego points into each of the rig's cameras with
     `rig.world_to_pixel` (reuse A11.5a - this reinforces the projection chain). Normalize pixels
     to grid_sample coords in `[-1, 1]` with the `align_corners=False` map
     `gx = 2*(u + 0.5)/W - 1`, `gy = 2*(v + 0.5)/H - 1` (expert-verified exact). Return `uv` of
     shape `(n_cam, nx, ny, n_heights, 2)` with the LAST dim ordered `(gx, gy) = (width, height)`
     - grid_sample reads the last dim as (x=W, y=H), so stack gx then gy, NOT (v, u). And `valid`
     bool `(n_cam, nx, ny, n_heights)` (in front of camera AND inside image bounds -
     `world_to_pixel` already returns that combined mask).
   - NORMALIZE BY THE FULL IMAGE SIZE `image_hw` (the W, H the rig's bounds mask uses), NOT by the
     feature-map size `Wf = img // stride`. grid_sample on a downsampled `feats` with the same
     `[-1, 1]` extent handles the stride itself; normalizing by `Wf` silently offsets every sample
     by the stride factor while the hit-mask (from `world_to_pixel`) still looks fine - a quiet
     garbage-features bug. State this in the docstring.

3. `SpatialCrossAttention` (HOLE in `forward`).
   - `__init__` (PROVIDED): a value projection `Linear(C, C)`, an output projection, and (when
     `offsets=True`) an offset head `Linear(C, n_heads * n_points * 2)` and an
     attention-weight head. Stores `offsets` flag, `n_points` (deformable samples per ref point,
     default 4).
   - `forward(self, query, feats, ref_uv, valid)` (HOLE):
     - `query`: `(nx, ny, C)` BEV queries; `feats`: `(n_cam, C, Hf, Wf)`; `ref_uv`, `valid` from
       hole 2.
     - SIMPLIFIED path (`offsets=False`): `F.grid_sample` each camera's `feats` at `ref_uv`
       (bilinear, `align_corners=False`) -> `(n_cam, C, nx, ny, n_heights)`. Then reduce with the
       SHARED helper below.
     - THE REDUCTION (expert blocker - put it in a single helper both paths call, do NOT inline
       a fixed `/n_heights`). Divide by the count of VALID heights per (cell, camera), not by
       `n_heights` - a pillar in-frame at 1 of 4 heights must not be 4x under-weighted:
       ```
       m = valid.float()                                  # (n_cam, nx, ny, n_heights)
       s = (sampled * m[:, None]).sum(-1)                 # (n_cam, C, nx, ny)
       per_cam = s / m.sum(-1).clamp_min(1)[:, None]      # mean over VALID heights
       cam_hit = (m.sum(-1) > 0)                          # (n_cam, nx, ny): view sees pillar
       out = (per_cam * cam_hit[:, None]).sum(0) / cam_hit.sum(0).clamp_min(1)[None]  # (C,nx,ny)
       no_hit = cam_hit.sum(0) == 0                       # (nx, ny)
       ```
       Output projection on `out`; residual add to `query`; on `no_hit` cells leave the query
       unchanged (the guarded path). This matches the paper's `|V_hit|` semantics (a view is hit
       if >=1 reference height projects in-frame).
     - DEFORMABLE path (`offsets=True`): predict per-head offsets and softmax weights from
       `query`, add offsets to `ref_uv` (in normalized units), grid_sample at the shifted
       locations, weight-sum over the `n_points` samples, then the SAME shared height/hit-view
       reduction. Keep the reference-point projection as the anchor; offsets are a learned delta.
       The value projection and the output projection MUST be shared with the simplified path (the
       only extra parameters under `offsets=True` are the offset head and the weight head), so
       that with a zero-initialized offset head the deformable forward is byte-equal to the
       simplified forward (test C relies on this).

4. `warp_bev(prev_bev, ego_delta, bev_grid)` (HOLE).
   - `prev_bev`: `(C, nx, ny)`; `ego_delta`: the SE(2) ego motion from t-1 to t (yaw + forward,
     lateral translation in meters). Resample `prev_bev` so a static WORLD point stays at the same
     ego BEV cell after ego motion, via `F.affine_grid(theta, ..., align_corners=False)` +
     `F.grid_sample(..., align_corners=False)`. Return `(C, nx, ny)`.
   - EXACT AXIS + SIGN (expert blocker - the plan must pin this; it is the high-probability silent
     bug). The BEV tensor is `(C, H=nx=forward, W=ny=lateral)`. `affine_grid`'s `theta` row 0 is
     the W (lateral, ego +y/left) axis and row 1 is the H (forward, ego +x) axis; the sampling
     grid's last dim is `(x=W, y=H)`. For a PURE forward ego translation of `k = forward_m / res`
     cells, the static world content must move to a LOWER forward index, and `affine_grid`
     already specifies a sampling (inverse) warp, so the normalized translation sign is `+`:
     `theta = [[1, 0, +2*k_y/nx? ...]]` - concretely the forward term goes in `theta[1, 2] =
     +2*k_x/nx` (NOT row 0, NOT minus) and the lateral term in `theta[0, 2] = +2*k_y/ny`.
     Verified numerically: a hot cell at forward index 2 with `k_x=+1` ends at index 1 only with
     `theta[1,2] = +2/nx`; `-2/nx` sends it to index 3 (the double-inverse bug). For the yaw,
     build the 2x2 rotation into `theta[:, :2]` consistently with the same (W, H) = (col 0, row 1)
     axis assignment. Zero ego motion is the identity. Document this mapping in the docstring.

5. `TemporalSelfAttention` (HOLE in `forward`).
   - `__init__` (PROVIDED): a small multi-head attention (reuse `nanovision.attention`) over the
     stack `{current query, warped history}`.
   - `forward(self, query, prev_bev_warped)` (HOLE): for each BEV cell, attend the query against
     the two-element set `{query[p], prev_bev_warped[p]}` (a 2-key attention), residual add. When
     `prev_bev_warped is None` (first frame) fall back to self-attention on the query alone.

6. `BEVFormerEncoder.forward` and `BEVFormerSeg.forward` (HOLE).
   - `__init__` (PROVIDED): a learnable BEV query embedding `(nx, ny, C)`, a list of layers each
     `{SpatialCrossAttention, TemporalSelfAttention, feed-forward}`, and (for `BEVFormerSeg`) a
     `seg_head: Conv2d(C, n_classes_or_1, 1)`. Stores precomputed `bev_reference_points`.
   - `forward(self, feats, rig, prev_bev=None, ego_delta=None)` (HOLE): project reference points;
     run each layer (TSA with the warped prev_bev, then SCA pulling from `feats`, then FFN);
     return the final BEV grid `(C, nx, ny)`. `BEVFormerSeg.forward` adds the seg head ->
     `(1, nx, ny)` logit (or `(n_classes, nx, ny)`). Document B handling (toy is B=1).

## Tests and exact pass conditions

A. `test_reference_points.py`
   - Pillar shape `(16, 16, 4, 3)`; the 4 z-values are exactly `linspace(z_min, z_max, 4)`; the
     `(x, y)` of all heights in a pillar match the `BEVGrid` cell center.
   - Projection: place a single 3D point on the optical axis of one toy camera at a known depth;
     assert it projects to that camera's principal point (`uv ~ (0, 0)` in normalized coords) and
     `valid=True` there, `valid=False` for a camera facing the other way. Round-trip a projected
     ego point back through `world_to_pixel` (exact, < 1e-4).

B. `test_spatial_cross_attention.py`
   - Shape: `query (16,16,C)`, `feats (n_cam,C,Hf,Wf)`, output `(16,16,C)`.
   - Hit-mask (CONSTRUCT the cell, do not pick a fixed index - FOV overlaps make an arbitrary cell
     2-view). Set each camera's `feats` to a distinct constant map. Compute the per-cell hit count
     from `project_reference_points` (`valid.any(-1).sum(0)`), select a cell where it equals 1
     (`(valid.any(-1).sum(0) == 1).nonzero()[0]`), and assert that cell's pooled value equals the
     single hit camera's constant. If no single-view cell exists at the default focal, the toy
     generator raises focal (narrows FOV, opens wedges between cameras); the test asserts at least
     one exists.
   - No-hit / divide-by-zero path: unit-test the reduction DIRECTLY (a full 360 ring sees every
     azimuth, so a no-hit cell is not reliably present in geometry). Build a synthetic `valid`
     all-False for one cell and assert `SCA(query, feats, ref_uv, valid)[cell] == query[cell]`.
   - `gradcheck` (float64) on `feats -> SCA(query, feats, ref_uv, valid).sum()` with fixed
     `ref_uv, valid` (grid_sample is differentiable; gradients flow to image features).

C. `test_deformable.py`
   - With `offsets=True` and the offset head zero-initialized, the deformable forward equals the
     `offsets=False` forward within 1e-5 - construct BOTH modules to SHARE the value + output
     projection weights (copy the state_dict of the shared submodules, or build one module and
     toggle a flag), so the comparison isolates the offsets and not a projection mismatch. The
     property holds because zero offsets make all `n_points` sample the same location and the
     softmax weights sum to 1 (expert-confirmed algebra), AND both paths use the same shared
     height/hit-view reduction helper.
   - A backward pass confirms the offset head's weights receive nonzero gradient.

D. `test_temporal.py`
   - `warp_bev` correctness: a BEV grid with a single hot cell at a known WORLD location, warped
     by a pure translation ego motion of k cells, moves the hot cell by exactly k cells (nearest
     within bilinear tolerance). A zero ego motion is the identity (< 1e-5).
   - Temporal-necessity test (the bare `temporal < no-temporal` comparison is a coin flip - the
     single-frame model already overfits per test E, so make temporal information NECESSARY and
     score only where it matters). Use `bev_multicam_scene` two-frame, but OCCLUDE the moving
     vehicle in the current frame's camera images (drop it from all frame-t renders; keep it in
     frame t-1 with the correct ego warp). Score BCE only on that vehicle's BEV cells. The
     TSA-enabled encoder recovers it from warped history; the no-temporal model cannot. Pass:
     `temporal_BCE < no_temporal_BCE - margin` (margin ~0.1 on the occluded cells) across a small
     seed set (e.g. 3 seeds), or require the mean gap to exceed 2 sigma. Report both numbers per
     seed. A single-seed strict `<` does NOT gate the build.
   - `bev_multicam_scene` must support this occlusion: an `occlude_moving` flag that renders the
     moving vehicle into frame t-1 only and returns its BEV cells (for scoring) and the ego_delta.

E. `test_bev_seg.py` (overfit, bounded)
   - One frame from `bev_multicam_scene` (n_cam=4, 16x16 BEV, M vehicles). Train `BEVFormerSeg`
     (simplified SCA, no temporal) with BCE-with-logits for <= 1500 steps, Adam lr ~1e-2. Pass:
     final BCE < 0.05 AND BEV IoU(logit > 0, gt) > 0.6. CPU, deterministic. Pre-measure; if it
     floors above threshold, report the number rather than thrash.

F. `test_forbidden_imports.py` (mirror A11.5b's tokenize scan over `bevformer.py` top + solution
   + the shim). Forbidden: `cv2.projectPoints`, `cv2.solvePnP`, `import kornia`, `from kornia`,
   and any prebuilt `nn.MultiheadAttention` / `F.scaled_dot_product_attention` /
   `nn.Transformer` (the attention must come from `nanovision.attention`, the A1 build).
   `F.grid_sample` and `F.affine_grid` are ALLOWED (they are the bilinear-sampling substrate, not
   the taught attention). State that in the test.

## Toy scene (orchestrator adds to `nanovision/data/toy.py` before the build)

`bev_multicam_scene(n_cams=4, n_vehicles=4, bev=(-8,8), res=1.0, img=32, stride=4, n_frames=1,
ego_step=1.0, focal=None, occlude_moving=False, seed=0, device="cpu")` returning a dict.
`focal` defaults to `img/2` (~90 deg FOV). VERIFIED standalone: a 4-camera cardinal ring never
produces 2-view cells (FOV boundaries meet on the diagonals), so every cell is single-view or
unseen at any focal; at `img/2` almost the whole grid is single-view (only the 4 far corners are
unseen). The SCA hit-mask test uses a single-view cell (plenty exist); multi-view averaging is
exercised by its UNIT test (a synthetic all-False / partial `valid`), not by this toy's geometry.
Returns:
- `images`: `(n_frames, n_cams, 3, img, img)` (squeeze frame dim for single-frame use). Cameras
  ring the ego at height 1.5 m, yaw uniformly over 360 deg (front, left, back, right for n=4),
  OpenCV. Camera axes from ego yaw `a`: `z_cam=(cos a, sin a, 0)`, `x_cam=(sin a, -cos a, 0)`,
  `y_cam=(0, 0, -1)`; `E = invert(make_transform(R_ce, (0,0,1.5)))` with columns
  `[x_cam, y_cam, z_cam]`.
- `K`: `(3, 3)` shared; `E`: `(n_cams, 4, 4)` per camera `T_cam_ego`.
- `bev_gt`: `(n_frames, nx, ny)` vehicle occupancy on the centered BEV grid.
- `ego_deltas`: `(n_frames, 3)` SE(2) ego motion between frames (zero for frame 0). For the
  temporal test, the ego moves forward `ego_step` m per frame and one vehicle moves 1 cell.
- `vehicles`: per-frame ego `(x, y)`.
- `occluded_cells`: when `occlude_moving=True`, the BEV cells of the vehicle that appears only in
  frame t-1 (for the temporal test's scoring region); the moving vehicle is rendered into frame
  t-1's images only, kept out of frame-t images, but still present in frame t-1's `bev_gt`. Its
  motion between frames is 1 cell, and `ego_deltas` carries the matching ego motion.
- Each vehicle rendered as a colored blob in whichever cameras see it (project centroid at
  z=0.75, paint if in-frame). Vehicles placed (rejection-sampled) so each is seen by at least one
  camera and lies inside the BEV grid. With the default `focal=img`, at least one BEV cell is
  single-view (the generator can assert this). Deterministic per seed.

Build a `CameraRig` from `K`, `E` per frame (the tests wrap them in `CameraRig` with
`image_sizes` so `world_to_pixel` returns the in-bounds mask). The orchestrator verifies the
generator standalone (cameras tile 360 deg, each vehicle seen by >=1 camera, projections in
frame, two-frame motion is 1 cell) before spawning the build. Mirror `bev_toy_scene` style.

## README (lecture notes per the skill)

Cover, with arXiv links VERIFIED by fetching `https://arxiv.org/abs/<id>` (these were checked
when writing this plan - re-confirm titles):
- BEVFormer, Li et al., ECCV 2022, arXiv 2203.17270 - BEV queries, spatial cross-attention at
  projected pillar reference points, temporal self-attention. (Note: 2203.17054 is BEVDet4D, NOT
  BEVFormer - do not confuse them, A11.5b's README cites both correctly.)
- Deformable DETR, Zhu et al., ICLR 2021, arXiv 2010.04159 - the deformable attention module SCA
  specializes (learned offsets + bilinear sampling).
- DETR3D, Wang et al., CoRL 2021, arXiv 2110.06922 - the sparse-query predecessor (3D object
  queries projected to cameras, no BEV grid).
- PETR, Liu et al., ECCV 2022, arXiv 2203.05625 - the cleaner contrast: unproject pixels to 3D,
  encode as position embeddings, global cross-attention, no BEV grid / no deformable / no
  pillars. Explain it in ~one paragraph as the ~30-line warm-up the note recommends, and note it
  in the viz/README even though the graded build is BEVFormer.
- BEVFormer v2 (arXiv 2211.10439) one paragraph: perspective supervision (an auxiliary 2D head)
  is what lets modern backbones transfer to BEV - if a frozen 2D backbone gives poor BEV results,
  the cause is the 3D-supervision gap, not the attention code.
- Sparse4D v2 (arXiv 2305.14018) and PETRv2 (arXiv 2206.01256) as the 2024-2026 context: sparse
  query methods (no dense BEV grid) now dominate camera-only 3D DETECTION on nuScenes; the dense
  BEV grid persists for MAP and OCCUPANCY (A11.5d/e), which is why this assignment is still the
  right prerequisite. Frame BEVFormer as the foundational query-pull mechanism, not 2026 SOTA.
- The organizing contrast, stated plainly: LSS PUSHES image features out into 3D via predicted
  depth; BEVFormer PULLS features in by projecting BEV reference points back to image space and
  sampling. Neither is strictly better; depth-push transfers to occupancy, query-pull accumulates
  temporal state. Forward-point to occupancy (A11.5d) reusing the dense BEV grid.
- Define jargon at first use: BEV query, reference point / pillar, hit view, spatial cross-
  attention, deformable attention (learned sampling offsets), temporal self-attention, ego-motion
  warp. Refer to prior concepts by NAME (the multi-head attention from the transformer assignment,
  the camera projection from the camera-geometry assignment, the depth-lift from lift-splat-shoot),
  never by assignment number.
- TOY DOES NOT OVERRIDE SCALE: the 16x16 toy with 4 cameras shows the mechanism composes; it is
  NOT evidence about the dense-vs-sparse tradeoff or temporal robustness at scale. State that the
  temporal self-attention's known fragility at long range (ego-localization noise accumulates) is
  a real at-scale finding the toy cannot exhibit.

Then the mandatory context-less style review (spawn a general-purpose subagent given ONLY the
README path + `~/.claude/CLAUDE.md`) and apply its edits.

## Verify (orchestrator, on disk, both modes)

- `NANOVISION_IMPL=solution .../python -m pytest assignments/a11_5c_bevformer/tests -q` fully green.
- Default mode fails ONLY at the holes (`NotImplementedError`), not collection/import errors;
  `test_forbidden_imports` passes both modes.
- Re-run A11.5b and A11.5a in their own sessions to confirm the new `toy.py` symbol and the
  `nanovision.bevformer` shim did not regress the shared library.
- Record the overfit BCE/IoU and the temporal vs no-temporal numbers.
