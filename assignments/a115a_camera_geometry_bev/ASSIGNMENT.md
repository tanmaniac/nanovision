# assignments/a115a_camera_geometry_bev/ASSIGNMENT.md

```yaml
id: a115a_camera_geometry_bev
title: Camera geometry & the BEV transform
module: 3.5
type: Core
estimated_learner_hours: 4
depends_on: [a00_harness]
builds_into_shared_lib:
  - nanovision.geometry.project_points
  - nanovision.geometry.unproject
  - nanovision.geometry.make_transform
  - nanovision.geometry.apply_transform
  - nanovision.geometry.invert_transform
  - nanovision.geometry.compose_transforms
  - nanovision.geometry.BEVGrid
  - nanovision.geometry.CameraRig
  - nanovision.geometry.ipm_to_bev
  - nanovision.data.nuscenes_mini.NuScenesMini
forbidden_imports:
  - cv2.projectPoints
  - cv2.solvePnP
  - cv2.warpPerspective
  - cv2.findHomography
  - kornia
fits_12gb: true
external_data: nuScenes v1.0-mini (~4GB, optional; tests run synthetic)
camera_axis_convention: OpenCV (+x right, +y down, +z forward)
```

## motivation
The geometry the rest of the AV module imports: the four SE(3) primitives,
`CameraRig`, the ego-centric `BEVGrid`, and `ipm_to_bev`, reused by LSS (A11.5b),
BEVFormer (A11.5c), and occupancy (A11.5d). The new content is nuScenes
coordinate-frame plumbing, the lidar/camera temporal offset, and the flat-ground
IPM baseline. Full treatment with paper links is in the README.

## background
Camera frame is OpenCV (+x right, +y down, +z forward); ego frame is x forward,
y left, z up; global is fixed ENU. Pinhole: u = fx X/Z + cx, v = fy Y/Z + cy;
inverse at depth d gives X = (u-cx) d/fx, Y = (v-cy) d/fy, Z = d. The four-step
lidar-to-camera chain is lidar sensor -> ego@lidar_time -> global ->
ego@cam_time -> camera; the naive shortcut reuses the lidar-time ego pose and
errs by the ego motion (~1.5 m at 30 m/s, 50 ms). nuScenes quaternions are
scalar-first (w, x, y, z); images are pre-undistorted (K exact, no distortion).
The BEV grid is ego-centric, default [-50, 50] m at 0.5 m (200x200).

## what_you_implement
- `project_points` / `unproject` (pinhole and inverse).
- `make_transform`, `apply_transform`, `invert_transform`, `compose_transforms`.
- `CameraRig.world_to_cam` / `cam_to_world` / `world_to_pixel`.
- `ipm_to_bev` (flat-ground warp into the BEV grid).

`BEVGrid` is provided; the nuScenes loader is provided boilerplate.

## tasks
- **Task 1 — projection** (file: `starter/geometry.py`, symbols:
  `project_points`, `unproject`): pinhole projection and its inverse. Teaches:
  the exact pinhole model on the OpenCV camera axes and that unproject undoes
  project at the known depth.
- **Task 2 — SE(3) primitives** (file: `starter/geometry.py`, symbols:
  `make_transform`, `apply_transform`, `invert_transform`, `compose_transforms`):
  the 4x4 toolkit. Teaches: SE(3) as homogeneous matrices, the structured
  inverse (R^T, -R^T t), and left-to-right composition. Reused by every later
  AV assignment.
- **Task 3 — CameraRig** (file: `starter/geometry.py`, symbol: `CameraRig`):
  per-camera projection with an in-front-and-in-bounds visibility mask.
  Teaches: the multi-camera rig abstraction (K + ego-to-camera extrinsic per
  camera) and which cameras see a given 3-D point.
- **Task 4 — ipm_to_bev** (file: `starter/geometry.py`, symbol: `ipm_to_bev`):
  warp images onto the ego ground plane via `grid_sample`. Teaches: the
  flat-ground homography baseline and its failure on elevated points.

## tests
Run in this order (see README "How to verify"):
1. `tests/test_projection.py` — reference values, axis point, unproject/project
   round-trip, float64 gradcheck. (shape + reference + gradcheck)
2. `tests/test_se3.py` — block layout, applied-point reference, inverse is
   identity, composition associativity and chaining, float64 gradcheck on
   `apply_transform`. (reference + gradcheck)
3. `tests/test_rig.py` — a synthetic 4-camera rig: a front point is visible only
   in the front camera and lands on the optical axis; a cube projects into the
   expected cameras; `cam_to_world` inverts `world_to_cam`. (reference)
4. `tests/test_ipm.py` — a ground marker warps back to its expected BEV cell;
   an elevated point is mapped past its footprint (the documented breakage).
5. `tests/test_temporal_offset.py` — the timestamp-correct chain differs from
   the naive one by the ego motion (1.5 m on synthetic poses); with no ego
   motion they agree.
6. `tests/test_nuscenes_loader.py` — the loader module imports without the
   devkit; the sample-shape check skips when NUSCENES_DATAROOT/devkit are absent.
7. `tests/test_forbidden_imports.py` — greps the solution and shared geometry
   module for cv2/kornia shortcuts.

## provided_boilerplate
`BEVGrid`, the full `nanovision.data.nuscenes_mini` loader (devkit plumbing,
image downsampling, calibration parsing), the synthetic-camera test helpers, and
`nanovision.gradcheck`. The learner writes only Tasks 1-4.

## compute_notes
CPU only, seconds. All tests use synthetic cameras; no dataset required. float32
geometry with float64 gradchecks on `project_points` and `apply_transform`. The
200x200 BEV grid and 400x224 images fit easily.

## stretch_goals
1. Coverage-count blending in `ipm_to_bev` (verticality-weighted instead of
   last-camera-wins).
2. The closed-form 3x3 ground-plane homography, checked against the `grid_sample`
   warp.
3. On real data, measure the naive-vs-correct pixel gap vs ego speed.

## further_reading
- Caesar et al., "nuScenes" (CVPR 2020), arXiv:1903.11027.
- nuScenes devkit schema_nuscenes.md (calibrated_sensor / ego_pose fields).
- Philion & Fidler, "Lift, Splat, Shoot" (ECCV 2020), arXiv:2008.05711.
- Li et al., "BEVFormer" (ECCV 2022), arXiv:2203.17270.
- Harley et al., "Simple-BEV" (ICRA 2023), arXiv:2206.07959.

## solution_notes
Camera axis convention is OpenCV (+x right, +y down, +z forward), matching
nuScenes. `CameraRig` stores ego-to-camera (world-to-camera) extrinsics; the
loader inverts the devkit's camera-to-ego record to match. The IPM round-trip
test is driven from a cell center (project its ground point to a pixel, paint a
marker, warp back) so the expected peak cell is exact rather than approximate.
The temporal-offset test uses identity lidar/camera extrinsics and a pure +x ego
translation so the 1.5 m shift is exactly reproducible.
```
