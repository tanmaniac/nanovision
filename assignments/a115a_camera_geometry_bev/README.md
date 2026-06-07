# A11.5a — Camera geometry and the BEV transform

## Motivation
This assignment builds the geometry substrate for the whole autonomous-driving
module. Lift-Splat-Shoot (A11.5b), BEVFormer (A11.5c), and occupancy (A11.5d)
all reuse the same four SE(3) primitives, the same `CameraRig`, and the same
ego-centric BEV grid you define here. The mechanism is not new to you: pinhole
projection and SE(3) composition are the bread and butter of stereo VO and SfM.
What is new is the nuScenes-specific plumbing - the three nested coordinate
frames, the four-step lidar-to-camera chain with its lidar/camera timestamp
offset, and the flat-ground IPM baseline that the learned BEV methods exist to
fix. The deliverable is a geometry library whose lidar overlay you can eyeball
on real data and trust downstream.

## Background

### Coordinate frames and conventions
The camera frame is OpenCV-style: +x right, +y down, +z forward into the scene.
This is the nuScenes camera convention, so the stored intrinsic K is the exact
pinhole matrix with no distortion terms (nuScenes images are pre-undistorted).
The ego frame is the vehicle body, right-handed with x forward, y left, z up;
its origin is the rear-axle midpoint. The global frame is a fixed ENU map frame.
All three matter because the lidar-to-camera chain visits all of them.

SE(3) transforms are 4x4 homogeneous matrices applied on the left,
`p' = T @ p_homogeneous`. A transform named `T_b_a` reads "a-to-b": it takes a
point in frame a and returns it in frame b.

### Pinhole projection (one-paragraph review)
A camera-frame point (X, Y, Z) with Z > 0 projects to a pixel by

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

and back-projects at depth d by X = (u - cx) d / fx, Y = (v - cy) d / fy, Z = d.
Back-projection is what every BEV lift depends on; the only place to get it
wrong is the sign/order when you chain it with the extrinsics.

### The four-step lidar-to-camera chain
LIDAR_TOP runs at 20 Hz, cameras at 12 Hz; at a keyframe the sweep and the image
were captured at slightly different times. The correct projection of a lidar
point into a camera is

    lidar sensor -> ego @ lidar_time -> global -> ego @ cam_time -> camera,

then apply K and divide by z, keeping z > 0 and in-bounds points. The naive
shortcut reuses the lidar-time ego pose for the camera step. The error is the
ego motion between the two timestamps: at 30 m/s and a ~50 ms offset that is
about 1.5 m, enough to visibly misalign points on objects at the field-of-view
edge. The temporal-offset test makes this concrete on synthetic poses.

### The pyquaternion convention
nuScenes stores rotations as scalar-first quaternions (w, x, y, z) in both
`calibrated_sensor` and `ego_pose`. Passing the components in the wrong order
gives a plausible-looking but wrong rotation. The point is SE(3) composition,
not quaternion algebra - build the 4x4 matrices immediately and chain them.

### The ego-centric BEV grid (the module-wide contract)
The BEV grid is always defined in the ego frame, centered on the vehicle. The
`BEVGrid` dataclass fixes the contract for the whole module: x forward, y left,
default [-50, 50] m on both axes at 0.5 m resolution (a 200x200 grid). LSS,
BEVFormer, and occupancy all reuse this; occupancy adds a z axis.

### Flat-ground IPM and why it breaks
Inverse perspective mapping maps each BEV ground cell to a single image pixel
under one assumption: every pixel is a point on the flat z = 0 ground plane. It
is exact for road markings and ground texture. It is wrong for anything
elevated: a feature at height h projects to the same pixel as a ground point
farther from the camera, so elevated objects are painted into a BEV cell beyond
their true footprint, smeared toward the camera. The homography cannot recover
height from one view; the fix is real depth (LSS) or explicit 3-D queries
(BEVFormer). The IPM test asserts both the correct ground mapping and this
breakage.

## What you'll implement
In `starter/geometry.py`: `project_points`/`unproject`, the four SE(3)
primitives (`make_transform`, `apply_transform`, `invert_transform`,
`compose_transforms`), `CameraRig` (`world_to_cam`, `cam_to_world`,
`world_to_pixel`), and `ipm_to_bev`. The `BEVGrid` dataclass is provided.
The nuScenes loader is provided boilerplate; you do not implement the devkit
plumbing.

## Tasks
1. `project_points` / `unproject` - pinhole projection and its inverse.
2. The four SE(3) primitives - 4x4 assembly, point application, structured
   inverse (R^T, -R^T t), and left-to-right composition.
3. `CameraRig.world_to_cam` / `cam_to_world` / `world_to_pixel` - per-camera
   projection with an in-front-and-in-bounds visibility mask.
4. `ipm_to_bev` - warp images onto the ego ground plane via `grid_sample`.

Each maps to a `raise NotImplementedError("A11.5a Task N: ...")` in `starter/`
and to one test file.

## How to verify
Run from the repo root with the `nanovision` env active, in this order:

    make test A=a115a_camera_geometry_bev      # your starter (red until filled)

The tests run projection -> SE(3) -> rig -> IPM -> temporal-offset, all on
synthetic cameras (no dataset needed). To confirm the reference passes and to
render the figures:

    make verify A=a115a_camera_geometry_bev    # reference solution (green)
    make viz    A=a115a_camera_geometry_bev    # writes PNGs to out/

The nuScenes loader test reports as skipped unless the dataset is present; that
is expected. The reference implementation is visible in `nanovision/geometry.py`;
read it if you get stuck.

## Dataset step zero (optional, only for the real-data overlay)
The geometry and all tests run without nuScenes. To render the real 6-camera
lidar overlay and stitched BEV in `viz.py`:

1. Create a nuScenes account and accept the license at
   https://www.nuscenes.org/nuscenes#download.
2. Download the `v1.0-mini` split (~4 GB) and extract it.
3. Install the dataset extra: `pip install -e ".[av]"` (pulls in
   `nuscenes-devkit` and `pyquaternion`).
4. Set `NUSCENES_DATAROOT` to the directory that contains the `v1.0-mini`
   folder, e.g. `export NUSCENES_DATAROOT=$HOME/data/nuscenes`.

With those set, `viz.py` renders the naive-vs-temporal-correct lidar overlay
and the stitched BEV; without them it falls back to a synthetic cube and a
ground-checkerboard warp.

## Compute notes
CPU only, seconds to run. Everything is tested on synthetic cameras at float32,
with float64 gradchecks on `project_points` and `apply_transform`. The 200x200
BEV grid and downsampled 400x224 images fit in memory with room to spare; no GPU
is needed for this assignment.

## Stretch goals
1. Add a per-cell coverage count to `ipm_to_bev` and blend overlapping cameras
   by viewing-ray verticality instead of last-camera-wins.
2. Implement the closed-form ground-plane homography H (3x3, pixel -> BEV) and
   check it agrees with the per-cell `grid_sample` warp on ground points.
3. On real data, pick 5 ground lidar points on a lane marking and measure the
   naive-vs-correct pixel gap as a function of ego speed.

## Further reading
- Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving"
  (CVPR 2020) - coordinate frames and the four-step projection chain.
- Philion and Fidler, "Lift, Splat, Shoot" (ECCV 2020) - the camera-rig and
  ego-centric BEV grid abstraction every later method reuses.
- Li et al., "BEVFormer" (ECCV 2022), Sec. 3.2 - 3-D reference points projected
  to 2-D image coordinates, the back half of the projection chain.
- Harley et al., "Simple-BEV" (ICRA 2023) - the 200x200x8 ego-centric grid.
