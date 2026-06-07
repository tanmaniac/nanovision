# A11.5a — Camera geometry and the BEV transform

Research date: 2026-06-06.

---

## 1. Key concepts a student must learn

### The three nuScenes coordinate frames and why all three matter

nuScenes defines three nested coordinate frames. Every piece of reasoning about multi-camera BEV perception requires moving data between all three.

**Sensor frame.** Raw sensor data lives here. For LIDAR_TOP, points are expressed in the lidar's own coordinate system. For each camera, the 3-D scene is expressed in the camera's own frame (z forward, x right, y down — the standard OpenCV camera convention). The calibrated_sensor record stores the rigid transform (translation + quaternion rotation) that maps sensor-frame coordinates into the ego frame.

**Ego frame.** The vehicle-body frame. Its origin is the midpoint of the rear axle; axes follow the vehicle's heading. This is the natural "scene center" for BEV grids because downstream perception algorithms (LSS, BEVFormer, occupancy networks) all define their BEV grid centered on the ego vehicle. The ego_pose record stores the rigid transform that maps ego-frame coordinates into the global frame.

**Global frame.** A fixed East-North-Up (ENU) map frame tied to the geographic region of the log. All 3-D annotations in nuScenes are stored in global coordinates. The z-component of ego_pose translation is always 0, reflecting that the vehicle moves on a nominally flat surface (this has implications for the IPM baseline).

The student must understand these frames not as an abstract formality but because the four-step lidar-to-camera projection chain (described below) visits all three, and because the BEV grid is always defined in the ego frame — not in any sensor frame and not in global.

### The calibrated_sensor and ego_pose records

`calibrated_sensor` fields: `translation` (float[3], meters, sensor origin in ego frame), `rotation` (float[4], quaternion w,x,y,z, sensor orientation in ego frame), `camera_intrinsic` (float[3,3], only populated for cameras).

`ego_pose` fields: `translation` (float[3], ego origin in global frame), `rotation` (float[4], quaternion w,x,y,z, ego orientation in global frame), `timestamp` (microseconds).

Both use the pyquaternion convention: scalar w first, then vector components (x, y, z). This matters when calling `Quaternion(w, x, y, z)` or reading from JSON. A common mistake is passing the components in the wrong order, which produces rotations that are visually plausible but wrong.

The transform from sensor to ego is: `p_ego = R_sensor * p_sensor + t_sensor`, where `R_sensor` is the rotation matrix from the calibrated_sensor quaternion and `t_sensor` is its translation. In matrix form this is a 4x4 SE(3) matrix. The devkit utility `transform_matrix(translation, rotation, inverse=False)` in `nuscenes.utils.geometry_utils` constructs this. The inverse kwarg builds the matrix for the reverse direction.

### The four-step lidar-to-camera projection

LIDAR_TOP runs at 20 Hz; cameras run at 12 Hz (full rate) with keyframes at 2 Hz. At a given keyframe timestamp, the lidar sweep and camera image were captured at different times. The correct four-step chain accounts for this temporal offset:

1. Lidar sensor frame -> ego frame at the lidar sweep timestamp, using the lidar's calibrated_sensor extrinsics.
2. Ego frame at lidar time -> global frame, using ego_pose at the lidar sweep timestamp.
3. Global frame -> ego frame at the camera image timestamp, using the inverse of ego_pose at the camera timestamp.
4. Ego frame at camera time -> camera sensor frame, using the inverse of the camera's calibrated_sensor extrinsics.

After step 4, points are in camera coordinates. Apply the 3x3 intrinsic matrix K and divide by z to get pixel coordinates. Filter to points with z > 0 (in front of the camera) and within image bounds.

The devkit's `map_pointcloud_to_image` implements this. The student must re-implement it from scratch in PyTorch using the raw JSON tables, so they understand every matrix multiply.

If steps 2 and 3 are collapsed by using the same ego_pose for both (a common shortcut), you introduce a small but real error because the vehicle moved between the lidar sweep and the camera capture. At highway speeds (30 m/s) and the ~50 ms worst-case offset, this is roughly 1.5 m. That is large enough to visibly misalign projected points on objects at the edge of the field of view.

### Camera intrinsics: the pinhole model and what nuScenes stores

nuScenes stores a 3x3 intrinsic matrix K per camera:

```
K = [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1 ]]
```

There are no distortion coefficients stored for the core nuScenes dataset — images are pre-undistorted and stored as rectified 1600x900 JPEGs. (This is distinct from the nuImages spin-off dataset, which stores distorted images plus distortion parameters.) The student should note this: the pinhole model is exact for the stored images; no radial/tangential correction is needed.

The projection from a 3-D camera-frame point `[X, Y, Z]` to pixel coordinates `[u, v]`:

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

And the back-projection (unproject) given pixel `[u, v]` and depth `d`:

```
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

Back-projection is the operation that LSS, BEVDepth, and BEVFormer all depend on: lifting 2-D image features to 3-D space. Getting the sign and order right (especially when combining with the extrinsic chain) is where most student errors occur.

### The nuScenes 6-camera rig

The six cameras are: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_BACK. They are mounted on the vehicle roof. Five cameras have a 70-degree horizontal FOV; CAM_BACK has a 110-degree FOV. Together they provide 360-degree horizontal coverage with overlapping fields of view between adjacent cameras. The inter-camera angles are approximately 55 degrees apart.

This rig is why a BEV grid centered on the ego vehicle sees useful coverage from all directions. For close-range objects (< 10 m), camera back-projection is most informative. For far-range (> 30 m), only the front cameras contribute.

A practical gotcha: in nuScenes-mini, `calibrated_sensor` has 120 entries across 10 scenes (roughly 12 per scene = 6 cameras + 1 LIDAR + 5 RADAR). Each scene can have slightly different calibration values because calibration was updated during the 6-month collection campaign. Students loading calibration must retrieve the calibrated_sensor token from the sample_data record for the specific camera at the specific timestamp — not assume a single global calibration.

### Naive flat-ground IPM and why it breaks

Inverse perspective mapping (IPM) uses a planar homography to transform a camera image into a bird's-eye-view image. The homography is derived from camera intrinsics and extrinsics under one assumption: every pixel corresponds to a point on a flat, horizontal ground plane.

Given the camera height h above the ground and the rotation of the camera relative to the ground normal, you can write a 3x3 homography H that maps image pixel coordinates directly to ground-plane BEV coordinates:

```
p_bev = H * p_image      (homogeneous coordinates)
```

This works perfectly for road markings and ground texture. It fails for anything elevated off the ground. A pedestrian's head is ~1.7 m above the ground; their feet are at ground level. IPM maps the feet to the correct BEV location but maps the head to a point ~5-10 m ahead of where the person is actually standing. Taller objects (trucks, buildings) appear stretched toward the camera in BEV, not at their true footprint.

This breakage is not a bug — it is the fundamental limit of the homography model. The flat-ground homography has no way to distinguish a feature at height h from a feature at height 0 that is slightly farther away; they project to the same image pixel. The disambiguation requires depth estimation (the LSS approach) or explicit 3-D queries (the BEVFormer approach).

The BEV grid in nuScenes-based research is universally defined in the ego frame, centered at the ego vehicle. A typical range is [-50 m, +50 m] in x (forward/backward) and [-50 m, +50 m] in y (left/right), discretized at 0.5 m per cell (a 200x200 grid). LSS uses a slightly smaller range for its segmentation task. BEVFormer uses a 100m range for detection. The vertical extent (z) is either discarded (for BEV segmentation tasks) or discretized into voxel columns of height ~8 m at 0.5 m per step (for occupancy tasks).

### The SE(3) / quaternion plumbing

Because the student has a strong prior background in SLAM, the rotation math is familiar. The new thing here is the specific quaternion convention and the nuScenes JSON layout. The devkit uses pyquaternion throughout; the quaternion multiplication and rotation-matrix-from-quaternion operations are provided by pyquaternion and should not be reimplemented (they are substrate, not the taught mechanism).

The SE(3) primitives the student must build from scratch:
- `make_transform(translation, quaternion_wxyz) -> Tensor[4,4]`: assembles the 4x4 matrix.
- `apply_transform(T, points) -> Tensor[N,3]`: applies it to a batch of 3-D points.
- `invert_transform(T) -> Tensor[4,4]`: exploits the SE(3) structure (transpose R, negate R^T t).
- `compose_transforms(*Ts) -> Tensor[4,4]`: chains a sequence of transforms.

These four primitives are the backbone of every subsequent AV assignment.

---

## 2. Mechanisms to implement from scratch

### Mechanism A: calibration loader and SE(3) primitives

**Task.** Write a `NuScenesCalibration` class that, given a sample token, loads all six camera intrinsics and extrinsics (as 4x4 SE(3) tensors) from the raw JSON tables without using any devkit shortcut functions for the matrix construction. Also load the LIDAR_TOP extrinsic.

Implement `make_transform`, `apply_transform`, `invert_transform`, `compose_transforms` in PyTorch (CPU, autograd-compatible).

**Verifiable.**
- Shape test: `make_transform(t, q).shape == (4,4)`.
- Inverse test: `compose_transforms(T, invert_transform(T))` is close to identity (residual < 1e-6).
- Consistency test: apply the devkit's `transform_matrix` to the same inputs and compare element-wise.
- `torch.autograd.gradcheck` on `apply_transform` with a small point batch.

### Mechanism B: lidar-to-image projection (calibration sanity check)

**Task.** Implement the full four-step chain: lidar sweep -> ego (at lidar time) -> global -> ego (at camera time) -> camera sensor frame -> pixel coordinates. Apply K to produce (u, v) pixel coordinates. Filter to in-front (z > 0) and in-bounds points. Overlay colored depth dots on the camera image.

For the first-pass version, skip the timestamp correction (use the same ego_pose for both lidar and camera times). Then add the timestamp-correct version and compare the two overlays to make the temporal parallax visible.

**Verifiable.**
- Visual: lidar points should land on the corresponding objects in the image. A car's lidar returns should cluster on the car body, not floating in empty space.
- Quantitative: pick 5 manually identified ground-plane lidar points; their projected (u, v) should match the visible road markings to within ~5 pixels.
- The temporal correction version should show visibly tighter alignment for moving objects compared to the single-ego_pose version.

### Mechanism C: flat-ground IPM and multi-camera BEV warp

**Task.** Implement a `GroundPlaneHomography` class that, given camera K and extrinsics (ego-to-camera SE3), computes the 3x3 homography mapping BEV grid coordinates (x, y in ego frame) to image pixel coordinates. Use `torch.nn.functional.grid_sample` with the resulting normalized grid to warp the camera image into the BEV canvas. Combine all six cameras into a single BEV image.

For the combination step, use a simple occupancy mask: each BEV cell gets its color from whichever camera last wrote to it (or from the camera whose viewing ray is most perpendicular to the ground in that cell).

**Verifiable.**
- Road markings should appear at approximately correct positions in the BEV canvas (they lie on the ground plane where IPM is correct).
- Tall objects (parked cars, pedestrians) should appear visibly distorted or smeared in the BEV canvas, with the distortion direction pointing toward the camera that saw them.
- The flat-ground breakage should be described in a one-paragraph comment in the code: which objects are broken, why, and what the correct fix is.

---

## 3. Assessment of the draft scope

### What is correct

The draft scope correctly identifies the core mechanism: pinhole projection and inverse, the full intrinsic/extrinsic chain for the 6-camera rig, ground-plane IPM as a naive BEV baseline, and a nuScenes-mini loader with calibration utils. These are the right primitives. The verifiable tasks (lidar overlay, multi-cam BEV warp, visible flat-ground failure) are well-chosen and self-evidently correct to evaluate visually.

The draft is also right to call out that the point of this assignment is to make the nuScenes-specific calibration plumbing concrete, not to re-teach pinhole cameras. The student's SLAM background means the math is not new; what is new is the JSON schema, the three-frame chain, and the temporal offset.

### What is missing or under-specified

**The four-step transform needs to be explicit.** The draft says "full intrinsic/extrinsic chain" but does not name the temporal offset issue. This is the most common source of confusion for experienced perception engineers working with nuScenes for the first time. The assignment should explicitly require the student to implement both the naive single-ego_pose version and the timestamp-correct version and observe the difference. This is a concrete, measurable exercise.

**The pyquaternion convention must be stated.** The draft does not mention that nuScenes uses (w, x, y, z) scalar-first ordering, that images are pre-undistorted (no distortion coefficients exist for the core dataset), or that each scene can have different calibration. All three are gotchas that experienced practitioners hit. They belong in the assignment text.

**The BEV grid definition needs to be made explicit.** The draft says "multi-cam images warp into a shared BEV grid" but does not specify what that grid is. Students coming to LSS (A11.5b) will be confused if the BEV grid definition is introduced there without grounding. The grid — ego-centric, defined in meters, typically [-50, 50] x [-50, 50] at 0.5m resolution — should be established in this foundational assignment.

**The SE(3) primitives should be named as a deliverable.** `make_transform`, `apply_transform`, `invert_transform`, `compose_transforms` are literally reused in every subsequent AV assignment (LSS, BEVFormer, occupancy). The draft mentions "calibration utils" but does not spell out that these four functions are the shared substrate. Being explicit about this shapes the student's code organization and prevents duplicated implementations.

### What should be cut

The draft's description "pinhole projection + inverse" as a major learning item is over-weighted for a student with this background. The student already knows the pinhole model. One paragraph of review is appropriate; half a lab is not. The assignment time saved here should be spent on the timestamp offset exercise and the BEV grid definition.

### What is outdated or mis-emphasized

Nothing in the draft scope is factually outdated — the nuScenes schema and coordinate frames have not changed since the dataset release. The devkit is maintained (last commit activity through 2025, still the standard for nuScenes access).

One subtle mis-emphasis: the draft says "quaternion extrinsics" as if the quaternion itself is the point. The point is SE(3) composition. Quaternions are just the storage format; the student should build SE(3) transform matrices immediately and work in 4x4 matrix form from there. Emphasizing quaternion operations risks students spending time on Rodrigues formulas when they should be doing matrix chain multiplication.

### Coordinate frame chain correctness

The draft does not explicitly describe the chain, but the elements it lists (calibrated_sensor, ego_pose, sensor-to-ego, ego-to-global) are all correct. The missing element is the temporal distinction between the lidar ego_pose and the camera ego_pose, which is distinct from the spatial chain. This should be added.

### Primitives needed by later assignments

LSS (A11.5b) directly reuses: `NuScenesCalibration`, the 4x4 extrinsic matrices (called E in LSS), K, and the BEV grid definition. It adds the depth lift on top of them.

BEVFormer (A11.5c) reuses: K, E, the ego-to-camera chain, and the BEV grid. It adds deformable attention with 3-D reference points projected to 2-D image coordinates — this projection step is identical to step 4 of the lidar-to-image chain.

Occupancy (A11.5d) reuses everything from A11.5b and adds a z axis to the BEV grid. The voxel-to-camera projection is the same four-step chain with a 3-D BEV query point instead of a lidar point.

Prediction (A11.5e) consumes the BEV feature map produced by A11.5b-d and does not directly use the geometry primitives, but it assumes the BEV grid definition (cell size, range, ego-centric frame) is consistent.

The foundation is therefore: `CameraRig` (stores K and SE3 extrinsics for all six cameras), the SE(3) transform primitives, and the BEV grid specification struct. These are the right named deliverables.

---

## 4. Connections to later assignments

**A11.5b (LSS)** is a direct extension of Mechanism C above. The only difference is that LSS does not use a flat-ground homography — it predicts per-pixel depth and unprojects image features into 3-D using the same K and extrinsics established here. If the student has wrong extrinsics in A11.5a, every downstream BEV result is wrong.

**A11.5c (BEVFormer)** replaces the explicit depth estimation with learned BEV queries. The spatial cross-attention computes, for each BEV grid cell, the 2-D image coordinates it would project to in each camera view. This is exactly the back-half of the lidar-to-image chain (ego frame -> camera sensor frame -> pixel). If the student cannot implement this projection cleanly, BEVFormer cannot be debugged.

**A11.5d (occupancy)** extends the BEV grid from 2-D (x, y) to 3-D (x, y, z). The camera-to-voxel projection is the same chain, just with query points at non-zero z. All occupancy methods (SurroundOcc, Occ3D) use a voxel grid still centered on the ego vehicle in the ego frame.

**A11.5e (prediction)** works on the BEV feature map produced by A11.5b/c/d. The prediction head must know the spatial extent and resolution of the BEV grid to interpret the feature map spatially. The BEV grid definition established in A11.5a is the shared contract.

The common failure mode across all four downstream assignments is a subtle sign error or axis swap in the extrinsic chain. If the lidar-to-camera overlay in A11.5a is visually verified to be correct, this class of error is caught before it can propagate.

---

## 5. Must-read sources

**Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving," CVPR 2020.** Primary reference for the dataset, coordinate frames, sensor specifications, and calibration methodology. The supplemental material describes the four-step lidar-to-camera projection chain explicitly.

**nuScenes devkit schema documentation, `docs/schema_nuscenes.md`, nutonomy/nuscenes-devkit (GitHub, maintained through 2025).** The authoritative field-by-field description of `calibrated_sensor`, `ego_pose`, `sample`, `sample_data`. Required reading before touching any JSON table.

**nuScenes devkit `nuscenes/utils/geometry_utils.py` (GitHub).** Four utility functions: `transform_matrix`, `view_points`, `box_in_image`, `points_in_box`. The student should read these before implementing their own versions — understanding what the reference implementation does is part of the exercise.

**Philion and Fidler, "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D," ECCV 2020.** Establishes the camera-rig abstraction (K + extrinsics per camera, ego-centric BEV grid) that every subsequent method reuses. The Section 3 geometry setup is the clearest published description of how to use nuScenes calibration for BEV perception.

**Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers," ECCV 2022.** The spatial cross-attention section (Sec. 3.2) shows exactly how 3-D BEV reference points are projected to 2-D image coordinates using K and extrinsics. Useful as a preview of why the projection primitives in A11.5a matter.

**Harley, Fang, and Fragkiadaki, "Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?," ICRA 2023.** Clear description of a 200x200x8 BEV voxel grid centered on ego with 0.5 m resolution. Good reference for the BEV grid definition and the bilinear-sampling lift strategy.

**Caesar et al., "nuScenes Revisited: Progress and Challenges in Autonomous Driving," arXiv 2512.02448, December 2025.** Retrospective from the dataset creators. Confirms that the ego-motion compensation in lidar sweeps is baked in and irreversible; documents the 99.7% confidence interval on sensor time offsets ([−4.5, 5.0] ms); notes the geographic overlap issue in train/val splits (relevant for any student building a sanity check on their BEV output).

---

## 6. Developments since 2024 relevant to the BEV foundation

**Metric3D v2 (2024) changes what "unproject" means.** Yin et al. (TPAMI 2024) showed that a foundation model trained on 16M+ images can produce zero-shot metric depth for arbitrary cameras by normalizing to a canonical focal length. This directly replaces the hand-crafted depth prediction in LSS with a pre-trained prior. For the course, this means the "lift" step in A11.5b can optionally use Metric3D v2 depth as a drop-in, making the depth ambiguity problem concrete by comparing learned depth vs. foundation-model depth vs. lidar depth.

**Temporal BEV and online mapping matured in 2023-2024.** Methods like MapTR v2 (2023) and StreamMapNet (2024) add recurrent BEV state across frames. The BEV grid coordinate system and the ego-motion compensation between frames are the exact same SE(3) machinery from A11.5a, just applied at inference time to shift the previous BEV state into the current ego frame. This connects A11.5a directly to any assignment on temporal fusion.

**Occupancy benchmarks consolidated in 2024.** Occ3D (NeurIPS 2024 proceedings) and SurroundOcc (ICCV 2023) both use the nuScenes ego-centric voxel grid. The standard occupancy grid for nuScenes is [-40 m, 40 m] in x/y and [-1 m, 5.4 m] in z, with 0.4 m voxels (200x200x16). This is a direct extension of the BEV grid definition from A11.5a. Students should see this continuity.

**SparseBEV and deformable attention (ICCV 2023, active through 2025).** SparseBEV achieves 67.5 NDS on nuScenes test using fully sparse 3-D queries rather than a dense BEV grid. The projection of sparse 3-D reference points to 2-D camera coordinates is identical to the four-step chain in A11.5a. The "dense BEV grid" established in A11.5a remains the easier conceptual entry point, but SparseBEV shows where the field went.

**nuScenes Revisited (Dec 2025)** confirms that the core dataset format, coordinate system, and devkit are stable. No schema changes. The paper documents that geographic overlap between train and val splits causes overfitting in map segmentation tasks, which is worth noting if students use the BEV canvas for any quantitative comparison.

**Gaussian Splatting BEV (2025-2026).** Several 2025-2026 works (e.g., "Reconstruction Matters: Learning Geometry-Aligned BEV Representation through 3D Gaussian Splatting," arXiv 2603.19193) replace the voxel splat with differentiable Gaussian primitives. These still require the same K, extrinsics, and BEV grid definition from A11.5a. The camera geometry foundation is unchanged; the aggregation step is new.
