# Classical SLAM: the parts the module points at but does not build

Reading-only notes attached to the classical-SLAM module (a14_0 through a14_5). The module
builds the geometric core: Lie groups, Kalman filtering, EKF-SLAM, multi-view geometry, ICP,
and factor-graph optimization. Four things sit just outside that core - they are what turns the
estimators into a working system, and each is a substantial topic on its own. These notes give
the key idea of each, why it matters, and what to read, without building them.

## IMU pre-integration and visual-inertial odometry

The pose-graph and bundle-adjustment back-end (a14_5) optimizes over camera poses connected by
relative-pose or reprojection factors. A real system on a moving platform also has an inertial
measurement unit (IMU): a gyroscope and accelerometer sampling at 100-1000 Hz, far faster than
the camera. The IMU measures angular velocity and linear acceleration in the body frame, which
in principle integrate to the pose - but integrating raw IMU between two camera frames depends
on the pose and velocity at the start of the interval, which are exactly the unknowns being
estimated. Naively, every time the optimizer adjusts a pose, all the IMU integrals downstream of
it would have to be recomputed.

Pre-integration is the trick that removes that coupling. The insight (Lupton and Sukkarieh 2012,
made manifold-correct by Forster, Carlone, Dellaert, Scaramuzza 2017) is to integrate the IMU in
the body frame of the first keyframe, producing a relative motion increment - a change in
rotation, velocity, and position - that does not depend on the absolute pose, only on the (small,
slowly varying) IMU biases. The increment becomes a single factor in the graph connecting two
keyframes, with an analytic Jacobian with respect to the biases so it can be corrected by a
first-order update rather than re-integrated when the bias estimate changes. The rotation part of
the increment lives on $SO(3)$, so the pre-integration and its covariance propagation use the
same right-Jacobian machinery as a14_0 and a14_5; this is why the manifold-correct version
matters and the early Euler-angle formulations drifted.

Visual-inertial odometry (VIO) fuses these IMU factors with visual factors (feature reprojection
or relative pose) in one estimator, filter-based (MSCKF, Mourikis and Roumeliotis 2007) or
smoothing-based (the pre-integration factor in a fixed-lag or full smoother). The payoff is what
the IMU adds that vision cannot: metric scale (a monocular camera recovers translation only up to
scale, as the multi-view assignment shows; the accelerometer's gravity-referenced measurements
fix it), robustness through motion blur and textureless stretches where features vanish, and a
high-rate pose estimate between frames. VIO is what runs on phones (ARKit, ARCore) and drones.

Read: Forster et al., "On-manifold preintegration for real-time visual-inertial odometry" (T-RO
2017); Mourikis and Roumeliotis, "A multi-state constraint Kalman filter for vision-aided inertial
navigation" (ICRA 2007); Qin, Li, Shen, "VINS-Mono" (T-RO 2018) for a complete monocular VIO.

## Time synchronization and extrinsic/temporal calibration

Multi-sensor fusion assumes the sensors agree on two things: where each is mounted relative to the
others (extrinsic calibration), and when each measurement was taken (temporal calibration). Both
are easy to get slightly wrong and expensive to get wrong.

Extrinsic calibration is the rigid transform between sensor frames - camera to IMU, camera to
LiDAR, camera to camera. The camera-geometry assignment (a11_5a) uses known extrinsics to place
cameras in a common rig; recovering them from data is calibration. The standard approach observes
a shared target (a checkerboard, an AprilTag grid) or shared motion and solves for the transform
that makes the sensors' observations consistent - a bundle-adjustment-style optimization over
$SE(3)$, again on the manifold.

Temporal calibration is the subtler one. Sensors are rarely hardware-triggered together, so there
is an unknown time offset between their clocks, and a few milliseconds matters: at typical driving
or hand-held speeds, a 5 ms skew between a camera and an IMU moving at 1 m/s is 5 mm of position
error injected into every fused measurement, and it biases the whole trajectory systematically
rather than averaging out. The fix is to treat the time offset as another variable in the
calibration optimization (Kalibr, Furgale et al. 2013, estimates the camera-IMU transform and the
time offset jointly by modeling the trajectory as a continuous-time spline). This ties back to the
camera-geometry assignment's temporal-offset exercise: the offset is not a nuisance to ignore but
a parameter to estimate.

Read: Furgale, Rehder, Siegwart, "Unified temporal and spatial calibration for multi-sensor
systems" (IROS 2013) and the Kalibr toolbox; Rehder et al. on camera-IMU calibration.

## Place recognition and loop-closure detection

The factor-graph assignment (a14_5) shows what a loop closure does: one edge tying a late pose back
to an early one, distributed by the optimizer to correct accumulated drift. It assumes the loop
closure is given. Finding it is a separate problem - place recognition - and it is what makes
long-term SLAM possible, because without re-recognizing visited places the trajectory drifts
without bound.

The problem is to decide, for the current view, whether it matches any past view in the map, fast
enough to query thousands of past frames in real time and reliably enough that a false match (which
adds a wrong edge and corrupts the whole graph, the same catastrophic failure as a bad data
association in EKF-SLAM) is rare. The classical solution is the bag-of-visual-words model: quantize
local features (the kind the multi-view front-end matches) against a precomputed vocabulary, and
represent each image as a histogram of word occurrences, so two images are compared by a cheap
vector similarity. DBoW2 (Galvez-Lopez and Tardos 2012) is the standard implementation, with an
inverted index for fast retrieval and a geometric-verification step (a RANSAC fundamental-matrix
check, exactly the multi-view assignment's estimator) to reject false matches before an edge is
added.

The learned successors replace the hand-built vocabulary with a global descriptor trained for
place recognition: NetVLAD (Arandjelovic et al. 2016) makes the bag-of-words aggregation
differentiable and learns it end to end, and modern transformer-based retrieval descriptors push
robustness to viewpoint and appearance change (day/night, seasons) much further. The pattern is the
same - a compact per-image descriptor plus geometric verification - with the descriptor learned
rather than engineered.

Read: Galvez-Lopez and Tardos, "Bags of binary words for fast place recognition in image sequences"
(T-RO 2012); Arandjelovic et al., "NetVLAD: CNN architecture for weakly supervised place
recognition" (CVPR 2016); Cummins and Newman, "FAB-MAP" (IJRR 2008) for the probabilistic
formulation.

## From classical to learned geometry

This module is deliberately the classical, geometric side of the field: explicit Lie-group state,
hand-derived Jacobians, estimators with provable structure. The rest of the repo is the learned
side - networks that regress geometry directly. The two are converging in 2026, and it is worth
seeing where they meet.

The multi-view assignment estimates two-view relative pose and a sparse point cloud from
correspondences through the eight-point algorithm, RANSAC, and triangulation. The learned analogue
regresses dense geometry directly from images: DUSt3R (Wang et al. 2024) and its successors take an
image pair and output per-pixel pointmaps in a common frame, collapsing matching, triangulation,
and relative pose into one forward pass, and VGGT (2025) extends this to many views with a
transformer that outputs cameras, depth, and points jointly. The repo's geometry-foundation-models
assignment (a10_5) covers these. They do not replace the geometry - the output is still 3D points
and camera poses - but they replace the hand-built front-end with a learned one that is robust where
correspondence-based methods fail (textureless regions, wide baselines, few views).

The same split appears in autonomous-driving perception. The BEV assignments (a11_5) lift camera
features into a bird's-eye-view grid with a learned depth distribution, where classical methods
would use calibrated geometry and explicit triangulation across the rig. And modern systems
increasingly put a learned front-end (feature matching like SuperGlue/LightGlue, learned
descriptors, learned depth) in front of a classical back-end (the same factor-graph optimization
this module builds), keeping the optimizer's guarantees while replacing its brittle hand-engineered
inputs. The factor graph in a14_5 is not obsolete; it is the back-end that learned front-ends still
feed. Knowing both sides - what the geometry must satisfy, and where a network does the front-end
job better - is the point of building the classical core by hand and reading about the learned
parts.

Read: the repo's a10_5 (geometry foundation models) and a11_5 (camera geometry and BEV); Wang et
al., "DUSt3R: geometric 3D vision made easy" (CVPR 2024); Sarlin et al., "SuperGlue" (CVPR 2020) and
Lindenberger et al., "LightGlue" (ICCV 2023) for learned matching feeding a classical back-end.
