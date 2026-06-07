# A10 — 3D Gaussian splatting: validation report

## 1. Key concepts the student must learn

### 3D Gaussian parameterization

Each Gaussian carries: a 3D mean position (3 floats), a covariance encoded as a 3D rotation quaternion (4 floats, normalized) and a scale vector (3 floats, log-space), an opacity scalar (sigmoid-activated), and spherical harmonic coefficients for color. The covariance factored as `C = R S S^T R^T` (where `S = diag(s)`) is the critical design choice: it keeps the covariance matrix positive semi-definite by construction and makes it differentiable through autograd. Storing the raw covariance 6-vector and reconstructing it inside the forward pass is the standard pattern in pure-PyTorch implementations.

### Projection of a 3D Gaussian to a 2D Gaussian via EWA / projective Jacobian

A 3D Gaussian projects to a 2D Gaussian in image space. The derivation comes from Zwicker et al.'s EWA Volume Splatting (2001) and EWA Splatting (2002): linearize the perspective projection around the Gaussian mean in camera space using its Jacobian `J` (a 2x3 matrix). The 2D covariance is then `Sigma_2D = J W Sigma_3D W^T J^T`, where `W` is the camera rotation and the third row/column are dropped to get the 2x2 screen-space covariance. This is the mathematical heart of the method and must be implemented by hand - no `torch.distributions` shortcut.

### Front-to-back alpha compositing

Gaussians are sorted by depth (front to back). The contribution of each Gaussian at a pixel is `alpha_i = opacity_i * G_i(x)`, where `G_i` is the 2D Gaussian evaluated at pixel `x`. Colors accumulate as `C = sum_i c_i * alpha_i * prod_{j<i} (1 - alpha_j)`. This is identical in structure to the transmittance accumulation in NeRF (from A9), but operates over sorted 2D Gaussians rather than along a ray with quadrature. The early termination condition (transmittance < 1e-4) is the same idea.

### Spherical harmonics for view-dependent color

Each Gaussian stores SH coefficients per color channel. Degree 0 (1 coefficient per channel, 3 total) gives view-independent color. Degree 3 (16 coefficients per channel, 48 total) captures specular-like effects. For a course implementation, degree 0 or degree 1 (4 coefficients per channel) is enough to demonstrate the mechanism. The view direction to the Gaussian center is evaluated against the SH basis functions at render time.

### Tile-based rasterization

The original implementation bins Gaussians into 16x16 pixel tiles and sorts per tile by depth. A pure-PyTorch version must either loop over tiles in Python (slow) or use larger tiles (64x64 is common) to reduce loop overhead. This is a practical limitation: the pure-PyTorch splatter will be slower than the CUDA version by 10-50x, but it will converge and reach competitive PSNR on a small synthetic scene.

### The optimization and densification loop

The original paper interleaves gradient descent (Adam, separate learning rates for position/SH/opacity/scale/rotation) with an adaptive density control (ADC) procedure:
- Every 100 iterations, Gaussians with high average 2D gradient magnitude are selected for densification.
- Small Gaussians (under-reconstructed regions) are cloned in place.
- Large Gaussians (over-reconstructed regions) are split into two with smaller scales sampled from the parent.
- Gaussians with opacity below a threshold are pruned.
- Opacity is periodically reset to encourage pruning of floaters.

This is a heuristic outer loop - it is not differentiable and cannot be autograd'd through. It is the part of the method that is hardest to re-derive from first principles and hardest to debug. See the note in section 3 about scope.

---

## 2. Mechanisms to implement from scratch

### 2.1 Scale+rotation -> covariance -> 2D covariance (EWA projection)

**Task:** Given a set of N=100 Gaussians with random `(q, s)` parameters and a camera `(K, R, t)`, compute the N 2x2 screen-space covariances. Run `torch.autograd.gradcheck` on the covariance computation with `q` and `s` as inputs, using double precision. Pass if the relative error is below 1e-4.

**What to implement:** `build_covariance_3d(q, s)`, `project_cov_to_2d(cov3d, cam_R, K, means_cam)`.

### 2.2 Alpha compositing forward pass

**Task:** Render a single 64x64 image from N depth-sorted Gaussians with fixed parameters (no backprop yet). Assert output shape is `(64, 64, 3)` and values in `[0, 1]`. Test with 1 Gaussian centered at the image center: verify the rendered image has a visible blob.

**What to implement:** `render_image(means2d, cov2d, colors, opacities, H, W)` using a nested loop or vectorized pixel distance computation.

### 2.3 Autograd through the forward

**Task:** With the full forward pass (3D mean + pose -> 2D Gaussian -> composite), run `torch.autograd.gradcheck` on opacity and position parameters with a 4x4 target image and N=10 Gaussians. Pass if the check passes at double precision.

### 2.4 Overfit a single image

**Task:** Initialize N=500 Gaussians randomly in front of a calibrated camera. Optimize `(means, log_s, q, opacity, color)` with Adam for 2000 steps against a single target image (e.g., a colorful synthetic object). Loss: L1 + 0.2 * D-SSIM (as in the paper). Assert that final PSNR >= 20 dB. Log PSNR every 100 steps; plot the curve.

### 2.5 Novel view synthesis from multiple posed images

**Task:** Load 5-10 posed images of a synthetic object (e.g., from NeRF-Synthetic Lego or a custom toy scene generated with Blender). Fit N=2000 Gaussians with no densification for 3000 steps. Render a held-out view. Report PSNR vs. the A9 NeRF on the same scene. Show that the 3DGS render at inference is measurably faster (wall-clock time for a single forward pass).

---

## 3. Assessment of the draft scope

### What is right

The draft correctly identifies the core mechanism: parameterization via scale+rotation, the EWA Jacobian projection, tile/alpha compositing, SH color, and the optimization loop. These are the right things to teach. The comparison with A9 NeRF at inference time is well-motivated.

### What needs adjustment

**Densification scope.** The draft lists "the optimization/densification loop" as a key concept to implement. Densification (clone/split/prune with opacity reset) is a heuristic outer loop with no gradients flowing through it. It cannot be validated by gradcheck, and debugging it on a small scene is time-consuming without producing insight proportional to effort. The recommendation is to treat it as a "read-and-understand" item: students read the ADC procedure in the paper, implement a simplified version (e.g., periodic pruning of low-opacity Gaussians only) for completeness, but the graded tasks focus on the differentiable forward pass. This is consistent with the course philosophy of "implement the mechanism."

**Pure-PyTorch performance expectations.** A pure-PyTorch splatter on a 12GB RTX 4080 can fit N=2000-5000 Gaussians and converge on a small synthetic scene (64x64 or 128x128 images). Without CUDA tiles, each render step on a 128x128 image with 2000 Gaussians takes roughly 0.5-2 seconds in Python loops. Training 3000 steps will take 30-60 minutes. This is feasible but should be stated explicitly so students do not expect real-time performance. The inference speed comparison with NeRF still holds because NeRF's inference is also slow without CUDA.

**The "visibly faster inference than A9 NeRF" claim.** This is correct and measurable even in pure-PyTorch, but the mechanism is different: 3DGS renders by evaluating a small number of 2D Gaussians per pixel, while the A9 NeRF runs an MLP hundreds of times per ray. The student should time both and report the ratio; it will likely be 5-20x faster even without CUDA tiles.

**SH degree in scope.** For a 12GB budget with a small scene, SH degree 0 (constant color per Gaussian) or degree 1 is sufficient to demonstrate the concept. Degree 3 (48 coefficients per Gaussian, the full original) is not needed for learning the mechanism and adds memory without clarity.

**2026 context to add:**

- **2DGS (Huang et al., SIGGRAPH 2024):** 2D oriented Gaussian disks as surface primitives. This is directly relevant to the target learner (robotics/AV) because it produces geometrically accurate surfaces that support meshing, normal estimation, and localization. It should be mentioned as a natural extension and contrast: 3D Gaussians are good at view synthesis, 2D Gaussians are better for surface reconstruction.

- **gsplat (Ye et al., 2024, JMLR 2025):** The nerfstudio library exposes both CUDA-accelerated and native-PyTorch backends for 3DGS. Students should know it exists and use it as a reference for correctness, but are forbidden from using its rasterizer in their own implementation (consistent with course rules). The pure-PyTorch backend is a useful reference for understanding what is achievable without CUDA.

- **Feed-forward / generalizable splatting (pixelSplat CVPR 2024, MVSplat ECCV 2024, AnySplat SIGGRAPH Asia 2025):** These methods predict Gaussian parameters from image features in a single forward pass, eliminating per-scene optimization. They are directly relevant to the AV engineer: scene reconstruction from a camera rig without optimization at inference. This should be a brief "where the field is going" note, not an implementation task.

- **Mip-Splatting (Yu et al., CVPR 2024 Best Student Paper):** Fixes aliasing artifacts from the 2D dilation filter in the original. Worth noting because the original 3DGS has visible artifacts at varying scales; Mip-Splatting is now the preferred baseline.

- **Bundle-adjusting Gaussian splatting:** Several 2024 papers (BAGS, InstantSplat, GloSplat) jointly optimize Gaussian parameters and camera poses, making the connection to bundle adjustment explicit. For a learner with a strong SfM background, this framing - 3DGS optimization as a generalization of bundle adjustment over a radiance field rather than over point correspondences - is the clearest pedagogical bridge.

### What to cut

The draft scope is otherwise well-calibrated. "Fit a small scene from posed images" and "render a novel view" are the right deliverables. Do not add SLAM or relocalization as implementation tasks; they belong in the AV module.

---

## 4. Connections to other modules

**Depends on A9 (NeRF / volume rendering):** The alpha compositing formula in 3DGS is the same emission-absorption model as NeRF. The transmittance product `T_i = prod_{j<i}(1 - alpha_j)` is identical; what changes is that the density contributions come from sorted 2D Gaussians rather than MLP-evaluated samples along a ray. Students who understand the A9 volume rendering integral will find the A10 compositing formula immediately recognizable.

**Depends on A9 camera geometry:** The projection of 3D Gaussian means through `K [R | t]` is the same as projecting 3D points. The Jacobian of perspective projection used in the EWA step is the same Jacobian the student computed for bundle adjustment. This is the key algebraic link: the covariance projection `J W Sigma W^T J^T` is a linearized version of exactly the point-projection Jacobian used in reprojection-error minimization.

**Feeds into the AV perception module (3D representations):** 3DGS is now a standard scene representation in AV sensor simulation (LiDAR-3DGS, TCLC-GS, EmerNeRF). The feed-forward generalizable methods (MVSplat, AnySplat) are direct competitors to NeRF-based reconstruction from camera rigs. 2DGS is used for mapping in visual SLAM systems. The student who can implement the differentiable forward pass has the foundation to understand all of these.

---

## 5. Must-read sources

1. **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM TOG / SIGGRAPH 2023.** The primary reference. Read sections 4 (representation), 5 (rendering), and 6 (optimization). The adaptive density control is in section 5.2. [arXiv:2308.04079]

2. **Zwicker et al., "EWA Volume Splatting," IEEE Visualization 2001.** The mathematical origin of the 3D-to-2D covariance projection. Specifically the local affine approximation (Eq. 9 in that paper) and the dropping of the third row/column to get a 2x2 screen-space covariance. The 2002 IEEE TVCG paper "EWA Splatting" is the journal version and is slightly more polished but covers the same derivation.

3. **Huang et al., "2D Gaussian Splatting for Geometrically Accurate Radiance Fields," SIGGRAPH 2024.** Required 2026 context. Shows that replacing 3D volumetric Gaussians with 2D surface disks gives geometrically consistent normals and better mesh extraction - directly relevant for AV mapping and SLAM.

4. **Ye et al., "gsplat: An Open-Source Library for Gaussian Splatting," JMLR 2025 (arXiv:2409.06765).** Read for: (a) the clean API design showing what primitives are needed, (b) the description of the native PyTorch backend (useful reference for the student's own implementation), (c) performance benchmarks to calibrate expectations.

5. **Yu et al., "Mip-Splatting: Alias-Free 3D Gaussian Splatting," CVPR 2024.** Addresses the main artifact of the original (dilation/aliasing under scale change). The 3D smoothing filter and 2D Mip filter are conceptually simple extensions of the base method. Worth reading because the original 3DGS code was patched to incorporate this in August 2024.

6. **Charatan et al., "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction," CVPR 2024 Oral / Best Paper Runner-Up.** Representative of the feed-forward paradigm: predicts Gaussian parameters from an image pair in one forward pass with no per-scene optimization. Directly relevant to the AV engineer. [arXiv:2312.12337]

7. **Chen et al., "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images," ECCV 2024.** Follows pixelSplat with a cost-volume architecture, 10x fewer parameters, and better cross-dataset generalization. Read alongside pixelSplat to understand the trajectory of feed-forward splatting.

**Flagged omissions in the original draft scope:** The draft does not mention EWA Splatting/Zwicker (a gap - it is the mathematical prerequisite for the Jacobian derivation), 2DGS (relevant for surfaces and the AV module), or gsplat (useful reference implementation). Mip-Splatting and the feed-forward methods (pixelSplat/MVSplat) are also absent.

---

## 6. 2024-2026 developments that change how this should be taught

**Anti-aliasing is now baseline (Mip-Splatting, CVPR 2024).** The original 3DGS has visible dilation artifacts when the camera is moved far from the training distribution. Mip-Splatting fixes this with a 3D frequency constraint and a 2D box filter. The official 3DGS repository was updated to include this in August 2024. A course assignment should use Mip-Splatting semantics (or at least note that degree-0 SH + no dilation filter reproduces the artifact).

**2DGS for surfaces (SIGGRAPH 2024).** 3D Gaussians represent volumetric density well but produce inconsistent normals and noisy geometry. 2D Gaussian disks (surfels) have an intrinsic surface normal, produce consistent depth across views, and support direct mesh extraction. For the AV/robotics learner, 2DGS is arguably more useful than 3DGS because it enables downstream tasks (localization, collision avoidance, map representation) that require reliable geometry. The teaching framing should include 2DGS as a brief extension: "swap the primitive, add a normal-consistency loss, get surfaces."

**gsplat as the community standard (2024-2025).** By 2025, gsplat (nerfstudio) is the de facto library for 3DGS research, analogous to what nerfstudio is for NeRF. It exposes a native-PyTorch backend that students can read for reference. Benchmarks show it is 10-44% faster and uses 44% less memory than the original implementation. Students should know it exists; they should not use its rasterizer.

**Feed-forward / generalizable splatting eliminates per-scene optimization (CVPR/ECCV 2024, SIGGRAPH Asia 2025).** pixelSplat, MVSplat, and AnySplat predict Gaussian parameters from image features in a single forward pass. This changes the framing of 3DGS from "per-scene optimization" (like NeRF) toward "feed-forward 3D perception" (like depth estimation or stereo). For an AV engineer, this matters: AnySplat (SIGGRAPH Asia 2025) works on uncalibrated image collections and does not require COLMAP poses. The course should at minimum note this trend.

**Bundle-adjusting Gaussian splatting (2024).** Multiple 2024 papers (BAGS, InstantSplat, GloSplat) jointly optimize Gaussian parameters and camera poses. This closes the loop with A9's camera geometry module and directly parallels bundle adjustment: instead of minimizing reprojection error over point correspondences, you minimize photometric error over the rendered Gaussian field with respect to both scene parameters and camera parameters. For a learner with SfM background, this is the clearest conceptual bridge and should be mentioned explicitly.

**3DGS in AV simulation and SLAM (2024-2025).** Systems like LiDAR-3DGS, TCLC-GS, and EmerNeRF use 3D Gaussians as the scene representation in sensor simulators for autonomous driving. 3DGS-based SLAM (SplatSLAM, MonoGS, LGS-SLAM) is an active research area. These are downstream applications of the core mechanism; they belong in the AV module but motivate why 3DGS is worth learning in depth.

**Compression and standardization (2025-2026).** SPZ v4 (May 2026) achieves 90% compression of Gaussian parameters. MPEG is standardizing Gaussian splatting codecs. The glTF KHR_gaussian_splatting extension is expected Q2 2026. These are infrastructure developments; they do not change what should be taught in a 2026 course, but they signal that 3DGS is now a production format, not just a research prototype.
