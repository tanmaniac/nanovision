# A9 — NeRF: Neural Radiance Fields

Validation report for the nanovision curriculum. Research current as of June 2026.

---

## 1. Key concepts a student must learn

### The volume rendering integral and its discretization

The continuous rendering integral computes the expected color of a ray r(t) = o + td as:

```
C(r) = ∫[t_n to t_f] T(t) · σ(r(t)) · c(r(t), d) dt
```

where `T(t) = exp(-∫[t_n to t] σ(r(s)) ds)` is accumulated transmittance and `σ` is volume density.

The quadrature approximation divides the ray into N intervals with midpoint samples. Each sample gets alpha value `α_i = 1 - exp(-σ_i δ_i)` where `δ_i` is the interval width. The rendered color is:

```
C_hat(r) = Σ_i T_i · α_i · c_i,   T_i = Π_{j<i} (1 - α_j)
```

This is exactly alpha compositing over ordered segments. Students must derive this discretization themselves from the continuous form - it is the single most important equation in the assignment.

### Alpha compositing along rays

The connection to classical Porter-Duff alpha compositing is direct and should be made explicit. Each sample (color c_i, opacity α_i) composites front-to-back, with `T_i` playing the role of "how much light has not yet been absorbed." The student should verify that summing all `T_i * α_i` equals 1 when the ray passes through opaque matter, and less than 1 for transmissive or partially sampled rays.

### Ray generation from camera intrinsics and extrinsics

For each pixel (u, v) in an image with intrinsic matrix K (focal lengths f_x, f_y, principal point c_x, c_y):

1. Lift pixel to normalized camera coordinates: `x_c = (u - c_x)/f_x`, `y_c = (v - c_y)/f_y`, `z_c = 1`.
2. Rotate to world frame using rotation R (from camera extrinsic `[R|t]`): `d_world = R * [x_c, y_c, z_c]^T`, normalized.
3. Ray origin is camera center `o = -R^T t`.

Students with SfM/SLAM backgrounds already own this derivation. The assignment should state this explicitly - camera geometry is prerequisite knowledge here, not new content. The exercise should focus on getting the sign conventions and coordinate frames right in code (OpenCV vs OpenGL conventions), not on deriving the pinhole model from scratch.

### Positional encoding for high-frequency detail

Standard MLPs suffer spectral bias: they fit low-frequency functions much faster than high-frequency ones. The fix from the original NeRF paper maps each scalar input coordinate x to:

```
γ(x) = [sin(2^0 π x), cos(2^0 π x), ..., sin(2^{L-1} π x), cos(2^{L-1} π x)]
```

Applied to each of the 3 position coordinates (typically L=10) and 2 viewing direction components (L=4), this gives a 60+24=84-dimensional input. The NeurIPS 2020 paper by Tancik et al. ("Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains") provides the theoretical justification via the neural tangent kernel: the Fourier mapping converts the NTK from a low-bandwidth non-stationary kernel to a stationary one with controllable bandwidth. Students should implement this as a fixed (non-trainable) layer and verify via gradcheck that gradients flow through it correctly.

### Coarse/fine hierarchical sampling

The original NeRF trains two MLPs. The coarse network uses N_c stratified uniform samples per ray. The weights w_i = T_i * α_i from the coarse pass form a probability distribution over the ray (after normalization). The fine network then draws N_f additional samples from this distribution (inverse CDF sampling), concentrating computation near surfaces. Final rendering uses all N_c + N_f samples.

For the tiny-NeRF variant, the coarse/fine split is optional. A single-network version with ~64-128 stratified samples is sufficient to learn the concept. The hierarchical scheme should be mentioned and optionally implemented as an extension, because it appears in every serious NeRF codebase and in Instant-NGP's proposal network.

### The MLP radiance field

The network maps (γ(x), γ(d)) -> (σ, RGB). The original architecture has density σ dependent only on position x (not view direction), while color c is view-dependent. This factorization enforces physically plausible geometry: a point has one density but can have different appearance from different angles (specular reflection, etc.). The skip connection at layer 5 (re-injecting γ(x)) is a standard implementation detail worth noting. For tiny-NeRF, a shallower MLP (4-6 layers, 128-256 hidden units) is sufficient.

---

## 2. Mechanisms to implement from scratch

### 2a. Positional encoding

**Task:** Implement `PositionalEncoding(L)` as a PyTorch `nn.Module` with no learnable parameters. For input tensor of shape `[..., D]`, output shape is `[..., 2*L*D]`.

**Verifiable check:** `torch.autograd.gradcheck` on a random input. Confirm output frequencies span `[1, 2^(L-1)]` via FFT on a 1D test signal.

### 2b. Radiance field MLP

**Task:** Implement `NeRFMLP(pos_L, dir_L, hidden_dim, n_layers)` that takes `(positions, directions)` tensors and outputs `(density, rgb)`. Density must go through a softplus or ReLU; RGB through sigmoid.

**Verifiable check:** Forward pass shape test with batch of rays x samples: input `[N_rays, N_samples, 3]` for positions, output `(σ: [N_rays, N_samples], c: [N_rays, N_samples, 3])`.

### 2c. Volume rendering

**Task:** Implement `volume_render(sigmas, colors, deltas)` that returns per-ray color using the quadrature formula. No library calls to differentiable rendering utilities.

**Verifiable check:**
- With one very opaque sample and transparent others, rendered color equals that sample's color.
- With all σ=0, rendered color is background (black or configurable).
- `gradcheck` on a small batch.
- Weights `T_i * α_i` sum to ≤1 for all rays; sum approaches 1 as σ → ∞.

### 2d. Ray sampling

**Task:** Implement `stratified_sample_rays(H, W, focal, c2w, near, far, N_samples)` that returns `(rays_o, rays_d, z_vals)` all from scratch using the pinhole model. Also implement `sample_along_rays(rays_o, rays_d, z_vals)` to compute 3D sample positions.

**Verifiable check:** For a known camera pose (identity), verify that ray through image center points along -z (or +z, depending on convention), and corner rays have the correct angle relative to focal length.

### 2e. Full tiny-NeRF training loop

**Target dataset:** The classic tiny-NeRF dataset (100 images of the Lego bulldozer at 100x100 resolution from the original NeRF Blender synthetic scenes). This fits comfortably on a 12GB GPU and trains in under 30 minutes at tiny scale.

**Minimum viable pipeline:**
1. Load images + poses (the `tiny_nerf_data.npz` file used in the original NeRF notebook is ~3MB).
2. For each training step, sample a batch of rays across all training images.
3. Sample points along rays, query MLP, volume-render.
4. MSE loss on pixel colors. Adam optimizer.
5. Every N steps, render a held-out view and compute PSNR.

**Verifiable check:**
- Overfit one batch (8 rays, 100 steps) to near-zero loss. This must work before training full scenes.
- After full training (~20k steps), held-out PSNR should exceed 20 dB on the Lego scene - the original tiny-NeRF notebook reaches ~22 dB.
- Rendered held-out view is visually recognizable as the Lego scene (not just a gray blob).

---

## 3. Assessment of the draft scope

### What is right

The draft scope is correct on the core mechanisms. All five elements it lists (positional encoding, MLP radiance field, volumetric rendering, fit tiny scene, render held-out view, coarse/fine sampling) belong in the assignment. The tiny Lego synthetic scene is the right vehicle: it is the canonical reference, the dataset is freely available in a single small file, and 100x100 resolution trains fast enough that students can iterate.

The stated purpose - "make the learner feel why splatting won (slow per-ray MLP queries)" - is pedagogically sound and honest. As of 2026, 3D Gaussian Splatting dominates real-time novel view synthesis benchmarks, and NeRF's role is primarily foundational. The assignment framing should be explicit about this: students are learning NeRF because it introduces volume rendering from first principles, not because it is the current production method.

### Camera geometry framing

The draft mentions "camera geometry introduced cleanly here and reused by A11.5a." This is correct that the geometry should appear here. However, given the target learner's SfM/SLAM background, the camera model should be a one-paragraph recap with the sign-convention warning (OpenCV vs OpenGL vs NeRF's convention for the c2w matrix), not a ground-up derivation. The actual learning content is the volume rendering and MLP, not the pinhole model. Treat ray generation as "apply your existing knowledge to a new code context."

### Missing: the spectral bias motivation

The draft mentions positional encoding but does not frame it as a solution to a specific failure mode. Students should briefly observe that training without positional encoding produces blurry results, then understand why (spectral bias / NTK bandwidth argument from Tancik et al. 2020). Omitting this leaves the encoding looking like an arbitrary engineering trick.

### Missing: the transmittance / Beer-Lambert connection

The formula T(t) = exp(-∫σ ds) is Beer-Lambert's law from optics/radiometry. Making this connection explicit (it also appears in atmosphere rendering, CT reconstruction, and radar) helps students recognize the equation in other contexts. It takes one sentence to say.

### Coarse/fine: mention but do not require

Hierarchical sampling is worth implementing as an optional extension but should not be a graded requirement for the tiny-NeRF variant. The core insight - that importance sampling concentrates compute near surfaces - can be explained and tested separately. The complexity cost of wiring up two networks and inverse-CDF sampling is high relative to the conceptual payoff at this stage.

### Instant-NGP hash encoding: mention as context, not implement

Instant-NGP (Muller et al., SIGGRAPH 2022) replaces sinusoidal positional encoding with a multiresolution hash table: feature vectors stored at voxel corners, looked up via hash functions, concatenated across resolution levels. Training time drops from hours to seconds. Students should know this exists and understand why it helps (the hash table is a learned spatial index that concentrates representational capacity near surfaces). However, implementing a multiresolution hash table from scratch is a separate assignment-sized effort. The right treatment is a one-page conceptual comparison after the student has finished tiny-NeRF: "what would you need to change to get Instant-NGP behavior?"

### Mip-NeRF aliasing: mention as limitation, not implement

Mip-NeRF (Barron et al., ICCV 2021) addresses the aliasing that appears when a NeRF trained at one scale is evaluated at another: rays are widened to cones, and the positional encoding is integrated over conical frustums (integrated positional encoding, IPE) rather than evaluated at point samples. Zip-NeRF (Barron et al., ICCV 2023) combines IPE with hash encodings for the current best-known NeRF-family result on standard benchmarks. For this assignment, mentioning aliasing as an observable artifact (visible when zooming in or rendering at different resolutions than training) and pointing to mip-NeRF as the fix is sufficient. No implementation needed.

### Is "prequel to splatting" the right pedagogical role in 2026?

Yes, but it needs to be stated without ambiguity. NeRF is worth one full assignment because:
1. Volume rendering (the integral, discretization, alpha compositing) is a fundamental technique that also appears in occupancy networks, NeuS/neural SDF, medical imaging, and atmospheric rendering.
2. The MLP-as-scene-representation paradigm is the ancestor of all implicit neural representations used through 2025.
3. Its computational cost (why it is slow) directly motivates the splatting representation in A10.

The risk is spending too much time on NeRF engineering at the expense of A10 Gaussian Splatting, which is more practically relevant in 2026. The assignment should be kept tight: tiny-NeRF in one Jupyter notebook, no hierarchical sampling required, no coarse/fine required. The payoff comes from the comparison to A10, not from building a production NeRF.

### Ordering note

The draft says camera geometry is "introduced cleanly here and reused by A11.5a." If the course has any SfM/MVG content before A9 (likely given the learner's background), camera geometry should already be established. If A9 is the first place cameras appear in the code, that is fine, but the assignment note should say "you are applying the pinhole model, not learning it."

---

## 4. Connections to other assignments

### Feeds A10 (3D Gaussian Splatting)

The rendering loop in A10 is the inversion of NeRF: instead of marching rays through a continuous density field, 3DGS rasterizes explicit 3D Gaussians onto the image plane. The key comparison to draw explicitly:

- NeRF: per-ray, per-sample MLP query. ~64-192 forward passes per pixel. Slow at inference.
- 3DGS: sort Gaussians by depth, alpha-composite front-to-back. One pass over explicit primitives. Fast at inference.

The alpha-compositing formula is identical. Students should write this side-by-side in the assignment notes so the structural equivalence is undeniable. The difference is representational, not conceptual.

The transmittance computation `T_i = Π(1 - α_j)` for j < i applies directly to 3DGS's tile-based rasterization. A9's gradcheck tests on the volume renderer can be reused to verify A10's splatting renderer on synthetic 1D data.

### Feeds A11.5d (occupancy / neural implicit surfaces)

NeRF's density σ is a soft occupancy: `σ(x) > 0` means "there is likely matter at x." Occupancy networks represent this as a hard binary classifier O(x) ∈ {0,1} or a continuous probability. The connection to NeuS (Wang et al., NeurIPS 2021) shows how SDF-based occupancy can be plugged into the same volume rendering framework: replace σ with a function derived from the signed distance `f_SDF(x)` via a Logistic/Sigmoid transform. The same rendering integral applies; only the density parameterization changes.

Students coming from A9 will recognize the rendering equation immediately when they see it in A11.5d. The assignment for A11.5d should cite A9's volume_render function as the shared infrastructure and ask students to swap in the SDF-derived density.

### Camera geometry reuse

The `stratified_sample_rays` function from A9 and the `c2w` matrix convention should be the canonical implementation reused in A10 and A11.5. These are infrastructure, not per-assignment novelties.

---

## 5. Must-read sources

**Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV 2020.** arXiv:2003.08934. The original paper; all implementation details are here including the positional encoding frequencies, MLP architecture, hierarchical sampling, and the volume rendering derivation in the appendix. Required reading.

**Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains," NeurIPS 2020.** arXiv:2006.10739. Provides the NTK-based theoretical explanation for why positional encoding works, and shows it is a special case of a broader family of random Fourier features. Read the first three sections; the full NTK derivation is optional for this course.

**Max, "Optical Models for Direct Volume Rendering," IEEE TVCG 1995.** The classical derivation of the volume rendering integral from radiative transfer. A single dense 8-page paper. Reading Section 2 (emission-absorption model) is sufficient to understand where the NeRF rendering equation comes from and why T(t) has the exponential form.

**Barron et al., "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields," ICCV 2021.** arXiv:2103.13415. Shows the aliasing artifact in vanilla NeRF and the cone-casting / integrated positional encoding fix. Read to understand NeRF's failure modes before seeing Instant-NGP and 3DGS supersede it.

**Muller et al., "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding," ACM Transactions on Graphics (SIGGRAPH) 2022.** arXiv:2201.05989. Replaces sinusoidal positional encoding with a learned multiresolution hash table; reduces NeRF training from hours to seconds. Read the encoding section (Section 3) to understand what changed architecturally and why. The official CUDA implementation is at github.com/NVlabs/instant-ngp.

**Wang et al., "NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction," NeurIPS 2021.** arXiv:2106.10689. Connects the NeRF volume rendering framework to SDF-based implicit surfaces, directly linking A9 to A11.5d. Read Sections 3-4 (the unbiased SDF-to-density mapping).

**Kerbl et al., "3D Gaussian Splatting for Real-Time Novel View Synthesis," SIGGRAPH 2023.** arXiv:2308.04079. The foundational 3DGS paper, required for A10. Read alongside A9 notes with the alpha-compositing formula side-by-side to see the shared mathematical structure.

---

## 6. 2024-2026 developments that change how this should be taught

### NeRF as foundational infrastructure, not frontier method

By 2025-2026, NeRF has been fully superseded for production novel view synthesis. Gaussian Splatting renders at 100+ FPS at equal or higher quality, and 3DGS has been adopted by industry tools (DJI Terra, Polycam, Luma AI) and standards bodies (OpenUSD, glTF). The assignment framing should not position NeRF as a competitive method. It is a pedagogical vehicle for volume rendering. State this in the assignment preamble.

### Zip-NeRF closed the gap between mip-NeRF and hash encodings

Zip-NeRF (Barron et al., ICCV 2023, arXiv:2304.06706) combines integrated positional encoding (from mip-NeRF) with multiresolution hash tables (from Instant-NGP), cutting error rates by 8-77% versus either predecessor and training 24x faster than mip-NeRF 360. This is the current state-of-the-art for NeRF-family methods. Mention it as the endpoint of the NeRF development arc, after which 3DGS became the dominant paradigm.

### Generative NeRFs and feed-forward reconstruction

2024-2025 saw a significant shift toward feed-forward reconstruction models (PixelSplat, MVSplat, DUST3R, MASt3R) that predict a 3D representation from 2 or 3 input images in a single forward pass without per-scene optimization. These are not NeRF-family methods technically, but they share the volume rendering renderer. Students should know this exists to understand that the slow per-scene optimization of NeRF is not an inherent property of volumetric rendering, just of the original representation.

### NerfBaselines (2024) for reproducible evaluation

The NerfBaselines benchmark (arXiv:2406.17345, 2024) standardizes evaluation protocols across NeRF and 3DGS methods, revealing that many published PSNR improvements were artifacts of inconsistent evaluation (image resolution, crop boundaries, background handling). For the assignment, this means the target PSNR for tiny-NeRF should be stated with the exact evaluation protocol used, not just a number from a paper that may have used different settings.

### The "coarse network is wasteful" finding

Multiple 2023-2024 papers (TermiNeRF, ProNeRF) showed that the original coarse/fine scheme wastes significant compute on the coarse pass. Modern NeRF implementations either skip the coarse network entirely (using a single network with proposal-based sampling, as in mip-NeRF 360 and nerfstudio's Nerfacto) or use a very lightweight coarse occupancy grid. For the assignment, this validates the decision to not require coarse/fine as a graded component: the original scheme is pedagogically clear but architecturally suboptimal.

### Volume rendering digest

Condor et al., "Volume Rendering Digest (for NeRF)," arXiv:2209.02417 (2022) is a clean standalone derivation of the discretized rendering equation. It was written specifically as a teaching resource and is more self-contained than the NeRF appendix. Recommend it to students who want the derivation spelled out step-by-step before reading the original paper.

---

*Validated June 2026. Sources: Mildenhall et al. 2020 (arXiv:2003.08934), Tancik et al. 2020 (NeurIPS 2020), Barron et al. 2021 mip-NeRF (arXiv:2103.13415), Muller et al. 2022 Instant-NGP (SIGGRAPH 2022), Wang et al. 2021 NeuS (NeurIPS 2021), Barron et al. 2023 Zip-NeRF (ICCV 2023), Kerbl et al. 2023 3DGS (SIGGRAPH 2023), Condor et al. 2022 Volume Rendering Digest (arXiv:2209.02417).*
