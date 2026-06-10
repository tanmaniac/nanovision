# A9 - NeRF (the prequel): build plan

Status: draft for expert review. Build subagent reads this whole file plus
`agent_build_guide.md` and mirrors the exemplar `assignments/a06_0_flow_matching` for
layout.

## What this assignment teaches

NeRF represents a scene as an MLP mapping a 3D position (and view direction) to a volume
density and color, and renders an image by marching rays through that field and alpha-
compositing samples. A9 is a pedagogical vehicle for volume rendering, not a competitive method:
by 2026 Gaussian splatting (A10) renders 100x faster at equal quality. The reason to build NeRF
is that the volume rendering integral, its discretization, and front-to-back alpha compositing
are foundational - they reappear in splatting (A10), occupancy / neural SDF (A11.5d), and
medical/atmospheric rendering. A9 also lets the learner feel why splatting won: NeRF needs ~64-192
MLP queries per pixel.

The three core mechanisms the student implements: the Fourier positional encoding (motivated by
the spectral-bias failure of a raw-coordinate MLP), the discretized volume renderer (the single
most important equation), and ray generation from camera intrinsics/extrinsics. The MLP and the
toy scene are provided.

## Convention decision (read this - the main trap)

The original NeRF uses the OpenGL camera convention: camera looks down -z, c2w matrices from the
Blender synthetic scenes. The rest of THIS course uses the OpenCV convention (camera looks down
+z), established in `nanovision.geometry` (A11.5a): `project_points` maps camera-frame points with
+z forward. A9 uses the OpenCV convention throughout, to stay consistent with the geometry/AV
stack it feeds. State this explicitly in the README: "the original NeRF uses OpenGL -z; this course
uses OpenCV +z for consistency with the camera-geometry module, so the ray directions and the
toy-scene poses both follow +z-forward." Getting this wrong silently flips the scene front-to-back.

## Reused, already in place (do NOT rebuild)

- `from nanovision.geometry import unproject, make_transform, apply_transform, invert_transform`:
  the A11.5a camera primitives (OpenCV convention). `unproject(px, depth, K)` lifts pixels to
  camera-frame 3D points; the transform helpers build/apply/invert 4x4 c2w matrices. Reuse these
  for ray generation - the pinhole model is prerequisite knowledge here, not new content.
- `nanovision.primitives` / `nanovision.trainer` / gradcheck as usual.

## Shared deliverable A9 OWNS (new shim)

A9 creates `render.py` (volume_render) and `rays.py` (ray generation), which A10 and A11.5d
reuse. These are shared OWNED files: import them ONLY via `nanovision.volume` (never bare), per
the import rule. The shim `nanovision/volume.py` is created at the orchestrator level (pointing at
`a09_nerf`'s `render` and `rays` modules) before the build; the build subagent just writes
`render.py` / `rays.py` and imports the shared symbols via `nanovision.volume` in its tests and
sibling files. `encoding.py` and `model.py` are assignment-LOCAL (import bare).

## Shapes (fix these numbers)

- Toy images: H = W = 16, 3 channels, ~6-8 posed views of one synthetic object. K is a 3x3
  pinhole intrinsic; near/far bound the scene depth.
- Rays: a batch of R rays, N = 32 stratified samples per ray.
- Positions along rays: (R, N, 3). Directions: (R, 3) broadcast to (R, N, 3).
- Positional encoding: position L_x = 6 (keep small for the toy; original uses 10), direction
  L_d = 4. With include_input, encoded position dim = 3 + 3*2*L_x = 39, direction = 3 + 3*2*L_d
  = 27.
- NeRFMLP: hidden 128, 4 layers, one skip connection re-injecting the encoded position. Outputs
  density sigma (R, N) and color rgb (R, N, 3).
- volume_render(sigmas (R,N), colors (R,N,3), deltas (R,N)) -> rendered color (R, 3), and the
  per-sample weights (R, N) for the optional fine-sampling extension and the ablation.

## Files (mirror the exemplar layout)

### `encoding.py` (holed; solution copy) - assignment-local

Hole:
- `PositionalEncoding(nn.Module).forward(self, x)`: Fourier features, no learnable parameters.
  For input (..., D), output (..., D*2*L) or (..., D + D*2*L) when `include_input=True`. Use the
  original NeRF mapping with the pi factor:
  $\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$,
  applied to each of the D input coordinates; concatenate the raw input first if include_input.
  `__init__` stores L, include_input, and the frequency bands `2^k` as a registered buffer
  (non-trainable). Only `forward` is the hole. Docstring: this fixes the MLP's spectral bias
  (low frequencies fit first); the bands let the network represent high-frequency detail.
  IMPORTANT input-range contract: the encoding's frequency schedule is only valid if inputs are
  normalized to roughly $[-1, 1]$ first. The caller (NeRFMLP) normalizes sample positions by a
  scene-bound constant (derived from the object extent / far plane) before encoding, and feeds
  unit-length directions. State this in both the encoding docstring and the MLP - an unnormalized
  position of magnitude ~4 with the top $2^{L-1}\pi$ band aliases badly. L_x is MEASURE-and-set:
  start at 6 but tie the final value to the spectral-bias ablation actually showing blur without
  the encoding (the toy object must have a sharp feature - see the scene below - or the ablation
  is vacuous).

### `render.py` (holed; solution copy) - shared OWNED (via nanovision.volume)

Hole:
- `volume_render(sigmas, colors, deltas, *, white_background=False)`: the discretized
  emission-absorption quadrature.
  - alpha: $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$, with sigma passed through a non-negativity
    activation by the CALLER (the MLP applies softplus/ReLU; volume_render takes raw nonneg
    sigmas - state this contract in the docstring so the activation is not double-applied).
  - transmittance: $T_i = \prod_{j<i}(1-\alpha_j)$ - the EXCLUSIVE cumulative product, with
    $T_0 = 1$. Implement as `T = torch.cumprod(torch.cat([ones(R,1), (1 - alpha)[:, :-1]],
    dim=1), dim=1)` - a leading column of ones, then the cumprod of all but the last
    $(1-\alpha)$. Do NOT add an epsilon inside the cumprod (it breaks the exact-equality tests),
    and do NOT use an inclusive cumprod then shift (double-counts). The `[:, :-1]` slice is the
    off-by-one sentinel; spell it out.
  - weights $w_i = T_i \alpha_i$; rendered color $C = \sum_i w_i c_i$. Return (color (R,3),
    weights (R,N)). If white_background, add $(1 - \sum_i w_i)$ as white.
  Docstring connects $T(t)=\exp(-\int\sigma\,ds)$ to Beer-Lambert's law. Note: `volume_render` is
  a plain function, not an nn.Module, so gradcheck it with `torch.autograd.gradcheck` directly
  (do not route through the module-oriented `check_gradients` helper).

### `rays.py` (holed; solution copy) - shared OWNED (via nanovision.volume)

Holes:
- `stratified_sample_rays(H, W, K, c2w, near, far, n_samples, *, perturb, generator)`: build a
  ray per pixel and stratified depth samples.
  - pixel grid -> camera-frame ray directions via the pinhole model (reuse
    `nanovision.geometry.unproject` at depth 1, or build directions
    $d_c = ((u-c_x)/f_x, (v-c_y)/f_y, 1)$ in OpenCV +z convention).
  - rotate directions to world with the c2w rotation and NORMALIZE rays_d to unit length; ray
    origin is the c2w translation (camera center). Return rays_o (R,3), rays_d (R,3, unit), and
    z_vals (R, n_samples) stratified in [near, far] (uniform bins, optional jitter when perturb).
    With unit rays_d, z is Euclidean distance along the ray and near/far are distances (not
    z-depths) - state this. The SAME unit rays_d is the direction fed to the MLP's color branch.
- `sample_along_rays(rays_o, rays_d, z_vals)`: points = rays_o[...,None,:] + rays_d[...,None,:] *
  z_vals[...,:,None] -> (R, n_samples, 3); and `deltas_from_z(z_vals)` -> consecutive differences
  `diff(z_vals)` with the last delta set to 1e10 so the final sample absorbs remaining
  transmittance (the original NeRF choice; with unit rays_d no $\lVert d\rVert$ factor is needed).
  Do NOT offer an unnormalized-rays variant - it would need a $\lVert d\rVert$ factor that the
  `deltas_from_z(z_vals)` signature cannot apply, silently breaking alpha.

### `model.py` (provided only) - assignment-local

- `NeRFMLP(pos_L, dir_L, hidden, n_layers, include_input)`: encodes positions and directions with
  `PositionalEncoding`, runs an MLP with one skip connection re-injecting the encoded position at
  the middle layer, outputs density via softplus (>=0) and view-dependent rgb via sigmoid (in
  [0,1]). Density depends only on position; color depends on position features + encoded
  direction (the NeRF factorization: one density per point, view-dependent appearance). Provided
  in full - the pedagogy is the encoding and the renderer, not the MLP plumbing.

### `config.py` (provided), `viz.py` (provided)

- config: H=W=16, n_views ~6, n_samples=32, pos_L=6, dir_L=4, hidden=128, n_layers=4, near, far,
  lr, white_background.
- viz: render the toy scene from a held-out pose before/after a short training run; show the
  spectral-bias ablation (train a few hundred steps with vs without positional encoding, render
  both); save to `out/`. Not graded. The full Lego scene and 20k-step / 22 dB PSNR target from the
  research note are explicitly OUT of scope (GPU, minutes-to-hours) - mention them in the README
  as the reference result, do not run them in tests.

### toy data: `nanovision.data.toy.nerf_synthetic_scene(...)` (NEW, provided)

Add to `nanovision/data/toy.py`. Generate a small, self-contained, CPU-cheap posed-image scene
WITHOUT any download, with GROUND TRUTH from an INDEPENDENT closed form (NOT volume_render -
avoid circularity, the expert's main concern):
- The object is a colored solid sphere (radius r, center at origin, constant interior density
  $\sigma_0$, a smooth position-dependent color $c_{\text{sphere}}$). The hard sphere boundary is
  the sharp feature the spectral-bias ablation needs.
- Place n_views cameras on a ring/arc around the object (OpenCV +z-forward c2w), shared pinhole K,
  near/far.
- Render each GT pixel by the CLOSED-FORM ray-sphere chord, written separately from volume_render:
  intersect the pixel ray with the sphere, get the interior chord length $\ell$ (0 if it misses),
  then $\alpha = 1 - e^{-\sigma_0 \ell}$ and composited color
  $C = \alpha\, c_{\text{sphere}} + (1-\alpha)\, c_{\text{bg}}$. This is the exact Beer-Lambert
  integral for a constant-density sphere - no per-sample loop, no call into volume_render. So a
  passing overfit/PSNR test proves the student's DISCRETIZED quadrature converges to the analytic
  integral, which is the real learning objective, not "matches its own renderer."
- Low res (16x16), renders in milliseconds. Return (images (n_views, H, W, 3) in [0,1], poses
  (n_views, 4, 4) c2w, K (3,3), near, far). A held-out view (last pose) is the eval target.
This scene generator is reusable by A10 (splatting fits the same posed images). Deterministic per
seed. The toy.py addition is the only edit outside the assignment dir besides the shim; flag it
for the orchestrator (no other assignment's toy data changes).

### `conftest.py`, `__init__.py`, `solution/`

Mirror the exemplar conftest (adjust the docstring file list). `solution/` holds the holed files:
`encoding.py`, `render.py`, `rays.py` (plus `__init__.py`). `model.py`, `config.py`, `viz.py`
top-level only.

## Tests (env python, CPU, seconds each; training-free exact checks preferred)

1. `test_encoding.py`: shape (..., D) -> (..., D + D*2*L) with include_input; gradcheck (float64)
   on `PositionalEncoding.forward`; and a frequency check - encode a 1D ramp, FFT the highest
   band's channel, confirm energy at the expected $2^{L-1}$ frequency (or a simpler exact check:
   the k-th sin channel of input x equals sin(2^k * pi * x)).
2. `test_volume_render.py` (the core exact checks, training-free):
   - one fully opaque sample (huge sigma) in front, others transparent -> rendered color equals
     that sample's color (atol ~1e-4).
   - all sigma = 0 -> rendered color is background (0, or white if white_background); weights sum
     to 0.
   - weights $w_i = T_i\alpha_i$ are nonneg and sum to <= 1 for every ray; as sigma -> large on a
     sample, the cumulative weight -> 1.
   - the exclusive-cumprod property: $T_0 = 1$ (the first sample is not attenuated). Assert the
     first weight equals alpha_0 exactly. Use a large-but-FINITE sigma (e.g. 1e3) for the opaque
     case, not inf.
   - the telescoping identity: $\sum_i w_i = 1 - \prod_i(1-\alpha_i)$ exactly (pins the whole
     transmittance structure in one assertion).
   - gradcheck (float64) on volume_render w.r.t. sigmas and colors, via `torch.autograd.gradcheck`
     directly (it is a function, not a module).
3. `test_rays.py`: for an identity c2w and centered principal point, the center pixel's ray
   direction is +z (OpenCV) within atol; a corner pixel's direction has the expected tangent
   $(u-c_x)/f_x$. `sample_along_rays` shapes (R,N,3); `deltas` last entry is the large constant;
   points lie on the ray (o + t d).
4. `test_mlp.py`: `NeRFMLP` forward shapes - positions (R,N,3), directions (R,3) -> sigma (R,N) >=
   0, rgb (R,N,3) in [0,1].
5. `test_overfit_ray.py`: overfit a small batch of rays from the toy scene (e.g. 64 rays, a few
   hundred steps) so the rendered pixel colors match the ground-truth pixels to near zero MSE.
   This exercises encoding -> MLP -> volume_render end to end. Report the measured floor; set the
   threshold from it (no-thrash). Keep it CPU-seconds.
6. `test_heldout_psnr.py` (default VIZ-ONLY, not graded): the graded correctness gate is the
   overfit test (5) plus the exact renderer checks (2). PSNR on a held-out view is seed-flaky in a
   CPU-seconds budget, so do NOT make it a graded assertion by default. Measure the held-out PSNR
   across ~5 seeds; only promote it to a graded threshold if the WORST seed clears a modest bar
   (~15 dB) with margin. Otherwise keep PSNR as a viz figure. Report the measured numbers either
   way.
7. `test_forbidden_imports.py`: one static scan over the holed files + solution + the
   `nanovision.volume` shim, mirroring the exemplar. Forbid importing a ready-made NeRF/renderer
   (nerfstudio, torch-ngp, pytorch3d renderer); forbid bare imports of the owned shared files
   (render.py / rays.py must be reached via nanovision.volume).

Solution mode all green; default mode fails only at the holes (NotImplementedError), except
`test_forbidden_imports` (passes both modes). Run with
`/home/tanmay/miniconda3/envs/nanovision/bin/python`.

## README (comprehensive lecture notes, per the skill, real LaTeX)

Fixed section order. Cover:
- Framing: NeRF is the volume-rendering vehicle, superseded by splatting for production (state it
  up front). Its cost (per-ray per-sample MLP queries) motivates A10.
- The volume rendering integral $C(r) = \int T(t)\sigma(r(t))c(r(t),d)\,dt$, the Beer-Lambert
  transmittance $T(t) = \exp(-\int\sigma\,ds)$, and the discretization to
  $\hat C = \sum_i T_i\alpha_i c_i$, $\alpha_i = 1-\exp(-\sigma_i\delta_i)$,
  $T_i = \prod_{j<i}(1-\alpha_j)$ - derive the discretization from the integral. State precisely:
  $\alpha_i = 1-\exp(-\sigma_i\delta_i)$ is the EXACT integral over a segment of constant $\sigma$,
  so the only approximation is the piecewise-constant assumption on $\sigma$ and $c$, not the
  per-segment formula. Make the Porter-Duff alpha-compositing equivalence explicit (this same
  compositing returns in A10 and A11.5d).
- Spectral bias and the Fourier encoding: a raw-coordinate MLP fits low frequencies first and
  renders blurry; $\gamma$ lifts coordinates to a band of frequencies so high-frequency detail is
  representable (Tancik et al. 2020, the NTK bandwidth argument in one paragraph).
- Ray generation: pinhole unprojection + c2w, the OpenCV vs OpenGL convention warning, that this
  is applying known camera geometry, not new content.
- The MLP radiance field: density-from-position, view-dependent color, the skip connection.
- Coarse/fine hierarchical sampling: explain importance sampling concentrates compute near
  surfaces (weights $w_i$ as a distribution, inverse-CDF resampling), mention it is optional and
  not graded; note modern NeRFs (mip-NeRF 360, Nerfacto) drop the original coarse network.
- Where this goes: A10 splatting reuses the SAME compositing $\sum_i T_i\alpha_i c_i$,
  $T_i=\prod_{j<i}(1-\alpha_j)$, but the $\alpha$ comes from a depth-sorted projected 2D Gaussian
  times an opacity, NOT from $1-\exp(-\sigma\delta)$ - the compositing is identical, the source of
  $\alpha$ differs and the ordering comes from a sort, not from marching $t$. Say this so the
  reader does not think 3DGS uses per-sample Beer-Lambert. A11.5d occupancy / NeuS is the cleaner
  carry-over: swap $\sigma$ for an SDF-derived density in the same ray-marched renderer. One-paragraph context for Instant-NGP (hash
  encoding, hours -> seconds), mip-NeRF (cone-cast IPE anti-aliasing), Zip-NeRF (the NeRF-arc
  endpoint), and feed-forward reconstruction (DUSt3R/MASt3R/pixelSplat) - mention, do not
  implement.

Verify every arXiv id by fetching `https://arxiv.org/abs/<id>`: NeRF 2003.08934, Fourier features
2006.10739, mip-NeRF 2103.13415, Instant-NGP 2201.05989, NeuS 2106.10689, Zip-NeRF 2304.06706,
3DGS 2308.04079, Volume Rendering Digest 2209.02417. Run the mandatory context-less style review
on the README before finishing.

## ASSIGNMENT.md

Concise builder contract in `TEMPLATE.md` format: holes (PositionalEncoding.forward,
volume_render, stratified_sample_rays + sample_along_rays/deltas), what is provided, the verify
command, the measured thresholds. Do not echo the README prose.

## Decisions resolved by the expert review (build to these)

The expert verified the volume-render math chain is correct. Resolutions, folded into the spec:

1. OpenCV +z-forward throughout (reuse nanovision.geometry.unproject for ray directions) is
   correct and carries no sign error, because the toy scene is self-generated in that convention
   (no OpenGL Blender poses imported). Confirmed.
2. Renderer confirmed: $\alpha_i=1-\exp(-\sigma_i\delta_i)$, exclusive cumprod with $T_0=1$,
   $w_i=T_i\alpha_i$, caller-applied softplus. Fix: NO epsilon in the cumprod; shift =
   `cumprod([ones, (1-alpha)[:, :-1]])`. Add the telescoping `sum_w == 1 - prod(1-alpha)` check.
   gradcheck the function directly.
3. $2^k\pi$ bands + include_input match the original NeRF - confirmed. REQUIRED: normalize
   positions to $[-1,1]$ before encoding (currently missing; the schedule is invalid otherwise).
   L_x measure-and-set, tied to the ablation showing blur; the sphere's hard boundary is the
   sharp feature that makes the ablation non-vacuous.
4. CIRCULARITY resolved: GT is rendered by an INDEPENDENT closed form (ray-sphere chord ->
   analytic Beer-Lambert alpha-composite), written separately from volume_render. A passing test
   then proves the discretized quadrature converges to the analytic integral - the real objective.
5. Held-out PSNR is VIZ-ONLY by default (seed-flaky in CPU-seconds); the graded gate is the
   overfit test + the exact renderer checks. Promote PSNR to graded only if the worst of ~5 seeds
   clears ~15 dB with margin.
6. Shared surface = volume_render + stratified_sample_rays + sample_along_rays/deltas_from_z via
   nanovision.volume. Do NOT share the positional encoding (NeRF-specific; 3DGS uses spherical
   harmonics, occupancy reuses the renderer). encoding.py stays assignment-local.

Also: ray signature uses `K` (full 3x3), not `focal`, to match nanovision.geometry.
