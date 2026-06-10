# assignments/a09_nerf/ASSIGNMENT.md

```yaml
id: a09_nerf
title: NeRF and the volume rendering integral
module: 3
type: Core
estimated_learner_hours: 8
depends_on: [a00_harness, a01_transformer, a11_5a_camera_geometry_bev]
builds_into_shared_lib: [nanovision.volume]
forbidden_imports:
  - import nerfstudio
  - from nerfstudio
  - import nerfacc
  - from nerfacc
  - import torch_ngp
  - from torch_ngp
  - import tinycudann
  - from tinycudann
  - import kaolin
  - from kaolin
  - import render        # owned shared file, reach via nanovision.volume
  - from render import
  - import rays          # owned shared file, reach via nanovision.volume
  - from rays import
fits_12gb: true
external_data: "none (one synthetic sphere scene, closed-form ground truth)"
```

## motivation
NeRF represents a scene as an MLP from a 3D position and view direction to volume density and
color, and renders by marching rays through the field and alpha-compositing samples. The
content is the volume rendering integral, its discretization, and front-to-back compositing,
which carry over to Gaussian splatting (A10) and occupancy / neural-SDF rendering (A11.5d).
This assignment builds the Fourier positional encoding (the fix for spectral bias), the discretized volume
renderer, and pinhole ray generation. The radiance-field MLP and the toy scene are provided.
See the README for the math.

## background
See the README. Convention: OpenCV +z forward (matching nanovision.geometry), not the original
NeRF's OpenGL -z. Volume renderer: alpha_i = 1 - exp(-sigma_i delta_i); exclusive transmittance
T_i = prod_{j<i}(1-alpha_j) with T_0 = 1; weights w_i = T_i alpha_i; C = sum_i w_i c_i. The MLP
applies softplus to sigma, so volume_render takes raw non-negative sigmas. Encoding:
gamma(p) = (sin(2^k pi p), cos(2^k pi p)) for k=0..L-1, plus raw input when include_input;
positions are normalized to ~[-1,1] (divide by scene_bound) before encoding. Rays_d is unit
length, so z is Euclidean distance and near/far are distances.

## what_you_implement
- PositionalEncoding.forward: the gamma(p) Fourier feature map with include_input, no params.
- volume_render: alpha, exclusive cumprod transmittance, weights, weighted color, optional
  white background.
- stratified_sample_rays / sample_along_rays / deltas_from_z: pinhole ray directions rotated
  to world and unit-normalized, stratified depth samples, points on the ray, segment lengths.

The NeRFMLP, config, viz, and the nerf_synthetic_scene toy data are provided.

## tasks
1. `PositionalEncoding.forward` (`encoding.py`): for x (..., D) return (..., D + D*2*L) with
   include_input, else (..., D*2*L). Per band k and coordinate, emit (sin(2^k pi x),
   cos(2^k pi x)); concatenate raw x first when include_input. bands are a non-trainable buffer.
2. `volume_render` (`render.py`): alpha = 1 - exp(-sigma*delta); T = cumprod(cat([ones(R,1),
   (1-alpha)[:, :-1]], dim=1)) (exclusive, T_0 = 1, no epsilon); weights = T*alpha;
   color = sum(weights * colors); add (1 - sum weights) as white if white_background. Return
   (color (R,3), weights (R,N)). Takes raw non-negative sigmas, does NOT re-activate.
3. `stratified_sample_rays` (`rays.py`): pixel grid -> camera dirs via
   nanovision.geometry.unproject at depth 1; rotate by c2w[:3,:3]; NORMALIZE rays_d to unit;
   rays_o = c2w[:3,3]; z_vals stratified in [near, far] (uniform bins, jitter when perturb).
4. `sample_along_rays` (`rays.py`): rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None].
5. `deltas_from_z` (`rays.py`): diff(z_vals) with the last delta = 1e10.

## tests
Run with `/home/tanmay/miniconda3/envs/nanovision/bin/python`.
1. `tests/test_encoding.py` - shape (..., D+D*2*L); per-band sin/cos equals the closed form
   exactly; float64 gradcheck.
2. `tests/test_volume_render.py` - opaque sample returns its color; all sigma=0 gives
   background and zero weights (white when white_background); weights nonneg and sum <= 1;
   first weight == alpha_0 exactly; large sigma saturates weight to 1; telescoping
   sum_w == 1 - prod(1-alpha) exactly; float64 gradcheck w.r.t. sigmas and colors via
   torch.autograd.gradcheck (it is a function, not a module).
3. `tests/test_rays.py` - center pixel ray is +z for identity c2w; corner ray tangent
   (u-cx)/fx; directions unit length; z_vals in [near, far] and sorted; sample_along_rays
   shape (R,N,3) and on the ray; deltas last entry is the large constant.
4. `tests/test_mlp.py` - NeRFMLP forward: sigma (R,N) >= 0, rgb (R,N,3) in [0,1].
5. `tests/test_overfit_ray.py` - overfit 256 toy rays end to end (encoding -> MLP ->
   volume_render) against closed-form ground truth; final MSE under threshold and a relative
   drop. The ground truth is the closed-form ray-sphere chord, not volume_render, so this
   shows the quadrature converges to the analytic integral.
6. `tests/test_forbidden_imports.py` - no nerfstudio/nerfacc/torch_ngp/tinycudann/kaolin/
   pytorch3d renderer; render.py and rays.py are not imported bare. Passes with the holes too.

Held-out-view PSNR is a viz figure, not a graded test (seed-flaky in a CPU-seconds budget);
the graded gate is the overfit test plus the exact renderer checks.

## provided_boilerplate
`model.py` `NeRFMLP` (encoded position trunk with one skip, softplus density, sigmoid
view-dependent color, positions normalized by scene_bound before encoding). `config.py`
`NeRFConfig`. `nanovision.data.toy.nerf_synthetic_scene` (one colored solid sphere, cameras on
a ring, OpenCV +z c2w, closed-form ray-sphere-chord ground truth). `viz.py` renders the
held-out view and the spectral-bias ablation (with vs without the Fourier encoding) to out/.

## compute_notes
16x16 images, 32 samples per ray, 128-wide MLP; every test runs on CPU in seconds. The overfit
test is 800 Adam steps on 256 rays. The exact renderer checks and the encoding/ray checks are
training-free. The full Lego scene (800x800, ~20k steps, ~22 dB PSNR) wants a GPU and is out of
scope; viz runs short CPU training.

## solution_notes
The exclusive cumprod is `cumprod([ones, (1-alpha)[:, :-1]])`: a leading column of ones for
T_0 = 1, then the running product of all but the last (1-alpha). No epsilon inside the cumprod
(it breaks the exact-equality tests) and no inclusive-cumprod-then-shift (it double-counts).
volume_render receives raw non-negative sigmas because the MLP already applied softplus; do not
re-activate. Normalizing rays_d to unit length makes z a Euclidean distance, so near/far are
distances and deltas need no ||d|| factor; the same unit rays_d feeds the MLP color branch.
Positions are divided by scene_bound before encoding because the 2^k frequency schedule aliases
on inputs of magnitude ~4. The toy ground truth comes from the closed-form ray-sphere chord
(analytic Beer-Lambert), not from volume_render, so a passing overfit test shows the discretized
quadrature converges to the analytic integral rather than to its own renderer. The shared shim
nanovision.volume is owned by A9; A10 and A11.5d reach volume_render and the ray helpers through
it.
