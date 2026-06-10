# A10.5 - geometry foundation models (DUSt3R-style): build plan

Status: draft for expert review. NOTE: there is NO pre-written research note for this assignment
(unlike A1-A10). The expert review must validate the whole mechanism against the real DUSt3R /
MASt3R literature, not just check this plan against a note. Build subagent reads this file plus
`agent_build_guide.md` and mirrors `assignments/a06_0_flow_matching` for layout.

## What this assignment teaches

DUSt3R (Wang et al., CVPR 2024) replaces the classic keypoint -> matching -> COLMAP ->
bundle-adjustment SfM pipeline with a single network: feed two images, get two per-pixel
pointmaps - the 3D coordinate of every pixel - BOTH expressed in the coordinate frame of the
first camera. No intrinsics, no poses, no matching as input; the geometry (relative pose, depth,
correspondence) falls out of the pointmaps afterward. This is the learned-geometry frame the
SfM/SLAM learner most needs to see.

The build part is the pointmap-regression mechanism: a Siamese ViT encoder (shared weights over
both images), two transformer decoders that CROSS-attend to each other's tokens (so each view's
prediction is informed by the other), two pointmap heads regressing per-pixel XYZ + a confidence,
and the confidence-weighted scale-normalized regression loss. The survey part (README): monocular
depth (DepthAnything v2, Marigold diffusion-depth), MASt3R (adds a matching head), VGGT
(multi-view 3D in one forward pass).

## The key idea to get right (validate with the expert)

Both pointmaps live in camera 1's frame. $X^{1,1}$ is image 1's pixels as 3D points in cam1
frame; $X^{2,1}$ is image 2's pixels as 3D points ALSO in cam1 frame (superscript = which image's
pixels, then which camera frame). So the network must implicitly recover the relative pose to put
image 2's points into image 1's frame. This shared-frame output is the whole trick - it makes the
two pointmaps directly comparable and lets relative pose / correspondence be read off afterward.

## Reused, already in place (do NOT rebuild)

- `from nanovision.vit import ViT` - the A2 ViT. `ViT.forward_features(img)` gives the (B, N, dim)
  patch grid used as the encoder tokens (shared weights for both images = Siamese).
- `from nanovision.transformer import TransformerDecoder` - the A1 decoder with `cross_attn=True`
  + `memory` for the cross-view attention (view-1 decoder attends to view-2 encoder tokens and
  vice versa).
- `from nanovision.geometry import unproject, project_points, make_transform, invert_transform,
  apply_transform` - the A11.5a camera primitives (OpenCV +z). depth_to_pointmap reuses unproject.
- `nanovision.data.toy.nerf_synthetic_scene(...)` - the posed colored-sphere scene. A10.5 derives
  GT pointmaps from its KNOWN sphere geometry (closed-form ray-sphere FRONT intersection per
  pixel, in cam1 frame), analogous to A9's closed-form GT. A pixel whose ray misses the sphere is
  invalid (masked in the loss). This GT helper is provided in the assignment (not toy.py).
- `nanovision.determinism.default_device` for the GPU viz; tests stay CPU.

## Shared deliverable A10.5 OWNS (adds to nanovision.geometry)

The pointmap/depth utilities land in `nanovision.geometry` (per the build checklist). They live in
a new owned file `a10_5_geometry_fm/geometry_fm.py`; the `nanovision/geometry.py` shim is extended
at the orchestrator level to import them ALONGSIDE the existing A11.5a symbols. Import them via
`nanovision.geometry`, never bare. Shared symbols: `depth_to_pointmap`, `pointmap_to_depth`,
`reproject_pointmap`.

## Shapes (fix these numbers)

- Two images (B, 3, 16, 16). ViT patch 4 -> 16 patch tokens, dim_v ~64.
- Decoder tokens (B, 16, dim). Pointmap head upsamples/reshapes the 16 patch tokens back to a
  per-pixel (or per-patch) grid. Simplest for the toy: predict a pointmap at PATCH resolution
  (4x4) - one 3D point per patch - so head output is (B, 4, 4, 3) pts + (B, 4, 4) conf, and GT is
  the sphere-surface point at each patch center. State this downscaling (real DUSt3R predicts at
  full pixel resolution via a DPT head; the toy predicts at patch resolution to stay tiny).
- Pointmaps pts1, pts2 each (B, 4, 4, 3) in cam1 frame; conf1, conf2 each (B, 4, 4); valid masks
  (B, 4, 4).

## Files (mirror the exemplar layout)

### `head.py` (holed; solution copy) - assignment-local

Hole:
- `PointmapHead.forward(self, tokens)`: an MLP mapping decoder tokens (B, N, dim) to per-token
  (X, Y, Z) and a confidence logit, reshaped to (B, h, w, 3) and (B, h, w). Confidence is made
  positive as $C = 1 + \exp(\text{conf\_logit})$ - the EXACT DUSt3R parameterization (paper text
  after Eq. 6), NOT softplus. $C \ge 1$. `__init__` (the MLP) provided; only forward is the hole.

### `loss.py` (holed; solution copy) - assignment-local

Holes:
- `normalize_scale(pts1, pts2, valid1, valid2)`: the DUSt3R scale normalization (paper Eq. 5).
  Compute ONE scalar $z$ over the UNION of valid points from BOTH pointmaps,
  $z = \frac{1}{|\mathcal{D}^1|+|\mathcal{D}^2|}\sum_{v}\sum_{i\in\mathcal{D}^v}\lVert X^{v}_i\rVert$,
  and divide both maps by that single $z$. This is critical: a per-pointmap scale would
  independently rescale view 1 and view 2 and destroy the relative scale between them, which is the
  exact information the shared-cam1-frame representation carries. Returns the two scaled pointmaps
  (or the scalar to apply).
- `pointmap_loss(pred_pts1, pred_pts2, gt_pts1, gt_pts2, pred_conf1, pred_conf2, valid1, valid2)`:
  the confidence-weighted regression loss over BOTH views. Scale-normalize the two predicted maps
  by a single $z$ (their joint scale) and the two GT maps by a single $\bar z$; per valid pixel
  $\ell_i = \lVert \hat X_i - \bar X_i \rVert$ (normalized-pointmap L2); total
  $\sum_{v\in\{1,2\}}\sum_i C_i^v \ell_i^v - \alpha \log C_i^v$ over valid pixels (mean),
  $\alpha = 0.2$ (paper value). The $-\alpha\log C$ term lets the network down-weight pixels it
  cannot predict (sky, occlusion) while paying a penalty, so confidence is learned, not free.
  Spell out the formula in the docstring. Add a test that rescaling ONLY view 2 changes the loss
  (guards against per-pointmap scaling).

### `geometry_fm.py` (holed; solution copy) - shared OWNED via nanovision.geometry

Holes:
- `depth_to_pointmap(depth, K)`: per-pixel back-projection to 3D camera-frame points (reuse
  `unproject` over the pixel grid). (B,H,W) -> (B,H,W,3). Build the pixel-center meshgrid with the
  SAME convention as the toy ($c_x=(W-1)/2$, $c_y=(H-1)/2$) or the round-trip test is off by half a
  pixel.
- `pointmap_to_depth(pts)`: the z-component (B,H,W,3) -> (B,H,W). (trivial inverse, the round-trip
  test anchor.)
- `reproject_pointmap(pts_cam1, T_1to2, K)`: transform cam1-frame points into cam2 (apply_transform
  with the relative pose), then project (project_points) to image-2 pixel coordinates.
  (B,H,W,3) -> (B,H,W,2). This is the consistency check: image-1 points reprojected into image 2
  should land where image 2 sees them.

### `model.py` (provided) - assignment-local

`GeometryFM(nn.Module)`: Siamese `ViT` encoder (shared weights, forward_features on both images),
two cross-attending decoders - decoder A runs on view-1 tokens with memory=view-2 tokens, decoder
B on view-2 tokens with memory=view-1 tokens - then two `PointmapHead`s. `forward(img1, img2)` ->
(pts1, conf1, pts2, conf2), all in cam1 frame. CRITICAL: do NOT use `TransformerDecoder` - it
hardcodes `causal=True`, and a causal mask over SPATIAL patch tokens is wrong (token 0 would see
only itself). Build the decoders from stacked `nanovision.transformer.TransformerBlock(dim,
n_heads, causal=False, cross_attn=True)` so self-attention is bidirectional over the patch grid and
each block cross-attends to the other view's memory. Provided wiring; the lesson is the pointmap
head + loss + cross-view structure, not the plumbing.

### toy GT helper (provided, in the assignment): `stereo_pointmap_gt(...)`

Given the sphere scene's geometry (radius, two camera poses, K), compute per-patch GT pointmaps:
ray through each patch center, FRONT ray-sphere intersection -> 3D point, expressed in cam1 frame
for BOTH views. View-2 points go cam2 -> world -> cam1 explicitly:
$X^{2,1} = T_1^{-1} T_2\, X^{2,2}$ where $T_i$ is cam-i-to-world (the toy's c2w poses), via
`invert_transform`/`apply_transform`. Plus the valid mask (ray hits the sphere). Provided
closed-form, no network. This is the GT the model overfits.

AVOID DEGENERACY (the expert's main concern): a single sphere centered at the origin viewed from a
symmetric ring gives two near-identical views, so the cross-attention contributes almost nothing
and the network can ignore the other view's memory - defeating the cross-view lesson. So OFF-CENTER
the sphere (translate it off the world origin) and use a WIDER camera baseline between the two
chosen views, so each sees a different surface portion and placing view-2 points in cam1 frame
genuinely requires the baseline. If that is still too easy, add a second sphere at a different
depth. The viz/overfit must include a CROSS-ATTENTION ABLATION: zero the cross-attention memory and
show the loss floor is higher with it disabled - MEASURE that cross-attention helps rather than
asserting it (per the back-claims-with-data rule). If it does not help, the toy is still degenerate
and must be enriched further.

### `config.py`, `viz.py` (provided)

- config: image size 16, patch 4, vit/decoder dims/depths, alpha=0.2, lrs, n_steps.
- viz (GPU via default_device): overfit one stereo pair, show the two predicted pointmaps as 3D
  scatter colored by confidence next to GT, and the reprojection-consistency error map. Save to
  out/. Not graded.

### `conftest.py`, `__init__.py`, `solution/`

Mirror the exemplar. `solution/` holds head.py, loss.py, geometry_fm.py (plus __init__.py).
model.py, config.py, viz.py top-level only.

## Tests (env python, CPU, seconds each)

1. `test_geometry_utils.py`: `depth_to_pointmap` then `pointmap_to_depth` round-trips the depth
   exactly; a point reprojected by `reproject_pointmap` into view 2 lands within atol at the pixel
   where view 2 observes it (use the toy scene's two poses and GT); gradcheck (float64) on
   depth_to_pointmap and reproject_pointmap.
2. `test_head.py`: `PointmapHead` output shapes (B,h,w,3) pts and (B,h,w) conf, confidence >= 1.
3. `test_loss.py`: gradcheck (float64) on `pointmap_loss`; scale-invariance - scaling BOTH pred
   maps and BOTH gt maps by the same factor leaves the loss ~unchanged (the joint normalize_scale
   property); the SHARED-scale guard - rescaling ONLY view 2 (not view 1) CHANGES the loss (proves
   the scale is joint, not per-pointmap); confidence behavior - for a fixed error, the loss is
   minimized at a finite confidence (not C->inf or C->1), confirming the $-\alpha\log C$ trade-off.
4. `test_overfit_stereo.py`: overfit the `GeometryFM` model on ONE toy stereo pair (a few hundred
   steps). The normalized pointmap error falls below a measured threshold AND the cross-view
   reprojection-consistency error (predicted view-1 points reprojected into view 2 vs GT view-2
   pixels) drops. Report both floors; set thresholds from them (no-thrash).
5. `test_forbidden_imports.py`: static scan; forbid importing a ready-made DUSt3R/MASt3R
   (dust3r, mast3r, croco packages); forbid bare imports of the owned shared geometry file.

Solution mode all green; default mode fails only at the holes, except test_forbidden_imports. Run
with `/home/tanmay/miniconda3/envs/nanovision/bin/python`.

## README (comprehensive lecture notes, per the skill, real LaTeX)

Fixed section order. Cover:
- The problem: classic SfM (keypoints -> matching -> COLMAP -> BA) is brittle on low-texture /
  wide-baseline / few images. DUSt3R regresses pointmaps directly, no intrinsics or poses needed
  as input.
- The pointmap representation: per-pixel 3D points, both images in camera 1's frame; why a shared
  frame makes pose/depth/correspondence recoverable afterward (relative pose by Procrustes/PnP
  between the two pointmaps, depth = z-channel, correspondence by nearest-neighbor in 3D).
- The architecture: Siamese ViT encoder (CroCo-pretrained in the real model - cross-view
  completion), two cross-attending decoders (each view sees the other), DPT pointmap heads. The
  course predicts at patch resolution; real DUSt3R at pixel resolution via DPT.
- The confidence-weighted loss: scale normalization (one shared scale over both pointmaps, so
  relative scale is preserved), the $\sum_i C_i \ell_i - \alpha\log C_i$ confidence loss, what
  confidence learns (down-weight unpredictable pixels: sky, occlusion, reflective surfaces).
- SCALE-GUARD sentences (course principle - toy must not override at-scale findings): state that
  if the single-sphere toy shows cross-attention or confidence contributing little, that is an
  artifact of the toy's simplicity, NOT a statement about DUSt3R. In real DUSt3R cross-view
  attention is load-bearing (it is how the second view's points get placed in the first camera's
  frame at all) and confidence measurably improves reconstruction on sky/occlusion/reflection. The
  cross-attention ablation in the toy is there to SHOW the mechanism helps, not to license the
  opposite conclusion.
- Global alignment (real DUSt3R, mention not implement): for >2 images, optimize a pose+scale per
  image so all pairwise pointmaps agree - a lightweight bundle-adjustment-like step over pointmaps
  rather than reprojection residuals. Make the BA parallel explicit for the SfM learner.
- Survey / where this goes: MASt3R (adds a dense feature-matching head + metric scale), MASt3R-SfM,
  monocular depth (DepthAnything v2's scale, Marigold's diffusion-as-depth), VGGT (a single
  feed-forward transformer for multi-view camera + depth + points in under a second), and the use
  in AV/robotics (pose-free reconstruction from camera rigs). Mention, do not implement.

Verify every arXiv id by fetching `https://arxiv.org/abs/<id>`: DUSt3R 2312.14132, MASt3R
2406.09756, CroCo 2210.10716, DepthAnything v2 2406.09414, Marigold 2312.02145, VGGT 2503.11651.
(Re-verify each title - this is a from-memory list with no research note behind it.) Run the
mandatory context-less style review on the README.

## ASSIGNMENT.md

Concise builder contract in `TEMPLATE.md` format: holes (PointmapHead.forward, normalize_scale,
pointmap_loss, depth_to_pointmap, pointmap_to_depth, reproject_pointmap), what is provided, the
verify command, the measured thresholds. Do not echo the README prose.

## Decisions resolved by the expert review (build to these - no research note existed, so the
review IS the validation)

The expert validated against the DUSt3R/MASt3R papers and verified all six arXiv ids correct.

1. Pointmap convention confirmed: both maps in cam1 frame ($X^{1,1}$, $X^{2,1}$), per-pixel 3D
   points, pose/depth/correspondence recovered afterward. The network does not regress an explicit
   pose - do not imply it does.
2. Loss: structure correct, TWO fixes folded in - (a) ONE shared scale over the union of both
   pointmaps' valid points (Eq. 5), not per-pointmap; (b) $C = 1 + \exp$, not softplus (paper text
   after Eq. 6). $\alpha=0.2$ confirmed. Loss sums over both views.
3. Architecture faithful. Patch-resolution output is an acceptable simplification (real DUSt3R is
   pixel-resolution via DPT). BLOCKER fixed: build decoders from non-causal
   TransformerBlock(causal=False, cross_attn=True), NOT the hardcoded-causal TransformerDecoder.
4. Toy non-circular (closed-form GT) but the centered sphere is DEGENERATE for cross-attention -
   off-center the sphere, widen the baseline, and add a cross-attention ablation that MEASURES the
   mechanism helps. Enrich (second sphere) if it does not.
5. Geometry utils + reprojection-consistency confirmed as the right shared set and verifiable
   result. Pin the pixel-center convention to the toy's $(W-1)/2$.
6. Survey framing sound (soften Marigold/DepthAnything metric-scale claims - both are
   scale/affine-invariant, not metric). BA-in-point-space parallel correct. All arXiv ids verified.
   Spell out author initials for the two Wang-led papers (DUSt3R: Shuzhe Wang; VGGT: Jianyuan Wang).
