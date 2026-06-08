# assignments/a10_5_geometry_fm/ASSIGNMENT.md

```yaml
id: a10_5_geometry_fm
title: Geometry foundation models (DUSt3R-style pointmap regression)
module: 4
type: Core
estimated_learner_hours: 8
depends_on: [a01_transformer, a02_vit, a09_nerf, a11_5a_camera_geometry_bev]
builds_into_shared_lib: [nanovision.geometry]
forbidden_imports:
  - import dust3r
  - from dust3r
  - import mast3r
  - from mast3r
  - import croco
  - from croco
  - import geometry_fm
  - from geometry_fm
fits_12gb: true
external_data: "none (the posed-sphere toy from nanovision.data.toy)"
```

## motivation
DUSt3R regresses two per-pixel pointmaps (the 3D point each pixel sees), both in the first
camera's frame, from an image pair, with no intrinsics, poses, or matching as input. Depth,
relative pose, and correspondence are read off the pointmaps afterward. This replaces the
classic keypoint -> match -> pose -> triangulate -> bundle-adjust SfM pipeline, which is
brittle on low-texture, wide-baseline, and few-image inputs, with one forward pass. You build
the regression mechanism: the pointmap head, the confidence-weighted scale-normalized loss,
and the depth/pointmap/reprojection utilities. See the README for the full treatment.

## background
A pointmap is an $H \times W \times 3$ map of camera-frame 3D points, the 3D form of a depth
map. DUSt3R outputs $X^{1,1}$ (image 1's pixels in cam1 frame) and $X^{2,1}$ (image 2's pixels
in cam1 frame), so placing view 2 in cam1's frame implicitly recovers the relative pose. The
loss uses ONE shared scale over both pointmaps' valid points (a per-map scale would destroy the
relative scale between cameras) and a learned confidence $C = 1 + \exp(\text{logit}) \ge 1$
with an $-\alpha\log C$ regularizer, $\alpha = 0.2$. Decoders are non-causal (patch tokens have
no order) and cross-attend to the other view.

## what_you_implement
- depth_to_pointmap, pointmap_to_depth, reproject_pointmap (`geometry_fm.py`): depth <-> pointmap
  and the cross-view reprojection check. Shared via `nanovision.geometry`.
- PointmapHead.forward (`head.py`): MLP token -> (XYZ, conf logit), reshape to grid,
  $C = 1 + \exp(\text{logit})$.
- normalize_scale, pointmap_loss (`loss.py`): the joint scale (DUSt3R Eq. 5) and the
  confidence-weighted regression loss over both views.

The Siamese ViT encoder, the two cross-attending decoders (`model.py`), the closed-form toy GT
(`toy_scene.py`), config, and viz are provided.

## tasks
1. `depth_to_pointmap` (`geometry_fm.py`): build the pixel-center meshgrid (u = column index,
   v = row index, principal point at $((W-1)/2, (H-1)/2)$ inside K), then `unproject`.
   (B,H,W) -> (B,H,W,3).
2. `pointmap_to_depth` (`geometry_fm.py`): the z-channel. (B,H,W,3) -> (B,H,W).
3. `reproject_pointmap` (`geometry_fm.py`): `apply_transform(T_1to2, pts)` then
   `project_points`. (B,H,W,3) -> (B,H,W,2).
4. `PointmapHead.forward` (`head.py`): run the provided MLP, split the 4 outputs into XYZ and a
   confidence logit, reshape to (B, grid, grid, 3) and (B, grid, grid), map the logit to
   $1 + \exp$.
5. `normalize_scale` (`loss.py`): mean L2 norm of valid points over BOTH stacked maps,
   (B,2,h,w,3) + (B,2,h,w) mask -> (B,) one scalar per pair.
6. `pointmap_loss` (`loss.py`): scale pred by $z$ and gt by $\bar z$ (each joint), per-pixel L2
   residual, $\sum_v \sum_i C_i \ell_i - \alpha\log C_i$ over valid pixels, mean over the valid
   count of both views.

## tests
Run with `/home/tanmay/miniconda3/envs/nanovision/bin/python`. Solution mode all green; default
mode fails only at the holes (NotImplementedError), except test_forbidden_imports.
1. `tests/test_geometry_utils.py` - depth round-trip exact; the pixel convention (pixel (i,j)
   projects to (u,v)=(j,i)); reprojection consistency on the toy GT (view-2 pointmap reprojected
   into image 2 lands on its patch centers, < 1e-3 px); float64 gradcheck of depth_to_pointmap
   and reproject_pointmap.
2. `tests/test_head.py` - shapes (B,grid,grid,3) and (B,grid,grid); confidence $\ge 1$ and can
   exceed 1.
3. `tests/test_loss.py` - float64 gradcheck; joint scale invariance (scaling both pred and both
   gt by a factor leaves the loss unchanged); the shared-scale guard (rescaling only view 2
   changes the loss); normalize_scale equals the mean valid-point norm; the confidence optimum
   is interior (C = alpha/ell).
4. `tests/test_overfit_stereo.py` - overfit GeometryFM on 8 toy stereo pairs (~2500 Adam steps);
   the normalized pointmap error falls below threshold with cross-attention and rises when
   cross-attention is disabled.
5. `tests/test_forbidden_imports.py` - no dust3r/mast3r/croco; no bare import of geometry_fm.
   Passes with the holes in place too.

## provided_boilerplate
`model.py` `GeometryFM` (Siamese ViT encoder via `nanovision.vit.ViT`, two non-causal
cross-attending decoders from `nanovision.transformer.TransformerBlock(causal=False,
cross_attn=True)`, two `PointmapHead`s; `forward(img1, img2, use_cross=True)` returns
pts1, conf1, pts2, conf2). `toy_scene.py` `stereo_pointmap_gt` (closed-form ray-sphere GT
pointmaps in cam1 frame, off-center sphere + wide baseline). `config.py` `GeometryFMConfig`.
`viz.py` (GPU via `default_device`: overfit, 3D pointmap scatter colored by confidence,
reprojection error map, cross-attention ablation bar chart).

## compute_notes
All graded tests run on CPU in seconds, except the overfit (~2500 Adam steps on 8 tiny pairs,
about a minute on CPU). Measured floors on this toy (8 view pairs, 2500 steps), stable across
seeds 0-2: normalized pointmap error about 0.007 with cross-attention and about 0.035 with it
disabled (~5x worse). Thresholds: `err_cross < 0.05` and `err_nocross > 1.5 * err_cross`, both
with margin. The single-pair cross-view reprojection-pixel error is numerically unstable (points
near the reprojected image plane blow up the pixel coordinate), so it is shown in the viz as an
error map but not asserted in the overfit test; reprojection consistency is checked exactly on
GT in `test_geometry_utils.py` instead.

## solution_notes
The pixel-center convention must match the toy: principal point $((W-1)/2, (H-1)/2)$ in K, and
the meshgrid maps pixel (i, j) -> (u, v) = (j, i), or the round-trip is off by half a pixel.
The loss scale is JOINT over both maps - stack pred1/pred2 (and gt1/gt2) on a new axis and call
normalize_scale once per pair; a per-map scale fails `test_shared_scale_guard`. Confidence is
$1 + \exp$, not softplus. The overfit needs varied relative poses across the batch (multiple
view pairs) for cross-attention to matter; a single fixed pair is degenerate because the network
bakes one pose into its weights and ignores the partner view. The off-center sphere
(`sphere_center=(0.7, 0, 0.4)`, `radius=1.6`) plus the wide baseline (`view1=0`, `view2=3` on an
8-camera ring) gives views that see different surface portions. The geometry utilities are
shared via `nanovision.geometry`; A9/A10 import that module, so the file must exist for the repo
to import even with the holes unfilled (the holes raise only when called).
