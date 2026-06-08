# assignments/a11_5c_bevformer/ASSIGNMENT.md

```yaml
id: a11_5c_bevformer
title: BEVFormer-style attention
module: 3.5
type: Core
estimated_learner_hours: 6
depends_on: [a00_harness, a01_transformer, a11_5a_camera_geometry_bev, a11_5b_lift_splat_shoot]
builds_into_shared_lib:
  - nanovision.bevformer.bev_reference_points
  - nanovision.bevformer.project_reference_points
  - nanovision.bevformer.SpatialCrossAttention
  - nanovision.bevformer.warp_bev
  - nanovision.bevformer.TemporalSelfAttention
  - nanovision.bevformer.BEVFormerEncoder
  - nanovision.bevformer.BEVFormerSeg
forbidden_imports:
  - nn.MultiheadAttention      # attention comes from nanovision.attention (the A1 build)
  - F.scaled_dot_product_attention
  - nn.Transformer
  - cv2.projectPoints
  - cv2.solvePnP
  - kornia
  # F.grid_sample and F.affine_grid are ALLOWED (the bilinear-sampling substrate).
fits_12gb: true
external_data: none (tests run on the synthetic bev_multicam_scene)
camera_axis_convention: OpenCV (+x right, +y down, +z forward); ego x forward, y left, z up
```

## motivation
BEVFormer (Li et al., ECCV 2022, arXiv:2203.17270) builds a bird's-eye-view feature grid by
PULLING image features in: each BEV cell projects a vertical pillar of 3D reference points back
into every camera and bilinear-samples the features there, the opposite of Lift-Splat-Shoot's
depth-PUSH. The dense BEV grid this produces is the canonical intermediate that occupancy
(A11.5d) and map/prediction (A11.5e) consume. The spatial cross-attention specializes the
deformable attention of Deformable DETR (arXiv:2010.04159); the temporal self-attention warps the
previous BEV grid by the ego motion and fuses it, recovering content the current cameras miss. The
README has the full treatment, including the LSS contrast and where sparse-query methods overtook
the dense grid for detection.

## background
A BEV cell center $(x, y)$ becomes a pillar of $n_{ref}$ points at heights $z \in [z_{min},
z_{max}]$ in the ego frame. Project each with the rig's `world_to_pixel` (E is $T_{cam\_ego}$);
normalize the pixel $(u, v)$ to grid_sample coords with the align_corners=False map $g_x = 2(u +
0.5)/W - 1$, $g_y = 2(v + 0.5)/H - 1$, last dim ordered $(g_x, g_y) = (\text{width},
\text{height})$, normalized by the FULL image size $W, H$ (not the feature-map size). Spatial
cross-attention samples each camera's feature map at these coords, averages over the in-frame
heights (divide by the count of valid heights, not $n_{ref}$), then over the cameras that see the
pillar (the $|V_{hit}|$ semantics); no-hit cells keep their query unchanged. The deformable path
adds learned offsets and softmax weights around each reference point. The ego-motion warp
resamples the previous BEV grid: for the tensor $(C, n_x, n_y)$ with $n_x$ along forward, a forward
ego translation of $k$ cells uses `theta[1, 2] = +2k/nx`, lateral `theta[0, 2] = +2k/ny`,
align_corners=False for both `affine_grid` and `grid_sample`. Temporal self-attention attends each
cell over $\{\text{query}, \text{warped history}\}$.

Shapes: reference pillars $(n_x, n_y, n_{ref}, 3)$; projected coords $(n_{cam}, n_x, n_y, n_{ref},
2)$ and mask $(n_{cam}, n_x, n_y, n_{ref})$; BEV query $(n_x, n_y, C)$; camera features $(n_{cam},
C, H_f, W_f)$; BEV grid $(C, n_x, n_y)$; seg logit $(n_{classes}, n_x, n_y)$. Toy: 4-camera ring,
centered 16x16 BEV grid, $n_{ref}=4$ heights from -5 to 3 m, 32x32 images.

## what_you_implement
- `bev_reference_points` (BEV cell centers -> ego-frame pillars at $n_{ref}$ heights).
- `project_reference_points` (ego pillars -> grid_sample coords + in-frame mask, reusing
  `nanovision.geometry`).
- `SpatialCrossAttention.forward` (the simplified bilinear path and the deformable-offset path,
  both calling the shared height/hit-view reduction helper).
- `warp_bev` (ego-motion affine warp of the previous BEV grid).
- `TemporalSelfAttention.forward` (2-key attention over query + warped history, reusing the
  multi-head attention).
- `BEVFormerEncoder.forward` and `BEVFormerSeg.forward` (assemble TSA -> SCA -> FFN, then the seg
  head).

`BEVFormerConfig`, the module `__init__`s (the projections, the offset/weight heads, the learnable
BEV query, the layer list, the seg head), the shared reduction helper, and `bev_multicam_scene`
are provided.

## tasks
- **Task 1 - reference pillars** (file: `bevformer.py`, symbol: `bev_reference_points`): tile each
  BEV cell center over $n_{ref}$ heights. Teaches: the 3D anchor points the query samples at.
- **Task 2 - projection** (file: `bevformer.py`, symbol: `project_reference_points`): project the
  pillars with `world_to_pixel`, normalize by the full image size, return coords + mask. Teaches:
  reuse of the camera projection and the align_corners=False normalization.
- **Task 3 - spatial cross-attention** (file: `bevformer.py`, symbol:
  `SpatialCrossAttention.forward`): bilinear-sample and reduce over heights and hit views; add the
  deformable-offset path sharing the same reduction. Teaches: the query-pull view transform and
  its deformable specialization.
- **Task 4 - ego-motion warp** (file: `bevformer.py`, symbol: `warp_bev`): affine-warp the previous
  BEV grid so static world content stays put. Teaches: the axis/sign of the inverse sampling warp.
- **Task 5 - temporal self-attention** (file: `bevformer.py`, symbol:
  `TemporalSelfAttention.forward`): attend each cell over query + warped history. Teaches: temporal
  fusion across frames.
- **Task 6 - assemble** (file: `bevformer.py`, symbols: `BEVFormerEncoder.forward`,
  `BEVFormerSeg.forward`): stack TSA -> SCA -> FFN and add the seg head. Teaches: how the encoder
  composes.

## tests
1. `tests/test_reference_points.py` - pillar shape $(16, 16, 4, 3)$, exact linspace heights, cell
   centers; an on-axis point hits the principal point and is in-frame for the front camera, out for
   the back camera; projection round-trip < 1e-4. (reference)
2. `tests/test_spatial_cross_attention.py` - shapes; a constructed interior single-view cell pools
   its one hit camera's constant; a synthetic all-False mask leaves the query unchanged; float64
   gradcheck on features -> SCA. (reference + gradcheck)
3. `tests/test_deformable.py` - with shared projections and a zero offset head, the deformable
   forward equals the bilinear forward < 1e-5; the offset head receives nonzero gradient.
4. `tests/test_temporal.py` - `warp_bev` identity at zero motion and exact cell shift under forward
   / lateral motion; the temporal model beats the no-temporal model on the occluded vehicle's BEV
   cells (mean BCE gap > 0.1 across 3 seeds).
5. `tests/test_bev_seg.py` - overfit `BEVFormerSeg` on one multi-cam frame, <= 1500 steps; BCE <
   0.05 and BEV IoU(logit > 0, gt) > 0.6. (overfit)
6. `tests/test_forbidden_imports.py` - tokenize scan over `bevformer.py` top + solution + the
   `nanovision.bevformer` shim for nn.MultiheadAttention / scaled_dot_product_attention /
   nn.Transformer / cv2 / kornia; grid_sample and affine_grid are allowed.

## provided_boilerplate
`BEVFormerConfig`; the `__init__`s of `SpatialCrossAttention` (value/output projections, offset and
weight heads), `TemporalSelfAttention`, `BEVFormerEncoder` (learnable BEV query, layer list,
precomputed reference points), `BEVFormerSeg` (seg head); the `_reduce_over_heights_and_views`
shared helper; `nanovision.data.toy.bev_multicam_scene`; `nanovision.attention.MultiHeadAttention`;
and `nanovision.geometry` (`CameraRig.world_to_pixel`, `BEVGrid`).

## compute_notes
CPU only for the graded tests, seconds each except the overfit (~18 s) and temporal (~6 short
training runs) tests. The overfit reaches BCE ~0 / IoU 1.0 in 1500 steps at Adam lr 1e-2. float32
throughout with a float64 gradcheck on the spatial cross-attention. `viz.py` uses the GPU when
present and keeps matplotlib tensors on CPU.

## stretch_goals
1. Turn on `offsets=True` in the config and overfit with the deformable path; compare convergence.
2. Add a second temporal frame (a 3-frame recurrent warp-and-fuse) and measure the occluded-cell
   recovery as history lengthens.
3. Replace the uniform height average with a learned per-height weight (a small attention over the
   $n_{ref}$ samples).

## further_reading
- Li et al., "BEVFormer" (ECCV 2022), arXiv:2203.17270.
- Zhu et al., "Deformable DETR" (ICLR 2021), arXiv:2010.04159.
- Wang et al., "DETR3D" (CoRL 2021), arXiv:2110.06922.
- Liu et al., "PETR" (ECCV 2022), arXiv:2203.05625; "PETRv2", arXiv:2206.01256.
- Yang et al., "BEVFormer v2" (CVPR 2023), arXiv:2211.10439.
- Lin et al., "Sparse4D v2" (2023), arXiv:2305.14018.

## solution_notes
The reduction divides per-camera by the count of valid heights (`m.sum(-1).clamp_min(1)`), not by
$n_{ref}$, then across cameras by the number of hit views; no-hit cells keep the input query via a
`torch.where` on the no-hit mask. The deformable path shares `value_proj` and `out_proj` with the
simplified path and zero-inits the offset head, so zero offsets make every sample land on the
reference point and the softmax weights (summing to 1) reproduce the bilinear sample exactly. The
warp sign is the trap: `affine_grid` builds an inverse (sampling) warp, so a forward ego motion of
$k$ cells uses `theta[1, 2] = +2k/nx` (a static world point moves to a LOWER forward index);
`-2k/nx` is the double-inverse bug. The single-view hit-mask test selects an INTERIOR cell (coords
< 0.95) because a corner cell projects to the image edge where bilinear sampling reads against the
zero-padding.
```
