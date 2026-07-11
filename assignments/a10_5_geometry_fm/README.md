# A10.5 - geometry foundation models (DUSt3R-style pointmap regression)

Two images of a scene feed into one network, and it returns, for every pixel, the 3D coordinate
of the surface that pixel sees. Both sets of 3D points come out in the same coordinate frame (the
first camera's), so depth, relative pose, and correspondence are read off the output afterward.
No intrinsics, no poses, and no feature matching go in. This is DUSt3R (Shuzhe Wang et al., CVPR
2024), which replaced a multi-stage geometry pipeline with a single forward pass.

This assignment builds the regression mechanism on a tiny posed-sphere toy: a Siamese ViT
encoder, two transformer decoders that cross-attend to each other's tokens, a head that
regresses per-pixel 3D points plus a confidence, and the confidence-weighted, scale-normalized
loss. The encoder and the cross-attending decoders are provided; the head, the loss, and the
depth/pointmap geometry utilities are the holes.

Required reading before starting:
- Wang et al. 2024, "DUSt3R: Geometric 3D Vision Made Easy",
  [arXiv:2312.14132](https://arxiv.org/abs/2312.14132).
- Weinzaepfel et al. 2022, "CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View
  Completion", [arXiv:2210.10716](https://arxiv.org/abs/2210.10716): the cross-view pretext task
  DUSt3R's encoder is pretrained on.

## Lecture notes

### The problem it replaces

Classic structure-from-motion (SfM) recovers 3D from images through a pipeline: detect keypoints,
match them across images, estimate the geometry (essential matrix, then poses) by RANSAC,
triangulate points, and refine everything with bundle adjustment (a nonlinear least-squares solve
over poses and 3D points minimizing reprojection error). COLMAP is the standard implementation.
The pipeline is accurate when it works, but each stage can fail and the failure propagates:
keypoint matching breaks on low-texture surfaces (blank walls, sky, repetitive patterns), wide
baselines (the same point looks too different between two far-apart cameras), and small image sets
(two views give few correspondences and a fragile two-view geometry). When matching returns too
few or wrong correspondences, pose estimation and triangulation have nothing to stand on.

DUSt3R removes the matching stage. It regresses geometry directly from raw pixels with a network
trained on many scenes, so the prior learned across that training data fills in where
correspondence search would have failed. The intrinsics and poses that the classic pipeline needs
as input (or estimates early and commits to) are not inputs at all; they fall out of the
network's output.

### The pointmap representation

A pointmap is a per-pixel map of 3D coordinates. For an image of shape $H \times W$, the pointmap
$X$ has shape $H \times W \times 3$: entry $X_{ij}$ is the 3D point, in some camera frame, of the
surface seen at pixel $(i, j)$. It is the geometric content of a depth map written in 3D instead
of as a scalar per pixel. Given intrinsics $K$, a depth map $D$ and a pointmap are
interchangeable: back-project each pixel to its 3D ray and scale by depth to get the point, or
read the point's $z$ coordinate to get the depth.

DUSt3R's move is which frame the two pointmaps live in. For an image pair, it outputs

$$X^{1,1} \in \mathbb{R}^{H \times W \times 3}, \qquad X^{2,1} \in \mathbb{R}^{H \times W \times 3}.$$

The superscript reads "which image's pixels, then which camera frame." $X^{1,1}$ is image 1's
pixels as 3D points in camera 1's frame. $X^{2,1}$ is image 2's pixels as 3D points, also in
camera 1's frame. So the network takes image 2's pixels and places them in image 1's coordinate
system. To do that it has to implicitly recover the relative pose between the two cameras, because
expressing image 2's geometry in image 1's frame requires knowing how the two cameras sit relative
to each other.

Both maps sharing one frame makes the downstream quantities recoverable by simple operations:

- Depth of image 1 is the $z$ channel of $X^{1,1}$ (and depth of image 2 needs $X^{2,2}$, the same
  network run with the roles swapped, or a transform of $X^{2,1}$ back into camera 2).
- Relative pose between the cameras comes from aligning $X^{2,1}$ (image 2's points in cam1 frame)
  with $X^{2,2}$ (the same points in cam2 frame): a rigid-alignment / Procrustes solve, or PnP
  between the 2D pixels and the 3D points.
- Correspondence between the two images is nearest-neighbor search in 3D: pixels whose pointmap
  entries are close in cam1 frame see the same surface point.

The network never outputs an explicit pose or correspondence. It regresses the two pointmaps, and
pose, depth, and matches are read off afterward.

### The architecture

```mermaid
flowchart TD
    I1[image 1] --> E1[ViT encoder]
    I2[image 2] --> E2[ViT encoder]
    E1 -. shared weights .- E2
    E1 --> F1["tokens f1 (B, N, d)"]
    E2 --> F2["tokens f2 (B, N, d)"]
    F1 --> D1[decoder 1: self-attn + cross-attn to f2]
    F2 --> D2[decoder 2: self-attn + cross-attn to f1]
    F2 -. memory .-> D1
    F1 -. memory .-> D2
    D1 --> H1[pointmap head]
    D2 --> H2[pointmap head]
    H1 --> P1["X^1,1 + conf (cam1 frame)"]
    H2 --> P2["X^2,1 + conf (cam1 frame)"]
```

A single ViT encoder runs on both images with shared weights (Siamese), turning each into a grid
of patch tokens of shape $(B, N, d)$, with $N$ the number of patches and $d$ the token dimension.
Sharing weights means the encoder learns one image representation, applied identically to both
views.

Two decoders, one per view, turn the tokens into the prediction. The view-1 decoder runs
self-attention over image 1's tokens and cross-attention to image 2's tokens as memory; the
view-2 decoder mirrors it. Cross-attention is how each view sees the other: the view-2 decoder
reads image 1's tokens, which lets it place image 2's points in image 1's frame. The two decoders
do not share weights, because their jobs differ - one predicts in its own frame, the other
predicts in the partner's frame.

The decoders are non-causal: bidirectional self-attention over the patch grid, then a
cross-attention sub-layer over the other view's memory, then the feed-forward. A causal mask would
be wrong here. Causal masking exists for sequences with a left-to-right order (language,
autoregressive generation) where a token may attend only to earlier tokens. Patch tokens have no
such order; patch 0 must see the whole grid.

Each decoder's output tokens go through a pointmap head: an MLP mapping each token to four
numbers, three for the 3D point and one for a confidence logit, reshaped back to the patch grid.
Predicting one point per patch (patch resolution) keeps the toy small. Real DUSt3R predicts one
point per pixel through a DPT head (a dense prediction head that upsamples transformer features
back to image resolution, Ranftl et al. 2021); the patch-resolution output here is a deliberate
simplification, not a different mechanism.

The encoder in real DUSt3R is pretrained with CroCo (Weinzaepfel et al. 2022), "cross-view
completion": mask patches of one image and reconstruct them using a second view of the same
scene. That pretext task teaches exactly the cross-view reasoning DUSt3R's decoders need, which is
why DUSt3R starts from CroCo weights rather than from scratch.

### The confidence-weighted, scale-normalized loss

Two pieces sit on top of a plain regression loss: scale normalization and a learned confidence.

Scale normalization handles the depth/scale ambiguity. From images alone, a scene and its
half-size copy at half the distance look identical, so the geometry is only fixed up to one global
scale. To compare a prediction with ground truth without penalizing a consistent overall-scale
difference, both are divided by their own scale factor before the residual is taken. The scale is
the mean distance of the valid points from the origin (DUSt3R Eq. 5):

$$z = \frac{1}{|\mathcal{D}^1| + |\mathcal{D}^2|} \sum_{v \in \{1, 2\}} \sum_{i \in \mathcal{D}^v} \lVert X^v_i \rVert,$$

where $\mathcal{D}^v$ is the set of valid pixels in pointmap $v$ and $\lVert X^v_i \rVert$ is the
Euclidean norm of the 3D point at pixel $i$.

The sum runs over both pointmaps with one shared $z$. A separate scale per map would rescale image
1 and image 2 independently, and that destroys the relative scale between the two cameras, which
is the exact information the shared-cam1-frame representation carries. The two views must be
scaled together so their relative geometry survives. The scale therefore is one scalar per image
pair, computed over the union of both maps' valid points.

The confidence term lets the network say "I cannot predict this pixel." With predicted pointmaps
scaled by $z$ and ground truth scaled by $\bar z$, the per-pixel residual is the Euclidean
distance between the scaled points,

$$\ell_i^v = \left\lVert \frac{\hat X_i^v}{z} - \frac{\bar X_i^v}{\bar z} \right\rVert,$$

and the total loss weights each residual by that pixel's confidence and subtracts a log term:

$$\mathcal{L} = \sum_{v \in \{1, 2\}} \sum_{i \in \mathcal{D}^v} \left( C_i^v\, \ell_i^v - \alpha \log C_i^v \right), \qquad \alpha = 0.2.$$

Confidence is $C = 1 + \exp(\text{logit})$, the DUSt3R parameterization (paper text after Eq. 6),
so $C \ge 1$ always. This is not softplus; substituting softplus changes the trade-off below.

The two terms trade off. For a fixed residual $\ell$, the per-pixel cost $C\ell - \alpha\log C$ is
minimized at $C = \alpha/\ell$: high confidence where the residual is small, low confidence where
it is large. The network can lower $C$ on pixels it cannot fit (sky, occluded regions, reflective
surfaces with no stable 3D point), paying the $-\alpha\log C$ penalty for doing so, and raise $C$
where it is sure. The confidence is learned from the data, not supplied. The optimum is interior,
neither $C \to 1$ nor $C \to \infty$.

### Why the cross-view path has to carry information

A single sphere centered at the world origin, viewed from a symmetric camera ring, is degenerate
for the cross-view lesson: the two views are near-identical, so placing view 2 in view 1's frame
needs almost nothing from view 1 and the cross-attention sits idle. Off-centering the sphere and
widening the baseline between the two views makes each view see a different portion of the
surface. Training on several view pairs at once, so the relative pose varies across the batch,
forces the cross-attention to carry information: with a single fixed pair the network could bake
one relative pose into its weights and ignore the other view; with the pose changing pair to pair
it cannot, and it has to read the partner tokens to place view 2 correctly.

In real DUSt3R the cross-view attention is the only path by which the second view's geometry
enters the first camera's frame, so removing it would leave the network unable to produce a
shared-frame pointmap pair at all. The confidence term improves reconstruction on the surfaces a
regression-only loss handles worst (sky, occlusion, reflective and transparent regions).

### Global alignment for more than two views

DUSt3R as built handles one image pair. For a collection of images it runs the network on many
pairs and then aligns all the pairwise pointmaps into one global reconstruction. The alignment
optimizes a pose and a scale per image so that, where two images overlap, their pointmaps agree in
a common world frame. The objective minimizes 3D disagreement between pointmaps directly, not
reprojection error in pixels as classic bundle adjustment does.

The parallel is exact: this is bundle adjustment moved into point space. Classic BA jointly
refines camera poses and 3D points to minimize where projected points land versus where features
were detected. DUSt3R's global alignment jointly refines per-image poses and scales to minimize
where the pointmaps disagree in 3D. The network has already done the hard part (producing
metric-consistent local geometry without correspondences), so the global step is a lightweight
least-squares solve rather than the fragile keypoint-driven optimization COLMAP runs.

### Where this goes

MASt3R (Leroy et al. 2024, [arXiv:2406.09756](https://arxiv.org/abs/2406.09756)) adds a dense
feature-matching head to DUSt3R: alongside the pointmaps it predicts local features whose
nearest-neighbor matches give pixel-accurate correspondences, which sharpens the geometry and
supports localization. MASt3R-SfM extends it to full unconstrained image collections.

Monocular depth networks solve the one-image case. Depth Anything V2 (Yang et al. 2024,
[arXiv:2406.09414](https://arxiv.org/abs/2406.09414)) trains a single-image depth predictor on
large mixed data and predicts relative (scale-invariant) depth, the depth up to an unknown global
scale and shift, not metric distance. Marigold (Ke et al. 2024,
[arXiv:2312.02145](https://arxiv.org/abs/2312.02145)) repurposes a pretrained image-diffusion
model as a depth estimator by fine-tuning it to denoise a depth map conditioned on the image, also
producing affine-invariant (scale-and-shift) depth. Both predict geometry up to an unknown
transform; DUSt3R's two-view setup is what fixes the relative scale between cameras that a single
image cannot.

VGGT (Wang et al. 2025, [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)) pushes the
feed-forward idea to many views at once: one transformer takes a set of images and predicts camera
intrinsics and extrinsics, depth maps, and point tracks together in a single pass, without the
pairwise-then-align structure. It is the multi-view generalization of the pointmap-regression idea
built here.

In robotics and autonomous driving these models give pose-free reconstruction from a camera rig:
feed the surround images and get consistent 3D geometry without a calibrated SfM run, the same
shared-frame trick applied to a vehicle's cameras instead of a stereo pair.

## The assignment

Fill these holes, in order. Each is one `NotImplementedError` with a matching test; the docstring in each file gives the signature, shapes, and constraints.

1. [`depth_to_pointmap()`](geometry_fm.py) in `geometry_fm.py`
2. [`pointmap_to_depth()`](geometry_fm.py) in `geometry_fm.py`
3. [`reproject_pointmap()`](geometry_fm.py) in `geometry_fm.py`
4. [`PointmapHead.forward()`](head.py) in `head.py`
5. [`normalize_scale()`](loss.py) in `loss.py`
6. [`pointmap_loss()`](loss.py) in `loss.py`

### Running and validating

Activate the environment (`conda activate nanovision`), then:

```
make test     A=a10_5_geometry_fm   # run the tests against your top-level files (red until the holes are filled)
make verify   A=a10_5_geometry_fm   # run the same tests against the reference solution/ (green from the start)
make viz      A=a10_5_geometry_fm   # render the figures from the reference solution
make viz-mine A=a10_5_geometry_fm   # render the figures from your own code (once the holes are filled)
```

`make test` is the command to run while working. It runs the suite in `tests/` against the
top-level files (the ones with the holes) and goes from red (the holes raise `NotImplementedError`)
to green as you fill them in. `make verify` runs the identical suite against the reference in
`solution/`: it sets `NANOVISION_IMPL=solution`, so the tests import the reference implementation
instead of the top-level files. `make verify` is green from the start, so it shows the target and
confirms the tests and the environment work before anything changes. The goal is to bring
`make test` to the same green as `make verify`.

The suite checks the depth round-trip (exact), the pixel convention, and the reprojection
consistency on the toy ground truth (view-2 points reprojected into image 2 land on their patch
centers to under $10^{-3}$ px), with a float64 gradcheck of the differentiable utilities; the
head's output shapes and that the confidence is $\ge 1$ and can exceed 1; the loss with a float64
gradcheck, the joint-scale invariance (scaling both prediction and both ground truth by a factor
leaves the loss unchanged), the shared-scale guard (rescaling only view 2 changes the loss, which
it cannot do if the scale were per-map), and the interior confidence optimum $C = \alpha/\ell$;
an overfit of the model on 8 toy stereo pairs; and a static scan blocking dust3r, mast3r, and
croco. The forbidden-imports scan passes with the holes in place.

`make viz` renders from the reference solution, so it works on a fresh checkout before any holes
are filled and shows the target figures. `make viz-mine` runs the same script against your
top-level code; it needs the holes filled, since it trains a model with them. Both write
`pointmaps.png` (predicted vs ground-truth pointmaps as a 3D scatter colored by confidence),
`reprojection.png` (the cross-view reprojection-consistency error map), and `cross_ablation.png`
(the error floor with vs without cross-attention) to `out/`, using matplotlib's headless Agg
backend so the commands behave the same over SSH, in WSL, and in CI with no display. The viz
trains on the GPU when one is present. Add `SHOW=1` (for example
`make viz-mine A=a10_5_geometry_fm SHOW=1`) to also open the figures in interactive windows when a
display is available.

What you should see when you run this. The overfit trains for about 2500 Adam steps on 8 tiny
pairs (about a minute on CPU). With cross-attention on, the normalized pointmap error falls to
around 0.007; with the cross-attention memory zeroed it rises to around 0.035, roughly 5x worse,
and the gap holds across seeds. The test asserts the cross-on error stays under 0.05 and the
cross-off error is at least 1.5x larger, both with margin. The cross-attention ablation thus
measures the effect rather than asserting it. The single-pair cross-view reprojection-pixel error
is numerically unstable (points near the reprojected image plane blow up the pixel coordinate), so
the viz shows it as an error map but the overfit test does not assert it; reprojection consistency
is checked exactly on the ground truth instead. These are toy artifacts on one smooth sphere with
16 patches and no pretraining. If on some configuration the cross-attention or the confidence term
looked like it contributed little, that would be an artifact of the toy's simplicity, not a
statement about DUSt3R.

## References

- Wang et al. 2024, DUSt3R, [arXiv:2312.14132](https://arxiv.org/abs/2312.14132).
- Weinzaepfel et al. 2022, CroCo, [arXiv:2210.10716](https://arxiv.org/abs/2210.10716).
- Ranftl et al. 2021, "Vision Transformers for Dense Prediction" (the DPT head),
  [arXiv:2103.13413](https://arxiv.org/abs/2103.13413).
- Leroy et al. 2024, MASt3R, [arXiv:2406.09756](https://arxiv.org/abs/2406.09756).
- Yang et al. 2024, Depth Anything V2, [arXiv:2406.09414](https://arxiv.org/abs/2406.09414).
- Ke et al. 2024, Marigold, [arXiv:2312.02145](https://arxiv.org/abs/2312.02145).
- Wang et al. 2025, VGGT, [arXiv:2503.11651](https://arxiv.org/abs/2503.11651).
