"""Semantic 3D occupancy prediction by inverting the volume rendering integral.

A voxel grid holds a per-voxel occupancy probability and per-voxel class logits. The same
front-to-back alpha compositing that the NeRF assignment uses to render a known density field
to a pixel runs backward here: rays cast into the grid accumulate occupancy into a rendered
depth and a rendered semantic vector, and the 2D supervision (depth, class) pulls the 3D field
into agreement. The bridge is the identity between the per-voxel opacity and the occupancy
probability: with density $\\sigma_i$ and segment length $\\delta_i$, the segment opacity
$\\alpha_i = 1 - e^{-\\sigma_i \\delta_i}$ is the occupancy probability of that segment, so a
sampled occupancy $o \\in [0, 1)$ converts to a density $\\sigma = -\\log(1 - o) / \\delta$ and
the NeRF renderer composites it unchanged.

Module layout: this is an assignment-LOCAL file (nothing imports it, so there is no
nanovision shim). It is imported bare by the tests through conftest. The alpha-compositing
kernel is REUSED from the NeRF assignment through nanovision.volume.volume_render - do not
re-implement the transmittance cumulative product here.

Shapes used throughout:
- Voxel features:      [B, C, Z, Y, X]   (channels first; Z is the depth/height axis)
- Class logits:        [B, n_classes, Z, Y, X]
- Occupancy grid:      [Z, Y, X]         occupancy probability in [0, 1]
- Semantic grid:       [n_classes, Z, Y, X]
- Rays:                rays_o, rays_d [R, 3];  z_vals [R, N]
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanovision.volume import deltas_from_z, sample_along_rays, volume_render


def bev_to_voxel(bev_feats: Tensor, n_z: int, conv: nn.Conv2d) -> Tensor:
    """Lift a flat BEV feature map to a voxel feature volume by pillar extrusion.

    A bird's-eye-view feature map has collapsed the height axis. Pillar extrusion restores it:
    a 1x1 convolution predicts a per-height feature distribution for every BEV cell (a learnable
    spread over Z), not a bare repeat of the same vector at every height.

    Args:
        bev_feats: [B, C, Y, X] BEV features.
        n_z: number of voxel layers along Z to produce.
        conv: a provided Conv2d(C, C * n_z, 1) that maps each BEV cell to C * n_z channels.

    Returns:
        [B, C, Z, Y, X] voxel features with Z = n_z.
    """
    B, C, Y, X = bev_feats.shape
    extruded = conv(bev_feats)                       # [B, C * n_z, Y, X]
    return extruded.reshape(B, C, n_z, Y, X)         # [B, C, Z, Y, X]


class OccupancyHead(nn.Module):
    """Per-voxel semantic classifier over a voxel feature volume.

    Two 3D convolutions with a nonlinearity between them map voxel features to class logits.
    The first 1x1x1 conv mixes channels, the second produces n_classes logits per voxel. Class 0
    is the free (unoccupied) class; classes 1..n_classes-1 are occupied categories.
    """

    def __init__(self, channels: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv3d(channels, n_classes, kernel_size=1)

    def forward(self, voxel_feats: Tensor) -> Tensor:
        """Map voxel features to per-voxel class logits.

        Args:
            voxel_feats: [B, C, Z, Y, X].

        Returns:
            [B, n_classes, Z, Y, X] class logits.
        """
        return self.conv2(self.act(self.conv1(voxel_feats)))


def inverse_frequency_weights(target: Tensor, n_classes: int, eps: float = 1.0) -> Tensor:
    """Per-class loss weights that grow as a class gets rarer.

    Free voxels dominate the grid (often >90%), so an unweighted cross-entropy collapses to
    predicting the majority class. Inverse-frequency weighting counters this: weight class $c$
    by $1 / (\\text{count}_c + \\varepsilon)$, then normalize so the weights have mean 1
    (equivalently sum to n_classes). The normalization is scale-only and does not change the
    ordering: the most frequent class (free) gets the smallest weight, rare classes the largest.

    Args:
        target: [B, Z, Y, X] long class indices in [0, n_classes).
        n_classes: number of classes.
        eps: additive smoothing on the count so an absent class does not divide by zero.

    Returns:
        [n_classes] float weights with mean 1, on target's device.
    """
    counts = torch.bincount(target.reshape(-1), minlength=n_classes).float()  # [n_classes]
    w = 1.0 / (counts + eps)
    w = w * (n_classes / w.sum())        # normalize to mean 1 (sum = n_classes)
    return w.to(device=target.device)


def weighted_ce_loss(logits: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """Class-weighted cross-entropy with the weighted-mean reduction that F.cross_entropy uses.

    The reduction is the weighted mean over voxels, $\\sum_v w_{t_v} \\ell_v / \\sum_v w_{t_v}$,
    where $\\ell_v$ is the per-voxel cross-entropy and $w_{t_v}$ is the weight of voxel $v$'s
    target class. This matches F.cross_entropy(logits, target, weight=weights) exactly. The plain
    $\\sum_v w_{t_v} \\ell_v / N$ reduction differs by a factor $\\sum w / N$ and would not match.

    Args:
        logits: [B, n_classes, Z, Y, X] class logits.
        target: [B, Z, Y, X] long class indices.
        weights: [n_classes] per-class weights.

    Returns:
        scalar loss.
    """
    w = weights.to(logits.dtype)
    n_classes = logits.shape[1]
    # Flatten the spatial axes so cross-entropy is over [M, n_classes] vs [M].
    logits_flat = logits.movedim(1, -1).reshape(-1, n_classes)   # [M, n_classes]
    target_flat = target.reshape(-1)                             # [M]
    log_probs = F.log_softmax(logits_flat, dim=-1)               # [M, n_classes]
    per_voxel = -log_probs.gather(1, target_flat[:, None]).squeeze(1)  # [M]
    w_target = w[target_flat]                                    # [M]
    return (w_target * per_voxel).sum() / w_target.sum()


def occupancy_iou(
    pred_labels: Tensor, target: Tensor, n_classes: int, ignore_free: bool = True
) -> Tensor:
    """Mean intersection-over-union over the occupied classes.

    For each class, IoU is $|pred = c \\cap target = c| / |pred = c \\cup target = c|$. The mean
    is taken over occupied classes only (class 0 = free is excluded when ignore_free, so the
    metric is not dominated by the ~95% free voxels). A class absent from both prediction and
    target (empty union) is excluded from the mean, the standard mIoU convention.

    Args:
        pred_labels: [...] predicted class indices.
        target: [...] same shape, ground-truth class indices.
        n_classes: number of classes.
        ignore_free: exclude class 0 from the mean.

    Returns:
        scalar mean IoU. If no class has a non-empty union, returns 0.
    """
    pred = pred_labels.reshape(-1)
    tgt = target.reshape(-1)
    start = 1 if ignore_free else 0
    ious = []
    for c in range(start, n_classes):
        p = pred == c
        t = tgt == c
        union = (p | t).sum()
        if union == 0:
            continue                    # empty union: class absent from both, exclude
        inter = (p & t).sum()
        ious.append(inter.float() / union.float())
    if not ious:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(ious).mean()


def render_occupancy_rays(
    occ: Tensor,
    sem: Tensor,
    rays_o: Tensor,
    rays_d: Tensor,
    z_vals: Tensor,
    grid_bounds: tuple,
    z_far: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render an occupancy grid to per-ray depth and semantics by alpha compositing.

    The rendering-supervision step. Sample points along each ray, trilinearly sample the
    occupancy and semantic grids at those points, convert occupancy to density so the reused
    NeRF kernel produces the exact compositing weights, then accumulate depth and semantics.

    Trilinear sampling axis order (the highest-risk line). The grid is fed to F.grid_sample as
    [N, C, Z, Y, X] = [N, C, D, H, W], so D maps to Z, H to Y, W to X. grid_sample's last grid
    dimension is ordered (gx, gy, gz) mapping to the (W, H, D) = (X, Y, Z) axes:

        gx = 2 * (px - x0) / (x1 - x0) - 1      # X axis (W)
        gy = 2 * (py - y0) / (y1 - y0) - 1      # Y axis (H)
        gz = 2 * (pz - z0) / (z1 - z0) - 1      # Z axis (D)
        grid = stack([gx, gy, gz], dim=-1)      # order MUST be (gx, gy, gz)

    align_corners=False matches the voxel_centers cell-center convention (center i at
    a + (i + 0.5) * cell). A wrong stack order silently transposes the field (Z=8 vs Y=X=32 still
    broadcasts to a valid-but-garbage sample), so the order is pinned.

    Density bridge: deltas = deltas_from_z(z_vals); sigma = -log(clamp(1 - o, min=1e-6)) / delta,
    so alpha = 1 - exp(-sigma * delta) = o exactly. The compositing weights come from
    nanovision.volume.volume_render (its second return); the color argument is a zeros dummy
    because semantics are composited separately, not through the RGB color path. Depth adds the
    leftover-transmittance term so miss rays reach z_far.

    Args:
        occ: [Z, Y, X] occupancy probability in [0, 1].
        sem: [n_classes, Z, Y, X] semantic logits or probs.
        rays_o, rays_d: [R, 3] ray origins and unit directions.
        z_vals: [R, N] sample distances along each ray.
        grid_bounds: ((x0, x1), (y0, y1), (z0, z1)) metric grid extent.
        z_far: far-plane distance for the leftover-transmittance depth term.

    Returns:
        depth: [R] rendered depth.
        sem_out: [R, n_classes] rendered semantics.
        weights: [R, N] compositing weights from the NeRF kernel.
    """
    (x0, x1), (y0, y1), (z0, z1) = grid_bounds
    Z, Y, X = occ.shape
    n_classes = sem.shape[0]
    R, N = z_vals.shape

    # Points along each ray: [R, N, 3].
    pts = sample_along_rays(rays_o, rays_d, z_vals)
    px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]

    # Normalized grid coordinates in [-1, 1]. Order is (gx, gy, gz) -> (X=W, Y=H, Z=D).
    gx = 2.0 * (px - x0) / (x1 - x0) - 1.0
    gy = 2.0 * (py - y0) / (y1 - y0) - 1.0
    gz = 2.0 * (pz - z0) / (z1 - z0) - 1.0
    grid = torch.stack([gx, gy, gz], dim=-1)            # [R, N, 3]
    grid5 = grid[None, :, :, None, :]                   # [1, R, N, 1, 3] -> 5D for grid_sample

    occ_vol = occ[None, None]                           # [1, 1, Z, Y, X] = [N,C,D,H,W]
    sem_vol = sem[None]                                 # [1, n_classes, Z, Y, X]

    o_i = F.grid_sample(occ_vol, grid5, mode="bilinear",
                        align_corners=False, padding_mode="zeros")  # [1, 1, R, N, 1]
    o_i = o_i.reshape(R, N)                             # [R, N]
    s_i = F.grid_sample(sem_vol, grid5, mode="bilinear",
                        align_corners=False, padding_mode="zeros")  # [1, n_classes, R, N, 1]
    s_i = s_i.reshape(n_classes, R, N).permute(1, 2, 0)            # [R, N, n_classes]

    # Density bridge: sigma = -log(1 - o) / delta so alpha = 1 - exp(-sigma*delta) = o exactly.
    deltas = deltas_from_z(z_vals)                      # [R, N]
    sigma = -torch.log(torch.clamp(1.0 - o_i, min=1e-6)) / deltas  # [R, N]

    # Reuse the NeRF alpha-compositing kernel only for its weights = T_i * alpha_i.
    dummy_colors = torch.zeros(R, N, 3, dtype=o_i.dtype, device=o_i.device)
    _, weights = volume_render(sigma, dummy_colors, deltas)        # weights [R, N]

    acc = weights.sum(-1)                               # [R]
    depth = (weights * z_vals).sum(-1) + (1.0 - acc) * z_far       # [R]
    sem_out = (weights[..., None] * s_i).sum(1)         # [R, n_classes]
    return depth, sem_out, weights
