"""Lift-Splat-Shoot: push image features out into 3D and splat them into a BEV grid.

Reference (filled) implementation. The holed top-level ``lift_splat.py`` carries the identical
docstrings and provided bodies; only the hole bodies below differ. The shared library
re-exports these symbols through ``nanovision.lift_splat``, so tests and the assignment's own
``viz.py`` import them via ``from nanovision.lift_splat import ...``, never by bare name.

The four mechanisms, in order:
- ``DepthLift``: two 1x1 conv heads turn a feature map into a per-pixel categorical depth
  distribution (D bins) and a context vector (C channels), then the outer product lifts the
  feature into a (D, C) volume per pixel - the central differentiable op of Lift-Splat-Shoot.
- ``frustum_points``: for every feature cell and every depth bin, back-project the pixel center
  to a 3D camera-frame point and transform it into the ego frame. Reuses the camera-geometry
  primitives (``unproject``, ``invert_transform``, ``apply_transform``).
- ``pillar_index`` + ``cumsum_pool``: the splat. Each lifted point falls into a BEV pillar
  (a flat integer index); ``cumsum_pool`` sums all points in a pillar with a sort+cumsum trick
  (no scatter_add), which is differentiable and avoids a per-point atomic add.
- ``LiftSplatShoot``: the assembled model (backbone -> lift -> frustum -> splat -> BEV encoder
  -> segmentation head).
- ``bevdepth_depth_loss``: BEVDepth's explicit depth supervision, a cross-entropy on the GT
  depth bin at labeled feature cells.

Conventions (fixed by ``nanovision.geometry`` and the toy substrate): ego frame x forward,
y left, z up; camera frame OpenCV x right, y down, z forward; the extrinsic ``E`` is
``T_cam_ego`` (ego -> camera). Depth bin centers are ``arange(d_min, d_max, d_step)``,
EXCLUSIVE of ``d_max`` (so d_min=1, d_max=9, d_step=1 gives D=8 centers [1..8], deepest
reachable point 8 m forward). The flat pillar index is ``ix * ny + iy`` everywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanovision.geometry import (
    BEVGrid,
    apply_transform,
    invert_transform,
    unproject,
)


class DepthLift(nn.Module):
    """Per-pixel categorical depth + context, lifted by an outer product.

    The depth head predicts D logits per pixel (a distribution over discrete depth bins);
    the context head predicts a C-channel feature per pixel. The lift forms the outer product
    of the softmax depth distribution with the context, giving a (D, C) volume per pixel: the
    context feature, scaled by the probability that the pixel's content sits at each depth.
    No depth label is needed - depth is a latent ("implicit") variable trained from the task
    loss in vanilla LSS.

    Args:
        c_in: input feature channels.
        D: number of depth bins.
        C_ctx: context channels.
    """

    def __init__(self, c_in: int, D: int, C_ctx: int):
        super().__init__()
        self.D = D
        self.C_ctx = C_ctx
        self.depth_head = nn.Conv2d(c_in, D, kernel_size=1)
        self.ctx_head = nn.Conv2d(c_in, C_ctx, kernel_size=1)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        """Map a feature map to depth logits and a context map.

        Args:
            feat: (B, c_in, Hf, Wf).

        Returns:
            depth_logits: (B, D, Hf, Wf).
            context: (B, C_ctx, Hf, Wf).
        """
        depth_logits = self.depth_head(feat)
        context = self.ctx_head(feat)
        return depth_logits, context

    def lift(self, feat: Tensor) -> Tensor:
        """Outer-product lift: (B, c_in, Hf, Wf) -> (B, D, C_ctx, Hf, Wf).

        Softmax the depth logits over the D bins, then multiply by the context map:
        ``volume[b, d, c, i, j] = softmax(depth_logits)[b, d, i, j] * context[b, c, i, j]``.
        """
        depth_logits, context = self.forward(feat)
        alpha = F.softmax(depth_logits, dim=1)             # (B, D, Hf, Wf)
        # Outer product over (depth bin d, context channel c) per pixel.
        volume = alpha[:, :, None] * context[:, None]      # (B, D, C, Hf, Wf)
        return volume


def frustum_points(pixel_xy: Tensor, depths: Tensor, K: Tensor, E: Tensor) -> Tensor:
    """Back-project a frustum of pixel centers at each depth into ego-frame 3D points.

    For each depth bin d and feature cell (i, j), back-project the pixel center to a
    camera-frame point with ``unproject(pixel, depth=d, K)``, then transform it into the ego
    frame with the inverse of E (E is T_cam_ego, so its inverse is T_ego_cam).

    Args:
        pixel_xy: (Hf, Wf, 2) image-pixel centers (u, v) per feature cell.
        depths: (D,) depth-bin centers.
        K: (3, 3) intrinsic.
        E: (4, 4) extrinsic T_cam_ego (ego -> camera).

    Returns:
        (D, Hf, Wf, 3) ego-frame points.
    """
    Hf, Wf, _ = pixel_xy.shape
    D = depths.shape[0]
    px = pixel_xy.reshape(-1, 2)                              # (Hf*Wf, 2)
    # Tile pixels over depths and depths over pixels: a (D, Hf*Wf) grid of (pixel, depth).
    px_rep = px[None].expand(D, -1, 2).reshape(-1, 2)         # (D*Hf*Wf, 2)
    d_rep = depths.reshape(D, 1).expand(D, Hf * Wf).reshape(-1)  # (D*Hf*Wf,)
    p_cam = unproject(px_rep, d_rep, K)                       # (D*Hf*Wf, 3) camera frame
    T_ego_cam = invert_transform(E)                          # E is T_cam_ego
    p_ego = apply_transform(T_ego_cam, p_cam)                # (D*Hf*Wf, 3) ego frame
    return p_ego.reshape(D, Hf, Wf, 3)


def pillar_index(pts_ego_xy: Tensor, bev_grid: BEVGrid) -> Tensor:
    """Map ego (x, y) points to flat BEV pillar indices, -1 for out-of-bounds.

    A point at ego (x, y) falls in cell ``ix = floor((x - x_min) / res)`` along forward x and
    ``iy = floor((y - y_min) / res)`` along lateral y. The flat index is ``ix * ny + iy``.
    Points outside ``[x_min, x_max) x [y_min, y_max)`` map to -1 (dropped by the pool).

    Args:
        pts_ego_xy: (N, 2) ego (x, y).
        bev_grid: the BEVGrid contract.

    Returns:
        (N,) long flat pillar index, -1 where out of bounds.
    """
    x = pts_ego_xy[..., 0]
    y = pts_ego_xy[..., 1]
    res = bev_grid.resolution
    ix = torch.floor((x - bev_grid.x_min) / res).long()
    iy = torch.floor((y - bev_grid.y_min) / res).long()
    in_bounds = (ix >= 0) & (ix < bev_grid.nx) & (iy >= 0) & (iy < bev_grid.ny)
    flat = ix * bev_grid.ny + iy
    return torch.where(in_bounds, flat, torch.full_like(flat, -1))


def cumsum_pool(feats: Tensor, idx: Tensor, n_bins: int) -> Tensor:
    """Sum features that share a bin index, via the sort+cumsum trick (no scatter_add).

    Drop points with ``idx < 0``. Sort the remaining points by ``idx`` (stable), cumsum the
    sorted features along N, then the sum for a bin is the cumsum at the last row of its
    equal-idx run minus the cumsum at the last row of the previous run. Equivalently, keep the
    last row of each run and take successive differences. Scatter each run's sum into
    ``out[bin]``. Differentiable wrt ``feats`` (the sort is a fixed permutation given ``idx``).

    Args:
        feats: (N, C).
        idx: (N,) long in [-1, n_bins).
        n_bins: number of output bins.

    Returns:
        (n_bins, C) pooled sums.
    """
    C = feats.shape[1]
    out = feats.new_zeros(n_bins, C)
    keep = idx >= 0
    if keep.sum() == 0:
        return out
    feats = feats[keep]
    idx = idx[keep]

    order = torch.argsort(idx, stable=True)
    idx_s = idx[order]
    feats_s = feats[order]
    cum = feats_s.cumsum(dim=0)                              # (M, C)

    # The last row of each equal-idx run: a run boundary is where idx changes (and the final row).
    N = idx_s.shape[0]
    boundary = torch.ones(N, dtype=torch.bool, device=idx_s.device)
    boundary[:-1] = idx_s[1:] != idx_s[:-1]                  # True at the last row of each run
    run_last = torch.nonzero(boundary, as_tuple=False).squeeze(1)  # indices of run-ends
    run_cum = cum[run_last]                                  # cumsum at each run's last row
    # Successive differences give the per-run sum (the first run is just its cumsum).
    run_sum = torch.cat([run_cum[:1], run_cum[1:] - run_cum[:-1]], dim=0)
    run_bins = idx_s[run_last]                               # the bin id of each run
    out = out.index_copy(0, run_bins, run_sum)
    return out


class LiftSplatShoot(nn.Module):
    """The assembled Lift-Splat-Shoot model: backbone -> lift -> splat -> BEV head.

    A tiny conv backbone downsamples the camera image to a feature grid; ``DepthLift`` lifts
    each feature cell into a (D, C) volume; ``frustum_points`` places every (depth, cell) entry
    in the ego frame; ``pillar_index`` + ``cumsum_pool`` splat the volume into a BEV grid; a
    small BEV encoder and a 1x1 segmentation head produce the per-cell occupancy logit.

    Batch handling: the toy is a single scene (B=1). ``forward`` accepts ``images`` of shape
    (B, 3, H, W) but the splat indexing is written for B=1; a multi-batch caller should loop.

    Args:
        cfg: an LSSConfig (provides image size, stride, channels, depth bins, BEV grid).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bev_grid = cfg.bev_grid()
        D = cfg.D
        C = cfg.c_ctx
        # Tiny conv backbone: stride down to the feature grid (img -> img // stride).
        layers = []
        c_prev = 3
        c_hidden = cfg.c_backbone
        n_down = 0
        s = cfg.stride
        while (1 << n_down) < s:
            n_down += 1
        for k in range(n_down):
            layers += [nn.Conv2d(c_prev, c_hidden, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            c_prev = c_hidden
        layers += [nn.Conv2d(c_prev, c_hidden, 3, padding=1), nn.ReLU(inplace=True)]
        self.backbone = nn.Sequential(*layers)
        self.depth_lift = DepthLift(c_in=c_hidden, D=D, C_ctx=C)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(C, 1, kernel_size=1)
        self.register_buffer("bins", cfg.bins())
        self.register_buffer("pixel_xy", cfg.pixel_xy())

    def forward(self, images: Tensor, K: Tensor, E: Tensor) -> Tensor:
        """Run one (or more) camera images through the full LSS pipeline.

        Steps for each camera:
        1. ``backbone(image)`` -> feature map (1, C_bb, Hf, Wf).
        2. ``depth_lift.lift`` -> volume (1, D, C, Hf, Wf).
        3. ``frustum_points(self.pixel_xy, self.bins, K, E)`` -> ego points (D, Hf, Wf, 3).
        4. Flatten the volume to (N, C) and the ego (x, y) to (N, 2) over (D, Hf, Wf).
        5. ``pillar_index`` -> idx (N,); ``cumsum_pool`` -> (nx*ny, C).
        6. Reshape to (C, nx, ny), run ``bev_encoder`` then ``seg_head``.

        Sum the pooled BEV contributions across cameras before the encoder.

        Args:
            images: (B, 3, H, W) - B handled as 1 in the toy.
            K: (3, 3) or (B, 3, 3) intrinsic.
            E: (4, 4) or (B, 4, 4) extrinsic T_cam_ego.

        Returns:
            (B, 1, nx, ny) occupancy logit.
        """
        B = images.shape[0]
        grid = self.bev_grid
        nx, ny = grid.nx, grid.ny
        C = self.cfg.c_ctx
        outs = []
        for b in range(B):
            img = images[b:b + 1]                            # (1, 3, H, W)
            Kb = K[b] if K.dim() == 3 else K
            Eb = E[b] if E.dim() == 3 else E

            feat = self.backbone(img)                        # (1, C_bb, Hf, Wf)
            volume = self.depth_lift.lift(feat)              # (1, D, C, Hf, Wf)
            _, D, _, Hf, Wf = volume.shape

            ego = frustum_points(self.pixel_xy, self.bins, Kb, Eb)  # (D, Hf, Wf, 3)
            # Flatten the volume to (N, C) and ego (x, y) to (N, 2) over (D, Hf, Wf).
            feats = volume[0].permute(0, 2, 3, 1).reshape(-1, C)    # (D*Hf*Wf, C)
            ego_xy = ego[..., :2].reshape(-1, 2)                    # (D*Hf*Wf, 2)

            idx = pillar_index(ego_xy, grid)                 # (N,)
            pooled = cumsum_pool(feats, idx, nx * ny)        # (nx*ny, C)
            bev = pooled.reshape(nx, ny, C).permute(2, 0, 1)  # (C, nx, ny)
            outs.append(bev)

        bev = torch.stack(outs, dim=0)                       # (B, C, nx, ny)
        bev = self.bev_encoder(bev)
        logit = self.seg_head(bev)                           # (B, 1, nx, ny)
        return logit


def bevdepth_depth_loss(depth_logits: Tensor, depth_bin_labels: Tensor, mask: Tensor) -> Tensor:
    """BEVDepth's explicit depth-supervision loss: CE on the GT depth bin at labeled cells.

    Cross-entropy over the D depth bins at the masked feature cells only, averaged over the
    labeled cells. With no labeled cell the loss is 0 (a no-label convention).

    Args:
        depth_logits: (B, D, Hf, Wf).
        depth_bin_labels: (B, Hf, Wf) long, GT bin index at each cell.
        mask: (B, Hf, Wf) bool, True at labeled cells.

    Returns:
        scalar loss (mean CE over labeled cells, or 0 if none).
    """
    B, D, Hf, Wf = depth_logits.shape
    # (B, D, Hf, Wf) -> (N, D) and (N,) over masked cells only.
    logits = depth_logits.permute(0, 2, 3, 1).reshape(-1, D)   # (B*Hf*Wf, D)
    labels = depth_bin_labels.reshape(-1)                       # (B*Hf*Wf,)
    m = mask.reshape(-1)
    if m.sum() == 0:
        return depth_logits.sum() * 0.0                        # 0, keeps the graph/dtype
    sel_logits = logits[m]
    sel_labels = labels[m]
    return F.cross_entropy(sel_logits, sel_labels)
