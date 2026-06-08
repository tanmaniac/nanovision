"""BEVFormer-style query-pull view transform (filled solution).

The shared library re-exports these symbols through ``nanovision.bevformer`` (the
orchestrator-owned shim), so tests and this assignment's ``viz.py`` import them via
``from nanovision.bevformer import ...``, never by bare name.

The mechanisms, in teaching order:
- ``bev_reference_points``: each BEV cell center becomes a vertical pillar of ``n_heights``
  3D points in the ego frame, the anchor locations the BEV query will sample features at.
- ``project_reference_points``: project those ego pillars into every camera with the rig's
  ``world_to_pixel``, turn the pixels into grid_sample coordinates, and return a per-camera
  in-frame mask.
- ``SpatialCrossAttention``: the query-pull view transform. Each BEV cell bilinear-samples the
  image features at its projected reference points and averages across reference heights and
  across the cameras that see the pillar (the spatial cross-attention). With ``offsets=True`` it
  predicts learned sampling offsets around each reference point (the deformable-attention path).
- ``warp_bev``: resample the previous frame's BEV feature grid by the ego motion so a static
  world point stays at the same ego BEV cell after the ego moves.
- ``TemporalSelfAttention``: each BEV cell attends over the two-element set {current query,
  warped history}, carrying state across frames.
- ``BEVFormerEncoder`` / ``BEVFormerSeg``: the assembled encoder (temporal self-attention ->
  spatial cross-attention -> feed-forward, stacked) and a BEV segmentation head.

Conventions (fixed by ``nanovision.geometry`` and the toy substrate): ego frame x forward,
y left, z up; camera frame OpenCV x right, y down, z forward; the extrinsic ``E`` is
``T_cam_ego`` (ego -> camera). The BEV tensor is laid out ``(C, nx, ny)`` with nx along ego x
(forward) and ny along ego y (lateral). The deformable and temporal attention reuse the
multi-head attention from the transformer assignment via ``nanovision.attention``; the only
prebuilt ops used are ``F.grid_sample`` and ``F.affine_grid`` (the bilinear-sampling substrate).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanovision.attention import MultiHeadAttention
from nanovision.geometry import BEVGrid, CameraRig


def bev_reference_points(
    bev_grid: BEVGrid, n_heights: int, z_min: float, z_max: float, dtype=torch.float32
) -> Tensor:
    """Build a vertical pillar of 3D reference points over every BEV cell, in the ego frame.

    Each cell center ``(x, y)`` (from ``BEVGrid.cell_centers``) is repeated at ``n_heights``
    z-values spaced uniformly in ``[z_min, z_max]`` (inclusive endpoints, ``linspace``). These
    pillars are the anchor 3D points the BEV query later projects into the cameras.

    Args:
        bev_grid: the centered BEV grid contract.
        n_heights: number of anchor heights per pillar.
        z_min, z_max: ego-z range of the anchor heights (meters).

    Returns:
        (nx, ny, n_heights, 3) ego-frame points; index [i, j, h] is the cell (i, j) center at
        the h-th height.
    """
    centers = bev_grid.cell_centers(dtype=dtype)          # (nx, ny, 2) ego (x, y)
    nx, ny, _ = centers.shape
    zs = torch.linspace(z_min, z_max, n_heights, dtype=dtype)  # (n_heights,)
    xy = centers[:, :, None, :].expand(nx, ny, n_heights, 2)   # (nx, ny, n_heights, 2)
    z = zs.view(1, 1, n_heights, 1).expand(nx, ny, n_heights, 1)
    return torch.cat([xy, z], dim=-1)                     # (nx, ny, n_heights, 3)


def project_reference_points(
    ref3d: Tensor, rig: CameraRig, image_hw: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Project ego pillar points into every rig camera and return grid_sample coords + a mask.

    Each ego point is projected with ``rig.world_to_pixel`` (the same projection chain the
    camera-geometry assignment defined), which returns pixel ``(u, v)`` and a combined mask of
    points in front of the camera and inside the image bounds. The pixels are normalized to
    grid_sample's ``[-1, 1]`` extent with the ``align_corners=False`` map

        gx = 2 * (u + 0.5) / W - 1,    gy = 2 * (v + 0.5) / H - 1

    where ``(W, H) = image_hw`` is the FULL image size. The last dim is ordered ``(gx, gy)``,
    i.e. (width, height), because ``F.grid_sample`` reads the last grid dim as x=width first.

    Normalize by the full image size, NOT by the downsampled feature-map size ``Wf = W // stride``.
    ``grid_sample`` on a feature map of any resolution maps the same ``[-1, 1]`` extent across the
    whole map, so it handles the stride itself. Normalizing by ``Wf`` instead would offset every
    sample by the stride factor while the hit-mask (from ``world_to_pixel``, computed in full-image
    pixels) still looked correct - a quiet garbage-features bug.

    Args:
        ref3d: (nx, ny, n_heights, 3) ego points from ``bev_reference_points``.
        rig: the camera rig built from the toy K + per-camera E (with image_sizes so the bounds
            mask is populated).
        image_hw: (W, H) full image size in pixels (the size the rig's bounds mask uses).

    Returns:
        uv: (n_cam, nx, ny, n_heights, 2) grid_sample coords, last dim (gx, gy).
        valid: (n_cam, nx, ny, n_heights) bool, True where the point is in front of the camera
            and inside the image bounds.
    """
    W, H = int(image_hw[0]), int(image_hw[1])
    nx, ny, nh, _ = ref3d.shape
    pts = ref3d.reshape(-1, 3)                             # (nx*ny*nh, 3)
    uv_cams, valid_cams = [], []
    for name in rig.names:
        px, valid = rig.world_to_pixel(name, pts)         # (P, 2), (P,)
        u, v = px[..., 0], px[..., 1]
        gx = 2.0 * (u + 0.5) / W - 1.0
        gy = 2.0 * (v + 0.5) / H - 1.0
        uv = torch.stack([gx, gy], dim=-1)                # (P, 2) ordered (width, height)
        uv_cams.append(uv.reshape(nx, ny, nh, 2))
        valid_cams.append(valid.reshape(nx, ny, nh))
    uv = torch.stack(uv_cams, dim=0)                      # (n_cam, nx, ny, nh, 2)
    valid = torch.stack(valid_cams, dim=0)                # (n_cam, nx, ny, nh)
    return uv, valid


def _reduce_over_heights_and_views(sampled: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Average sampled features over valid heights, then over the cameras that see the pillar.

    Shared by the simplified and deformable spatial-cross-attention paths so the two agree
    exactly when the offset head is zero. A pillar in-frame at only 1 of ``n_heights`` heights
    must not be averaged as if all heights contributed, so the per-camera mean divides by the
    count of VALID heights, not by ``n_heights``. A camera is a "hit view" for a cell if at least
    one reference height projects in-frame; the cross-camera mean divides by the number of hit
    views (the paper's ``|V_hit|`` semantics). Cells no camera sees are flagged so the caller can
    leave their query unchanged.

    Args:
        sampled: (n_cam, C, nx, ny, n_heights) bilinear-sampled features at the reference points.
        valid: (n_cam, nx, ny, n_heights) bool in-frame mask.

    Returns:
        out: (C, nx, ny) reduced features (0 on no-hit cells).
        no_hit: (nx, ny) bool, True where no camera sees the cell.
    """
    m = valid.float()                                          # (n_cam, nx, ny, n_heights)
    s = (sampled * m[:, None]).sum(-1)                         # (n_cam, C, nx, ny)
    per_cam = s / m.sum(-1).clamp_min(1)[:, None]              # mean over VALID heights
    cam_hit = (m.sum(-1) > 0)                                  # (n_cam, nx, ny)
    out = (per_cam * cam_hit[:, None]).sum(0) / cam_hit.sum(0).clamp_min(1)[None]  # (C, nx, ny)
    no_hit = cam_hit.sum(0) == 0                               # (nx, ny)
    return out, no_hit


class SpatialCrossAttention(nn.Module):
    """The query-pull view transform: BEV cells bilinear-sample image features at their pillars.

    Each BEV cell projects its 3D reference pillar into every camera and samples the camera
    feature map at those locations, then averages over reference heights and over hit views.
    This is "cross-attention" with the attention weights fixed by geometry (a uniform average over
    the valid samples), the spatial cross-attention of BEVFormer.

    With ``offsets=True`` the module predicts ``n_points`` learned sampling offsets per head
    around each reference point and softmax weights over them (the deformable-attention path of
    Deformable DETR). The value and output projections are shared with the simplified path, so a
    zero-initialized offset head makes the deformable forward byte-equal to the simplified one.

    Args:
        dim: feature channels C (BEV query and image features share C).
        n_heads: number of deformable heads (only used when offsets=True).
        offsets: turn on the learned-deformable-offset path.
        n_points: deformable samples per reference point per head (offsets=True only).
    """

    def __init__(self, dim: int, n_heads: int = 4, offsets: bool = False, n_points: int = 4):
        super().__init__()
        self.dim = dim
        self.offsets = offsets
        self.n_heads = n_heads
        self.n_points = n_points
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        if offsets:
            # Offsets in normalized grid units per (head, point), and a weight per (head, point).
            self.offset_head = nn.Linear(dim, n_heads * n_points * 2)
            self.weight_head = nn.Linear(dim, n_heads * n_points)
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)

    def _value_maps(self, feats: Tensor) -> Tensor:
        """Apply the value projection to each camera feature map. (n_cam, C, Hf, Wf) -> same."""
        n_cam, C, Hf, Wf = feats.shape
        v = feats.permute(0, 2, 3, 1).reshape(-1, C)
        v = self.value_proj(v).reshape(n_cam, Hf, Wf, C).permute(0, 3, 1, 2)
        return v.contiguous()

    def forward(self, query: Tensor, feats: Tensor, ref_uv: Tensor, valid: Tensor) -> Tensor:
        """Pull image features into BEV cells at the projected reference points.

        Args:
            query: (nx, ny, C) BEV queries.
            feats: (n_cam, C, Hf, Wf) per-camera feature maps.
            ref_uv: (n_cam, nx, ny, n_heights, 2) grid_sample coords from project_reference_points.
            valid: (n_cam, nx, ny, n_heights) bool in-frame mask.

        Returns:
            (nx, ny, C) updated queries (residual add; no-hit cells keep the input query).
        """
        n_cam, C, Hf, Wf = feats.shape
        nx, ny, nh = valid.shape[1], valid.shape[2], valid.shape[3]
        value = self._value_maps(feats)                       # (n_cam, C, Hf, Wf)

        if not self.offsets:
            grid = ref_uv.reshape(n_cam, nx, ny * nh, 2)
            sampled = F.grid_sample(
                value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )                                                 # (n_cam, C, nx, ny*nh)
            sampled = sampled.reshape(n_cam, C, nx, ny, nh)
            out, no_hit = _reduce_over_heights_and_views(sampled, valid)
        else:
            nhd, npt = self.n_heads, self.n_points
            off = self.offset_head(query).reshape(nx, ny, nhd, npt, 2)          # (nx,ny,H,P,2)
            w = self.weight_head(query).reshape(nx, ny, nhd, npt)
            w = torch.softmax(w, dim=-1)                                        # (nx,ny,H,P)
            # Sampling locations: reference point + learned offset, per head and point.
            # ref_uv is (n_cam, nx, ny, nh, 2); broadcast the per-cell offsets across cameras and
            # reference heights, and average the head outputs (heads share the same value map).
            base = ref_uv[:, :, :, :, None, None, :]          # (n_cam, nx, ny, nh, 1, 1, 2)
            delta = off[None, :, :, None, :, :, :]            # (1, nx, ny, 1, H, P, 2)
            loc = base + delta                                # (n_cam, nx, ny, nh, H, P, 2)
            grid = loc.reshape(n_cam, nx, ny * nh * nhd * npt, 2)
            sampled = F.grid_sample(
                value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )                                                 # (n_cam, C, nx, ny*nh*H*P)
            sampled = sampled.reshape(n_cam, C, nx, ny, nh, nhd, npt)
            # Weighted sum over the P points, then mean over the H heads -> per (cam, height).
            wb = w[None, None, :, :, None, :, :]              # (1,1,nx,ny,1,H,P)
            sampled = (sampled * wb).sum(-1).mean(-1)         # (n_cam, C, nx, ny, nh)
            out, no_hit = _reduce_over_heights_and_views(sampled, valid)

        # Output projection + residual; leave no-hit cells as the input query.
        upd = self.out_proj(out.permute(1, 2, 0))             # (nx, ny, C)
        res = query + upd
        return torch.where(no_hit[:, :, None], query, res)


def warp_bev(prev_bev: Tensor, ego_delta: Tensor, bev_grid: BEVGrid) -> Tensor:
    """Resample the previous BEV feature grid by the ego motion (the ego-motion warp).

    A static world point sits at a different ego BEV cell once the ego moves, so the previous
    frame's features must be resampled into the current ego frame before temporal fusion. The
    BEV tensor is ``(C, H=nx=forward, W=ny=lateral)``. ``F.affine_grid``'s ``theta`` row 0 is the
    W axis (lateral, ego +y / left) and row 1 is the H axis (forward, ego +x); the sampling grid's
    last dim is ``(x=W, y=H)``.

    ``affine_grid`` builds a SAMPLING (inverse) warp: for output cell p it gives the source cell to
    read. After a forward ego translation of ``k_x = forward_m / res`` cells, a static world point
    that was at forward index i must appear at a LOWER current index ``i - k_x``; reading that
    output cell from source index ``i`` needs a normalized translation of ``+2*k_x/nx`` in
    ``theta[1, 2]`` (the H row). ``-2*k_x/nx`` would send it to ``i + k_x`` (the double-inverse
    bug). The lateral term goes in ``theta[0, 2] = +2*k_y/ny``. Yaw rotates the 2x2 block of
    ``theta`` consistently with the same (W=col0, H=row1) assignment. Zero ego motion is the
    identity warp.

    Args:
        prev_bev: (C, nx, ny) previous-frame BEV features.
        ego_delta: (3,) SE(2) ego motion (forward_m, lateral_m, yaw_rad) from t-1 to t.
        bev_grid: the BEV grid (for resolution and nx/ny).

    Returns:
        (C, nx, ny) the previous features resampled into the current ego frame.
    """
    C, nx, ny = prev_bev.shape
    res = bev_grid.resolution
    fwd_m, lat_m, yaw = float(ego_delta[0]), float(ego_delta[1]), float(ego_delta[2])
    k_x = fwd_m / res          # forward shift in cells (H axis, row 1)
    k_y = lat_m / res          # lateral shift in cells (W axis, row 0)

    cos, sin = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    dtype, device = prev_bev.dtype, prev_bev.device
    # theta acts on the normalized output grid (x=W, y=H). Row 0 = W (lateral), row 1 = H (fwd).
    # The 2x2 rotation is the inverse (transpose) rotation of the ego yaw, consistent with the
    # inverse-warp convention; for the toy's pure-translation tests yaw is 0 so it is identity.
    theta = torch.zeros(1, 2, 3, dtype=dtype, device=device)
    theta[0, 0, 0] = cos
    theta[0, 0, 1] = sin
    theta[0, 1, 0] = -sin
    theta[0, 1, 1] = cos
    theta[0, 0, 2] = 2.0 * k_y / ny     # lateral, W axis
    theta[0, 1, 2] = 2.0 * k_x / nx     # forward, H axis

    grid = F.affine_grid(theta, (1, C, nx, ny), align_corners=False)
    warped = F.grid_sample(
        prev_bev[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return warped[0]


class TemporalSelfAttention(nn.Module):
    """Attend each BEV cell over {current query, warped history} (temporal self-attention).

    For each BEV cell the query attends over a two-element key/value set: itself and the warped
    previous-frame feature at the same cell. This carries state across frames, letting the encoder
    recover content that the current cameras miss (an occluded object still in the warped history).
    On the first frame ``prev_bev_warped`` is None and this falls back to self-attention on the
    query alone. The attention is the multi-head attention from the transformer assignment.

    Args:
        dim: feature channels C.
        n_heads: attention heads.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads)

    def forward(self, query: Tensor, prev_bev_warped: Tensor | None) -> Tensor:
        """Temporal self-attention with residual add.

        Args:
            query: (nx, ny, C) current BEV queries.
            prev_bev_warped: (C, nx, ny) warped history, or None on the first frame.

        Returns:
            (nx, ny, C) updated queries.
        """
        nx, ny, C = query.shape
        q = query.reshape(nx * ny, 1, C)                      # (P, 1, C): one query per cell
        if prev_bev_warped is None:
            kv = q                                             # self-attention on the query alone
        else:
            prev = prev_bev_warped.permute(1, 2, 0).reshape(nx * ny, 1, C)
            kv = torch.cat([q, prev], dim=1)                  # (P, 2, C): {query, history}
        out = self.attn(q, kv=kv)                             # (P, 1, C)
        return query + out.reshape(nx, ny, C)


class _FeedForward(nn.Module):
    """A two-layer position-wise MLP with a residual, applied per BEV cell."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class _EncoderLayer(nn.Module):
    """One BEVFormer encoder layer: temporal self-attention -> spatial cross-attention -> FFN."""

    def __init__(self, dim: int, n_heads: int, offsets: bool, n_points: int, ffn_hidden: int):
        super().__init__()
        self.tsa = TemporalSelfAttention(dim, n_heads)
        self.sca = SpatialCrossAttention(dim, n_heads, offsets=offsets, n_points=n_points)
        self.ffn = _FeedForward(dim, ffn_hidden)


class BEVFormerEncoder(nn.Module):
    """The assembled BEVFormer encoder: a learnable BEV query refined by stacked layers.

    A learnable BEV query embedding ``(nx, ny, C)`` is the starting state. Each layer runs
    temporal self-attention against the warped previous BEV grid, then spatial cross-attention
    pulling from the camera features, then a feed-forward. The 3D reference pillars are precomputed
    once (they are fixed by the BEV grid and the anchor heights). The encoder returns the final
    dense BEV feature grid ``(C, nx, ny)``, the canonical intermediate later occupancy and
    map/prediction assignments consume.

    Args:
        cfg: a BEVFormerConfig (BEV grid, channels, heads, layers, image size, anchor heights).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bev_grid = cfg.bev_grid()
        nx, ny, C = self.bev_grid.nx, self.bev_grid.ny, cfg.dim
        self.query_embed = nn.Parameter(torch.zeros(nx, ny, C))
        nn.init.normal_(self.query_embed, std=0.02)
        self.layers = nn.ModuleList(
            _EncoderLayer(C, cfg.n_heads, cfg.offsets, cfg.n_points, cfg.ffn_hidden)
            for _ in range(cfg.n_layers)
        )
        ref = bev_reference_points(self.bev_grid, cfg.n_heights, cfg.z_min, cfg.z_max)
        self.register_buffer("ref3d", ref)                    # (nx, ny, n_heights, 3)

    def forward(
        self,
        feats: Tensor,
        rig: CameraRig,
        prev_bev: Tensor | None = None,
        ego_delta: Tensor | None = None,
    ) -> Tensor:
        """Run the encoder for one frame (toy B=1).

        Args:
            feats: (n_cam, C, Hf, Wf) per-camera feature maps for the current frame.
            rig: the camera rig (with image_sizes) for projecting the reference points.
            prev_bev: (C, nx, ny) previous-frame BEV features, or None on the first frame.
            ego_delta: (3,) SE(2) ego motion t-1 -> t, used to warp prev_bev. Ignored if
                prev_bev is None.

        Returns:
            (C, nx, ny) the final dense BEV feature grid.
        """
        image_hw = (self.cfg.img, self.cfg.img)
        ref_uv, valid = project_reference_points(self.ref3d, rig, image_hw)

        if prev_bev is not None:
            if ego_delta is None:
                ego_delta = torch.zeros(3, device=feats.device, dtype=feats.dtype)
            prev_warped = warp_bev(prev_bev, ego_delta, self.bev_grid)
        else:
            prev_warped = None

        query = self.query_embed
        for layer in self.layers:
            query = layer.tsa(query, prev_warped)
            query = layer.sca(query, feats, ref_uv, valid)
            query = layer.ffn(query)
        return query.permute(2, 0, 1).contiguous()            # (C, nx, ny)


class BEVFormerSeg(nn.Module):
    """BEVFormer encoder plus a 1x1 BEV segmentation head.

    Args:
        cfg: a BEVFormerConfig.
        n_classes: output channels of the segmentation head (1 for binary occupancy).
    """

    def __init__(self, cfg, n_classes: int = 1):
        super().__init__()
        self.encoder = BEVFormerEncoder(cfg)
        self.seg_head = nn.Conv2d(cfg.dim, n_classes, kernel_size=1)

    def forward(
        self,
        feats: Tensor,
        rig: CameraRig,
        prev_bev: Tensor | None = None,
        ego_delta: Tensor | None = None,
    ) -> Tensor:
        """Encode the BEV grid and apply the segmentation head.

        Args:
            feats: (n_cam, C, Hf, Wf) per-camera feature maps.
            rig: the camera rig.
            prev_bev: (C, nx, ny) previous BEV features or None.
            ego_delta: (3,) ego motion or None.

        Returns:
            (n_classes, nx, ny) BEV segmentation logits.
        """
        bev = self.encoder(feats, rig, prev_bev, ego_delta)   # (C, nx, ny)
        return self.seg_head(bev[None])[0]                    # (n_classes, nx, ny)
