"""Train camera-only, LiDAR-only, and fused BEV seg heads and compare their IoU. Provided.

This is the measurement behind the fusion-beats-single claim. Each modality alone is built to be
insufficient on the toy: the camera BEV feature localizes a vehicle only to its lateral column
(depth ambiguity), and the LiDAR feature cannot tell a vehicle blob from a clutter blob (matched
geometry and density). The comparison trains three tiny heads across several toy scenes and
reports the mean vehicle-occupancy IoU on HELD-OUT scenes, so the numbers measure whether each
modality's signal generalizes, not whether a net can memorize one scene.

Not a hole: the three heads and the training loop are provided. The LiDAR and fused branches call
the mechanisms the learner fills in (``LidarPillarEncoder``, ``BEVFuser``); the camera branch is a
small conv over the toy's camera BEV feature.
"""

import torch
import torch.nn as nn

from fusion import BEVFuser, LidarPillarEncoder

from nanovision.data import toy


def _iou(pred_bool, gt_bool):
    inter = (pred_bool & gt_bool).sum().float()
    union = (pred_bool | gt_bool).sum().float()
    return (inter / union.clamp(min=1)).item()


def _make_scenes(seeds, n_ground, device):
    return [toy.bev_fusion_scene(seed=int(s), n_ground=n_ground, device=device) for s in seeds]


def compare_modalities(
    cfg,
    train_seeds=range(12),
    test_seeds=range(100, 108),
    steps: int = 400,
    lr: float = 1e-2,
    pos_weight: float = 30.0,
    n_ground: int = 80,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train the three heads and return their held-out mean IoU.

    Args:
        cfg: a FusionConfig (grid, channels, class count).
        train_seeds: toy-scene seeds for the training set.
        test_seeds: toy-scene seeds for the held-out evaluation set.
        steps: full-batch training steps per head.
        lr: Adam learning rate.
        pos_weight: BCE positive-class weight (the toy is ~2% occupied, so weight the positives).
        n_ground: ground points per scene (passed through to the toy).
        seed: RNG seed for parameter init (determinism).
        device: torch device.

    Returns:
        dict with mean held-out IoU under keys "camera", "lidar", "fused", and per-scene lists
        under "camera_scenes", "lidar_scenes", "fused_scenes".
    """
    grid = cfg.bev_grid()
    train = _make_scenes(train_seeds, n_ground, device)
    test = _make_scenes(test_seeds, n_ground, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    def build_camera():
        # Camera-only branch: a small conv over the toy's camera BEV feature (semantics, but the
        # depth-ambiguous column smear means it cannot localize the forward cell).
        net = nn.Sequential(
            nn.Conv2d(cfg.n_classes, cfg.fuse_hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.fuse_hidden, cfg.fuse_hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.fuse_hidden, 1, 1),
        ).to(device)
        return (lambda sc: net(sc["cam_bev"][None])), list(net.parameters())

    def build_lidar():
        # LiDAR-only branch: the pillar encoder (geometry, but no class signal to reject clutter).
        enc = LidarPillarEncoder(grid, cfg.lidar_hidden, cfg.lidar_channels).to(device)
        head = nn.Conv2d(cfg.lidar_channels, 1, 1).to(device)
        return (lambda sc: head(enc(sc["lidar"])[None])), \
            list(enc.parameters()) + list(head.parameters())

    def build_fused():
        # Fused branch: BEVFuser over the camera BEV feature and the LiDAR BEV feature.
        enc = LidarPillarEncoder(grid, cfg.lidar_hidden, cfg.lidar_channels).to(device)
        fuser = BEVFuser(cfg.n_classes, cfg.lidar_channels, cfg.fuse_hidden, cfg.fuse_channels).to(device)
        head = nn.Conv2d(cfg.fuse_channels, 1, 1).to(device)
        return (lambda sc: head(fuser(sc["cam_bev"], enc(sc["lidar"]))[None])), \
            list(enc.parameters()) + list(fuser.parameters()) + list(head.parameters())

    out = {}
    for name, builder in [("camera", build_camera), ("lidar", build_lidar), ("fused", build_fused)]:
        torch.manual_seed(seed)  # same init draw order for each branch
        fwd, params = builder()
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = sum(loss_fn(fwd(sc), sc["bev_gt"][None, None]) for sc in train) / len(train)
            loss.backward()
            opt.step()
        with torch.no_grad():
            scenes = [_iou(fwd(sc) > 0.0, sc["bev_gt"][None, None] > 0.5) for sc in test]
        out[name] = sum(scenes) / len(scenes)
        out[f"{name}_scenes"] = scenes
    return out
