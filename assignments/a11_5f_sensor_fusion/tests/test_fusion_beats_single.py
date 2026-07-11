"""Fusion beats either single modality on held-out scenes.

The toy is built so neither modality alone recovers vehicle occupancy: the camera BEV feature
localizes a vehicle only to its lateral column (depth ambiguity), and the LiDAR feature cannot
separate a vehicle blob from a clutter blob (matched geometry and density). A head trained on one
modality therefore hits a ceiling that generalization to held-out scenes exposes, while a fused
head that keeps LiDAR geometry and camera semantics clears it.

The margins are floored well below the measured gap. Measured on this config across six parameter
inits, the held-out mean IoU gap was at least +0.30 over camera-only and +0.18 over LiDAR-only;
the asserts below require only +0.10 and +0.05. See compare.py for the training protocol and the
README for the full per-init distribution.
"""

from config import FusionConfig
from compare import compare_modalities


def test_fused_beats_each_single_modality():
    cfg = FusionConfig()
    res = compare_modalities(cfg, seed=0)
    camera, lidar, fused = res["camera"], res["lidar"], res["fused"]

    assert fused > 0.4, f"fused IoU {fused:.3f} too low (camera {camera:.3f}, lidar {lidar:.3f})"
    assert fused > camera + 0.10, f"fused {fused:.3f} vs camera {camera:.3f}"
    assert fused > lidar + 0.05, f"fused {fused:.3f} vs lidar {lidar:.3f}"
