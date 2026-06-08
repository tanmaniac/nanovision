"""Per-voxel classifier: shape, gradients, and an overfit to toy voxel labels."""

import torch

from config import OccConfig
from occupancy import OccupancyHead, inverse_frequency_weights, occupancy_iou, weighted_ce_loss

from nanovision.data import toy


def test_shape():
    B, C, Z, Y, X, n_classes = 2, 16, 8, 32, 32, 4
    feats = torch.randn(B, C, Z, Y, X)
    head = OccupancyHead(C, n_classes)
    out = head(feats)
    assert out.shape == (B, n_classes, Z, Y, X)


def test_gradcheck():
    torch.manual_seed(0)
    B, C, Z, Y, X, n_classes = 1, 2, 2, 2, 2, 4
    feats = torch.randn(B, C, Z, Y, X, dtype=torch.float64, requires_grad=True)
    head = OccupancyHead(C, n_classes).double()
    assert torch.autograd.gradcheck(lambda f: head(f), (feats,))


def test_overfit_voxel_labels():
    """Optimize the head plus a learnable feature volume to match the toy voxel classes.

    The feature volume stands in for the trainable BEV encoder upstream, so it is optimized
    alongside the head. Pass condition is occupied-class IoU > 0.85. Boxes are >= 2 voxels thick
    per side, so occupied count dominates the boundary, keeping the IoU stable.
    """
    torch.manual_seed(0)
    cfg = OccConfig()
    scene = toy.occupancy_toy_scene(grid=cfg.grid, bounds=cfg.grid_bounds,
                                    n_classes=cfg.n_classes, n_boxes=cfg.n_boxes, seed=0)
    sem_gt = scene["sem_gt"]                          # [Z, Y, X] long
    target = sem_gt[None]                             # [1, Z, Y, X]

    C = cfg.bev_channels
    feats = torch.randn(1, C, *cfg.grid, requires_grad=True)   # learnable feature volume
    head = OccupancyHead(C, cfg.n_classes)
    weights = inverse_frequency_weights(target, cfg.n_classes)
    opt = torch.optim.Adam(list(head.parameters()) + [feats], lr=1e-2)

    for _ in range(300):
        logits = head(feats)
        loss = weighted_ce_loss(logits, target, weights)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = head(feats).argmax(1)                 # [1, Z, Y, X]
    iou = occupancy_iou(pred, target, cfg.n_classes, ignore_free=True)
    assert iou > 0.85, f"occupied IoU {iou.item():.3f}"
