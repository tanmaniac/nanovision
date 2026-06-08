"""Weighted CE, inverse-frequency weights, mean IoU, and the free-class-collapse lesson."""

import torch
import torch.nn.functional as F
from torch import nn

from occupancy import (
    inverse_frequency_weights,
    occupancy_iou,
    weighted_ce_loss,
)


def test_inverse_frequency_ordering():
    # Class 0 is 90%, classes 1-3 share 10%.
    target = torch.zeros(1, 10, 10, 10, dtype=torch.long)
    flat = target.reshape(-1)
    flat[:100] = 1
    flat[100:130] = 2
    flat[130:150] = 3
    w = inverse_frequency_weights(target, 4)
    assert w.shape == (4,)
    # Free (majority) gets the smallest weight; rarer classes get larger.
    assert w[0] == w.min()
    assert w[1] < w[2] < w[3]          # counts 100 > 30 > 20 -> weights increase
    # Normalization convention: mean 1 (sum = n_classes).
    assert torch.allclose(w.sum(), torch.tensor(4.0), atol=1e-5)


def test_weighted_ce_matches_f_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 3, 3)
    target = torch.randint(0, 4, (2, 3, 3))
    weights = torch.tensor([0.2, 1.0, 2.0, 0.8])
    ours = weighted_ce_loss(logits, target, weights)
    ref = F.cross_entropy(logits, target, weight=weights)
    assert torch.allclose(ours, ref, atol=1e-6), f"ours {ours.item()} vs F {ref.item()}"


def test_weighted_ce_gradcheck():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 2, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 4, (1, 2, 2))
    weights = torch.tensor([0.5, 1.5, 2.0, 1.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda l: weighted_ce_loss(l, target, weights), (logits,))


def test_occupancy_iou_exact():
    # 1D toy: 4 classes. Count intersections and unions by hand.
    pred = torch.tensor([0, 1, 1, 2, 2, 2, 3, 0])
    tgt = torch.tensor([0, 1, 2, 2, 2, 0, 3, 0])
    # class 1: pred {1,2} tgt {1} -> inter 1, union 2 -> 0.5
    # class 2: pred {3,4,5} tgt {2,3,4} -> inter {3,4}=2, union {2,3,4,5}=4 -> 0.5
    # class 3: pred {6} tgt {6} -> inter 1 union 1 -> 1.0
    # mean over occupied = (0.5 + 0.5 + 1.0)/3
    iou = occupancy_iou(pred, tgt, 4, ignore_free=True)
    assert torch.allclose(iou, torch.tensor((0.5 + 0.5 + 1.0) / 3), atol=1e-6)

    # With the free class included: class 0 pred {0,7} tgt {0,5,7} -> inter 2, union 3 = 2/3.
    # Mean over all 4 classes = (2/3 + 0.5 + 0.5 + 1.0)/4. Confirms ignore_free actually drops
    # class 0 from the averaged set (a different denominator and member set).
    iou_with_free = occupancy_iou(pred, tgt, 4, ignore_free=False)
    assert torch.allclose(iou_with_free, torch.tensor((2 / 3 + 0.5 + 0.5 + 1.0) / 4), atol=1e-6)


def test_empty_union_excluded():
    # Class 3 absent from both pred and target -> excluded, not counted as IoU 0.
    pred = torch.tensor([0, 1, 1, 2])
    tgt = torch.tensor([0, 1, 2, 2])
    # class 1: inter 1 union 2 = 0.5; class 2: inter 1 union 2 = 0.5; class 3 empty -> excluded.
    iou = occupancy_iou(pred, tgt, 4, ignore_free=True)
    assert torch.allclose(iou, torch.tensor(0.5), atol=1e-6)


def test_free_class_collapse_lesson():
    """Unweighted CE leaves the rare occupied classes unpredicted; weighting recovers their recall.

    A single linear Conv3d classifier (no deep head, so capacity cannot trivially memorize)
    trained for a short budget from a shared init. The input is a weak class hint (a unit-scale
    one-hot of the class buried in unit-variance noise, padded with pure-noise channels), so the
    rare classes are learnable but not handed over. Unweighted CE sits in the predict-majority
    basin and barely fires on the occupied classes; weighted CE pulls them out.

    The contrast is measured by occupied-class RECALL, not IoU. A linear classifier trained with
    inverse-frequency weights trades precision for recall: it over-predicts the rare classes, so
    its false-positive count inflates the IoU union and the occupied IoU can sit BELOW the
    unweighted run even though the rare classes are now actually detected. Recall isolates the
    teaching point, which is that the rare class is detected at all. Measured gap at this config
    is ~0.57 (unweighted recall ~0.19, weighted ~0.76); pass requires the gap to exceed 0.3.
    """
    torch.manual_seed(0)
    Z, Y, X, C, n_classes = 8, 24, 24, 4, 3
    # A tiny occupied fraction (~1.6%): two small solid boxes of classes 1 and 2.
    target = torch.zeros(1, Z, Y, X, dtype=torch.long)
    target[0, 3:6, 5:9, 5:9] = 1
    target[0, 2:5, 15:18, 15:18] = 2
    occ_frac = (target > 0).float().mean().item()
    assert occ_frac <= 0.05, f"occupied fraction {occ_frac:.3f} too high"

    # Weak class hint: a unit-scale one-hot in unit noise, plus pure-noise padding channels.
    oh = F.one_hot(target, n_classes).permute(0, 4, 1, 2, 3).float()
    feats = 1.5 * oh + torch.randn(1, n_classes, Z, Y, X)
    feats = torch.cat([feats, torch.randn(1, C - n_classes, Z, Y, X)], dim=1)

    occupied = target > 0

    def recall(weighted: bool, steps: int = 80):
        torch.manual_seed(1)                         # shared init across both runs
        clf = nn.Conv3d(C, n_classes, 1)
        w = inverse_frequency_weights(target, n_classes) if weighted else torch.ones(n_classes)
        opt = torch.optim.Adam(clf.parameters(), lr=5e-2)
        for _ in range(steps):
            loss = weighted_ce_loss(clf(feats), target, w)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = clf(feats).argmax(1)
        correct = ((pred == target) & occupied).sum().float()
        return (correct / occupied.sum()).item()

    unweighted_recall = recall(weighted=False)
    weighted_recall = recall(weighted=True)
    gap = weighted_recall - unweighted_recall
    assert gap > 0.3, (
        f"weighted recall {weighted_recall:.3f} - unweighted {unweighted_recall:.3f} = {gap:.3f}"
    )
