"""Shared training loop for the prediction tests and viz. Provided, not a hole.

Trains a MultimodalTrajectoryHead on one fixed batch (the shared-context ambiguous scene, or the
distinct-context overfit scene) under one of three winner-take-all recipes, and returns the
trained head. Kept out of the test files so test_wta_beats_single_mode, test_modes_specialize,
test_overfit, and viz all drive the head the same way.
"""

import torch

from predict import MultimodalTrajectoryHead, wta_loss


def train_head(
    scene,
    *,
    n_modes,
    recipe,
    steps=400,
    lr=5e-3,
    tau0=3.0,
    anneal_frac=0.6,
    cls_weight=1.0,
    dim=64,
    n_layers=2,
    n_heads=4,
    roi_size=3,
    radius=1.0,
    seed=0,
    device="cpu",
):
    """Train a head on one batch and return it.

    recipe is one of:
      - "hard":     temperature=None throughout (canonical min-of-N).
      - "soft":     fixed temperature=tau0 throughout (pure soft assignment).
      - "annealed": temperature decays linearly tau0 -> 0 over the first anneal_frac of steps,
                    then hard (None) for the rest. This is the headline working recipe.
    K=1 with recipe "hard" is the single-mode baseline (the winner is the only mode).
    """
    torch.manual_seed(seed)
    bev = scene["bev_feat"].to(device)
    centers = scene["centers"].to(device)
    gt = scene["futures_local"].to(device)
    in_ch = scene["C"]
    horizon = scene["horizon"]

    head = MultimodalTrajectoryHead(
        in_ch, dim=dim, n_modes=n_modes, horizon=horizon, n_layers=n_layers,
        n_heads=n_heads, roi_size=roi_size, radius=radius,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    hard_start = int(anneal_frac * steps)
    for step in range(steps):
        if recipe == "hard":
            temp = None
        elif recipe == "soft":
            temp = tau0
        elif recipe == "annealed":
            if step < hard_start:
                temp = tau0 * (1.0 - step / max(1, hard_start))
                if temp < 1e-4:
                    temp = None
            else:
                temp = None
        else:
            raise ValueError(f"unknown recipe {recipe!r}")
        trajs, scores = head(bev, centers)
        loss = wta_loss(trajs, scores, gt, temperature=temp, cls_weight=cls_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head
