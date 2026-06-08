"""The ego-motion warp and the temporal-self-attention necessity test.

warp_bev moves a static-world hot cell by exactly the ego motion (a forward ego translation moves
the cell to a LOWER forward index; lateral moves it along the column axis). Zero ego motion is the
identity.

The necessity test makes temporal information NECESSARY: the moving vehicle is occluded in the
current frame's images (dropped from all frame-t renders) but present in frame t-1 with the correct
ego warp. BCE is scored only on that vehicle's current-frame BEV cells. The temporal model recovers
it from the warped history; the no-temporal model (history zeroed) cannot. A single-seed strict
inequality is a coin flip near the floor, so the test averages over a seed set and requires the
mean gap to clear a margin.
"""

import torch

from config import BEVFormerConfig

from nanovision.data import toy
from nanovision.geometry import CameraRig
from nanovision.bevformer import BEVFormerSeg, warp_bev


def _rig(scene, cfg):
    K, E = scene["K"], scene["E"]
    Ks = {f"cam{i}": K for i in range(cfg.n_cams)}
    Es = {f"cam{i}": E[i] for i in range(cfg.n_cams)}
    sizes = {f"cam{i}": (cfg.img, cfg.img) for i in range(cfg.n_cams)}
    return CameraRig(Ks, Es, sizes)


def test_warp_identity_and_shift():
    cfg = BEVFormerConfig()
    grid = cfg.bev_grid()
    C, nx, ny = 2, grid.nx, grid.ny
    bev = torch.zeros(C, nx, ny)
    bev[:, 5, 8] = 1.0                                   # hot at forward index 5, column 8

    # Zero ego motion is the identity.
    out0 = warp_bev(bev, torch.tensor([0.0, 0.0, 0.0]), grid)
    assert torch.allclose(out0, bev, atol=1e-5)

    # Forward ego motion of 2 m (k_x = 2 cells at res 1.0) moves the hot cell to index 5 - 2 = 3.
    out_f = warp_bev(bev, torch.tensor([2.0, 0.0, 0.0]), grid)
    fi, fj = divmod(int(out_f[0].argmax()), ny)
    assert (fi, fj) == (3, 8)

    # Lateral ego motion of 3 m moves the column from 8 to 8 - 3 = 5.
    out_l = warp_bev(bev, torch.tensor([0.0, 3.0, 0.0]), grid)
    li, lj = divmod(int(out_l[0].argmax()), ny)
    assert (li, lj) == (5, 5)


def _backbone(cfg):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, cfg.dim, 3, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(cfg.dim, cfg.dim, 3, stride=2, padding=1), torch.nn.ReLU(),
    )


def _train_and_score(cfg, seed, temporal):
    """Train BEVFormerSeg on the two-frame occlusion scene; return BCE on the occluded cells."""
    torch.manual_seed(seed)
    scene = toy.bev_multicam_scene(
        n_cams=cfg.n_cams, n_vehicles=4, img=cfg.img, stride=cfg.stride, focal=cfg.f,
        n_frames=2, ego_step=1.0, occlude_moving=True, seed=seed,
    )
    rig = _rig(scene, cfg)
    imgs = scene["images"]                               # (2, n_cam, 3, img, img)
    gt = scene["bev_gt"]                                 # (2, nx, ny)
    ego_delta = scene["ego_deltas"][1]                  # frame-1 ego motion
    occ = scene["occluded_cells"]                        # (k, 2) current-frame cells
    if occ.numel() == 0:
        # No occluded cell placed for this seed; skip by returning equal scores.
        return None

    model = BEVFormerSeg(cfg)
    bb = _backbone(cfg)
    params = list(model.parameters()) + list(bb.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    lf = torch.nn.BCEWithLogitsLoss()

    for _ in range(800):
        feats0 = bb(imgs[0])
        prev = model.encoder(feats0, rig)               # (C, nx, ny) frame t-1 BEV
        feats1 = bb(imgs[1])
        if temporal:
            logit = model(feats1, rig, prev_bev=prev.detach(), ego_delta=ego_delta)
        else:
            logit = model(feats1, rig)                   # no history
        # Train on the full current-frame occupancy (both frames' supervision).
        loss = lf(logit[0], gt[1]) + lf(model(feats0, rig)[0], gt[0])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        feats0 = bb(imgs[0]); prev = model.encoder(feats0, rig)
        feats1 = bb(imgs[1])
        if temporal:
            logit = model(feats1, rig, prev_bev=prev, ego_delta=ego_delta)[0]
        else:
            logit = model(feats1, rig)[0]
        ii, jj = occ[:, 0], occ[:, 1]
        target = gt[1][ii, jj]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit[ii, jj], target)
    return bce.item()


def test_temporal_recovers_occluded_vehicle():
    cfg = BEVFormerConfig()
    seeds = [0, 1, 2]
    gaps, temporals, no_temporals = [], [], []
    for s in seeds:
        bt = _train_and_score(cfg, s, temporal=True)
        bn = _train_and_score(cfg, s, temporal=False)
        if bt is None or bn is None:
            continue
        temporals.append(bt); no_temporals.append(bn); gaps.append(bn - bt)

    assert len(gaps) >= 2, "too few seeds placed an occluded cell"
    mean_gap = sum(gaps) / len(gaps)
    # The temporal model recovers the occluded vehicle; the no-temporal one cannot. Require the
    # mean BCE gap on the occluded cells to clear a margin across the seed set.
    assert mean_gap > 0.1, (
        f"mean gap {mean_gap:.3f}; temporal {temporals}, no_temporal {no_temporals}"
    )
