"""After annealed K=6 training, at least n_intentions modes win and their endpoints are
mutually separated, so the modes cover the distinct intentions rather than collapsing.

Uses the ANNEALED recipe, not hard WTA: hard WTA's dead modes make the winning-mode count
init-dependent on this single-shared-input toy (the measured table and compare_recipes figure in the
README show that fragility). Measured on the annealed solution head: 3 winning modes with pairwise
endpoint separation > 4.3 m, robust across seeds.
"""

import torch

from _train import train_head
from config import PredConfig

from nanovision.data import toy

_N_INTENTIONS = 3


def test_modes_specialize():
    cfg = PredConfig()
    scene = toy.pred_toy_scene(
        n_intentions=_N_INTENTIONS, channels=cfg.in_ch, horizon=cfg.horizon,
        shared_context=True, seed=0,
    )
    gt = scene["futures_local"]
    head = train_head(scene, n_modes=cfg.n_modes, recipe="annealed", steps=400, seed=0,
                      tau0=cfg.tau0, anneal_frac=cfg.anneal_frac)

    with torch.no_grad():
        trajs, _ = head(scene["bev_feat"], scene["centers"])              # (B, K, T, 2)
        fde = torch.linalg.norm(trajs[:, :, -1] - gt[:, None, -1], dim=-1)  # (B, K)
        winners = fde.argmin(dim=1)                                       # (B,)
        win_ids = winners.unique()

    assert win_ids.numel() >= _N_INTENTIONS, (
        f"only {win_ids.numel()} modes win; expected >= {_N_INTENTIONS} (the distinct intentions)"
    )

    # Winning modes' endpoints must be mutually separated (cover distinct intentions, not collapse).
    endpoints = []
    for w in win_ids:
        member = (winners == w).nonzero(as_tuple=True)[0][0]
        endpoints.append(trajs[member, w, -1])
    endpoints = torch.stack(endpoints)                                   # (n_win, 2)
    d = torch.cdist(endpoints, endpoints)
    min_sep = d[d > 0].min().item()
    assert min_sep > 2.0, f"winning endpoints too close ({min_sep:.2f} m); modes collapsed"
