"""Headline lesson: a single mode is forced to average the futures, while a K-mode head trained so
every mode gets early gradient (soft -> hard annealing) recovers the distinct futures.

shared_context=True puts every agent at the same BEV cell with the same feature but a balanced fan
of distinct intentions, so the future is genuinely ambiguous given the input. With identical input,
a single mode minimizing mean error converges to the conditional mean of the intentions - a
trajectory through the middle of all of them, far from each. The annealed K-mode head covers them.

This test asserts only what is robustly true for any correct implementation, on multiple seeds:
  single   (K=1, hard)             min_ade ~ 1.17 m, seed-independent (the conditional-mean floor).
  annealed (K=6, soft -> hard)     min_ade < 0.15 m on every seed (full coverage, all 3 intentions).
So coverage beats averaging by a wide margin, regardless of init.

What is NOT asserted here, on purpose: the hard-WTA dead-mode collapse is real but FRAGILE - it
depends on init, and on this toy the solution head collapses (2 of 6 modes, min_ade ~0.56) on
roughly 3 of 8 random seeds and reaches full coverage on the rest. Pinning a single "collapse seed"
into an assertion would both misrepresent the typical behavior and be brittle against a valid but
arithmetically different student implementation. That fragility is shown instead in viz.py
(compare_recipes panel) and described in the README with the measured per-seed statistics, where it
can be honest about being intermittent. At scale, hard min-of-N is the standard working loss because
diverse per-scene context keeps modes alive; this single-shared-input toy is the amplified worst case.
"""

import torch

from _train import train_head
from config import PredConfig
from predict import min_ade

from nanovision.data import toy


def _ade(head, scene, gt):
    with torch.no_grad():
        return min_ade(head(scene["bev_feat"], scene["centers"])[0], gt).item()


def test_wta_beats_single_mode():
    cfg = PredConfig()
    scene = toy.pred_toy_scene(
        channels=cfg.in_ch, horizon=cfg.horizon, shared_context=True, seed=0
    )
    gt = scene["futures_local"]

    single = train_head(scene, n_modes=1, recipe="hard", steps=400, seed=0,
                        tau0=cfg.tau0, anneal_frac=cfg.anneal_frac)
    ade_single = _ade(single, scene, gt)
    # Single mode is forced to the conditional mean of the intentions (mode averaging).
    assert ade_single > 1.0, f"single-mode min_ade {ade_single:.3f} should sit near 1.17 m"

    # The working multimodal recipe recovers coverage on every init we try (robust, not seed-picked).
    for seed in (0, 7):
        annealed = train_head(scene, n_modes=cfg.n_modes, recipe="annealed", steps=400, seed=seed,
                              tau0=cfg.tau0, anneal_frac=cfg.anneal_frac)
        ade_annealed = _ade(annealed, scene, gt)
        assert ade_annealed < 0.15, (
            f"annealed min_ade {ade_annealed:.3f} (seed {seed}) should be near zero (full coverage)"
        )
        # Coverage beats averaging by a wide margin.
        assert ade_annealed < 0.3 * ade_single, (
            f"annealed ({ade_annealed:.3f}) should be far below single ({ade_single:.3f})"
        )
