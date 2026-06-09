"""Distinct inputs -> the head maps each agent to its own future and overfits to ~0.

shared_context=False spreads agents to distinct cells with distinct yaws and feature signatures,
so the RoI features identify each agent and the head can fit every future exactly. This also
exercises the agent-centric rotation: a wrong rotation sign fails the fit here instead of passing
on a near-identity.
"""

import torch

from _train import train_head
from config import PredConfig
from predict import min_ade

from nanovision.data import toy


def test_overfit_distinct_context():
    cfg = PredConfig()
    scene = toy.pred_toy_scene(
        channels=cfg.in_ch, horizon=cfg.horizon, shared_context=False, seed=0
    )
    head = train_head(
        scene, n_modes=cfg.n_modes, recipe="annealed", steps=800, lr=5e-3,
        tau0=cfg.tau0, anneal_frac=cfg.anneal_frac, dim=cfg.dim, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, roi_size=cfg.roi_size, radius=cfg.radius, seed=0,
    )
    with torch.no_grad():
        trajs, _ = head(scene["bev_feat"], scene["centers"])
        ade = min_ade(trajs, scene["futures_local"]).item()
    assert ade < 0.1, f"distinct-context min_ade {ade:.4f} should overfit below 0.1 m"
