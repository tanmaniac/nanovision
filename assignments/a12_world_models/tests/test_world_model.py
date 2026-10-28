"""WorldModel.loss: the loss dict and shapes, gradient paths, and an overfit check on a fixed tiny
batch (loose bound; no training-return assertion)."""

import torch

from ._helpers import synthetic_batch, tiny_cfg
from world_model import WorldModel


def _run(cfg, data, wm):
    h0, z0 = wm.rssm.initial(data["image"].shape[0])
    return wm.loss(data, h0, z0)


def test_loss_dict_and_shapes():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    wm = WorldModel(cfg)
    data = synthetic_batch(cfg, B=2, T=6)
    losses, h, z, feat = _run(cfg, data, wm)
    assert list(losses) == ["dyn", "rep", "image", "rew", "con"]
    for k, v in losses.items():
        assert v.shape == () and torch.isfinite(v), k
    assert losses["dyn"] >= cfg.kl_free and losses["rep"] >= cfg.kl_free
    assert h.shape == (2, 6, cfg.deter) and z.shape == (2, 6, cfg.stoch, cfg.discrete)
    assert feat.shape == (2, 6, wm.rssm.feat_dim) and feat.requires_grad
    torch.testing.assert_close(feat, wm.rssm.get_feat(h, z))


def test_image_loss_is_summed_squared_error():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    wm = WorldModel(cfg)
    data = synthetic_batch(cfg, B=2, T=4)
    losses, h, z, _ = _run(cfg, data, wm)
    recon = wm.decoder(h, z)
    expect = ((recon - data["image"]) ** 2).sum((-3, -2, -1)).mean()
    torch.testing.assert_close(losses["image"], expect)


def test_reward_loss_reaches_encoder():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    wm = WorldModel(cfg)
    # The reward head's output layer starts at zero, which blocks the gradient; un-zero it.
    torch.nn.init.normal_(wm.reward.last.weight, std=0.3)
    losses, _, _, _ = _run(cfg, synthetic_batch(cfg), wm)
    losses["rew"].backward()
    assert wm.encoder.layers[0].weight.grad.abs().sum() > 0       # through the posterior's straight-through z
    assert wm.decoder.sp2.weight.grad is None


def test_overfit_fixed_batch():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    wm = WorldModel(cfg)
    data = synthetic_batch(cfg, B=2, T=8)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    scale = cfg.loss_scales
    hist = []
    for _ in range(150):
        losses, _, _, _ = _run(cfg, data, wm)
        total = sum(v * scale[k] for k, v in losses.items())
        opt.zero_grad()
        total.backward()
        opt.step()
        hist.append({k: float(v.detach()) for k, v in losses.items()})
    first, last = hist[0], {k: sum(h[k] for h in hist[-10:]) / 10 for k in hist[0]}
    assert last["image"] < 0.5 * first["image"], (first, last)
    assert last["rew"] < 0.5 * first["rew"], (first, last)
    assert last["dyn"] < 3.0 and last["rep"] < 3.0, last
