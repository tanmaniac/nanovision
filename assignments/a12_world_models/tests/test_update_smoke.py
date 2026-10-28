"""Two full training updates (world model, imagination, actor, critic, replay value loss, optimizer)
on the tiny config: every loss is finite and every trained parameter moves. Two, because the reward
and critic heads start with a zero output layer, which blocks the gradient to their hidden layers
until the first step has moved it."""

import torch

import _train
from ._helpers import synthetic_batch, tiny_cfg


def test_two_updates():
    cfg = tiny_cfg(batch_size=2, batch_length=8)
    torch.manual_seed(0)
    agent = _train.Agent(cfg)
    before = {n: p.detach().clone() for n, p in agent.named_parameters() if not n.startswith("slow_critic.")}
    data = synthetic_batch(cfg, B=2, T=8)
    h0, z0 = agent.wm.rssm.initial(2)
    agent.update(data, h0, z0)
    metrics = agent.update(data, h0, z0)
    for k in ["loss/dyn", "loss/rep", "loss/image", "loss/rew", "loss/con", "loss/policy", "loss/value",
              "loss/repval", "loss/total"]:
        assert torch.isfinite(metrics[k]), k
    assert agent.nf_skipped == 0
    moved = [n for n, p in agent.named_parameters() if n in before and not torch.equal(p, before[n])]
    assert len(moved) == len(before), set(before) - set(moved)
