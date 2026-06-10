"""Float64 gradcheck of cfm_target and the flow velocity network in the loss."""

import torch

from config import GradcheckConfig
from flow import FlowHead, cfm_target, flow_loss


def test_cfm_target_gradcheck():
    a = torch.randn(3, 2, 2, dtype=torch.float64, requires_grad=True)
    z0 = torch.randn(3, 2, 2, dtype=torch.float64)
    t = torch.rand(3, 1, 1, dtype=torch.float64)
    # Gradcheck both outputs wrt a_chunk (z_t and the target v both depend on it).
    assert torch.autograd.gradcheck(lambda x: cfm_target(x, z0, t), (a,))


def test_flow_head_differentiable_wrt_zt_and_t():
    cfg = GradcheckConfig()
    torch.manual_seed(0)
    head = FlowHead(cfg, cond_in=5).double()
    z_t = torch.randn(2, cfg.chunk, cfg.act_dim, dtype=torch.float64, requires_grad=True)
    t = torch.rand(2, 1, 1, dtype=torch.float64, requires_grad=True)
    c = torch.randn(2, 5, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda z, tt: head(z, tt, c), (z_t, t))


def test_flow_loss_gradcheck_wrt_action():
    cfg = GradcheckConfig()
    torch.manual_seed(0)
    head = FlowHead(cfg, cond_in=5).double()
    a = torch.randn(3, cfg.chunk, cfg.act_dim, dtype=torch.float64, requires_grad=True)
    c = torch.randn(3, 5, dtype=torch.float64)

    # Re-seed the generator each call so the sampled (z0, t) are identical, making the loss a
    # deterministic function of the action chunk for gradcheck. This exercises the full loss path
    # (cfm_target, the network forward, the MSE) end to end.
    def f(x):
        return flow_loss(head, x, c, generator=torch.Generator().manual_seed(0))

    assert torch.autograd.gradcheck(f, (a,), eps=1e-6, atol=1e-4)
