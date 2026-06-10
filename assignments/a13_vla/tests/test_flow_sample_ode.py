"""Euler ODE sampling is exact for a constant velocity field (training-free oracle).

The straight CFM path has a constant velocity v = a* - z0. A velocity field that returns that
constant integrates from the SAME z0 to a* exactly with forward Euler at any step count (no
discretization error for a constant field). This checks the integrator wiring (start at t=0, dt =
1/n_steps, n_steps updates) without depending on any training.

Note: only the analytic constant field is exactly step-count-invariant. A TRAINED flow sampler is
only approximately so, because its learned field is not literally constant along a trajectory; the
viz caption says this and does not assert it.
"""

import torch

from flow import flow_sample


class ConstFieldHead:
    """A stand-in head whose velocity is a fixed constant a* - z0, ignoring z, t, c.

    flow_sample starts its own z ~ N(0, I); to make the oracle exact we seed the same generator it
    uses so the field's z0 matches the integrator's start. act_dim mirrors a real FlowHead.
    """

    def __init__(self, a_star, z0):
        self.act_dim = a_star.shape[-1]
        self._v = a_star - z0

    def __call__(self, z, t, c):
        return self._v


def test_constant_field_integrates_exactly():
    B, H, A = 4, 3, 2
    g_seed = 12345
    a_star = torch.randn(B, H, A, generator=torch.Generator().manual_seed(1))
    # Reproduce the z0 that flow_sample will draw with this generator, so the constant field is
    # exactly a_star - z0_start and the straight line lands on a_star.
    z0 = torch.randn(B, H, A, generator=torch.Generator().manual_seed(g_seed))
    head = ConstFieldHead(a_star, z0)
    c = torch.zeros(B, 5)
    for n_steps in (1, 5, 10, 50):
        out = flow_sample(head, c, H, n_steps, generator=torch.Generator().manual_seed(g_seed))
        assert torch.allclose(out, a_star, atol=1e-5), n_steps
