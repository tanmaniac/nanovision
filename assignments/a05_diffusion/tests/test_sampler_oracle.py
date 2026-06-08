"""Sampler correctness without training: a perfect-prediction oracle must reconstruct x0.

The oracle returns the construction prediction at the current x_t for a known x0 (the eps,
x0, or v that is consistent with x_t given x0). With that, x0_hat = x0 at every step, so
both samplers' final step (abar_prev := 1) returns x0 exactly. Deterministic DDIM (eta=0)
and DDPM both reconstruct x0 to float precision, with no training.
"""

import torch

from sampling import ddim_sample, ddpm_sample
from schedule import cosine_alpha_bar


class Oracle:
    """Returns the prediction (eps / x0 / v) consistent with x0 at the current x_t."""

    def __init__(self, x0, alphas_bar, kind):
        self.x0, self.abar, self.kind = x0, alphas_bar, kind
        self.null_index = 0

    def __call__(self, x, t, labels=None):
        abar_t = self.abar[t].view(-1, 1, 1, 1)
        a, b = abar_t.sqrt(), (1 - abar_t).sqrt()
        eps = (x - a * self.x0) / b
        if self.kind == "eps":
            return eps
        if self.kind == "x0":
            return self.x0.expand_as(x)
        return a * eps - b * self.x0          # v


def test_ddim_oracle_reconstructs_x0():
    T = 30
    _, abar = cosine_alpha_bar(T)
    x0 = torch.rand(3, 1, 16, 16) * 2 - 1     # in [-1, 1]
    ts = list(range(T - 1, -1, -1))
    for kind in ("eps", "x0", "v"):
        out = ddim_sample(Oracle(x0, abar, kind), x0.shape, abar, ts, kind=kind, eta=0.0,
                          clip_x0=False, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(out, x0, atol=1e-5), kind


def test_ddpm_oracle_reconstructs_x0():
    T = 30
    _, abar = cosine_alpha_bar(T)
    x0 = torch.rand(3, 1, 16, 16) * 2 - 1
    for kind in ("eps", "v"):
        out = ddpm_sample(Oracle(x0, abar, kind), x0.shape, abar, kind=kind, clip_x0=False,
                          generator=torch.Generator().manual_seed(0))
        assert torch.allclose(out, x0, atol=1e-5), kind
