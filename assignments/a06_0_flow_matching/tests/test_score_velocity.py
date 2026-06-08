"""The score-velocity relation matches the conditional score of the linear path.

For x_t = (1-t)x0 + t*x1 with x0 ~ N(0, I), the conditional is x_t | x1 ~ N(t*x1, (1-t)^2 I),
so the conditional score is -(x_t - t*x1)/(1-t)^2. The relation score = (t*v - x_t)/(1-t)
with v = x1 - x0 must reproduce it. t is sampled away from 1 (the score is singular there).
"""

import torch

from flow import score_from_velocity
from path import linear_path, linear_velocity


def test_score_matches_conditional_score():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(64, 2, generator=g)
    x1 = torch.randn(64, 2, generator=g)
    t = 0.05 + 0.85 * torch.rand(64, generator=g)        # in [0.05, 0.90], away from t=1
    x_t = linear_path(x0, x1, t)
    v = linear_velocity(x0, x1)

    got = score_from_velocity(v, x_t, t)
    tv = t.view(-1, 1)
    expected = -(x_t - tv * x1) / (1.0 - tv) ** 2
    assert torch.allclose(got, expected, atol=1e-5)
