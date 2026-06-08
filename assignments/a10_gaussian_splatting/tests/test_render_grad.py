"""Float64 gradcheck through the full forward: means_w + pose -> project -> splat_render.

Gradient inputs are opacity_logits and means_w ONLY. depths is deliberately excluded: the
depth sort key reaches the output only through argsort, which is piecewise-constant in the
parameters and carries no gradient, so a depths-only loss would error. The sort order is
non-differentiable; only the gathered per-Gaussian values are.
"""

import torch

from gaussian import build_covariance_3d
from project import perspective_jacobian, project_cov_to_2d
from render import splat_render

from nanovision.geometry import apply_transform, project_points

_K = torch.tensor([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _forward(means_w, opacity_logits, quats, log_scales, color_logits, w2c):
    means_cam = apply_transform(w2c, means_w)
    means2d = project_points(means_cam, _K)
    depths = means_cam[:, 2]
    cov3d = build_covariance_3d(quats, log_scales)
    J = perspective_jacobian(means_cam, _K)
    cov2d = project_cov_to_2d(cov3d, w2c[:3, :3], J)
    colors = torch.sigmoid(color_logits)
    opac = torch.sigmoid(opacity_logits)
    return splat_render(means2d, cov2d, colors, opac, depths, 4, 4, bg=0.1)


def test_full_forward_gradcheck():
    n = 4
    g = torch.Generator().manual_seed(0)
    means_w = (0.3 * torch.randn(n, 3, generator=g, dtype=torch.float64))
    means_w[:, 2] += 4.0
    means_w = means_w.requires_grad_(True)
    opacity_logits = (0.5 * torch.randn(n, generator=g, dtype=torch.float64)).requires_grad_(True)
    quats = torch.tensor([[1.0, 0.1, -0.05, 0.07]], dtype=torch.float64).repeat(n, 1)
    log_scales = torch.log(torch.full((n, 3), 0.4, dtype=torch.float64))
    color_logits = 0.3 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    # Camera at the origin looking down +z, so world-to-camera is the identity. Passing w2c
    # directly keeps this gradcheck exercising A10's own forward, not the camera inverse.
    w2c = torch.eye(4, dtype=torch.float64)

    def f(m, o):
        return _forward(m, o, quats, log_scales, color_logits, w2c)

    assert torch.autograd.gradcheck(f, (means_w, opacity_logits), atol=1e-4)
