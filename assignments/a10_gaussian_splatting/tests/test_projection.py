"""The EWA projection: the Jacobian matches finite differences of project_points, the 2D
covariance stays a valid symmetric positive-definite 2x2, and the EWA step gradchecks.
"""

import torch

from project import perspective_jacobian, project_cov_to_2d

from nanovision.geometry import project_points

_K = torch.tensor([[50.0, 0.0, 8.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def test_jacobian_matches_finite_difference():
    # A few camera-space means in front of the camera (z > 0).
    means = torch.tensor([[0.2, -0.3, 4.0], [-0.5, 0.4, 3.5], [0.1, 0.1, 5.0]],
                         dtype=torch.float64)
    J = perspective_jacobian(means, _K)
    assert J.shape == (3, 2, 3)
    for i in range(means.shape[0]):
        m = means[i].clone().requires_grad_(True)
        Jfd = torch.autograd.functional.jacobian(lambda p: project_points(p[None], _K)[0], m)
        assert torch.allclose(J[i], Jfd, atol=1e-4), f"row {i}: {J[i]} vs {Jfd}"


def test_cov2d_symmetric_positive_definite():
    n = 6
    g = torch.Generator().manual_seed(1)
    quat = torch.zeros(n, 4, dtype=torch.float64)
    quat[:, 0] = 1.0
    quat = quat + 0.3 * torch.randn(n, 4, generator=g, dtype=torch.float64)
    from gaussian import build_covariance_3d
    cov3d = build_covariance_3d(quat, 0.1 * torch.randn(n, 3, generator=g, dtype=torch.float64))
    W = torch.eye(3, dtype=torch.float64)
    means_cam = torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float64).repeat(n, 1)
    J = perspective_jacobian(means_cam, _K)
    cov2d = project_cov_to_2d(cov3d, W, J)
    assert cov2d.shape == (n, 2, 2)
    assert torch.allclose(cov2d, cov2d.transpose(-1, -2), atol=1e-10)
    eigs = torch.linalg.eigvalsh(cov2d)
    assert (eigs > 0).all(), f"non-PD 2D cov, min eig {eigs.min().item()}"


def test_project_cov_gradcheck():
    n = 3
    quat = torch.tensor([[1.0, 0.1, -0.1, 0.05]], dtype=torch.float64).repeat(n, 1)
    quat = quat.requires_grad_(True)
    log_s = (0.1 * torch.randn(n, 3, dtype=torch.float64)).requires_grad_(True)
    W = torch.eye(3, dtype=torch.float64)
    means_cam = torch.tensor([[0.1, -0.2, 4.0]], dtype=torch.float64).repeat(n, 1)
    J = perspective_jacobian(means_cam, _K)
    from gaussian import build_covariance_3d

    def f(q, ls):
        return project_cov_to_2d(build_covariance_3d(q, ls), W, J)

    assert torch.autograd.gradcheck(f, (quat, log_s))
