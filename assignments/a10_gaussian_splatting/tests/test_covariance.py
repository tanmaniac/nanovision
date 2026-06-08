"""The scale+rotation 3D covariance: symmetry, PSD, the eigenvalue identity, gradcheck.

The eigenvalue test uses DISTINCT scales (so the sorted eigenvalues are identifiable) and
O(1)-magnitude quaternions. gradcheck through the quaternion normalization is singular at
near-zero quaternions, so every test seeds quats near (1, 0, 0, 0).
"""

import torch

from gaussian import build_covariance_3d, quat_to_rotmat


def _good_quats(n, *, dtype=torch.float64, seed=0):
    """O(1) quaternions: identity plus small noise. Far from the singular q=0."""
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(n, 4, dtype=dtype)
    q[:, 0] = 1.0
    q = q + 0.3 * torch.randn(n, 4, generator=g, dtype=dtype)
    return q


def test_covariance_symmetric_psd():
    quats = _good_quats(8)
    log_scales = 0.2 * torch.randn(8, 3, dtype=torch.float64)
    cov = build_covariance_3d(quats, log_scales)
    assert cov.shape == (8, 3, 3)
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-10)
    eigs = torch.linalg.eigvalsh(cov)
    assert (eigs >= -1e-8).all(), f"non-PSD: min eig {eigs.min().item()}"


def test_eigenvalues_equal_scales_squared():
    # A pure rotation maps the eigenvalues of Sigma = R diag(s^2) R^T to s^2 exactly.
    quats = _good_quats(5, seed=3)
    log_s = torch.log(torch.tensor([[0.3, 0.7, 1.4]], dtype=torch.float64)).repeat(5, 1)
    cov = build_covariance_3d(quats, log_s)
    eigs = torch.sort(torch.linalg.eigvalsh(cov), dim=-1).values
    target = torch.sort(torch.exp(log_s) ** 2, dim=-1).values
    assert torch.allclose(eigs, target, atol=1e-8)


def test_quat_to_rotmat_orthogonal():
    quats = _good_quats(6, seed=5)
    R = quat_to_rotmat(quats)
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(R @ R.transpose(-1, -2), eye.expand(6, 3, 3), atol=1e-10)
    assert torch.allclose(torch.linalg.det(R), torch.ones(6, dtype=torch.float64), atol=1e-10)


def test_quat_to_rotmat_gradcheck():
    quats = _good_quats(3, seed=7).requires_grad_(True)
    assert torch.autograd.gradcheck(quat_to_rotmat, (quats,))


def test_build_covariance_gradcheck():
    quats = _good_quats(3, seed=9).requires_grad_(True)
    log_scales = (0.1 * torch.randn(3, 3, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(build_covariance_3d, (quats, log_scales))
