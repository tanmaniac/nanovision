"""The bundle-adjustment Schur complement: marginalizing the block-diagonal landmark block and
back-substituting gives exactly the dense solution of the same system."""

import numpy as np

from _impl import schur_solve


def _ba_system(seed, n_pose_blocks=4, pose_dim=6, n_lm=6, lm_block=3):
    """A symmetric positive-definite H with BA structure: dense pose block, dense pose-landmark
    cross blocks, and a strictly block-diagonal landmark block (no landmark-landmark coupling,
    which is what the Schur complement exploits)."""
    gen = np.random.default_rng(seed)
    n_pose = n_pose_blocks * pose_dim
    n = n_pose + n_lm * lm_block
    J = gen.normal(size=(n + 30, n))
    H = J.T @ J + 0.5 * np.eye(n)  # PSD
    # Zero the landmark-landmark off-diagonal blocks.
    for a in range(n_lm):
        for b in range(n_lm):
            if a != b:
                ra, rb = n_pose + a * lm_block, n_pose + b * lm_block
                H[ra:ra + lm_block, rb:rb + lm_block] = 0.0
    b = gen.normal(size=n)
    return H, b, n_pose, lm_block


def test_schur_equals_dense_solve():
    for seed in range(5):
        H, b, n_pose, lm_block = _ba_system(seed)
        d_schur = np.asarray(schur_solve(H, b, n_pose, lm_block))
        d_dense = np.linalg.solve(H, b)
        assert np.abs(d_schur - d_dense).max() < 1e-9


def test_schur_residual_is_zero():
    H, b, n_pose, lm_block = _ba_system(7)
    d = np.asarray(schur_solve(H, b, n_pose, lm_block))
    assert np.linalg.norm(H @ d - b) < 1e-9
