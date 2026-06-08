"""The straight-through estimator: the gradient passes through the argmin as identity.

The quantization argmin has zero gradient almost everywhere, so a finite-difference check
would disagree with autograd by design - that is the whole point of the estimator. The right
check is direct: the autograd gradient of the quantized output w.r.t. the encoder output is
exactly identity, and the forward value is the hard codebook lookup.
"""

import torch

from config import VQConfig

from nanovision.quantize import VectorQuantizer


def test_gradient_is_identity():
    cfg = VQConfig()
    torch.manual_seed(0)
    q = VectorQuantizer(cfg.num_codes, cfg.code_dim, cfg.beta)
    z_e = torch.randn(2, cfg.code_dim, 3, 3, requires_grad=True)
    assert z_e.grad is None

    z_q_ste, _, _ = q(z_e)
    z_q_ste.sum().backward()                       # backprop only through the STE path
    assert torch.allclose(z_e.grad, torch.ones_like(z_e), atol=1e-6)


def test_forward_is_hard_quantize():
    cfg = VQConfig()
    torch.manual_seed(1)
    q = VectorQuantizer(cfg.num_codes, cfg.code_dim, cfg.beta)
    z_e = torch.randn(2, cfg.code_dim, 3, 3)
    z_q_ste, idx, _ = q(z_e)
    z_q_hard = q.codebook(idx).permute(0, 3, 1, 2)
    assert torch.allclose(z_q_ste, z_q_hard, atol=1e-6)
