"""Float64 gradcheck of kl_divergence and a single DiTBlock.forward.

reparameterize is deliberately NOT gradchecked: it draws a fresh eps per call, so the
finite-difference reference forward uses a different eps than the analytic pass and the
check is ill-posed. reparameterize is covered by the shape and overfit tests instead.
"""

import torch

from config import DiTConfig
from dit import DiTBlock
from vae import kl_divergence


def test_kl_gradcheck():
    mu = torch.randn(2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    logvar = torch.randn(2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(kl_divergence, (mu, logvar))


def test_ditblock_gradcheck():
    torch.manual_seed(0)
    d, n_heads, mlp_ratio = 8, 2, 2
    block = DiTBlock(d, n_heads, mlp_ratio).double()
    # Perturb the zero-init adaLN Linear so the block is not the trivial identity map: a
    # gradcheck through an exact identity would pass but test nothing.
    with torch.no_grad():
        block.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
        block.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
    x = torch.randn(2, 5, d, dtype=torch.float64, requires_grad=True)
    c = torch.randn(2, d, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: block(a, b), (x, c))
