"""Task 2 reference: patch embedding via Conv2d equals unfold-then-linear.

The ViT patch projection is a single linear map applied to each flattened p x p x C
patch. A Conv2d with kernel = stride = p computes exactly that, with its weight
(dim, C, p, p) reshaped to a (C*p*p, dim) matrix. This test reconstructs the linear
form from the same conv weights and checks the two agree.
"""

import torch
import torch.nn.functional as F

from vit import PatchEmbed


def test_conv_equals_unfold_linear():
    torch.manual_seed(0)
    B, C, p, dim, side = 2, 3, 4, 16, 32
    pe = PatchEmbed(in_chans=C, dim=dim, patch=p)
    x = torch.randn(B, C, side, side)

    conv_out = pe(x)  # (B, N, dim)

    # Unfold into patches: (B, C*p*p, N), each column is one flattened patch.
    patches = F.unfold(x, kernel_size=p, stride=p)          # (B, C*p*p, N)
    patches = patches.transpose(1, 2)                       # (B, N, C*p*p)
    # Conv weight (dim, C, p, p) -> (C*p*p, dim); F.unfold orders rows as
    # (C, p_row, p_col), which matches the conv weight's (C, p, p) flatten.
    W = pe.proj.weight.reshape(dim, C * p * p).t()          # (C*p*p, dim)
    linear_out = patches @ W + pe.proj.bias                 # (B, N, dim)

    assert conv_out.shape == linear_out.shape
    assert torch.allclose(conv_out, linear_out, atol=1e-5), (
        (conv_out - linear_out).abs().max().item()
    )
