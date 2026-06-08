"""Task 1 reference values: hand-built inputs with a known answer.

With one-hot keys and a query equal to one of those keys, the scaled dot product
is largest for the matching key, so softmax concentrates on it and attention
returns (approximately) that key's value. With a large temperature (scaling up
the magnitude) the selection becomes effectively exact.
"""

import torch

from nanovision.attention import scaled_dot_product_attention


def test_onehot_keys_select_single_value():
    # 3 keys, one-hot in a 3-dim space, scaled up so the softmax is near-hard.
    scale = 50.0
    k = torch.eye(3).view(1, 1, 3, 3) * scale
    v = torch.tensor([[10.0, 0.0, 0.0],
                      [0.0, 20.0, 0.0],
                      [0.0, 0.0, 30.0]]).view(1, 1, 3, 3)
    # query matches key index 1
    q = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3) * scale
    out, attn = scaled_dot_product_attention(q, k, v)
    assert attn.argmax(-1).item() == 1
    assert attn[0, 0, 0, 1] > 0.99
    assert torch.allclose(out[0, 0, 0], v[0, 0, 1], atol=1e-2)


def test_uniform_keys_give_mean():
    # identical keys -> uniform attention -> output is the mean of the values
    k = torch.ones(1, 1, 4, 5)
    v = torch.randn(1, 1, 4, 5)
    q = torch.ones(1, 1, 1, 5)
    out, attn = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(attn, torch.full((1, 1, 1, 4), 0.25), atol=1e-6)
    assert torch.allclose(out[0, 0, 0], v[0, 0].mean(0), atol=1e-6)
