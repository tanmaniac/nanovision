"""Task 1 random masking: keep count, mask complement, and the unshuffle round-trip.

Runs after gradcheck. Verifies that random_masking keeps exactly round((1-r)N)
tokens, the mask has the complementary count of ones, and that the
[kept; masked-placeholder] set unshuffled by ids_restore returns the original
token order.
"""

import torch


def test_keep_and_mask_counts():
    from mae import random_masking
    torch.manual_seed(0)
    B, N, D, r = 4, 64, 16, 0.75
    x = torch.randn(B, N, D)
    x_kept, mask, ids_restore = random_masking(x, r)
    n_keep = round((1 - r) * N)          # 16
    assert x_kept.shape[1] == n_keep
    assert torch.all(mask.sum(dim=1) == (N - n_keep))   # 48 masked per row
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_unshuffle_round_trip():
    """Gathering the [kept; placeholder] set by ids_restore inverts the shuffle.

    Build the full shuffled set as [x_kept; masked-rows-of-x] in shuffled order,
    then gather by ids_restore. Where mask==0 (kept positions) the result must equal
    the original x, proving ids_restore places visible tokens back at their original
    grid positions.
    """
    from mae import random_masking
    torch.manual_seed(0)
    B, N, D, r = 2, 64, 8, 0.75
    x = torch.randn(B, N, D)
    x_kept, mask, ids_restore = random_masking(x, r)
    n_keep = x_kept.shape[1]

    # Placeholder for the masked slots; only the kept slots are checked below.
    placeholder = torch.zeros(B, N - n_keep, D)
    full_shuffled = torch.cat([x_kept, placeholder], dim=1)
    restored = torch.gather(full_shuffled, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))

    keep = (mask == 0).unsqueeze(-1).expand(-1, -1, D)
    assert torch.allclose(restored[keep], x[keep], atol=1e-6)


def test_deterministic_with_seed():
    from mae import random_masking
    x = torch.randn(3, 64, 8)
    torch.manual_seed(123)
    a = random_masking(x, 0.75)
    torch.manual_seed(123)
    b = random_masking(x, 0.75)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)
