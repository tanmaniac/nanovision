"""The assembled ViT overfits one batch of synthetic images. Run last.

End-to-end wiring signal: with every mechanism correct, the full forward + training
loop drives cross-entropy on a single fixed batch of random images with random
labels to near zero. No dataset download - the batch is torch.randn under a fixed
seed. Both pool modes ("cls" and "mean") must overfit.
"""

import pytest
import torch
import torch.nn.functional as F

from vit import ViT

from nanovision.determinism import set_seed
from nanovision.trainer import Trainer


@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_vit_overfits_one_batch(pool):
    set_seed(0)
    B, num_classes = 8, 10
    x = torch.randn(B, 3, 32, 32)
    y = torch.randint(0, num_classes, (B,))

    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4,
                num_classes=num_classes, n_registers=4, pool=pool)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = lambda logits, target: F.cross_entropy(logits, target)

    tr = Trainer(model, opt, loss_fn, device="cpu")
    losses = tr.overfit_one_batch((x, y), steps=500)
    assert losses[-1] < 0.02, f"pool={pool} should memorize the batch; final {losses[-1]}"
    assert losses[-1] < losses[0]
