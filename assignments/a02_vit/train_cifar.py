"""Provided wiring for the optional real CIFAR-10 run (no holes here).

This is assignment glue, not a graded mechanism: it builds the ViT and hands it to
nanovision.Trainer. The learner does not write the training loop. The gating signal
is overfit-one-batch (see tests/test_overfit.py); this script is for the optional
short real run with the DeiT-style recipe described in the README.

Run (downloads CIFAR-10, ~170MB, into ./data):
    NANOVISION_IMPL=solution python solution/train_cifar.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from vit import ViT  # noqa: E402

from nanovision.determinism import set_seed  # noqa: E402
from nanovision.trainer import Trainer  # noqa: E402


def build_model(cfg):
    return ViT(
        img_size=cfg.img_size, patch=cfg.patch, in_chans=cfg.in_chans, dim=cfg.dim,
        depth=cfg.depth, n_heads=cfg.n_heads, mlp_ratio=cfg.mlp_ratio,
        num_classes=cfg.num_classes, n_registers=cfg.n_registers, pool=cfg.pool,
    )


def train_cifar(cfg, max_steps: int = 1000, batch_size: int = 128, lr: float = 1e-3,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Short supervised CIFAR-10 run. Returns the loss history.

    This is a sanity wiring run, not a convergence run: a tiny ViT from scratch
    underfits CIFAR-10 without the full DeiT augmentation recipe (RandAugment,
    MixUp, CutMix, label smoothing, stochastic depth). The point is that the model
    trains end to end, not the final accuracy. See the README for the recipe.
    """
    from nanovision.data.images import cifar10

    set_seed(cfg.seed)
    train_ds = cifar10(train=True, download=True)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    loss_fn = lambda logits, y: F.cross_entropy(logits, y, label_smoothing=0.1)
    trainer = Trainer(model, opt, loss_fn, device=device)
    return trainer.fit(loader, max_steps=max_steps)


if __name__ == "__main__":
    from config import ViTConfig

    cfg = ViTConfig()
    history = train_cifar(cfg, max_steps=200)
    print(f"ran {len(history)} logged steps; last loss {history[-1][1]:.3f}")
