"""Render the VQ tokenizer: reconstructions, codebook usage, and autoregressive samples.

Run from the repo root: `python -m assignments.a06_5_vq_tokenizer.viz` (or
`make viz A=a06_5_vq_tokenizer`). Writes three figures to out/:
  recon.png      - original images next to their VQ-VAE reconstructions.
  codebook.png   - the code-usage histogram and the perplexity (the collapse diagnostic).
  samples.png    - images decoded from token grids sampled by the autoregressive prior.
"""

import os
import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import VQConfig  # noqa: E402
from prior import TokenPrior, ar_nll, ar_sample  # noqa: E402
from vqvae import VQVAE, vq_vae_loss  # noqa: E402

from nanovision.data import toy  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402
from nanovision.quantize import codebook_perplexity  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _train_vae(cfg, x, steps=2000):
    vae = VQVAE(cfg).to(x.device)
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        x_hat, idx, vq = vae(x)
        vq_vae_loss(x, x_hat, vq).backward()
        opt.step()
    return vae


def _train_prior(cfg, indices, steps=1500):
    prior = TokenPrior(cfg).to(indices.device)
    opt = torch.optim.Adam(prior.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        ar_nll(prior, indices).backward()
        opt.step()
    return prior


def main():
    torch.manual_seed(0)
    cfg = VQConfig()
    dev = default_device()
    x, _ = toy.diffusion_image_batch(16, num_classes=3, size=cfg.img_size,
                                     channels=cfg.channels, seed=0)
    x = x.to(dev)
    vae = _train_vae(cfg, x)
    with torch.no_grad():
        x_hat, idx, _ = vae(x)
        ppl = codebook_perplexity(idx, cfg.num_codes).item()

    # Reconstructions.
    n = 8
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for j in range(n):
        axes[0, j].imshow(x[j, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1); axes[0, j].axis("off")
        axes[1, j].imshow(x_hat[j, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("original"); axes[1, 0].set_ylabel("reconstruction")
    plt.tight_layout(); finish(_OUT / "recon.png")

    # Codebook usage.
    counts = torch.bincount(idx.reshape(-1), minlength=cfg.num_codes).cpu().numpy()
    plt.figure(figsize=(5, 3.2))
    plt.bar(range(cfg.num_codes), counts)
    plt.xlabel("code"); plt.ylabel("usage count")
    plt.title(f"perplexity {ppl:.1f} / {cfg.num_codes}")
    plt.tight_layout(); finish(_OUT / "codebook.png")

    # Autoregressive samples decoded to images.
    prior = _train_prior(cfg, idx)
    with torch.no_grad():
        grids = ar_sample(prior, n, (cfg.grid, cfg.grid), cfg.num_codes,
                          generator=torch.Generator(device=dev).manual_seed(1), device=str(dev))
        samples = vae.decode_indices(grids)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.2))
    for j in range(n):
        axes[j].imshow(samples[j, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1); axes[j].axis("off")
    plt.tight_layout(); finish(_OUT / "samples.png")

    print(f"wrote figures to {_OUT}; perplexity {ppl:.2f}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
