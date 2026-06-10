"""Render the two-stage latent-diffusion pipeline: train the VAE on a batch and show
originals vs reconstructions, then train the DiT on the encoded latents and sample one
image per class through the frozen VAE decoder. Provided, not graded.

Run from the repo root: `python -m assignments.a07_latent_dit.viz` (or
`make viz A=a07_latent_dit`). Writes a figure to out/. Runs in solution mode (the
nanovision.attention shim loads the filled multi-head attention).
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

from config import DiTConfig  # noqa: E402
from dit import DiT  # noqa: E402
from flow import cfm_loss, euler_sample  # noqa: E402
from vae import KLVAE, vae_loss  # noqa: E402

from nanovision.data import toy  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _train_vae(cfg, x, steps=1500):
    vae = KLVAE(cfg)
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    for _ in range(steps):
        x_hat, mu, logvar = vae(x)
        total, recon, kl = vae_loss(x, x_hat, mu, logvar, cfg.beta)
        opt.zero_grad(); total.backward(); opt.step()
    return vae, recon.item(), kl.item()


def _train_dit(cfg, latents, labels, steps=3000):
    g = torch.Generator().manual_seed(0)
    model = DiT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = latents.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(32, n),), generator=g)
        x1, y = latents[idx], labels[idx]
        x0 = torch.randn_like(x1)
        t = 0.05 + 0.9 * torch.rand(x1.shape[0], generator=g)
        opt.zero_grad(); cfm_loss(model, x0, x1, y, t).backward(); opt.step()
    return model


def main():
    torch.manual_seed(0)
    cfg = DiTConfig()
    x, labels = toy.diffusion_image_batch(64, num_classes=cfg.num_classes,
                                          size=cfg.image_size, channels=cfg.channels, seed=0)

    vae, recon, kl = _train_vae(cfg, x)
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(x)
        latents = mu                                 # use the mean latent for the DiT
        x_hat, _, _ = vae(x)

    model = _train_dit(cfg, latents, labels)
    model.eval()
    cls = torch.arange(cfg.num_classes)
    C, hw = cfg.latent_dim, cfg.image_size // cfg.f
    with torch.no_grad():
        x0 = torch.randn(cfg.num_classes, C, hw, hw,
                         generator=torch.Generator().manual_seed(1))
        z_samp = euler_sample(model, x0, cls, cfg.n_steps)
        samples = vae.decode(z_samp)

    # Top row: originals vs reconstructions (first num_classes images). Bottom: DiT samples.
    fig, axes = plt.subplots(3, cfg.num_classes, figsize=(2 * cfg.num_classes, 6.2))
    for j in range(cfg.num_classes):
        axes[0, j].imshow(x[j, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[0, j].set_title(f"orig (class {int(labels[j])})"); axes[0, j].axis("off")
        axes[1, j].imshow(x_hat[j, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[1, j].set_title("VAE recon"); axes[1, j].axis("off")
        axes[2, j].imshow(samples[j, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[2, j].set_title(f"DiT sample (class {j})"); axes[2, j].axis("off")
    plt.tight_layout(); finish(_OUT / "latent_dit.png")
    print(f"wrote figure to {_OUT}; VAE recon {recon:.3f}, KL {kl:.3f}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
