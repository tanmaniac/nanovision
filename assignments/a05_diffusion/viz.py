"""Render the diffusion mechanism: schedules, forward noising, the conditioning of the
three parameterizations, and samples from a briefly-trained model.

Run from the repo root: `python -m assignments.a05_diffusion.viz` (or
`make viz A=a05_diffusion`). Writes four figures to the assignment's out/ directory:
  schedules.png   - alpha_bar vs t for the linear and cosine schedules.
  noising.png     - one image carried through the forward process at increasing t.
  conditioning.png - the eps/x0 inversion magnitude blowing up near the schedule
                     endpoints while v stays bounded (why v-prediction is preferred).
  samples.png     - a class-conditional sample grid, DDPM-vs-DDIM and a CFG sweep, from a
                     model trained for a few thousand steps on the toy shapes.

The training run is short, so samples are recognizable shapes, not crisp. A real run uses
an EMA copy of the weights for sampling; raw-weight samples look worse than EMA ones.
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

from config import DiffusionConfig  # noqa: E402
from diffusion import diffusion_loss, q_sample, to_x0_eps  # noqa: E402
from sampling import ddim_sample, ddpm_sample  # noqa: E402
from schedule import cosine_alpha_bar, gather, linear_alpha_bar  # noqa: E402
from unet import TimeEmbeddedUNet  # noqa: E402

from nanovision.data.toy import diffusion_image_batch  # noqa: E402
from nanovision.determinism import default_device  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _schedules(T=1000):
    _, lin = linear_alpha_bar(T)
    _, cos = cosine_alpha_bar(T)
    plt.figure(figsize=(5, 3.5))
    plt.plot(lin.numpy(), label="linear")
    plt.plot(cos.numpy(), label="cosine")
    plt.xlabel("t"); plt.ylabel(r"$\bar\alpha_t$"); plt.legend(); plt.tight_layout()
    finish(_OUT / "schedules.png")


def _noising(T=1000):
    _, abar = cosine_alpha_bar(T)
    x0, _ = diffusion_image_batch(n=1, seed=3)
    ts = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    fig, axes = plt.subplots(1, len(ts), figsize=(2 * len(ts), 2.2))
    eps = torch.randn_like(x0)
    for ax, ti in zip(axes, ts):
        x_t = q_sample(x0, torch.tensor([ti]), eps, abar)
        ax.imshow(x_t[0, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t={ti}"); ax.axis("off")
    plt.tight_layout(); finish(_OUT / "noising.png")


def _conditioning(T=1000):
    # For a fixed (x0, eps), the eps- and x0-inversions divide by sqrt(1-abar) or
    # sqrt(abar), which blow up at the endpoints; v stays bounded.
    _, abar = cosine_alpha_bar(T)
    x0 = torch.ones(1, 1, 4, 4); eps = torch.randn(1, 1, 4, 4)
    t = torch.arange(T)
    abar_t = gather(abar, t)                              # (T,1,1,1)
    x_t = abar_t.sqrt() * x0 + (1 - abar_t).sqrt() * eps
    # magnitude of each predicted quantity that a unit model error maps into x0:
    eps_inv = 1.0 / (1 - abar).clamp(min=1e-8).sqrt()
    x0_inv = 1.0 / abar.clamp(min=1e-8).sqrt()
    v_inv = torch.ones_like(eps_inv)
    plt.figure(figsize=(5, 3.5))
    plt.semilogy(eps_inv.numpy(), label="eps inversion")
    plt.semilogy(x0_inv.numpy(), label="x0 inversion")
    plt.semilogy(v_inv.numpy(), label="v (bounded)")
    plt.xlabel("t"); plt.ylabel("error amplification"); plt.legend(); plt.tight_layout()
    finish(_OUT / "conditioning.png")


def _train_and_sample(steps=3000):
    torch.manual_seed(0)
    cfg = DiffusionConfig()
    dev = default_device()  # 3000 UNet steps + sampling: the heaviest demo, GPU pays off here
    T = 200
    _, abar = cosine_alpha_bar(T)
    abar = abar.to(dev)
    x0, labels = diffusion_image_batch(n=64, num_classes=cfg.num_classes, size=cfg.img_size,
                                       channels=cfg.channels, seed=0)
    x0, labels = x0.to(dev), labels.to(dev)
    model = TimeEmbeddedUNet(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator(device=dev).manual_seed(0)
    for step in range(steps):
        idx = torch.randint(0, x0.shape[0], (16,), generator=gen, device=dev)
        opt.zero_grad()
        loss = diffusion_loss(model, x0[idx], abar, kind="v", num_classes=cfg.num_classes,
                              cfg_drop_prob=0.1, min_snr_gamma=cfg.min_snr_gamma,
                              labels=labels[idx], generator=gen)
        loss.backward(); opt.step()
    model.eval()

    shape = (cfg.num_classes, cfg.channels, cfg.img_size, cfg.img_size)
    cls = torch.arange(cfg.num_classes, device=dev)
    ws = [1.0, 3.0, 7.5]
    fig, axes = plt.subplots(len(ws) + 1, cfg.num_classes, figsize=(2 * cfg.num_classes, 2 * (len(ws) + 1)))
    with torch.no_grad():
        ddpm = ddpm_sample(model, shape, abar, kind="v", labels=cls, guidance=1.0,
                           generator=torch.Generator(device=dev).manual_seed(1))
        for j in range(cfg.num_classes):
            axes[0, j].imshow(ddpm[j, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
            axes[0, j].set_title(f"DDPM cls {j}"); axes[0, j].axis("off")
        ts = list(range(T - 1, -1, -4))
        for i, w in enumerate(ws):
            grid = ddim_sample(model, shape, abar, ts, kind="v", eta=0.0, labels=cls,
                               guidance=w, generator=torch.Generator(device=dev).manual_seed(1))
            for j in range(cfg.num_classes):
                axes[i + 1, j].imshow(grid[j, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
                axes[i + 1, j].set_title(f"DDIM w={w}"); axes[i + 1, j].axis("off")
    plt.tight_layout(); finish(_OUT / "samples.png")


if __name__ == "__main__":
    _schedules()
    _noising()
    _conditioning()
    _train_and_sample()
    print(f"wrote figures to {_OUT}")
    if SHOW:
        plt.show()
