"""A3 viz: MAE reconstruction and the DINO collapse curves, headless.

Always produces PNGs in out/ with no download. Path (a) overfits a small MAE on a
synthetic image and plots original / masked / reconstructed patches. Path (b) runs
the three DINO collapse variants (full / no-centering / no-sharpening) and plots
teacher entropy vs step - the module's signature figure. Path (c), only if timm is
installed and DINO weights are reachable, loads a pretrained DINO ViT and shows its
[CLS] attention; it falls back cleanly and prints a message if timm/internet are
missing.

Run with: make viz A=a03_0_ssl  (uses the reference solution).
"""

import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt  # noqa: E402  (sets the matplotlib backend)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Make the assignment's modules importable when run as a script (no pytest conftest).
# Render uses NANOVISION_IMPL=solution, so solution/ goes on last (highest priority)
# for the bare imports; the top level still provides config.py.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "solution"))

from backbone import patchify, unpatchify  # noqa: E402
from config import SSLConfig  # noqa: E402
from dino import build_student_teacher, dino_step, teacher_entropy  # noqa: E402
from mae import MAE  # noqa: E402

from nanovision.determinism import set_seed  # noqa: E402

OUT = Path(__file__).parent / "out"


def _synthetic_image():
    """A smooth low-frequency RGB image (so MAE has structure to reconstruct)."""
    set_seed(0)
    low = torch.randn(1, 3, 8, 8)
    img = F.interpolate(low, size=(32, 32), mode="bicubic", align_corners=False)
    img = (img - img.amin()) / (img.amax() - img.amin())
    return img


def mae_reconstruction(out_path):
    """Overfit a small MAE on one image and plot original / masked / reconstruction."""
    cfg = SSLConfig()
    img = _synthetic_image()
    model = MAE(img_size=cfg.img_size, patch=cfg.patch, in_chans=cfg.in_chans,
                enc_dim=cfg.enc_dim, enc_depth=cfg.enc_depth, enc_heads=cfg.enc_heads,
                dec_dim=cfg.dec_dim, dec_depth=cfg.dec_depth, dec_heads=cfg.dec_heads,
                mask_ratio=cfg.mask_ratio)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.mae_lr)
    for _ in range(cfg.mae_steps):
        torch.manual_seed(0)               # fixed mask -> clean memorization
        loss, _, _ = model(img)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        _, pred, mask = model(img)         # pred is per-patch-normalized; mask (1, N)

    # Build a masked-input visualization (gray out the masked patches) and a
    # reconstruction that keeps visible patches and fills masked ones from pred.
    patch = cfg.patch
    grid = model.grid
    target_raw = patchify(img, patch)                         # (1, N, p*p*C) raw pixels
    # De-normalize the prediction back to pixel scale per patch using target stats.
    mean = target_raw.mean(dim=-1, keepdim=True)
    std = target_raw.var(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6).sqrt()
    pred_pixels = pred * std + mean
    m = mask.unsqueeze(-1)
    recon_patches = target_raw * (1 - m) + pred_pixels * m    # visible kept, masked filled
    masked_patches = target_raw * (1 - m) + 0.5 * m           # gray on masked

    recon = unpatchify(recon_patches, patch, cfg.in_chans, grid)[0].clamp(0, 1)
    masked = unpatchify(masked_patches, patch, cfg.in_chans, grid)[0].clamp(0, 1)
    orig = img[0]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, im, title in zip(
        axs,
        [orig, masked, recon],
        ["original", f"masked ({int(cfg.mask_ratio * 100)}%)", "reconstruction"],
    ):
        ax.imshow(im.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"A3 MAE overfit (synthetic), masked-patch MSE {loss.item():.3e}")
    fig.tight_layout()
    finish(out_path)
    return loss.item()


def _collapse_curves():
    """Run the three DINO variants and return (steps, full, no_center, no_sharp)."""
    cfg = SSLConfig()
    cfg.ema_momentum = cfg.collapse_momentum
    set_seed(0)
    img = torch.randn(cfg.overfit_batch, 3, 32, 32)

    def run(use_centering, teacher_temp):
        set_seed(0)
        student, teacher = build_student_teacher(cfg)
        center = torch.zeros(1, cfg.out_dim)
        opt = torch.optim.Adam(student.parameters(), lr=cfg.dino_lr)
        ents = []
        for _ in range(cfg.dino_steps):
            _, center, tc = dino_step(student, teacher, center, img, cfg, opt,
                                      use_centering=use_centering, teacher_temp=teacher_temp)
            read_center = center if use_centering else torch.zeros_like(center)
            ents.append(teacher_entropy(tc, read_center, teacher_temp).item())
        return ents

    full = run(True, cfg.teacher_temp)
    no_center = run(False, cfg.teacher_temp)
    no_sharp = run(True, 1.0)
    return list(range(cfg.dino_steps)), full, no_center, no_sharp, np.log(cfg.out_dim)


def dino_collapse(out_path):
    steps, full, no_center, no_sharp, log_k = _collapse_curves()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(steps, full, label="full DINO")
    ax.plot(steps, no_center, label="no centering")
    ax.plot(steps, no_sharp, label="no sharpening (high teacher temp)")
    ax.axhline(log_k, color="gray", ls="--", lw=1, label="log K (uniform)")
    ax.axhline(0.0, color="black", ls=":", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("teacher entropy")
    ax.set_title("A3 DINO collapse: centering and sharpening each prevent a collapse")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    finish(out_path)
    return full[-1], no_center[-1], no_sharp[-1]


def pretrained_dino_attention(out_path):
    """Optional: a pretrained DINO ViT [CLS] attention map, if timm + weights load.

    Returns True if it produced the figure, False if it fell back. Never raises.
    """
    try:
        import timm
    except Exception as e:  # noqa: BLE001
        print(f"timm not available ({e}); skipping the pretrained-DINO attention map.")
        return False
    try:
        model = timm.create_model("vit_small_patch16_224.dino", pretrained=True,
                                  num_classes=0).eval()
    except Exception as e:  # noqa: BLE001
        print(f"DINO weights unreachable ({e}); skipping the pretrained attention map.")
        return False

    set_seed(0)
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feats = model.forward_features(img)
    tokens = feats[0, 1:] if feats.ndim == 3 else feats[0]
    side = int(round(tokens.shape[0] ** 0.5))
    norms = tokens.norm(dim=-1)[: side * side].reshape(side, side).numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(norms, cmap="inferno")
    ax.set_title("pretrained DINO patch-token norm")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    finish(out_path)
    print(f"wrote pretrained-DINO attention map to {out_path}")
    return True


def main():
    OUT.mkdir(exist_ok=True)
    mae_mse = mae_reconstruction(OUT / "mae_reconstruction.png")
    e_full, e_nc, e_ns = dino_collapse(OUT / "dino_collapse.png")
    did = pretrained_dino_attention(OUT / "dino_attention.png")
    msg = (f"MAE masked-patch MSE {mae_mse:.3e}; DINO end entropy full {e_full:.2f} "
           f"no-center {e_nc:.2f} no-sharp {e_ns:.2f} - wrote "
           f"{OUT}/mae_reconstruction.png, {OUT}/dino_collapse.png")
    if did:
        msg += f", {OUT}/dino_attention.png"
    print(msg)


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
