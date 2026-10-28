"""A3 viz: MAE reconstruction and the DINO collapse curves, headless.

Always produces PNGs in out/ with no download. Path (a) overfits a small MAE on a
synthetic image and plots original / masked / reconstructed patches. Path (b) runs
the three DINO collapse variants (full / no-centering / no-sharpening) and plots
teacher entropy vs step - the module's signature figure. Path (c), only if timm is
installed and the DINO weights are reachable (about 85 MB, downloaded once), loads the
pretrained DINO ViT-S/8 and plots its last-layer class-token self-attention, one map per
head, on matplotlib's bundled sample photo grace_hopper.jpg at 448x448 (DINO's emergent
object attention, Caron et al. 2021, Fig. 1), and prints how much of each head's
attention falls on the subject's head and cap versus the background. It falls back
cleanly and prints a message if timm/internet are missing.

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

from nanovision.determinism import default_device, set_seed  # noqa: E402

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
    dev = default_device()
    img = _synthetic_image().to(dev)
    model = MAE(img_size=cfg.img_size, patch=cfg.patch, in_chans=cfg.in_chans,
                enc_dim=cfg.enc_dim, enc_depth=cfg.enc_depth, enc_heads=cfg.enc_heads,
                dec_dim=cfg.dec_dim, dec_depth=cfg.dec_depth, dec_heads=cfg.dec_heads,
                mask_ratio=cfg.mask_ratio).to(dev)
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
        ax.imshow(im.permute(1, 2, 0).cpu().numpy())
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
    dev = default_device()
    img = torch.randn(cfg.overfit_batch, 3, 32, 32, device=dev)

    def run(use_centering, teacher_temp):
        set_seed(0)
        student, teacher = build_student_teacher(cfg)
        student, teacher = student.to(dev), teacher.to(dev)
        center = torch.zeros(1, cfg.out_dim, device=dev)
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


# Rough regions of matplotlib's grace_hopper.jpg (512 wide x 600 high), as fractions
# (x0, y0, x1, y1) of width and height. HEAD covers the face and cap; BACKGROUND is the
# flag on the left plus the blue backdrop on the right, both above the shoulders.
_HEAD_BOX = (150 / 512, 20 / 600, 375 / 512, 340 / 600)
_BACKGROUND_BOXES = [(0 / 512, 0 / 600, 140 / 512, 330 / 600),
                     (380 / 512, 0 / 600, 512 / 512, 300 / 600)]


def _box_mask(boxes, grid):
    """Boolean (grid, grid) mask of the patches whose centers fall inside any box."""
    c = (np.arange(grid) + 0.5) / grid
    xs, ys = np.meshgrid(c, c)
    m = np.zeros((grid, grid), bool)
    for x0, y0, x1, y1 in boxes:
        m |= (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return m


def pretrained_dino_cls_attention(out_path, model_name: str = "vit_small_patch8_224.dino",
                                  img_size: int = 448):
    """Optional: DINO's emergent result, the class token's last-layer self-attention.

    Loads a pretrained DINO ViT from timm (default ViT-S/8, `vit_small_patch8_224.dino`,
    about 85 MB of weights; the 8x8-patch ViT-S is the model in Caron et al. 2021, Fig. 1),
    runs it on matplotlib's bundled sample photo `grace_hopper.jpg` resized to
    img_size x img_size (default 448; the paper's figures use 480p), and reads the
    attention weights of the class token over the patch tokens in the last block, one
    map per head. The weights are recomputed
    from the last block's attention input (q, k, softmax) captured by a forward pre-hook.

    Also prints, per head, the fraction of the class token's patch attention that falls
    on the head-and-cap region and on the background (flag and backdrop), next to the
    area fraction of each region. Returns the per-head stats dict, or None if timm or the
    weights are unavailable. Never raises on a missing dependency.
    """
    try:
        import timm
    except Exception as e:  # noqa: BLE001
        print(f"timm not available ({e}); skipping the pretrained-DINO attention maps.")
        return None
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=0,
                                  dynamic_img_size=True).eval()
    except Exception as e:  # noqa: BLE001
        print(f"DINO weights unreachable ({e}); skipping the pretrained-DINO attention maps.")
        return None

    import matplotlib.cbook as cbook
    from PIL import Image

    dev = default_device()
    model = model.to(dev)
    photo = Image.open(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)).convert("RGB")
    photo = np.asarray(photo.resize((img_size, img_size), Image.BICUBIC), dtype=np.float32) / 255.0
    mean = np.array(model.pretrained_cfg["mean"], dtype=np.float32)
    std = np.array(model.pretrained_cfg["std"], dtype=np.float32)
    x = torch.from_numpy((photo - mean) / std).permute(2, 0, 1)[None].to(dev)

    attn_mod = model.blocks[-1].attn
    captured = {}
    handle = attn_mod.register_forward_pre_hook(lambda mod, args: captured.update(x=args[0]))
    with torch.no_grad():
        model(x)
        handle.remove()
        t = captured["x"]                                         # (1, 1 + g*g, dim)
        B, N, C = t.shape
        H = attn_mod.num_heads
        qkv = attn_mod.qkv(t).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k = attn_mod.q_norm(qkv[0]), attn_mod.k_norm(qkv[1])
        attn = ((q * attn_mod.scale) @ k.transpose(-2, -1)).softmax(dim=-1)  # (1, H, N, N)
    g = img_size // model.patch_embed.patch_size[0]               # patches per side
    cls = attn[0, :, 0, N - g * g:]                               # (H, g*g), skip class token
    cls = (cls / cls.sum(dim=-1, keepdim=True)).reshape(H, g, g).cpu().numpy()

    fg = _box_mask([_HEAD_BOX], g)
    bg = _box_mask(_BACKGROUND_BOXES, g)
    stats = {"head_area": fg.mean(), "background_area": bg.mean(),
             "head_mass": [float(a[fg].sum()) for a in cls],
             "background_mass": [float(a[bg].sum()) for a in cls]}

    fig, axs = plt.subplots(1, H + 1, figsize=(2.0 * (H + 1), 2.4))
    axs[0].imshow(photo)
    axs[0].set_title("input", fontsize=9)
    for h in range(H):
        axs[h + 1].imshow(cls[h], cmap="inferno")
        axs[h + 1].set_title(f"head {h}: {100 * stats['head_mass'][h]:.0f}% on head",
                             fontsize=7)
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Pretrained DINO {model_name} at {img_size}x{img_size}: last-layer "
                 "class-token attention per head "
                 f"(head-and-cap box is {100 * stats['head_area']:.0f}% of the image)",
                 fontsize=9)
    fig.tight_layout()
    finish(out_path)
    for h in range(H):
        print(f"head {h}: attention mass on head-and-cap box {stats['head_mass'][h]:.3f} "
              f"(area {stats['head_area']:.3f}), on background {stats['background_mass'][h]:.3f} "
              f"(area {stats['background_area']:.3f})")
    print(f"wrote pretrained-DINO class-token attention maps to {out_path}")
    return stats


def main():
    OUT.mkdir(exist_ok=True)
    mae_mse = mae_reconstruction(OUT / "mae_reconstruction.png")
    e_full, e_nc, e_ns = dino_collapse(OUT / "dino_collapse.png")
    did = pretrained_dino_cls_attention(OUT / "dino_attention.png")
    msg = (f"MAE masked-patch MSE {mae_mse:.3e}; DINO end entropy full {e_full:.2f} "
           f"no-center {e_nc:.2f} no-sharp {e_ns:.2f} - wrote "
           f"{OUT}/mae_reconstruction.png, {OUT}/dino_collapse.png")
    if did is not None:
        msg += f", {OUT}/dino_attention.png"
    print(msg)


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
