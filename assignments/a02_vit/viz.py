"""A2 viz: ViT overfit loss curve and a CLS-token attention map, headless.

Always produces PNGs in out/ with no download. Path (a) overfits a synthetic batch
and plots the loss curve. Path (b) renders the [CLS]-token attention over the patch
grid on a synthetic image via an attention rollout. Path (c), only if timm is
installed and DINOv2 weights are reachable, loads vit_small_patch14_dinov2 and its
register variant and saves their attention-map difference; it falls back cleanly to
the synthetic path and prints a message if timm/internet are missing.

Run with: make viz A=a02_vit  (uses the reference solution).
"""

import sys
from pathlib import Path

from nanovision.viz import SHOW, finish, plt, plot_loss_curve  # noqa: E402  (sets the matplotlib backend)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Make the assignment's ViT importable when run as a script (no pytest conftest).
# Render uses NANOVISION_IMPL=solution, so solution/ goes on last (highest priority)
# for the bare imports; the top level still provides config.py.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "solution"))

from vit import ViT  # noqa: E402

from nanovision.attention import scaled_dot_product_attention  # noqa: E402
from nanovision.determinism import set_seed  # noqa: E402
from nanovision.trainer import Trainer  # noqa: E402

OUT = Path(__file__).parent / "out"


def overfit_curve(out_path):
    set_seed(0)
    B, num_classes = 8, 10
    x = torch.randn(B, 3, 32, 32)
    y = torch.randint(0, num_classes, (B,))
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4,
                num_classes=num_classes, n_registers=4, pool="cls")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    tr = Trainer(model, opt, lambda logits, t: F.cross_entropy(logits, t))
    losses = tr.overfit_one_batch((x, y), steps=500)
    plot_loss_curve(losses, out_path, title="A2 ViT overfit (synthetic, cross-entropy)")
    return losses[-1]


def _cls_attention_rollout(model, x):
    """Attention rollout from the [CLS] token to the patch tokens.

    Hooks every encoder block's self-attention to capture its (B, H, S, S) weights,
    averages heads, adds the identity (residual) and renormalizes per layer, then
    multiplies the layer matrices. The CLS row over the patch columns is the
    rollout attention used as a saliency map. Follows Abnar & Zuidema (2020).
    """
    weights = []

    def make_hook():
        def hook(module, inp, out):
            # Recompute the attention weights from the module's own projections so
            # we get the (B, H, S, S) map (the block returns only the output).
            x_in = inp[0]
            B, S, _ = x_in.shape
            q = module.q_proj(x_in).view(B, S, module.n_heads, module.head_dim).transpose(1, 2)
            k = module.k_proj(x_in).view(B, S, module.n_kv_heads, module.head_dim).transpose(1, 2)
            v = module.v_proj(x_in).view(B, S, module.n_kv_heads, module.head_dim).transpose(1, 2)
            if module.n_kv_heads != module.n_heads:
                r = module.n_heads // module.n_kv_heads
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            _, attn = scaled_dot_product_attention(q, k, v)
            weights.append(attn.detach())
        return hook

    handles = [block.attn.register_forward_hook(make_hook()) for block in model.encoder.blocks]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    S = weights[0].shape[-1]
    rollout = torch.eye(S)
    for attn in weights:
        a = attn[0].mean(0)               # average heads -> (S, S)
        a = a + torch.eye(S)              # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    n_patches = model.n_patches
    grid = model.grid
    cls_to_patches = rollout[0, 1 : 1 + n_patches]   # CLS row over patch tokens
    return cls_to_patches.reshape(grid, grid).numpy()


def attention_map(out_path):
    set_seed(0)
    model = ViT(img_size=32, patch=4, dim=64, depth=2, n_heads=4, n_registers=4)
    # A synthetic image with a bright off-center blob the CLS token can latch onto.
    yy, xx = np.mgrid[0:32, 0:32]
    blob = np.exp(-((xx - 21) ** 2 + (yy - 11) ** 2) / (2 * 5.0 ** 2))
    img = np.stack([blob, 0.3 * blob, 1.0 - blob], 0).astype(np.float32)
    x = torch.from_numpy(img).unsqueeze(0)

    rollout = _cls_attention_rollout(model, x)
    rollout_up = F.interpolate(
        torch.from_numpy(rollout)[None, None], size=(32, 32), mode="bilinear",
        align_corners=False
    )[0, 0].numpy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(np.transpose(img, (1, 2, 0)))
    axs[0].set_title("synthetic input")
    axs[0].axis("off")
    axs[1].imshow(np.transpose(img, (1, 2, 0)))
    im = axs[1].imshow(rollout_up, cmap="inferno", alpha=0.6)
    axs[1].set_title("[CLS] attention rollout (untrained)")
    axs[1].axis("off")
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    fig.tight_layout()
    finish(out_path)


def dinov2_register_comparison(out_path):
    """Optional: DINOv2 vs DINOv2-reg attention, if timm and weights are reachable.

    Returns True if it produced the figure, False if it fell back. Never raises.
    """
    try:
        import timm
    except Exception as e:  # noqa: BLE001
        print(f"timm not available ({e}); skipping the DINOv2 register comparison.")
        return False
    try:
        plain = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                   num_classes=0).eval()
        reg = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True,
                                num_classes=0).eval()
    except Exception as e:  # noqa: BLE001
        print(f"DINOv2 weights unreachable ({e}); skipping the register comparison.")
        return False

    set_seed(0)
    img = torch.randn(1, 3, 518, 518)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (name, model) in zip(axs, [("DINOv2", plain), ("DINOv2-reg", reg)]):
        with torch.no_grad():
            feats = model.forward_features(img)
        # Per-token feature norm; the high-norm artifacts are what registers remove.
        if isinstance(feats, dict):
            feats = feats.get("x_norm_patchtokens", next(iter(feats.values())))
        tokens = feats[0]
        n = tokens.shape[0]
        side = int(round((n) ** 0.5))
        norms = tokens.norm(dim=-1)[: side * side].reshape(side, side).numpy()
        im = ax.imshow(norms, cmap="viridis")
        ax.set_title(f"{name} patch-token norm")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    finish(out_path)
    print(f"wrote DINOv2 register comparison to {out_path}")
    return True


def main():
    OUT.mkdir(exist_ok=True)
    final = overfit_curve(OUT / "vit_overfit.png")
    attention_map(OUT / "cls_attention.png")
    did_dino = dinov2_register_comparison(OUT / "dinov2_registers.png")
    msg = f"final ViT overfit loss {final:.3e} - wrote {OUT}/vit_overfit.png, {OUT}/cls_attention.png"
    if did_dino:
        msg += f", {OUT}/dinov2_registers.png"
    print(msg)


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
