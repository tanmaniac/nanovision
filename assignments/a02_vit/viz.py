"""A2 viz: ViT overfit loss curve, a CLS-token attention map, and DINOv2 token norms.

Writes PNGs to out/. Path (a) overfits a synthetic batch and plots the loss curve.
Path (b) renders the [CLS]-token attention over the patch grid on a synthetic image
via an attention rollout. Both need no download. Path (c), only if timm is installed
and the DINOv2 weights are reachable (vit_base_patch14_dinov2 and its reg4 variant,
about 331 MB each, cached after the first run), plots the patch-token norms at an
intermediate encoder block on matplotlib's bundled grace_hopper.jpg, with and without
register tokens; it prints a message and skips the figure if timm or the weights are
unavailable.

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
from nanovision.determinism import default_device, set_seed  # noqa: E402
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
    tr = Trainer(model, opt, lambda logits, t: F.cross_entropy(logits, t),
                 device=str(default_device()))
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


DINO_BLOCK = 9  # 1-based encoder block whose output norms are plotted


def _grace_hopper(res: int = 518) -> torch.Tensor:
    """matplotlib's bundled sample photo, resized and ImageNet-normalized: (1, 3, res, res)."""
    import matplotlib
    from PIL import Image

    path = Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    img = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((x - mean) / std)[None]


def dinov2_register_comparison(out_path):
    """Optional: patch-token norms of DINOv2 ViT-B with and without registers.

    Runs vit_base_patch14_dinov2 and vit_base_patch14_reg4_dinov2 (timm, about 331 MB
    of weights each) on matplotlib's bundled grace_hopper.jpg at 518x518, captures the
    output of encoder block DINO_BLOCK with a forward hook (before the final norm),
    drops the class and register tokens, and plots each patch token's norm on the
    37x37 grid. Returns True if it produced the figure, False if timm or the weights
    are unavailable. Never raises on a missing dependency.
    """
    try:
        import timm
    except Exception as e:  # noqa: BLE001
        print(f"timm not available ({e}); skipping the DINOv2 register comparison.")
        return False
    try:
        plain = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                                   num_classes=0).eval()
        reg = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True,
                                num_classes=0).eval()
    except Exception as e:  # noqa: BLE001
        print(f"DINOv2 weights unreachable ({e}); skipping the register comparison.")
        return False

    dev = default_device()  # the pretrained ViT-B forward at 518x518 is the heavy part
    img = _grace_hopper().to(dev)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(((img[0].cpu() * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                    + torch.tensor([0.485, 0.456, 0.406])[:, None, None])
                   .clamp(0, 1).permute(1, 2, 0).numpy()))
    axs[0].set_title("input (518x518)")
    axs[0].axis("off")
    for ax, (name, model) in zip(axs[1:], [("DINOv2 ViT-B", plain), ("DINOv2 ViT-B reg4", reg)]):
        model = model.to(dev)
        captured = {}
        handle = model.blocks[DINO_BLOCK - 1].register_forward_hook(
            lambda mod, inp, out: captured.__setitem__("x", out.detach()))
        with torch.no_grad():
            model.forward_features(img)
        handle.remove()
        # timm puts the class token and any register tokens first; drop them so only
        # the patch tokens are laid out on the grid.
        tokens = captured["x"][0, model.num_prefix_tokens:]
        n = tokens.shape[0]
        side = int(round(n ** 0.5))
        assert side * side == n, f"{name}: {n} patch tokens do not form a square grid"
        norms = tokens.norm(dim=-1).float()
        med = norms.median().item()
        n_out = int((norms > 10 * med).sum().item())
        print(f"{name} block {DINO_BLOCK}: median norm {med:.1f}, max {norms.max().item():.1f} "
              f"({norms.max().item() / med:.1f}x median), {n_out} tokens above 10x median")
        im = ax.imshow(norms.reshape(side, side).cpu().numpy(), cmap="viridis")
        ax.set_title(f"{name}, block {DINO_BLOCK} patch-token norm")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        model.cpu()
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
