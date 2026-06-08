"""Render the VLM toy: train briefly, show generated captions, and the grounding ablation.

Run from the repo root: `python -m assignments.a08_vlm.viz` (or `make viz A=a08_vlm`). It
trains the VLM on a small image_text_batch (stage 1 then stage 2), prints the generated
caption for each image next to its target, and saves a bar chart of the caption loss under
three conditions: the full model, the visual tokens zeroed, and the projector re-initialized
to random weights. Writes figures to out/. Runs in solution mode (the shims load the filled
ViT and decoder). Not graded.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_impl))

from config import VLMConfig  # noqa: E402
from projector import MLPProjector  # noqa: E402
from vlm import VLM, prepend_visual, vlm_loss  # noqa: E402

from nanovision.transformer import build_causal_mask  # noqa: E402
from nanovision.data import toy  # noqa: E402

_OUT = _here / "out"
_OUT.mkdir(exist_ok=True)


def _train(model, img, tok, stage, steps, lr=3e-3):
    model.set_stage(stage)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits, n_visual = model(img, tok)
        loss = vlm_loss(logits, tok, n_visual)
        loss.backward()
        opt.step()
    return loss.item()


def _loss_with_visual(model, img, tok, mode):
    feats = model.vit.forward_features(img)
    if mode == "random_projector":
        proj = MLPProjector(model.cfg.vit_dim, model.cfg.dim_l)
        vis = proj(feats)
    else:
        vis = model.connector(feats)
        if mode == "zeroed":
            vis = torch.zeros_like(vis)
    seq = prepend_visual(vis, model.embed(tok))
    mask = build_causal_mask(seq.shape[1])
    logits = model.head(model.norm(model.decoder(seq, mask=mask)))
    return vlm_loss(logits, tok, vis.shape[1]).item()


def main():
    torch.manual_seed(0)
    cfg = VLMConfig()
    model = VLM(cfg)
    img, tok = toy.image_text_batch(
        batch=4, size=cfg.img_size, channels=cfg.in_chans, n_classes=cfg.n_classes,
        n_attrs=cfg.n_attrs, vocab_size=cfg.vocab_size, max_len=cfg.max_len, seed=0,
    )

    s1 = _train(model, img, tok, stage=1, steps=300)
    s2 = _train(model, img, tok, stage=2, steps=300)
    print(f"stage 1 final loss {s1:.4f}   stage 2 final loss {s2:.5f}")

    gen = model.generate(img, max_new_tokens=3)
    for i in range(img.shape[0]):
        print(f"image {i}: target {tok[i, :3].tolist()}  generated {gen[i].tolist()}")

    losses = {
        "full": _loss_with_visual(model, img, tok, "full"),
        "visual zeroed": _loss_with_visual(model, img, tok, "zeroed"),
        "random projector": _loss_with_visual(model, img, tok, "random_projector"),
    }
    print("grounding ablation:", {k: round(v, 4) for k, v in losses.items()})

    plt.figure(figsize=(4.8, 3.2))
    plt.bar(list(losses), list(losses.values()), color=["C0", "C3", "C1"])
    plt.ylabel("caption cross-entropy")
    plt.title("grounding ablation")
    plt.tight_layout()
    plt.savefig(_OUT / "grounding_ablation.png", dpi=120)
    plt.close()

    # Images with their generated captions.
    fig, axes = plt.subplots(1, img.shape[0], figsize=(2.2 * img.shape[0], 2.6))
    for i in range(img.shape[0]):
        axes[i].imshow(img[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i].set_title(f"gen {gen[i].tolist()}\ntgt {tok[i, :3].tolist()}", fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(_OUT / "captions.png", dpi=120)
    plt.close()
    print(f"wrote figures to {_OUT}")


if __name__ == "__main__":
    main()
