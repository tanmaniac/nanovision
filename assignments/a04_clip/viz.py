"""Render the CLIP/SigLIP dual encoder: the similarity matrix and a small-batch note.

Run from the repo root: `python -m assignments.a04_clip.viz` (or `make viz A=a04_clip`).
Writes two figures: the image-text cosine-similarity matrix before and after training
(the diagonal should light up as matched pairs align), and an alignment-vs-batch-size
curve for InfoNCE vs the sigmoid loss. The curve is a demonstration, not a benchmark:
overfitting a tiny fixed batch is exactly where InfoNCE is strong, so both losses align
at small N here. The real SigLIP advantage is representation quality during large-scale
training, which a 12GB toy cannot reproduce.
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

from config import CLIPConfig  # noqa: E402
from losses import clip_loss, siglip_loss  # noqa: E402
from model import CLIPModel  # noqa: E402

from nanovision.data.toy import image_text_batch  # noqa: E402
from nanovision.determinism import default_device, set_seed  # noqa: E402


def _sim(model, imgs, toks):
    with torch.no_grad():
        fi, ft = model(imgs, toks)
        return (fi @ ft.T).cpu().numpy()


def _train(cfg, imgs, toks, loss_name, steps):
    set_seed(0)
    model = CLIPModel(cfg).to(imgs.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(steps):
        fi, ft = model(imgs, toks)
        if loss_name == "clip":
            loss = clip_loss(fi, ft, model.logit_scale_clamped())
        else:
            loss = siglip_loss(fi, ft, model.logit_scale_clamped(), model.bias)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main() -> None:
    set_seed(0)
    cfg = CLIPConfig()
    dev = default_device()
    out = _here / "out"
    out.mkdir(exist_ok=True)

    imgs, toks = image_text_batch(batch=cfg.overfit_batch, size=cfg.img_size,
                                  channels=cfg.in_chans, n_classes=cfg.n_classes,
                                  n_attrs=cfg.n_attrs, vocab_size=cfg.vocab_size,
                                  max_len=cfg.max_len, seed=0)
    imgs, toks = imgs.to(dev), toks.to(dev)

    before = CLIPModel(cfg).to(dev)
    sim_before = _sim(before, imgs, toks)
    trained = _train(cfg, imgs, toks, "siglip", cfg.steps)
    sim_after = _sim(trained, imgs, toks)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, mat, title in ((axes[0], sim_before, "before training"),
                           (axes[1], sim_after, "after (SigLIP)")):
        im = ax.imshow(mat, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("text"); ax.set_ylabel("image")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("image-text cosine similarity (diagonal = matched pairs)", fontsize=11)
    fig.tight_layout()
    sim_path = out / "similarity_matrix.png"
    finish(sim_path)

    # Alignment vs batch size for both losses (a demonstration, see the caption).
    ns = [2, 4, 8, 16]
    align = {"clip": [], "siglip": []}
    for n in ns:
        bimgs, btoks = image_text_batch(batch=n, size=cfg.img_size, channels=cfg.in_chans,
                                        n_classes=cfg.n_classes, n_attrs=cfg.n_attrs,
                                        vocab_size=cfg.vocab_size, max_len=cfg.max_len, seed=1)
        bimgs, btoks = bimgs.to(dev), btoks.to(dev)
        for name in ("clip", "siglip"):
            m = _train(cfg, bimgs, btoks, name, 200)
            with torch.no_grad():
                fi, ft = m(bimgs, btoks)
                s = fi @ ft.T
                frac = (s.argmax(dim=1) == torch.arange(n, device=dev)).float().mean().item()
            align[name].append(frac)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ns, align["clip"], "o-", label="InfoNCE (CLIP)")
    ax.plot(ns, align["siglip"], "s-", label="sigmoid (SigLIP)")
    ax.set_xlabel("batch size N"); ax.set_ylabel("fraction of rows aligned")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("overfit alignment vs N (both align; gap needs scale)", fontsize=9)
    fig.tight_layout()
    curve_path = out / "alignment_vs_batch.png"
    finish(curve_path)

    print(f"wrote {sim_path}, {curve_path}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
