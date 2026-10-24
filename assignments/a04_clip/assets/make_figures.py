"""Render the README figures for the CLIP/SigLIP notes.

three_pair_batch.png. Our three-pair batch: three (image, caption) pairs drawn by the
assignment's toy data generator (red top-left, red bottom-right, green top-left), a
hand-picked 3x3 cosine-similarity matrix S for them, the row-wise softmax of 10 S (what
InfoNCE scores), and the entrywise sigmoid of 10 S - 5 (what the sigmoid loss scores).

perfect_model_loss.png. The InfoNCE loss of an idealized perfect model
(matched cosine +1, every mismatched cosine -1, which no real embedding reaches for N >= 3;
the curve is a lower bound), log(1 + (N-1) exp(-2 s)), against the logit
scale s, for several batch sizes N, with log N (the loss of a uniform guess) dashed.

Run from this directory:

    python make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanovision.data.toy import image_text_batch  # noqa: E402

# All 16 (class, attribute) combinations, then pick the three pairs by their caption tokens.
imgs, toks = image_text_batch(batch=16, size=16, channels=3, n_classes=4, n_attrs=4,
                              vocab_size=32, max_len=8, seed=0)
wanted = [(1, 5), (1, 8), (2, 5)]  # (class token, attribute token)
names = ["red, top-left", "red, bottom-right", "green, top-left"]
idx = [next(i for i in range(16) if (toks[i, 0].item(), toks[i, 1].item()) == w) for w in wanted]

S = np.array([[0.6, 0.3, 0.2],
              [0.4, 0.5, -0.4],
              [0.2, -0.4, 0.7]])
s, b = 10.0, -5.0
L = s * S
P_soft = np.exp(L) / np.exp(L).sum(axis=1, keepdims=True)
P_sig = 1.0 / (1.0 + np.exp(-(L + b)))

fig = plt.figure(figsize=(11, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
for k, (i, name) in enumerate(zip(idx, names)):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(imgs[i].permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_title(f"pair {k + 1}: image {k + 1}\ncaption {k + 1} = {list(toks[i, :3].tolist())}"
                 f"\n({name})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

panels = [(S, "cosine similarity S", -1, 1, "coolwarm"),
          (P_soft, "InfoNCE: softmax of 10 S\nalong each row (rows sum to 1)", 0, 1, "viridis"),
          (P_sig, "sigmoid loss: sigmoid(10 S - 5)\nfor each entry on its own", 0, 1, "viridis")]
for k, (M, title, lo, hi, cmap) in enumerate(panels):
    ax = fig.add_subplot(gs[1, k])
    ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi)
    for r in range(3):
        for c in range(3):
            v = M[r, c]
            txt = f"{v:+.1f}" if k == 0 else f"{v:.3f}"
            dark = (abs(v) > 0.6) if k == 0 else (v < 0.6)
            ax.text(c, r, txt, ha="center", va="center", fontsize=10,
                    color="white" if dark else "black")
    ax.set_xticks(range(3), [f"caption {c + 1}" for c in range(3)], fontsize=8)
    ax.set_yticks(range(3), [f"image {r + 1}" for r in range(3)], fontsize=8)
    ax.set_title(title, fontsize=10)
fig.suptitle("Our three-pair batch: matched pairs on the diagonal, negatives off it", fontsize=12)
fig.tight_layout()
fig.savefig("three_pair_batch.png", dpi=110)
plt.close(fig)

# Perfect-model InfoNCE loss vs the logit scale.
scales = np.linspace(1, 100, 400)
fig, ax = plt.subplots(figsize=(6.5, 4))
for N, color in ((8, "tab:blue"), (1024, "tab:orange"), (32768, "tab:green")):
    loss = np.log1p((N - 1) * np.exp(-2 * scales))
    ax.plot(scales, loss, color=color, label=f"N = {N}")
    ax.axhline(np.log(N), color=color, ls="--", lw=0.8)
    ax.text(30, np.log(N) + 0.1, f"log {N} = {np.log(N):.2f} (uniform guess)", va="bottom",
            fontsize=8, color=color)
ax.axvline(1 / 0.07, color="gray", ls=":", lw=1)
ax.text(1 / 0.07 + 1, 6, "CLIP init\ns = 1/0.07", fontsize=8, color="gray")
ax.set_xscale("log")
ax.set_ylim(-0.3, 11.5)
ax.set_xlabel("logit scale s = 1/temperature (log axis)")
ax.set_ylabel("InfoNCE loss, idealized perfect model (nats)")
ax.set_title("Idealized perfect model (lower bound): zero loss needs a large s", fontsize=10)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.tight_layout()
fig.savefig("perfect_model_loss.png", dpi=110)
plt.close(fig)
print("wrote three_pair_batch.png, perfect_model_loss.png")
