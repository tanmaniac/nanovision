"""Render the README figures for the VLM notes.

toy_task.png. All 16 (class, attribute) combinations of the toy captioning data
(nanovision.data.toy.image_text_batch with batch=16), each image titled with its caption's
first three tokens [class, attribute, EOS].

sequence_layout.png. The combined sequence for our blue-blob example (caption
[3, 7, 31, 0, 0, 0, 0, 0], 16 visual tokens): what enters each of the 24 positions, and the
target each position is trained to predict after the one-step shift. Only positions 15, 16,
and 17 have a target; the rest carry the ignore index.

Run from this directory:

    python make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanovision.data import toy  # noqa: E402

# Figure 1: the toy task.
img, tok = toy.image_text_batch(batch=16, seed=0)
order = sorted(range(16), key=lambda i: (int(tok[i, 0]), int(tok[i, 1])))
fig, axes = plt.subplots(4, 4, figsize=(7.2, 8.0))
for ax, i in zip(axes.flat, order):
    ax.imshow(img[i].permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_title(str(tok[i, :3].tolist()), fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
for r in range(4):
    axes[r, 0].set_ylabel(f"class token {r + 1}", fontsize=9)
fig.suptitle("Toy captioning data: caption = [class, attribute, EOS]", fontsize=11)
fig.tight_layout()
fig.savefig("toy_task.png", dpi=110)
plt.close(fig)

# Figure 2: the combined sequence and the shifted targets.
N, L = 16, 8
caption = [3, 7, 31, 0, 0, 0, 0, 0]
inputs = [f"v{j}" for j in range(N)] + [str(t) for t in caption]
labels = [None] * N + [t if t != 0 else None for t in caption]
targets = labels[1:] + ["-"]  # position i is trained to predict labels[i + 1]

fig, ax = plt.subplots(figsize=(13, 2.6))
w = 1.0
for i in range(N + L):
    vis = i < N
    ax.add_patch(Rectangle((i * w, 1.0), w, 0.8, facecolor="#cfe0f3" if vis else "#f6dcc0",
                           edgecolor="k", lw=0.6))
    ax.text(i * w + 0.5, 1.4, inputs[i], ha="center", va="center", fontsize=8)
    t = targets[i]
    sup = t not in (None, "-")
    ax.add_patch(Rectangle((i * w, 0.0), w, 0.8, facecolor="#b9e4b0" if sup else "#eeeeee",
                           edgecolor="k", lw=0.6))
    ax.text(i * w + 0.5, 0.4, str(t) if sup else ("none" if t == "-" else "-100"),
            ha="center", va="center", fontsize=7 if not sup else 9,
            fontweight="bold" if sup else "normal")
    ax.text(i * w + 0.5, 2.0, str(i), ha="center", va="center", fontsize=7, color="0.4")
ax.text(-0.3, 1.4, "input", ha="right", va="center", fontsize=9)
ax.text(-0.3, 0.4, "target", ha="right", va="center", fontsize=9)
ax.text(-0.3, 2.0, "position", ha="right", va="center", fontsize=8, color="0.4")
ax.text(N / 2, 2.4, "visual tokens (from the image)", ha="center", fontsize=9)
ax.text(N + L / 2, 2.4, "text embeddings (caption)", ha="center", fontsize=9)
ax.set_xlim(-2.2, N + L + 0.2)
ax.set_ylim(-0.2, 2.7)
ax.axis("off")
ax.set_title("Our blue-blob example: only positions 15, 16, 17 are supervised "
             "(green); grey cells are ignored", fontsize=10)
fig.tight_layout()
fig.savefig("sequence_layout.png", dpi=110)
plt.close(fig)
print("wrote toy_task.png, sequence_layout.png")
