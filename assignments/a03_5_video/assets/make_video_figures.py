"""Render the README figures for the video MAE notes.

toy_clip_masks.png. One toy clip from nanovision.data.toy.video_batch (6 frames, 16x16,
three moving blobs), cut into 2x4x4 tubelets (a 3 x 4 x 4 grid of 48 tokens). Row 1 is
the clip. Row 2 keeps 6 tubelets chosen independently at random (random masking);
row 3 keeps 2 tubes of 3 tubelets (tube masking). Both hide 42 of 48 tubelets. Hidden tubelets are grayed out; a red outline marks a
hidden tubelet whose spatial position is visible in some other temporal step.

twelve_tokens.png. Our twelve-token clip: T' = 3 temporal steps by S' = 4 spatial
positions, tokens numbered idx = t' * S' + s. Left, random masking keeping 3
tokens; right, tube masking keeping spatial position 1 in every step.

Run from this directory:

    python make_video_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanovision.data.toy import video_batch  # noqa: E402

T, size, p, t = 6, 16, 4, 2
Tp, g = T // t, size // p
S = g * g
N = Tp * S

vid = video_batch(batch=8, n_frames=T, size=size, channels=3, n_blobs=3, seed=0)[0].numpy()

rng = np.random.default_rng(3)
keep_random = np.zeros(N, bool)
keep_random[rng.choice(N, 6, replace=False)] = True
keep_tube = np.zeros(N, bool)
tube_s = rng.choice(S, 2, replace=False)
for tp in range(Tp):
    keep_tube[tp * S + tube_s] = True


def leaks(keep):
    k = keep.reshape(Tp, S)
    return (~k) & k.any(axis=0, keepdims=True)


fig, axes = plt.subplots(3, T, figsize=(1.5 * T, 5.0))
rows = [("clip", None), ("random masking", keep_random), ("tube masking", keep_tube)]
for r, (label, keep) in enumerate(rows):
    for f in range(T):
        ax = axes[r, f]
        img = vid[:, f].transpose(1, 2, 0).clip(0, 1).copy()
        tp = f // t
        if keep is not None:
            k = keep.reshape(Tp, g, g)[tp]
            lk = leaks(keep).reshape(Tp, g, g)[tp]
            for h in range(g):
                for w in range(g):
                    if not k[h, w]:
                        img[h * p:(h + 1) * p, w * p:(w + 1) * p] = 0.55
        ax.imshow(img, interpolation="nearest")
        if keep is not None:
            for h in range(g):
                for w in range(g):
                    if lk[h, w]:
                        ax.add_patch(Rectangle((w * p - 0.5, h * p - 0.5), p, p, fill=False,
                                               ec="red", lw=1.5))
        for v in range(1, g):
            ax.axhline(v * p - 0.5, color="white", lw=0.4)
            ax.axvline(v * p - 0.5, color="white", lw=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(f"frame {f}  (t'={tp})", fontsize=8)
        if f == 0:
            ax.set_ylabel(label, fontsize=8)
n_leak_r = int(leaks(keep_random).sum())
n_leak_t = int(leaks(keep_tube).sum())
fig.suptitle(f"42 of 48 tubelets hidden. Red: hidden, but the same position is visible at "
             f"another t' ({n_leak_r} random, {n_leak_t} tube)", fontsize=9)
fig.tight_layout()
fig.savefig("toy_clip_masks.png", dpi=130)
plt.close(fig)
print("toy_clip_masks: leaks random", n_leak_r, "tube", n_leak_t)

# Our twelve-token clip.
Tp2, S2 = 3, 4
keep_r = np.zeros(Tp2 * S2, bool)
keep_r[[0, 5, 11]] = True
keep_t = np.zeros(Tp2 * S2, bool)
keep_t[[1, 5, 9]] = True
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, keep, title in [(axes[0], keep_r, "random masking (3 kept)"),
                        (axes[1], keep_t, "tube masking (spatial position 1 kept)")]:
    lk = ((~keep.reshape(Tp2, S2)) & keep.reshape(Tp2, S2).any(0, keepdims=True))
    for tp in range(Tp2):
        for s in range(S2):
            idx = tp * S2 + s
            kept = keep[idx]
            ax.add_patch(Rectangle((s, Tp2 - 1 - tp), 0.95, 0.95,
                                   fc="tab:blue" if kept else "0.85",
                                   ec="red" if lk[tp, s] else "none", lw=2))
            ax.text(s + 0.475, Tp2 - 1 - tp + 0.475, str(idx), ha="center", va="center",
                    fontsize=12, color="white" if kept else "black")
    ax.set_xlim(-0.1, S2)
    ax.set_ylim(-0.1, Tp2)
    ax.set_xticks([i + 0.475 for i in range(S2)])
    ax.set_xticklabels([f"s={s}\n(h'={s // 2}, w'={s % 2})" for s in range(S2)], fontsize=8)
    ax.set_yticks([Tp2 - 1 - tp + 0.475 for tp in range(Tp2)])
    ax.set_yticklabels([f"t'={tp}" for tp in range(Tp2)], fontsize=9)
    ax.set_title(title, fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
fig.text(0.5, 0.01, "blue: kept   gray: hidden   red outline: hidden, same s kept at another t'",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("twelve_tokens.png", dpi=130)
plt.close(fig)
print("twelve_tokens written")
