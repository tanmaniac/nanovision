"""Render the README figures for the self-supervised learning notes.

mae_masking.png. A 32x32 picture cut into the 8x8 grid of 4x4 patches, the same picture with 75% of
the patches masked at random, and the 16 visible patches laid out as the token sequence the MAE
encoder receives.

eight_tokens.png. Our eight-token example: ids_shuffle = [5, 2, 7, 0, 3, 6, 1, 4], keep 4. Shows the
shuffled order with the kept slots, and how gathering by ids_restore carries the mask back to grid
order.

three_image_batch.png. Our three-image batch: teacher logits for 3 images over K = 4 prototypes,
turned into teacher distributions three ways (centering with a high temperature, sharpening with no
centering, centering and sharpening).

None of these uses the assignment code. Run from this directory:

    python make_ssl_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- mae_masking.png
S, P = 32, 4
G = S // P
yy, xx = np.mgrid[0:S, 0:S]
img = np.zeros((S, S, 3))
img[:] = (0.55 + 0.35 * yy / S)[..., None] * np.array([0.55, 0.75, 1.0])  # sky gradient
img[yy > 22] = [0.35, 0.6, 0.3]                                            # ground
disk = (yy - 10) ** 2 + (xx - 23) ** 2 < 25
img[disk] = [1.0, 0.85, 0.2]                                               # sun
house = (yy >= 14) & (yy <= 26) & (xx >= 5) & (xx <= 15)
img[house] = [0.75, 0.3, 0.25]
roof = (yy >= 8) & (yy < 14) & (np.abs(xx - 10) <= (yy - 8))
img[roof] = [0.4, 0.2, 0.15]
img[(yy >= 19) & (yy <= 26) & (xx >= 9) & (xx <= 11)] = [0.3, 0.2, 0.1]   # door

rng = np.random.default_rng(3)
N = G * G
n_keep = round(0.25 * N)
ids_shuffle = np.argsort(rng.random(N))
keep = np.sort(ids_shuffle[:n_keep])
mask = np.ones(N, bool)
mask[keep] = False

masked = img.copy()
for i in range(N):
    if mask[i]:
        r, c = divmod(i, G)
        masked[r * P:(r + 1) * P, c * P:(c + 1) * P] = 0.5

fig = plt.figure(figsize=(11, 3.6))
ax1 = fig.add_axes([0.02, 0.12, 0.25, 0.78])
ax2 = fig.add_axes([0.30, 0.12, 0.25, 0.78])
ax3 = fig.add_axes([0.60, 0.30, 0.38, 0.40])
for ax, im, title in [(ax1, img, "image, 8x8 grid of 4x4 patches"),
                      (ax2, masked, "75% of patches masked (gray)")]:
    ax.imshow(im, interpolation="nearest")
    for k in range(1, G):
        ax.axhline(k * P - 0.5, color="white", lw=0.5)
        ax.axvline(k * P - 0.5, color="white", lw=0.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
strip = np.ones((P, n_keep * (P + 1) - 1, 3))
for j, i in enumerate(keep):
    r, c = divmod(i, G)
    strip[:, j * (P + 1):j * (P + 1) + P] = img[r * P:(r + 1) * P, c * P:(c + 1) * P]
ax3.imshow(strip, interpolation="nearest")
ax3.set_title("encoder input: the 16 visible patches only", fontsize=10)
ax3.set_xticks([j * (P + 1) + 1.5 for j in range(n_keep)])
ax3.set_xticklabels([str(i) for i in keep], fontsize=7)
ax3.set_yticks([])
ax3.set_xlabel("patch index in the grid (row-major)", fontsize=8)
for s in ax3.spines.values():
    s.set_visible(False)
fig.savefig("mae_masking.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- eight_tokens.png
ids_shuffle = np.array([5, 2, 7, 0, 3, 6, 1, 4])
ids_restore = np.argsort(ids_shuffle)
nk = 4
mask_shuf = np.array([0] * nk + [1] * (8 - nk))
mask_grid = mask_shuf[ids_restore]
kept_c, masked_c = "#4c9be8", "#d9d9d9"

fig, ax = plt.subplots(figsize=(9, 3.3))
y_top, y_bot = 2.0, 0.0
for slot, tok in enumerate(ids_shuffle):
    col = kept_c if slot < nk else masked_c
    ax.add_patch(Rectangle((slot - 0.4, y_top - 0.35), 0.8, 0.7, color=col))
    ax.text(slot, y_top + 0.05, f"token {tok}", ha="center", va="center", fontsize=8)
    ax.text(slot, y_top - 0.22, f"mask {mask_shuf[slot]}", ha="center", va="center", fontsize=7)
    ax.text(slot, y_top + 0.5, f"slot {slot}", ha="center", fontsize=7, color="gray")
for pos in range(8):
    col = kept_c if mask_grid[pos] == 0 else masked_c
    ax.add_patch(Rectangle((pos - 0.4, y_bot - 0.35), 0.8, 0.7, color=col))
    ax.text(pos, y_bot + 0.05, f"token {pos}", ha="center", va="center", fontsize=8)
    ax.text(pos, y_bot - 0.22, f"mask {mask_grid[pos]}", ha="center", va="center", fontsize=7)
    ax.text(pos, y_bot - 0.6, f"restore {ids_restore[pos]}", ha="center", fontsize=7, color="gray")
    ax.annotate("", xy=(pos, y_bot + 0.37), xytext=(ids_restore[pos], y_top - 0.37),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
ax.text(-0.6, y_top, "shuffled\norder", ha="right", va="center", fontsize=9)
ax.text(-0.6, y_bot, "grid\norder", ha="right", va="center", fontsize=9)
ax.text(1.5, y_top + 0.85, "kept (first 4 slots)", ha="center", fontsize=9, color=kept_c)
ax.text(5.5, y_top + 0.85, "masked", ha="center", fontsize=9, color="gray")
ax.set_xlim(-1.6, 7.6)
ax.set_ylim(-0.9, 3.1)
ax.axis("off")
ax.set_title("Our eight-token example: grid position i reads shuffled slot ids_restore[i]", fontsize=10)
fig.tight_layout()
fig.savefig("eight_tokens.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- three_image_batch.png
logits = np.array([[0.9, 0.6, 0.0, 0.0],
                   [0.9, 0.0, 0.6, 0.0],
                   [0.9, 0.0, 0.0, 0.6]])
center = logits.mean(axis=0)


def softmax(g, tau):
    z = np.exp((g - g.max(axis=-1, keepdims=True)) / tau)
    return z / z.sum(axis=-1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


settings = [("centering, tau = 1\n(no sharpening)", logits - center, 1.0),
            ("no centering, tau = 0.04\n(sharpening only)", logits, 0.04),
            ("centering, tau = 0.04\n(both, as in DINO)", logits - center, 0.04)]
fig, axs = plt.subplots(4, 3, figsize=(9, 7.2), sharey=True)
colors = ["#7f7f7f", "#4c9be8", "#e8834c", "#59b26b"]
for j, (title, g, tau) in enumerate(settings):
    p = softmax(g, tau)
    for i in range(3):
        ax = axs[i, j]
        ax.bar(range(4), p[i], color=colors)
        ax.set_ylim(0, 1.3)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["1", "2", "3", "4"], fontsize=8)
        ax.text(3.45, 1.12, f"H = {entropy(p[i]):.2f}", ha="right", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"image {'ABC'[i]}", fontsize=9)
        if i == 0:
            ax.set_title(title, fontsize=9)
    ax = axs[3, j]
    pm = p.mean(axis=0)
    ax.bar(range(4), pm, color=colors, alpha=0.6)
    ax.set_ylim(0, 1.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1", "2", "3", "4"], fontsize=8)
    ax.text(3.45, 1.12, f"H = {entropy(pm):.2f}", ha="right", fontsize=8)
    ax.set_xlabel("prototype", fontsize=8)
    if j == 0:
        ax.set_ylabel("batch average", fontsize=9)
fig.suptitle("Our three-image batch: teacher distributions over K = 4 prototypes", fontsize=10)
fig.tight_layout()
fig.savefig("three_image_batch.png", dpi=130)
plt.close(fig)
print("wrote mae_masking.png, eight_tokens.png, three_image_batch.png")
