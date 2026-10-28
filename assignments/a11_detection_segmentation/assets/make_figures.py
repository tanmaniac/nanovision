"""Render the README figures for our two-square example and the IoU/GIoU comparison.

two_square_example.png. Left: a 32x32 image in the style of the toy dataset (a red square of
class 0 and a blue square of class 2 on black), the two ground-truth boxes (dashed white), and
three predicted query boxes. Right: the 3x2 matching cost matrix
C[i, j] = -p_i[c_j] + 5 ||b_i - b_j||_1 - 2 GIoU(b_i, b_j) on normalized cxcywh boxes, with the
Hungarian assignment outlined.

giou_vs_offset.png. IoU and GIoU between a fixed 8x8 box and a second 8x8 box slid
diagonally past it (the same offset in x and in y).

Self-contained (numpy, scipy, matplotlib); it does not use the assignment code. Run from this
directory:

    python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

SIZE = 32


def to_xyxy(b):
    cx, cy, w, h = b
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_giou(a, b):
    a, b = to_xyxy(a), to_xyxy(b)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = lambda x: (x[2] - x[0]) * (x[3] - x[1])
    union = area(a) + area(b) - inter
    ew = max(a[2], b[2]) - min(a[0], b[0])
    eh = max(a[3], b[3]) - min(a[1], b[1])
    enc = ew * eh
    iou = inter / union
    return iou, iou - (enc - union) / enc


# Our two-square example, in pixels (cx, cy, w, h); divided by 32 for the cost.
gt = np.array([[8, 8, 8, 8], [24, 24, 8, 8]], dtype=float)
gt_cls = [0, 2]
queries = np.array([[24, 23, 8, 8], [9, 8, 8, 8], [8, 10, 10, 10]], dtype=float)
probs = np.array([[0.05, 0.05, 0.80, 0.10],
                  [0.70, 0.05, 0.05, 0.20],
                  [0.50, 0.05, 0.05, 0.40]])

C = np.zeros((3, 2))
for i in range(3):
    for j in range(2):
        l1 = np.abs(queries[i] - gt[j]).sum() / SIZE
        _, g = iou_giou(queries[i] / SIZE, gt[j] / SIZE)
        C[i, j] = -probs[i, gt_cls[j]] + 5 * l1 - 2 * g
rows, cols = linear_sum_assignment(C)
print("cost matrix\n", np.round(C, 2))
print("assignment", list(zip(rows.tolist(), cols.tolist())), "total", round(C[rows, cols].sum(), 4))

img = np.zeros((SIZE, SIZE, 3))
img[4:12, 4:12] = [1, 0, 0]
img[20:28, 20:28] = [0, 0, 1]

fig, (ax, axc) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1]})
ax.imshow(img, extent=(0, SIZE, SIZE, 0))
for j, b in enumerate(gt):
    x0, y0, x1, y1 = to_xyxy(b)
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", ls="--", lw=2))
    ax.text(x0 + 9.5 if j == 0 else x0 - 8, y1 + 1.6 if j == 1 else y0 + 7.5, f"object {j} (class {gt_cls[j]})", color="white", fontsize=8)
qcolors = ["gold", "lime", "magenta"]
for i, b in enumerate(queries):
    x0, y0, x1, y1 = to_xyxy(b)
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=qcolors[i], lw=1.5))
    ax.text(x1 + 0.3, y1 if i == 2 else y0 + 1, f"q{i}", color=qcolors[i], fontsize=9)
ax.set_xlim(0, SIZE)
ax.set_ylim(SIZE, 0)
ax.set_title("Ground truth (dashed) and three query predictions")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

im = axc.imshow(C, cmap="viridis_r")
for i in range(3):
    for j in range(2):
        axc.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                 color="white" if C[i, j] > 2 else "black", fontsize=11)
for r, c in zip(rows, cols):
    axc.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="red", lw=3))
axc.set_xticks([0, 1], ["object 0", "object 1"])
axc.set_yticks([0, 1, 2], ["q0", "q1", "q2"])
axc.set_title("Cost C[i, j], Hungarian pairs in red")
fig.colorbar(im, ax=axc, fraction=0.046)
fig.tight_layout()
fig.savefig("two_square_example.png", dpi=120)

# IoU and GIoU as a box slides past another.
a = np.array([0.0, 0.0, 8.0, 8.0])
offs = np.linspace(-24, 24, 481)
ious, gious = zip(*[iou_giou(a, np.array([d, d, 8.0, 8.0])) for d in offs])
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(offs, ious, label="IoU", lw=2)
ax.plot(offs, gious, label="GIoU", lw=2)
ax.axhline(0, color="gray", lw=0.5)
ax.axvspan(-24, -8, color="gray", alpha=0.12)
ax.axvspan(8, 24, color="gray", alpha=0.12)
ax.text(-16, 0.55, "disjoint:\nIoU = 0", ha="center", fontsize=9)
ax.text(16, 0.55, "disjoint:\nIoU = 0", ha="center", fontsize=9)
ax.set_xlabel("offset d of the second 8x8 box, moved by (d, d) pixels")
ax.set_ylabel("score")
ax.set_title("IoU is flat once the boxes separate; GIoU keeps decreasing")
ax.legend(loc="lower center")
fig.tight_layout()
fig.savefig("giou_vs_offset.png", dpi=120)
print("wrote two_square_example.png, giou_vs_offset.png")
