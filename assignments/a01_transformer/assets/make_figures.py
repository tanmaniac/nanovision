"""Render the README figures for the transformer notes.

prev_char_head.png. Our previous-character head: one attention head with a two-channel query and
key (a single RoPE pair) that turns a quarter turn per position (theta = pi/2). Every key has the
same content, b = (2, 0), and the query is a = (0, -2). Left panel: the keys of the four
characters "the " after rotation by n * theta, and the query at position 3 after rotation by
3 * theta; it points the same way as the key at position 2. Middle and right panels: the causal
attention weights of all four query positions without position information (keys and query left
unrotated) and with RoPE.

softmax_scaling.png. For random queries and keys with independent unit-variance entries, the
average largest attention weight over 32 keys, as a function of the head width d_h, with raw
dot products and with dot products divided by sqrt(d_h).

Run from this directory:

    python make_figures.py
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def rot(phi):
    return np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])


# ----- our previous-character head -----
chars = ["t", "h", "e", "␣"]
S = len(chars)
theta = math.pi / 2
a = np.array([0.0, -2.0])  # query before rotation
b = np.array([2.0, 0.0])  # every key before rotation
mask = np.triu(np.full((S, S), -np.inf), k=1)

scores_plain = np.array([[a @ b for n in range(S)] for m in range(S)]) / math.sqrt(2)
scores_rope = np.array(
    [[(rot(m * theta) @ a) @ (rot(n * theta) @ b) for n in range(S)] for m in range(S)]
) / math.sqrt(2)
w_plain = softmax(scores_plain + mask)
w_rope = softmax(scores_rope + mask)

fig, (ax_p, ax_n, ax_r) = plt.subplots(1, 3, figsize=(12.5, 4.7))

colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
for n in range(S):
    k = rot(n * theta) @ b
    ax_p.annotate("", xy=k, xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color=colors[n], lw=2))
    off = {0: (2.2, 0.35), 1: (0.0, 2.45), 2: (-2.2, 0.35), 3: (0.0, -2.45)}[n]
    ax_p.text(*off, f"key, pos {n} ('{chars[n]}')", color=colors[n], ha="center", va="center",
              fontsize=9)
q = rot(3 * theta) @ a
ax_p.annotate("", xy=0.75 * q, xytext=(0, 0),
              arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5, ls="--"))
ax_p.text(-1.4, -0.4, "query, pos 3", color="black", ha="center", fontsize=9)
ax_p.set_xlim(-3.6, 3.6)
ax_p.set_ylim(-3.0, 3.0)
ax_p.set_aspect("equal")
ax_p.axhline(0, color="gray", lw=0.5)
ax_p.axvline(0, color="gray", lw=0.5)
ax_p.set_title("After rotation: query at pos 3\npoints along the key at pos 2")
ax_p.set_xticks([])
ax_p.set_yticks([])

for ax, w, title in ((ax_n, w_plain, "Attention weights, no position\n(every allowed key ties)"),
                     (ax_r, w_rope, "Attention weights with RoPE\n(each query picks the key one back)")):
    ax.imshow(w, cmap="viridis", vmin=0, vmax=1)
    for m in range(S):
        for n in range(S):
            txt = "-" if n > m else f"{w[m, n]:.2f}"
            ax.text(n, m, txt, ha="center", va="center",
                    color="white" if w[m, n] < 0.6 else "black", fontsize=10)
    ax.set_xticks(range(S), [f"{n} '{c}'" for n, c in enumerate(chars)])
    ax.set_yticks(range(S), [f"{m} '{c}'" for m, c in enumerate(chars)])
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_title(title)

fig.tight_layout()
fig.savefig("prev_char_head.png", dpi=120)
plt.close(fig)

# ----- softmax saturation versus head width -----
rng = np.random.default_rng(0)
widths = [2, 4, 8, 16, 32, 64, 128, 256]
n_keys, trials = 32, 2000
raw, scaled = [], []
for dh in widths:
    qv = rng.standard_normal((trials, 1, dh))
    kv = rng.standard_normal((trials, n_keys, dh))
    s = (qv * kv).sum(-1)
    raw.append(softmax(s).max(-1).mean())
    scaled.append(softmax(s / math.sqrt(dh)).max(-1).mean())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(widths, raw, "o-", label=r"raw scores $q \cdot k$")
ax.plot(widths, scaled, "s-", label=r"scaled scores $q \cdot k / \sqrt{d_h}$")
ax.axhline(1 / n_keys, color="gray", ls=":", label="uniform (1/32)")
ax.set_xscale("log", base=2)
ax.set_xlabel(r"head width $d_h$")
ax.set_ylabel("average largest attention weight")
ax.set_ylim(0, 1.02)
ax.set_title("Random queries and keys, 32 keys")
ax.legend()
fig.tight_layout()
fig.savefig("softmax_scaling.png", dpi=120)
plt.close(fig)

print("raw", np.round(raw, 3))
print("scaled", np.round(scaled, 3))
print("rope weights row 3", np.round(w_rope[3], 4))
