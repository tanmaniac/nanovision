"""Plotting helpers. A0 ships `plot_loss_curve`; later assignments add
attention-map, BEV-grid, and splat-render helpers to this module.
"""

import csv
import os
from pathlib import Path

import matplotlib

# Headless by default: write PNGs, never open a window. Set NANOVISION_VIZ_SHOW=1
# (e.g. `make viz SHOW=1`) to use an interactive backend and open windows too.
SHOW = os.environ.get("NANOVISION_VIZ_SHOW") == "1"
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def finish(out_path, *, dpi: int = 120):
    """Save the current figure to out_path (always), and in SHOW mode keep it open so a
    final `plt.show()` can display it. In headless mode close it to free memory."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    if not SHOW:
        plt.close()
    return out_path


def plot_loss_curve(losses, out_path, title: str = "loss", logy: bool = True):
    """Plot a loss curve and save it to out_path.

    `losses` may be a path to a `loss.csv` (columns step,loss), a list of floats,
    or a list of (step, loss) pairs.
    """
    steps, vals = _coerce(losses)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, vals)
    if logy and min(vals) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if not SHOW:
        plt.close(fig)
    return out_path


def _coerce(losses):
    if isinstance(losses, (str, Path)):
        steps, vals = [], []
        with open(losses, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                steps.append(int(row[0]))
                vals.append(float(row[1]))
        return steps, vals
    seq = list(losses)
    if seq and isinstance(seq[0], (tuple, list)):
        return [int(s) for s, _ in seq], [float(v) for _, v in seq]
    return list(range(len(seq))), [float(v) for v in seq]
