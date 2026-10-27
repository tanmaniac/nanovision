"""Copy the training-run figures from ../out/ into this directory and shrink the GIFs for the README.

The PNGs and GIFs in ../out/ are written by viz.py from one training run (see the README). The
GIFs there are 4-6 MB; this script keeps every third frame (each shown three times as long, so the
playback speed is unchanged) and requantizes to 48 colors, which brings each under 2 MB.

Run from this directory after `make viz A=a12_world_models`:

    python make_gifs.py
"""

import shutil
from pathlib import Path

from PIL import Image

OUT = Path("../out")
STEP = 3      # keep every third frame
COLORS = 48
for name in ["learning_curve.png", "imagination_vs_real.png", "imagination_divergence.png"]:
    shutil.copy(OUT / name, name)
    print("copied", name)

for name in ["policy_vs_imagination.gif", "random_vs_imagination.gif"]:
    src = Image.open(OUT / name)
    duration = src.info.get("duration", 80)
    frames = []
    for i in range(0, src.n_frames, STEP):
        src.seek(i)
        frames.append(src.convert("RGB").quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT,
                                                   dither=Image.Dither.NONE))
    frames[0].save(name, save_all=True, append_images=frames[1:], duration=STEP * duration, loop=0,
                   optimize=True)
    print(f"wrote {name}: {len(frames)} frames, {Path(name).stat().st_size / 1e6:.2f} MB")
