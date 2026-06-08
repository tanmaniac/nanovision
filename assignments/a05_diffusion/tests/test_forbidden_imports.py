"""Anti-cheat: the diffusion mechanism is built from scratch, no prebuilt schedulers/UNets.

Scans the A5 top-level files and the solution for diffusion libraries (diffusers,
k_diffusion, torchsde) and prebuilt scheduler/UNet symbols. Comments and string literals
are stripped via tokenize, so naming a forbidden symbol in a docstring does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "schedule.py",
    _ASSIGN / "diffusion.py",
    _ASSIGN / "sampling.py",
    _ASSIGN / "unet.py",
    _ASSIGN / "solution" / "schedule.py",
    _ASSIGN / "solution" / "diffusion.py",
    _ASSIGN / "solution" / "sampling.py",
    _ASSIGN / "solution" / "unet.py",
]

_FORBIDDEN = [
    r"import\s+diffusers",
    r"from\s+diffusers",
    r"import\s+k_diffusion",
    r"from\s+k_diffusion",
    r"import\s+torchsde",
    r"DDPMScheduler",
    r"DDIMScheduler",
    r"UNet2DModel",
    r"UNet2DConditionModel",
]

_SKIP = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
         tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}


def _code_only(path: Path) -> str:
    parts = []
    with open(path) as f:
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type in _SKIP:
                continue
            parts.append(tok.string)
    return " ".join(parts)


def test_no_forbidden_imports():
    for f in _FILES:
        code = _code_only(f)
        for pat in _FORBIDDEN:
            assert re.search(pat, code) is None, f"forbidden pattern {pat!r} found in {f}"
