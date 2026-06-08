"""Anti-cheat: the splat rasterizer is built from scratch, no ready-made splatter.

Scans the A10 top-level holed files and the solution for prebuilt Gaussian-splatting /
rasterization libraries. Comments and string literals are stripped via tokenize, so a
forbidden name in a docstring does not trip it. This passes with the holes in place too
(it is a static scan, not an import).
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "gaussian.py",
    _ASSIGN / "project.py",
    _ASSIGN / "render.py",
    _ASSIGN / "densify.py",
    _ASSIGN / "solution" / "gaussian.py",
    _ASSIGN / "solution" / "project.py",
    _ASSIGN / "solution" / "render.py",
]

_FORBIDDEN = [
    r"import\s+gsplat",
    r"from\s+gsplat",
    r"import\s+diff_gaussian_rasterization",
    r"from\s+diff_gaussian_rasterization",
    r"import\s+nerfstudio",
    r"from\s+nerfstudio",
    r"import\s+simple_knn",
    r"from\s+simple_knn",
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
