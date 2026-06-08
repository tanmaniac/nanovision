"""Anti-cheat: the encoding, renderer, and ray generation are built from scratch.

Scans the A9 holed files, their solutions, and the nanovision.volume shim for prebuilt
NeRF / renderer libraries, and for bare imports of the owned shared files (render.py and
rays.py must be reached through nanovision.volume, never imported by bare name). Comments and
string literals are stripped via tokenize, so a forbidden name in a docstring does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]
_SHIM = _ASSIGN.parents[1] / "nanovision" / "volume.py"

_FILES = [
    _ASSIGN / "encoding.py",
    _ASSIGN / "render.py",
    _ASSIGN / "rays.py",
    _ASSIGN / "solution" / "encoding.py",
    _ASSIGN / "solution" / "render.py",
    _ASSIGN / "solution" / "rays.py",
    _SHIM,
]

_FORBIDDEN = [
    r"import\s+nerfstudio",
    r"from\s+nerfstudio",
    r"import\s+nerfacc",
    r"from\s+nerfacc",
    r"import\s+torch_ngp",
    r"from\s+torch_ngp",
    r"import\s+tinycudann",
    r"from\s+tinycudann",
    r"from\s+pytorch3d\S*\s+import.*[Rr]ender",
    r"import\s+kaolin",
    r"from\s+kaolin",
    # The owned shared files must be reached via nanovision.volume, not bare.
    r"^\s*import\s+render\b",
    r"^\s*from\s+render\s+import",
    r"^\s*import\s+rays\b",
    r"^\s*from\s+rays\s+import",
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


def _raw_lines(path: Path) -> str:
    # For the bare-import patterns we want line starts, so scan raw source with comments
    # and strings removed line by line is overkill; instead strip full-line comments only.
    out = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            out.append(line)
    return "".join(out)


def test_no_forbidden_imports():
    for f in _FILES:
        code = _code_only(f)
        raw = _raw_lines(f)
        for pat in _FORBIDDEN:
            target = raw if pat.startswith("^") else code
            flags = re.MULTILINE if pat.startswith("^") else 0
            assert re.search(pat, target, flags) is None, f"forbidden pattern {pat!r} in {f}"
