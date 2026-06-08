"""Anti-cheat: the VLM bridge is built from scratch, on top of the in-course modules.

Scans the A8 top-level holed files and their solution copies for two things:
  1. prebuilt VLM/processor libraries (transformers, a timm VLM head) that would hand the
     student the connector and the LM wiring;
  2. bare imports of the owned shared modules (vit, transformer, attention) - those must come
     through the nanovision.* shims, never `import vit` / `from transformer import ...`.
Comments and string literals are stripped via tokenize, so a forbidden name in a docstring
does not trip it.
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_FILES = [
    _ASSIGN / "projector.py",
    _ASSIGN / "resampler.py",
    _ASSIGN / "vlm.py",
    _ASSIGN / "anyres.py",
    _ASSIGN / "solution" / "projector.py",
    _ASSIGN / "solution" / "resampler.py",
    _ASSIGN / "solution" / "vlm.py",
]

_FORBIDDEN = [
    r"import\s+transformers",
    r"from\s+transformers",
    r"import\s+timm",
    r"from\s+timm",
    # Owned shared modules must be imported via nanovision.*, not bare.
    r"(?<!\.)\bimport\s+vit\b",
    r"from\s+vit\s+import",
    r"(?<!\.)\bimport\s+transformer\b",
    r"from\s+transformer\s+import",
    r"(?<!\.)\bimport\s+attention\b",
    r"from\s+attention\s+import",
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
