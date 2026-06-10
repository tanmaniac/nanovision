"""Anti-cheat: no RL / world-model library, and no bare cross-assignment import.

Scans the four holed top-level modules and their solution copies (A12 is a leaf, so there is no
nanovision shim and nothing is imported via nanovision.*). Comments and string literals are stripped
via tokenize, so a forbidden name in a docstring does not trip the scan. Passes in both modes (it is
a static file scan that does not import the holed code).
"""

import re
import tokenize
from pathlib import Path

_ASSIGN = Path(__file__).resolve().parents[1]

_MODULES = ["nets.py", "rssm.py", "world_model.py", "actor_critic.py"]
_FILES = [_ASSIGN / m for m in _MODULES] + [_ASSIGN / "solution" / m for m in _MODULES]

# Whole RL / world-model frameworks that would do the lesson for the student.
_FORBIDDEN_LIBS = [
    r"\bgym\b",
    r"\bgymnasium\b",
    r"\bdreamerv3\b",
    r"\bstable_baselines3\b",
    r"\btianshou\b",
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


def test_no_forbidden_libraries():
    for f in _FILES:
        assert f.exists(), f"expected file not found: {f}"
        code = _code_only(f)
        for pat in _FORBIDDEN_LIBS:
            assert re.search(pat, code) is None, f"forbidden pattern {pat!r} in {f}"


def test_no_bare_cross_assignment_import():
    # A12 is a leaf: it imports only its own local modules (nets, rssm, world_model, actor_critic,
    # config, env), torch, numpy, and (in viz/_train only) nanovision.determinism. No top-level
    # holed module may import another assignment's code by bare name (a01_..., a05_..., etc.).
    for f in _FILES:
        code = _code_only(f)
        # No "from aXX..." or "import aXX..." cross-assignment imports (assignment ids start with a
        # digit after the leading 'a', e.g. a01_transformer).
        assert not re.search(r"\b(from|import)\s+a\d\w*", code), f"cross-assignment import in {f}"
