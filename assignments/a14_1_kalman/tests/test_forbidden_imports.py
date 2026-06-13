"""Static scan: the filters must be implemented from scratch, not by calling an
existing estimation/solver library. Scans the C++ sources (top-level + solution) with
comments and string literals stripped. Passes in both impl modes (never imports the
build)."""

import re
from pathlib import Path

_here = Path(__file__).parent.parent

FORBIDDEN = ("sophus", "manif", "ceres", "gtsam", "g2o")

CPP_SOURCES = [
    _here / "kalman.cpp",
    _here / "solution" / "kalman.cpp",
]


def _strip_comments_and_strings(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)  # block comments
    src = re.sub(r"//[^\n]*", " ", src)  # line comments
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)  # string literals
    return src


def test_no_forbidden_libraries():
    for path in CPP_SOURCES:
        assert path.exists(), f"missing source {path}"
        code = _strip_comments_and_strings(path.read_text()).lower()
        includes = re.findall(r"#\s*include\s*[<\"]([^>\"]+)[>\"]", code)
        for inc in includes:
            for bad in FORBIDDEN:
                assert bad not in inc, f"{path.name} includes forbidden library: {inc}"
