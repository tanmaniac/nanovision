"""Static scan: ICP must be implemented from scratch, not by calling an existing registration
or solver library. Open3D is allowed as a Python oracle in the tests, but the C++ mechanism
may not include it (or PCL, nanoflann, etc.). Scans the sources with comments and string
literals stripped."""

import re
from pathlib import Path

_here = Path(__file__).parent.parent

FORBIDDEN = ("open3d", "pcl/", "nanoflann", "sophus", "ceres", "gtsam", "g2o")

CPP_SOURCES = [
    _here / "icp.cpp",
    _here / "models.cpp",
    _here / "solution" / "icp.cpp",
]


def _strip_comments_and_strings(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    return src


def test_no_forbidden_libraries():
    for path in CPP_SOURCES:
        assert path.exists(), f"missing source {path}"
        code = _strip_comments_and_strings(path.read_text()).lower()
        includes = re.findall(r"#\s*include\s*[<\"]([^>\"]+)[>\"]", code)
        for inc in includes:
            for bad in FORBIDDEN:
                assert bad not in inc, f"{path.name} includes forbidden library: {inc}"
