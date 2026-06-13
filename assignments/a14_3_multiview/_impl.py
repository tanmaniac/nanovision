"""Build-on-import shim: compile the C++ for the active NANOVISION_IMPL and re-export it.
Same pattern as the other a14 assignments; see a14_0/_impl.py for the full comment."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).parent


def _build_and_load():
    impl = "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else "student"
    modname = "_a14_3_solution" if impl == "solution" else "_a14_3_student"
    build_dir = _here / "build" / impl
    build_dir.mkdir(parents=True, exist_ok=True)

    import pybind11

    if not (build_dir / "CMakeCache.txt").exists():
        subprocess.run(
            [
                "cmake",
                "-S", str(_here),
                "-B", str(build_dir),
                f"-DNV_IMPL={impl}",
                f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
            stdout=sys.stderr,
        )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j"],
        check=True,
        stdout=sys.stderr,
    )

    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    return importlib.import_module(modname)


_module = _build_and_load()

globals().update({k: getattr(_module, k) for k in dir(_module) if not k.startswith("_")})
