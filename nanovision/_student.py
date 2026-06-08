"""Loader that sources shared symbols from the assignment the student edits.

The shared library (`nanovision.primitives`, `nanovision.attention`, ...) does not
hold the implementations any more. Each shared module is a thin shim that calls
`load(assignment, module)` to pull the symbols out of the assignment directory
where the student wrote them. The student's own work becomes the library that
later assignments import.

Default (no env var): import the top-level file the student edits, e.g.
`assignments.a00_harness.primitives`.
NANOVISION_IMPL=solution: import the reference, e.g.
`assignments.a00_harness.solution.primitives`.

The repo root is on `sys.path` via pytest's `pythonpath = ["."]` (tests) or by
running scripts as modules from the repo root (`python -m assignments.aXX.viz`).
There is no editable install; nothing here is meant to import from outside the repo.
"""

import importlib
import os


def load(assignment: str, module: str):
    sub = "solution." if os.environ.get("NANOVISION_IMPL") == "solution" else ""
    return importlib.import_module(f"assignments.{assignment}.{sub}{module}")
