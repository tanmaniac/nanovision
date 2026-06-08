"""Put the assignment-local code on sys.path so the tests can import it by name.

The holed files are encoding.py, render.py, rays.py. By default they resolve to the
top-level files the student edits; with NANOVISION_IMPL=solution they resolve to
solution/. model.py, config.py, and viz.py live only at the top level (provided), so the
top-level dir is always on the path too.

render.py and rays.py are shared OWNED files: tests and sibling files import their symbols
through nanovision.volume, never by bare name. encoding.py and model.py are assignment-local
and are imported bare.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
for _p in (_here, _impl):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
