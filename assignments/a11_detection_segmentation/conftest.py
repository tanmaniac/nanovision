"""Put the assignment-local code on sys.path so the tests can import it by name.

Local files (boxes.py, matcher.py, loss.py, model.py, config.py) are imported with a
bare name. By default they resolve to the top-level files the student edits; with
NANOVISION_IMPL=solution the holed files (boxes.py, matcher.py, loss.py) resolve to
solution/. model.py, config.py, and viz.py live only at the top level (provided), so the
top-level dir is always on the path too.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
for _p in (_here, _impl):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
