"""Put the assignment-local code on sys.path so the tests can import it by name.

Local files (video_mae.py, backbone.py, config.py, ...) are imported with a bare
name. By default they resolve to the top-level files the student edits; with
NANOVISION_IMPL=solution they resolve to solution/. config.py lives only at the top
level, so the top-level dir is always on the path too, at lower priority.

The shared symbol (TubeletEmbedding) is NOT imported bare here; it comes from
nanovision.transformer, which nanovision/_student.py routes the same way.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
for _p in (_here, _impl):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
