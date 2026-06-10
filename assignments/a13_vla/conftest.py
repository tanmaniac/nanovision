"""Put the assignment-local code on sys.path so the tests can import it by name.

Local files (env.py, config.py, flow.py, bc.py, ddpm.py) are imported with a bare name. flow.py,
bc.py, and ddpm.py are holed: by default they resolve to the top-level files the student edits;
with NANOVISION_IMPL=solution they resolve to solution/. env.py and config.py live only at the top
level (provided), so the top-level dir is always on the path too. A13 is a leaf (nothing imports
it), so there is no nanovision shim.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
# Insert _here first, then _impl, so _impl ends up at sys.path[0] (highest priority) and its
# solution/<file>.py shadows the holed top-level copy under NANOVISION_IMPL=solution. env.py and
# config.py live only at the top level, so _here keeps them importable in both modes. When
# NANOVISION_IMPL is unset, _impl == _here and this is a single insert.
for _p in (_here, _impl):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
