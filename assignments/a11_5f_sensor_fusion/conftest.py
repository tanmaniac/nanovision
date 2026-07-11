"""Put the assignment-local code on sys.path so the tests can import it by name.

The holed mechanism files (fusion.py, transfuser.py) and the provided helpers (config.py,
compare.py, viz.py) are imported by bare name. In solution mode the solution/ dir is inserted
ahead of the top-level dir, so solution/fusion.py and solution/transfuser.py shadow the holed
top-level copies while the provided helpers, which live only at the top level, still resolve.
The shared primitives this assignment reuses (nanovision.lift_splat, nanovision.geometry,
nanovision.transformer) are NOT imported bare - they come through their nanovision.* shims, which
nanovision/_student.py routes to each owner's top-level (default) or solution/ copy keyed on
NANOVISION_IMPL.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
# _here first (lower priority), then _impl: in solution mode solution/ shadows the holed
# top-level modules, while config.py still resolves from the top level.
for _p in (_here, _impl):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
