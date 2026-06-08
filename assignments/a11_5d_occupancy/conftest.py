"""Put the assignment-local code on sys.path so the tests can import it by name.

Local files (config.py) are imported with a bare name; they resolve to the top-level files the
student edits. occupancy.py is also assignment-LOCAL here (nothing imports it, so there is no
nanovision shim), so it too is imported bare and resolves to the top-level (default) or solution/
copy keyed on NANOVISION_IMPL. The top-level dir is always on the path (config.py lives only
there); solution/ is added first under NANOVISION_IMPL=solution so its occupancy.py shadows the
holed top-level one.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
# _impl first (higher priority) so solution/occupancy.py shadows the holed top-level one;
# _here second so config.py still resolves from the top level.
for _p in (_here, _impl):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
