"""Put the assignment dir on sys.path so tests can `from _impl import ...` and
`from sim import ...` by bare name. _impl.py does the CMake build keyed on NANOVISION_IMPL."""

import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
