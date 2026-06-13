"""Put the assignment dir on sys.path so tests can `from _impl import ...` by bare name.

_impl.py does the actual CMake build keyed on NANOVISION_IMPL; this just makes it
importable. The C++ analog of the Python assignments' conftest.
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
