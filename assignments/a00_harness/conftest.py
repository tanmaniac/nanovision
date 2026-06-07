"""Put starter/ or solution/ on sys.path per NANOVISION_IMPL.

Tests import the modules under test by bare name (e.g. `from primitives import
LayerNorm`); this conftest resolves that name to either starter/ (default) or
solution/ so the same test file proves both implementations.
"""

import os
import sys
from pathlib import Path

_impl = os.environ.get("NANOVISION_IMPL", "starter")
_dir = Path(__file__).parent / _impl
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))
