"""Put the assignment-local code on sys.path so the tests can import it by name.

A12 is a leaf assignment with no nanovision shim: every module here is imported by bare name.
The provided files (env.py, config.py, viz.py, _train.py) live only at the top level. The holed
files (nets.py, rssm.py, world_model.py, actor_critic.py) have a top-level copy the student edits
and a solution/ answer key. The top-level dir is always on the path (config.py/env.py live only
there); solution/ is added first under NANOVISION_IMPL=solution so its copies shadow the holed
top-level ones.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_impl = _here / "solution" if os.environ.get("NANOVISION_IMPL") == "solution" else _here
# _impl first (higher priority) so solution/<mod>.py shadows the holed top-level one;
# _here second so config.py / env.py still resolve from the top level.
for _p in (_here, _impl):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
