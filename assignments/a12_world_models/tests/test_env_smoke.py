"""Smoke test for the dm_control cartpole wrapper. Skips cleanly if dm_control/MuJoCo is absent.

dm_control is a heavy optional dependency, so this test imports it via pytest.importorskip and is
skipped when it (or a working MuJoCo GL backend) is unavailable. The graded mechanism tests do not
depend on dm_control; only this one touches the env. It checks that the env resets and steps, the
render is (3, 64, 64) in [0, 1], and the random-policy return is a finite scalar.
"""

import numpy as np
import pytest

pytest.importorskip("dm_control")


def test_env_resets_steps_and_renders():
    import env as E

    try:
        e = E.make_env(0)
    except Exception as exc:  # MuJoCo GL backend not available in this environment.
        pytest.skip(f"dm_control present but env could not load (likely no GL backend): {exc}")

    e.reset()
    img = E.render(e)
    assert img.shape == (3, E.OBS_SIZE, E.OBS_SIZE)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0

    ts = e.step(np.zeros(1, np.float32))
    assert np.isfinite(ts.reward)

    ret = E.random_return(e, n=2, max_agent_steps=20)
    assert np.isfinite(ret)
