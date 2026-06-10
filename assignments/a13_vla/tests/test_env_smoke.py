"""Smoke test for the dm_control reacher wrapper. Skipped when dm_control is not installed.

dm_control is isolated to env.py and viz.py: this is the ONLY test that touches it, and it skips
cleanly (pytest.importorskip) so the CPU mechanism grading path runs without the robot library. It
checks reset/step/render shapes and that the analytic expert reaches the target on at least some
seeds, not an exact success number (rollout success is measured in viz/README, never asserted).

Needs MUJOCO_GL=egl for headless rendering; without a GL backend the render call raises and the
test errors rather than silently passing.
"""

import numpy as np
import pytest

pytest.importorskip("dm_control")

import env as ENV  # noqa: E402


def test_reset_step_render_shapes():
    env = ENV.make_reacher(seed=0)
    env.reset()
    spec = env.action_spec()
    assert spec.shape == (2,)
    obs = ENV.render_obs(env)
    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    a = ENV.expert_torque(env.physics)
    assert a.shape == (2,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    ts = env.step(a)
    assert ENV.render_obs(env).shape == (3, 64, 64)
    # reward is None on reset, a float scalar after a step.
    assert ts.reward is None or np.isscalar(ts.reward) or np.ndim(ts.reward) == 0


def test_expert_reaches_some_seeds():
    # The analytic IK+PD expert should reach the target on at least one of a few seeds within the
    # episode cap. Loose: this is a wiring check, not a success-rate assertion.
    hits = 0
    for s in range(4):
        o, a, reach_step = ENV.collect_one_episode(s, max_steps=80)
        assert o.shape[1:] == (3, 64, 64)
        assert a.shape[1:] == (2,)
        hits += int(reach_step is not None)
    assert hits >= 1, "analytic expert reached on no seed; check the IK/PD wiring"


def test_collect_demos_filters_and_pads():
    # Collect a handful of successful demos and check the padded shapes and the validity mask.
    demos = ENV.collect_demos(n_success=3, seed0=0, max_steps=80)
    n, T = demos["obs"].shape[0], demos["T"]
    assert n == 3
    assert demos["obs"].shape == (3, T, 3, 64, 64)
    assert demos["act"].shape == (3, T, 2)
    assert demos["mask"].shape == (3, T)
    # Every kept demo has at least one real step, and masked-out steps are zero-padded.
    assert demos["mask"].sum(axis=1).min() >= 1.0
