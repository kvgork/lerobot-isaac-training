"""Cross-package parity guard for the SO-101 absolute joint-limit floor.

The conservative joint floor (notably ``elbow_flex >= -10 deg``, which stops the arm
driving itself into the table) is declared in up to three places, in three separate
git repos:

1. ``lerobot_isaac_deploy.arm_motor_writer`` — the original, as numpy min/max arrays.
2. ``robot_data_runner.safety_limits`` — "ported verbatim" per its own docstring.
3. ``lerobot_isaac_env.joint_limits`` — added 2026-09-10 so the SIM side has the same
   numbers; before that the simulation had no equivalent at all, which is a silent
   sim2real action-space mismatch (a policy learns a command in sim that hardware
   quietly clips).

Safety-critical constants copied across repos drift. This test is the mechanical
guard: if any copy changes without the others, it fails here rather than becoming a
deployment surprise. It is the same lesson as the duplicated predicate registries
merged on the same date — a value whose whole job is to be authoritative must have
exactly one definition, or a test proving the copies agree.

Meta is the right home for this: it is the one live workspace member that legitimately
sees every sibling. Packages that are not installed are skipped, never failed, so this
still passes in a partial environment.
"""

from __future__ import annotations

import pytest

from lerobot_isaac_env.joint_limits import (
    DEFAULT_JOINT_LIMITS_DEG as SIM_LIMITS,
)
from lerobot_isaac_env.joint_limits import (
    SO101_JOINT_NAMES as SIM_NAMES,
)


def test_sim_declares_the_table_avoidance_floor():
    """The whole point of the sim-side copy: elbow_flex must carry the -10 deg floor."""
    assert SIM_LIMITS["elbow_flex"] == (-10.0, 90.0)


def test_sim_covers_every_canonical_joint():
    assert set(SIM_LIMITS) == set(SIM_NAMES)
    assert len(SIM_NAMES) == 6


def test_gripper_is_percent_not_degrees():
    """Gripper is RANGE_0_100 regardless of the follower's use_degrees setting."""
    assert SIM_LIMITS["gripper"] == (0.0, 100.0)


def test_every_bound_is_ordered():
    for name, (lo, hi) in SIM_LIMITS.items():
        assert lo < hi, f"{name} has lo >= hi: {lo} >= {hi}"


def test_parity_with_deploy_arm_motor_writer():
    """Sim copy must equal the original numpy arrays in arm_motor_writer."""
    amw = pytest.importorskip(
        "lerobot_isaac_deploy.arm_motor_writer",
        reason="lerobot-isaac-deploy not installed in this environment",
    )
    names = list(amw.SO101_JOINT_NAMES)
    assert names == list(SIM_NAMES), "canonical joint ORDER diverged"

    lo = amw._DEFAULT_JOINT_LIMITS_MIN
    hi = amw._DEFAULT_JOINT_LIMITS_MAX
    for i, name in enumerate(names):
        sim_lo, sim_hi = SIM_LIMITS[name]
        assert float(lo[i]) == pytest.approx(sim_lo), (
            f"{name} lower bound diverged: deploy={float(lo[i])} sim={sim_lo}"
        )
        assert float(hi[i]) == pytest.approx(sim_hi), (
            f"{name} upper bound diverged: deploy={float(hi[i])} sim={sim_hi}"
        )


def test_parity_with_robot_data_runner_safety_limits():
    """Sim copy must equal the runner's client-side clamp table."""
    sl = pytest.importorskip(
        "robot_data_runner.safety_limits",
        reason="robot-data-runner not installed in this environment",
    )
    assert list(sl.SO101_JOINT_NAMES) == list(SIM_NAMES), (
        "canonical joint ORDER diverged"
    )
    assert sl.DEFAULT_JOINT_LIMITS_DEG == SIM_LIMITS, (
        "runner and sim joint-limit tables diverged"
    )


def test_clip_is_opt_in_and_off_by_default(monkeypatch):
    """Importing the module must never change environment behaviour on its own."""
    from lerobot_isaac_env import joint_limits

    monkeypatch.delenv(joint_limits.ENABLE_ENV_VAR, raising=False)
    assert joint_limits.hw_limits_enforced() is False
    monkeypatch.setenv(joint_limits.ENABLE_ENV_VAR, "1")
    assert joint_limits.hw_limits_enforced() is True
    monkeypatch.setenv(joint_limits.ENABLE_ENV_VAR, "off")
    assert joint_limits.hw_limits_enforced() is False


def test_clip_enforces_the_elbow_floor():
    """A sim action outside the hardware floor is what would be silently clipped."""
    from lerobot_isaac_env.joint_limits import clip_to_hw_limits

    out = clip_to_hw_limits([0.0, 0.0, -40.0, 0.0, 0.0, 50.0])
    assert out[2] == pytest.approx(-10.0)
    # in-range values must pass through untouched
    assert out[0] == pytest.approx(0.0)
    assert out[5] == pytest.approx(50.0)


def test_clip_handles_batches_and_does_not_mutate_input():
    from lerobot_isaac_env.joint_limits import clip_to_hw_limits

    src = [[0.0, 0.0, -40.0, 0.0, 0.0, 50.0], [0.0, 0.0, 45.0, 0.0, 0.0, 150.0]]
    out = clip_to_hw_limits(src)
    assert out.shape == (2, 6)
    assert out[0][2] == pytest.approx(-10.0)
    assert out[1][5] == pytest.approx(100.0)
    assert src[0][2] == -40.0, "input must not be mutated"


def test_clip_rejects_wrong_width():
    from lerobot_isaac_env.joint_limits import clip_to_hw_limits

    with pytest.raises(ValueError):
        clip_to_hw_limits([0.0, 0.0, 0.0])
