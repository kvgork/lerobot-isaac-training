"""Canary: every sim entrypoint must fail closed on a missing action-scale JSON.

`lerobot_isaac_env.so101_env_cfg.load_action_scale_dict()` is strictly opt-in: with
LEROBOT_ISAAC_ACTION_SCALE_JSON unset it silently falls back to a uniform scale of 0.5.
At 0.5 any joint needing more than 0.5 rad of travel from default requires |action| > 1,
which the residual blend clamps to 1 — so the arm physically cannot descend to grasp
height. That silent fallback burned four 13-hour runs.

Three entrypoints guarded themselves inline and three did not, so the trap stayed live
on half of them (master plan Track A, item A0.5). Listing the scripts mechanically means
a NEW sim entrypoint that forgets the guard fails here instead of failing overnight.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"

#: Scripts that build or drive an Isaac sim env and therefore depend on the action scale.
SIM_ENTRYPOINTS = [
    "launch_residual_rl.sh",
    "_residual_smoke_gate.sh",
    "launch_warmstart_full.sh",
    "launch_warmstart_cur.sh",
    "_run_autoresearch_wm_isaac.sh",
]

GUARD = SCRIPTS / "_action_scale_guard.sh"


def test_the_shared_guard_exists():
    assert GUARD.is_file(), f"missing {GUARD}"


def test_guard_fails_closed_not_open():
    """The guard must exit on a missing file, not warn and continue."""
    text = GUARD.read_text()
    assert "exit 1" in text, "guard must exit, not merely warn"
    assert "export LEROBOT_ISAAC_ACTION_SCALE_JSON" in text


@pytest.mark.parametrize("name", SIM_ENTRYPOINTS)
def test_sim_entrypoint_is_guarded(name):
    path = SCRIPTS / name
    if not path.is_file():
        pytest.skip(f"{name} not present in this checkout")
    # Strip comments before matching. The first version of this canary passed on the
    # explanatory COMMENT above the source line, which also names the guard file — the
    # same self-satisfying-match bug that the safe_connect canary hit. A canary that
    # matches its own documentation cannot fail.
    code = "\n".join(line.split("#", 1)[0] for line in path.read_text().splitlines())
    guarded = (
        "_action_scale_guard.sh" in code or "LEROBOT_ISAAC_ACTION_SCALE_JSON" in code
    )
    assert guarded, (
        f"{name} drives a sim env but never references the action scale, so it takes "
        f"the silent 0.5 default. Source scripts/_action_scale_guard.sh near the top."
    )
