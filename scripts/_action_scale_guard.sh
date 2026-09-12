#!/usr/bin/env bash
# Shared fail-closed guard for LEROBOT_ISAAC_ACTION_SCALE_JSON.
#
# WHY THIS EXISTS
# ---------------
# `lerobot_isaac_env.so101_env_cfg.load_action_scale_dict()` is STRICTLY OPT-IN: with
# LEROBOT_ISAAC_ACTION_SCALE_JSON unset it silently falls back to the historical uniform
# scale of 0.5. At 0.5 any joint that must move more than 0.5 rad from default needs
# |action| > 1, which the residual blend clamps to 1 — so the arm physically cannot
# descend to grasp height. That silent fallback burned four 13-hour runs.
#
# Three entrypoints guarded themselves inline; `launch_warmstart_full.sh`,
# `launch_warmstart_cur.sh` and `_run_autoresearch_wm_isaac.sh` did not, so the same
# trap stayed live on every other sim entrypoint (master plan, Track A item A0.5).
#
# USAGE
#   source "$(dirname "${BASH_SOURCE[0]}")/_action_scale_guard.sh"
#
# Resolves the path (respecting an existing value), FAILS LOUDLY if the file is absent,
# and exports it. Uses an explicit `exit 1` rather than relying on `set -e`, because
# `_run_autoresearch_wm_isaac.sh` runs under `set -uo pipefail` with no `-e`.

_ASG_ROOT="${_ASG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LEROBOT_ISAAC_ACTION_SCALE_JSON="${LEROBOT_ISAAC_ACTION_SCALE_JSON:-$_ASG_ROOT/outputs/action_scale.json}"

if [ ! -f "$LEROBOT_ISAAC_ACTION_SCALE_JSON" ]; then
  echo "[action-scale-guard] FATAL: action-scale JSON missing:" >&2
  echo "    $LEROBOT_ISAAC_ACTION_SCALE_JSON" >&2
  echo "  Without it the sim env silently uses scale=0.5 for every arm joint, which" >&2
  echo "  makes the descent to grasp height unreachable. Regenerate with:" >&2
  echo "    python scripts/_gen_sim_demos.py --measure_scale" >&2
  exit 1
fi

export LEROBOT_ISAAC_ACTION_SCALE_JSON
echo "[action-scale-guard] using $LEROBOT_ISAAC_ACTION_SCALE_JSON"
