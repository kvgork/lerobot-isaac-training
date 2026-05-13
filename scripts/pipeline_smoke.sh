#!/usr/bin/env bash
# pipeline_smoke.sh
# ==================
# Seven-stage end-to-end smoke test for the lerobot-isaac-training pipeline.
#
# Reproduces the exact stages exercised in the prior manual smoke (see
# outputs/pipeline-test-2026-05-13/*.log) so they can be re-run on demand
# from CI or from a fresh checkout. Each stage runs against `--dry_run`
# (or the smallest equivalent invocation) so total wall-clock fits inside
# a 15-minute budget.
#
# Usage:
#   bash scripts/pipeline_smoke.sh
#   OUTPUT_DIR=outputs/my-smoke bash scripts/pipeline_smoke.sh
#
# Environment:
#   OUTPUT_DIR  Optional. Defaults to outputs/pipeline-test-<timestamp>/
#
# Exit codes:
#   0  all stages PASS or DRYRUN (expected outcome on a healthy tree)
#   1  one or more unexpected FAILs
#
# Stage map:
#   1  data-prep         pusht dataset present + version v3.0 + episode count
#   2  bridge            lerobot_world_model_bridge state-only HDF5 round-trip
#   3  train             lerobot-train diffusion dry-run via adapters
#   4  wm-train          dreamerv3 dry-run via lerobot_isaac_adapters.train
#   5  metric-contract   5-arch metric-emission contract via train_wrapper
#   6  autoresearch      autoresearch tests + one ml-executor iteration
#   7  dashboard         lerobot_isaac_dashboard.report renders manifest

set -uo pipefail

# --- config ------------------------------------------------------------------
WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE_ROOT"

TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pipeline-test-${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

PIXI_ENV_DEFAULT_PY="$WORKSPACE_ROOT/.pixi/envs/default/bin/python"
PIXI_ENV_DASHBOARD_PY="$WORKSPACE_ROOT/.pixi/envs/dashboard/bin/python"

DATASET_ROOT="$WORKSPACE_ROOT/datasets/pusht"
BRIDGE_OPS="/home/koen/tools/claude_code/skills/lerobot_world_model_bridge"

# Known-failing stages (until upstream fixes land). Stage names listed here
# are reported as FAIL but the script continues and still exits 0 unless an
# UNEXPECTED FAIL occurs.
KNOWN_FAILURES=()

# --- timing + reporting helpers ----------------------------------------------
_t0=$(date +%s)
TOTAL_STAGES=7
declare -a STAGE_RESULTS

_now()    { date +%s; }
_elapsed() { echo $(( $(_now) - _t0 )); }
_stage_dur() { echo $(( $(_now) - $1 )); }

_report() {
    # _report <stage_no> <name> <result> <duration_s>
    local n="$1" name="$2" result="$3" dur="$4"
    printf '[stage %d/%d] %-18s : %-7s (%ss)\n' "$n" "$TOTAL_STAGES" "$name" "$result" "$dur"
    STAGE_RESULTS+=("$n|$name|$result|$dur")
}

_known_failure() {
    local needle="$1"
    for kf in "${KNOWN_FAILURES[@]+"${KNOWN_FAILURES[@]}"}"; do
        [ "$kf" = "$needle" ] && return 0
    done
    return 1
}

# --- pre-flight --------------------------------------------------------------
if [ ! -x "$PIXI_ENV_DEFAULT_PY" ]; then
    echo "ERROR: pixi default env python not found at $PIXI_ENV_DEFAULT_PY" >&2
    echo "       Run: pixi install" >&2
    exit 2
fi
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: pusht dataset not present at $DATASET_ROOT" >&2
    echo "       Acquire datasets/pusht/ before running smoke." >&2
    exit 2
fi

echo "pipeline_smoke.sh starting"
echo "  workspace : $WORKSPACE_ROOT"
echo "  output    : $OUTPUT_DIR"
echo "  budget    : 15min wall-clock"
echo

# --- stage 1: data-prep ------------------------------------------------------
s=$(_now); log="$OUTPUT_DIR/stage1-data-prep.log"
{
    du -sh "$DATASET_ROOT" 2>&1 | head -1
    "$PIXI_ENV_DEFAULT_PY" -c "
import json
from pathlib import Path
root = Path('$DATASET_ROOT')
info = json.loads((root / 'meta' / 'info.json').read_text())
print(f\"version={info.get('codebase_version', '?')}\")
print(f\"episodes={info.get('total_episodes', '?')}\")
print(f\"frames={info.get('total_frames', '?')}\")
"
} > "$log" 2>&1
rc=$?
dur=$(_stage_dur "$s")
if [ "$rc" -eq 0 ] && grep -q "^version=" "$log"; then
    _report 1 "data-prep" PASS "$dur"
else
    _report 1 "data-prep" FAIL "$dur"
fi

# --- stage 2: bridge ---------------------------------------------------------
s=$(_now); log="$OUTPUT_DIR/stage2-bridge.log"
"$PIXI_ENV_DEFAULT_PY" -c "
import sys
sys.path.insert(0, '$BRIDGE_OPS')
from operations import lerobot_to_worldmodel
res = lerobot_to_worldmodel(
    dataset_path='$DATASET_ROOT',
    output_path='$OUTPUT_DIR/pusht_state.hdf5',
    output_format='hdf5',
    image_keys=[],
    max_episodes=2,
    normalize_actions=True,
)
print(f'SUCCESS: {res.success}')
print(f'ERROR:   {res.error}')
print(f'DATA:    {res.data}')
sys.exit(0 if res.success else 1)
" > "$log" 2>&1
rc=$?
dur=$(_stage_dur "$s")
if [ "$rc" -eq 0 ]; then
    _report 2 "bridge" PASS "$dur"
elif _known_failure "bridge"; then
    _report 2 "bridge" FAIL "$dur"
    echo "      (known failure — see KNOWN_FAILURES in this script)"
else
    _report 2 "bridge" FAIL "$dur"
fi

# --- stage 3: train (dry-run) ------------------------------------------------
s=$(_now); log="$OUTPUT_DIR/stage3-train.log"
"$PIXI_ENV_DEFAULT_PY" -m lerobot_isaac_adapters.train \
    --target_arch diffusion \
    --dataset lerobot/pusht \
    --output_dir "$OUTPUT_DIR/stage3-diffusion" \
    --steps 20 \
    --batch_size 2 \
    --lr 1e-4 \
    --seed 42 \
    --dry_run > "$log" 2>&1
rc=$?
dur=$(_stage_dur "$s")
if [ "$rc" -eq 0 ] && grep -q "lerobot-train" "$log"; then
    _report 3 "train" DRYRUN "$dur"
else
    _report 3 "train" FAIL "$dur"
fi

# --- stage 4: wm-train (dreamerv3 dry-run) -----------------------------------
s=$(_now); log="$OUTPUT_DIR/stage4-wm-train.log"
"$PIXI_ENV_DEFAULT_PY" -m lerobot_isaac_adapters.train \
    --target_arch dreamerv3 \
    --dataset outputs/wm_data \
    --output_dir "$OUTPUT_DIR/stage4-dreamerv3" \
    --steps 20 \
    --batch_size 4 \
    --lr 1e-4 \
    --seed 42 \
    --dry_run > "$log" 2>&1
rc=$?
dur=$(_stage_dur "$s")
if [ "$rc" -eq 0 ] && grep -q "sheeprl" "$log"; then
    _report 4 "wm-train" DRYRUN "$dur"
else
    _report 4 "wm-train" FAIL "$dur"
fi

# --- stage 5: metric-contract (5 archs, dry-run) -----------------------------
s=$(_now); log="$OUTPUT_DIR/stage5-metric-contract.log"
: > "$log"
rc=0
for arch in smolvla act diffusion dreamerv3 le_world_model; do
    last=$(
        "$PIXI_ENV_DEFAULT_PY" -m lerobot_isaac_autoresearch.train_wrapper \
            --target_arch "$arch" \
            --dataset "$OUTPUT_DIR/fake_$arch" \
            --output_dir "$OUTPUT_DIR/stage5-${arch}" \
            --dry_run 2>>"$log" | tail -n 1
    )
    echo "[$arch] last_line=$last" >> "$log"
    echo "$last" | grep -qE '^[A-Za-z_]+=[0-9.eE+\-]+$' || rc=1
done
dur=$(_stage_dur "$s")
if [ "$rc" -eq 0 ]; then
    _report 5 "metric-contract" PASS "$dur"
else
    _report 5 "metric-contract" FAIL "$dur"
fi

# --- stage 6: autoresearch (tests + one executor iteration) ------------------
s=$(_now); log="$OUTPUT_DIR/stage6-autoresearch.log"
{
    "$PIXI_ENV_DEFAULT_PY" -m pytest \
        "$WORKSPACE_ROOT/archive/packages/lerobot-isaac-autoresearch/tests/" \
        -q --no-header 2>&1 || true
    "$PIXI_ENV_DEFAULT_PY" -m lerobot_isaac_autoresearch.train_wrapper \
        --target_arch smolvla \
        --dataset "$OUTPUT_DIR/fake_iter" \
        --output_dir "$OUTPUT_DIR/stage6-iter" \
        --dry_run 2>&1 | tail -3
} > "$log" 2>&1
rc=$?
dur=$(_stage_dur "$s")
if [ "$rc" -eq 0 ] && grep -q " passed" "$log"; then
    _report 6 "autoresearch" PASS "$dur"
else
    _report 6 "autoresearch" FAIL "$dur"
fi

# --- stage 7: dashboard (report render) --------------------------------------
s=$(_now); log="$OUTPUT_DIR/stage7-dashboard.log"
DASH_OUT="$OUTPUT_DIR/stage7-dashboard"
mkdir -p "$DASH_OUT"
if [ -x "$PIXI_ENV_DASHBOARD_PY" ]; then
    DASHBOARD_PY="$PIXI_ENV_DASHBOARD_PY"
else
    DASHBOARD_PY="$PIXI_ENV_DEFAULT_PY"
fi
"$DASHBOARD_PY" -m lerobot_isaac_dashboard.report \
    --workspace "$WORKSPACE_ROOT" \
    --output-dir "$DASH_OUT" > "$log" 2>&1
rc=$?
dur=$(_stage_dur "$s")
if [ "$rc" -eq 0 ] && [ -f "$DASH_OUT/report.html" ]; then
    _report 7 "dashboard" PASS "$dur"
else
    _report 7 "dashboard" FAIL "$dur"
fi

# --- summary -----------------------------------------------------------------
total=$(_elapsed)
echo
echo "pipeline_smoke.sh complete in ${total}s"
echo "logs: $OUTPUT_DIR/stage<N>-<name>.log"
echo

# Determine exit code: only UNEXPECTED FAILs are fatal.
exit_rc=0
for row in "${STAGE_RESULTS[@]}"; do
    IFS='|' read -r n name result _ <<< "$row"
    if [ "$result" = "FAIL" ]; then
        if _known_failure "$name"; then
            echo "  (known) stage $n $name : FAIL — tolerated"
        else
            echo "  UNEXPECTED FAIL: stage $n $name"
            exit_rc=1
        fi
    fi
done

exit "$exit_rc"
