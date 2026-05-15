#!/usr/bin/env bash
# Episode sample-complexity sweep.
#
# For each N in {2,5,10,15,17}:
#   1. subsample SO-101 dataset to first N training-pool episodes
#   2. train diffusion 30 min with the AR-v3 best config (bs=8 lr=1e-4 seed=42)
#   3. open-loop eval against held-out eps 17/18/19 of the ORIGINAL dataset
#   4. write a per-trial eval JSON
#
# Output: outputs/episode-sweep-<ts>/N<K>/{policy-diffusion,logs,system_metrics}
#         outputs/episode-sweep-<ts>/N<K>-eval.json
#
# Driven by ENV vars:
#   SECONDS_PER_EXP  per-trial budget (default 1800 = 30 min)
#   GRID             space-separated Ns (default "2 5 10 15 17")
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

TS="$(date +%Y-%m-%d-%H%M%S)"
SWEEP_ROOT="$WORKSPACE/outputs/episode-sweep-$TS"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-1800}"
GRID="${GRID:-2 5 10 15 17}"
SRC_DATASET="$WORKSPACE/datasets/kvgork/so101-pickplace1"

mkdir -p "$SWEEP_ROOT/logs"
echo "[sweep] root=$SWEEP_ROOT grid='$GRID' seconds_per_exp=$SECONDS_PER_EXP"
echo "[sweep] src=$SRC_DATASET"

for N in $GRID; do
    echo "════════════════════════════════════════════════════════════"
    echo "[sweep N=$N] starting at $(date +%H:%M:%S)"
    DST_DATASET="$WORKSPACE/datasets/_sweep_so101_N${N}"
    TRIAL_DIR="$SWEEP_ROOT/N${N}"

    PYTHONNOUSERSITE=1 "$WORKSPACE/.pixi/envs/default/bin/python" \
        "$WORKSPACE/scripts/_subsample_dataset.py" \
        --src "$SRC_DATASET" \
        --dst "$DST_DATASET" \
        --n-episodes "$N" \
        --eval-holdout 3 \
        > "$SWEEP_ROOT/logs/N${N}-subsample.log" 2>&1

    bash "$WORKSPACE/scripts/run_full_pipeline.sh" \
        --train-minutes "$(( SECONDS_PER_EXP / 60 ))" \
        --run-dir "$TRIAL_DIR" \
        --dataset "$DST_DATASET" \
        --skip-synthetic \
        --skip-worldmodel \
        --skip-eval \
        --skip-dashboard \
        > "$SWEEP_ROOT/logs/N${N}-pipeline.log" 2>&1
    pipeline_rc=$?

    # Eval against the ORIGINAL dataset's held-out 17/18/19
    CKPT=$(find "$TRIAL_DIR/policy-diffusion/checkpoints" \
        -maxdepth 2 -name pretrained_model -type d 2>/dev/null | sort | tail -1)
    if [ -n "${CKPT:-}" ] && [ -d "$CKPT" ]; then
        "$WORKSPACE/.pixi/envs/train-policy/bin/python" \
            "$WORKSPACE/scripts/_open_loop_eval.py" \
            --policy_path "$CKPT" \
            --dataset_root "$SRC_DATASET" \
            --n_episodes 3 \
            --output_json "$SWEEP_ROOT/N${N}-eval.json" \
            --task_label "so101-N${N}-open-loop-mse" \
            --run_id "episode-sweep-N${N}" \
            > "$SWEEP_ROOT/logs/N${N}-eval.log" 2>&1
    else
        echo "[sweep N=$N] NO CHECKPOINT under $TRIAL_DIR — eval skipped" >&2
    fi

    rm -rf "$DST_DATASET"
    echo "[sweep N=$N] done at $(date +%H:%M:%S) (pipeline_rc=$pipeline_rc)"
done

echo "════════════════════════════════════════════════════════════"
echo "[sweep] all trials done at $(date +%H:%M:%S)"
echo "[sweep] aggregate:"
"$WORKSPACE/.pixi/envs/default/bin/python" - <<PY
import json, pathlib
root = pathlib.Path("$SWEEP_ROOT")
rows = []
for jf in sorted(root.glob("N*-eval.json")):
    j = json.loads(jf.read_text())
    rows.append({
        "N": int(jf.stem.split("N")[1].split("-")[0]),
        "pc_success": j["pc_success"],
        "mse": j["_metadata"]["mse"],
        "n_frames": j["_metadata"]["n_frames_evaluated"],
    })
rows.sort(key=lambda r: r["N"])
header = f"{'N':>4} {'pc_success':>14} {'mse':>10} {'n_frames':>10}"
print(header); print("-" * len(header))
for r in rows:
    print(f"{r['N']:>4} {r['pc_success']:>14.7f} {r['mse']:>10.2f} {r['n_frames']:>10}")
PY
echo "[sweep] root: $SWEEP_ROOT"
