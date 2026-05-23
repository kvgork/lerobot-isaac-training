#!/usr/bin/env bash
# Backfill held-out rollout MSE for an existing WM autoresearch session.
# Iterates every trial's sheeprl ckpt, runs wm_rollout, writes
# trial_${i}_rollout.json, augments history.jsonl with `eval_rollout_loss`.
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SESSION_ID="${1:?usage: $0 <session_id> [hdf5_path]}"
HDF5_CACHE="${2:-outputs/wm_data/so101-pickplace1_dreamerv3.hdf5}"
SLUG="wm-dreamerv3"
AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
PY="$WORKSPACE/.pixi/envs/train-dreamer/bin/python"

[ -d "$AR_DIR" ] || { echo "ERROR: AR dir not found: $AR_DIR" >&2; exit 2; }
[ -f "$HDF5_CACHE" ] || { echo "ERROR: HDF5 not found: $HDF5_CACHE" >&2; exit 2; }

echo "[backfill] session=$SESSION_ID slug=$SLUG"
echo "[backfill] hdf5=$HDF5_CACHE"

# Iterate every trial folder in logs/runs/dreamer_v3/custom_hdf5 matching this session.
for run_dir in "$WORKSPACE/logs/runs/dreamer_v3/custom_hdf5/trial_"*"_${SESSION_ID}"; do
    [ -d "$run_dir" ] || continue
    trial_idx=$(basename "$run_dir" | sed -E "s/^trial_([0-9]+)_.*$/\1/")
    ckpt_file=$(find "$run_dir" -name "ckpt_*.ckpt" 2>/dev/null | sort -V | tail -1)
    [ -n "$ckpt_file" ] || { echo "[backfill] trial=$trial_idx: no ckpt — skip"; continue; }

    # Stage a deploy-format dir so detect_policy_kind() recognises it:
    # both .hydra/config.yaml AND a ckpt_*.ckpt directory under one root.
    staged="$run_dir/staged"
    mkdir -p "$staged/.hydra" "$staged/checkpoint"
    cp -f "$WORKSPACE/outputs/autoresearch-$SLUG/trial_${trial_idx}/.hydra/config.yaml" "$staged/.hydra/config.yaml" 2>/dev/null \
        || cp -f "$run_dir/.hydra/config.yaml" "$staged/.hydra/config.yaml" 2>/dev/null
    ln -sf "$(readlink -f "$ckpt_file")" "$staged/checkpoint/$(basename "$ckpt_file")"

    rollout_dir="$run_dir/rollout"
    eval_json="$AR_DIR/trial_${trial_idx}_rollout.json"
    log_file="$AR_DIR/trial_${trial_idx}_rollout.log"
    mkdir -p "$rollout_dir"

    echo "[backfill] trial=$trial_idx running rollout…"
    "$PY" - <<PY > "$log_file" 2>&1
import json, sys
from pathlib import Path
from lerobot_isaac_deploy.wm_rollout import rollout
try:
    rollout(
        checkpoint_path="$staged",
        dataset_root="$HDF5_CACHE",
        output_dir="$rollout_dir",
        horizon_steps=50,
        n_seed_episodes=2,
    )
except Exception as e:
    print("ROLLOUT_FAIL:", type(e).__name__, str(e)[:300], file=sys.stderr)
    sys.exit(3)
sp = Path("$rollout_dir") / "rollout_summary.json"
d = json.loads(sp.read_text())
out = {
    "trial": $trial_idx,
    "mean_recon_loss": d.get("mean_recon_loss"),
    "n_frames_total": d.get("n_frames_total"),
    "n_seed_episodes": d.get("n_seed_episodes"),
    "horizon": d.get("horizon"),
    "checkpoint": d.get("checkpoint"),
    "hdf5_path": d.get("hdf5_path"),
}
Path("$eval_json").write_text(json.dumps(out, indent=2))
print(f"trial={$trial_idx} mean_recon_loss={out['mean_recon_loss']:.6f}")
PY
    rc=$?
    if [ "$rc" -eq 0 ]; then
        metric=$("$PY" -c "import json;d=json.load(open('$eval_json'));print(d['mean_recon_loss'])" 2>/dev/null)
        echo "[backfill] trial=$trial_idx mean_recon_loss=$metric"
    else
        echo "[backfill] trial=$trial_idx FAILED rc=$rc — see $log_file"
    fi
done

# Re-merge into history.jsonl: keep existing rows, add `eval_rollout_loss` field.
echo "[backfill] merging into history.jsonl"
"$PY" - <<PY
import json
from pathlib import Path
ar = Path("$AR_DIR")
hist_path = ar / "history.jsonl"
bak = ar / "history.jsonl.pre_rollout_backfill"
if not bak.exists():
    bak.write_text(hist_path.read_text())
rows = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
new_best = None
new_best_metric = None
for r in rows:
    i = r.get("trial")
    eval_p = ar / f"trial_{i}_rollout.json"
    if not eval_p.is_file():
        continue
    d = json.loads(eval_p.read_text())
    mrl = d.get("mean_recon_loss")
    if mrl is None:
        continue
    r["eval_rollout_loss"] = float(mrl)
    r["metric_value"] = float(mrl)
    r["metric_name"] = "wm_rollout_recon_loss"
    r["metric_kind"] = "wm_rollout_holdout_mse"
    if new_best_metric is None or mrl < new_best_metric:
        new_best_metric = mrl
        new_best = r
with hist_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
if new_best:
    (ar / "best.json").write_text(json.dumps({
        "trial": new_best["trial"],
        "metric_value": new_best["metric_value"],
        "metric_kind": new_best["metric_kind"],
        "metric_name": new_best["metric_name"],
        "config": new_best.get("config", {}),
        "source": "scraped_from_wm_rollout",
    }, indent=2))
    print(f"best trial={new_best['trial']} metric={new_best_metric:.6f}")
PY

echo "[backfill] done — $AR_DIR/history.jsonl updated"
