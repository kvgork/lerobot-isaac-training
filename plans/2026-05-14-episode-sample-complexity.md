# Episode Sample-Complexity Sweep — How Many Demos for "Good"?

**Created:** 2026-05-14
**Trigger:** user request: "afterwards determine how many episodes of data
gathering would be needed to get a good result".
**Depends on:** the 4h policy + 4h WM run launched at 2026-05-14 19:39
(`outputs/long-train-2026-05-14-diffusion-dreamerv3-4h/`) — its full-budget
loss / pc_success on the 20-episode SO-101 dataset is the **upper-bound
reference point** for the sweep.
**Goal:** find the smallest `n_episodes` that gets within X% of the
20-episode reference, and report N + the knee curve.

---

## 1. Definitions

- **n_episodes**: number of source teleop episodes used for training.
  The SO-101 dataset has 20 — held-out last 3 are eval, training pool is
  17 (k=0..16 shuffled indices).
- **Held-out eval set:** episodes 17, 18, 19 (last 3). Same as the
  open-loop eval used everywhere else in this workspace.
- **"Good result" target:** within 10% of the reference `pc_success`
  AND within 20% of the reference `raw_loss`.
  Reference comes from the 4h run on all 20 episodes (TBD).

---

## 2. Sweep Grid

| k | n_train_eps | budget (min) | Why |
|---|-------------|--------------|-----|
| 0 | 2           | 30           | Floor — what does 2 demos look like? |
| 1 | 5           | 30           | Quarter dataset |
| 2 | 10          | 30           | Half |
| 3 | 15          | 30           | Three-quarters |
| 4 | 17          | 30           | Full training pool (matches main runs) |

Each sweep trial uses the SAME hyperparam config (bs=8 lr=1e-4 seed=42,
the AR-v3 best). Diffusion only. Eval on held-out episodes 17/18/19.

Total wall-clock: 5 × 30 min ≈ 2.5 h.

---

## 3. Implementation

### 3.1 Subsample helper

A new helper writes a `meta/episodes.parquet` overlay that only references
the first `k` episodes. Diffusion training reads this overlay; everything
else (data parquet, info.json) stays unchanged.

```python
# scripts/_subsample_dataset.py
import pyarrow.parquet as pq, pyarrow as pa, json, shutil
from pathlib import Path

def subsample(src: Path, dst: Path, n_episodes: int, eval_holdout: int = 3) -> None:
    """Copy `src` → `dst`, then trim meta/episodes to first n_episodes.

    Held-out episodes (last eval_holdout) are EXCLUDED from train; the
    eval script reads them by hard-coded index list.
    """
    shutil.copytree(src, dst)
    ep_shards = sorted((dst / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for s in ep_shards:
        t = pq.read_table(s)
        df = t.to_pandas()
        train_pool = df.iloc[: -eval_holdout]
        kept = train_pool.head(n_episodes)
        pq.write_table(pa.Table.from_pandas(kept), s)
    # info.json total_episodes update
    info = json.loads((dst / "meta" / "info.json").read_text())
    info["total_episodes"] = int(n_episodes)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
```

(Implementation note: this is a destructive copy. For a smaller diff,
do the subsample in-memory and write only `meta/episodes/`. Either way
works.)

### 3.2 Sweep runner

```bash
# scripts/_run_episode_sweep.sh — to be written
TRIAL_OUT=outputs/episode-sweep-2026-05-14
mkdir -p "$TRIAL_OUT"

for N in 2 5 10 15 17; do
    DST="datasets/so101-pickplace1-N${N}"
    .pixi/envs/default/bin/python scripts/_subsample_dataset.py \
        --src datasets/kvgork/so101-pickplace1 \
        --dst "$DST" --n-episodes "$N"

    bash scripts/run_full_pipeline.sh \
        --train-minutes 30 \
        --run-dir "$TRIAL_OUT/N${N}" \
        --dataset "$DST" \
        --skip-synthetic --skip-worldmodel

    # Eval against the ORIGINAL dataset's held-out 17/18/19
    .pixi/envs/train-policy/bin/python scripts/_open_loop_eval.py \
        --policy_path "$TRIAL_OUT/N${N}/policy-diffusion/checkpoints/last/pretrained_model" \
        --dataset_root datasets/kvgork/so101-pickplace1 \
        --n_episodes 3 \
        --output_json "$TRIAL_OUT/N${N}-eval.json" \
        --task_label "so101-N${N}-open-loop-mse" \
        --run_id "episode-sweep-N${N}"

    rm -rf "$DST"     # keep workspace clean
done
```

### 3.3 Analysis

After all 5 trials, aggregate:

```python
# scripts/_summarize_episode_sweep.py
import json, pandas as pd
rows = []
for N in (2, 5, 10, 15, 17):
    eval_json = json.load(open(f"outputs/episode-sweep-2026-05-14/N{N}-eval.json"))
    rows.append({
        "n_episodes": N,
        "pc_success": eval_json["pc_success"],
        "mse":        eval_json["_metadata"]["mse"],
        "n_frames":   eval_json["_metadata"]["n_frames_evaluated"],
    })
df = pd.DataFrame(rows)
print(df.to_string())
# Find first N where pc_success ≥ 0.9 × reference (from 4h run)
```

Append the table + recommendation to this plan once the sweep finishes.

---

## 4. Caveats

- **Open-loop action-MSE is saturated** (per `plans/2026-05-14-post-ar-next-steps.md`).
  Differences across N may again be within MSE noise.
  → Closed-loop hardware eval (`robot-data-run`) is the real signal.
  Plan to run hardware eval on the **N=2** and **N=17** checkpoints
  at minimum — visible difference there confirms the sweep is meaningful.
- **Episode quality varies.** Subsampling the FIRST k episodes may be
  biased if recorder skill improved over time. Mitigation: also run a
  randomized-shuffle variant with seed=42.
- **Held-out set is small (3 eps, 1306 frames).** Variance high.
  Consider 5-fold CV when budget allows.

---

## 5. Output

When sweep completes:

1. Append a table to this plan: `N | loss | pc_success | Δ vs reference`.
2. Write a knee-curve plot to `outputs/episode-sweep-2026-05-14/knee.png`.
3. Write a **recommendation paragraph** to this plan ending in a
   one-sentence answer: "Collect at least N episodes for a good result."
4. Snapshot dashboard so the sweep run shows up in compare reports.

---

## 6. Execution Plan

| Step | Wait condition | Action |
|------|----------------|--------|
| 0 | — | This plan is committed; nothing else to do until step 1. |
| 1 | Current 4h+4h run done (`outputs/long-train-2026-05-14-diffusion-dreamerv3-4h/dashboard/manifest.json` exists). Record its loss + pc_success as the **reference**. |
| 2 | Step 1 done | Write `scripts/_subsample_dataset.py` + `scripts/_run_episode_sweep.sh`. |
| 3 | Step 2 done | Run sweep (~2.5h foreground OR background with poll). |
| 4 | Step 3 done | Append results table + recommendation to this plan. |
| 5 | Step 4 done | Optional: closed-loop hardware eval on N=2 and N=17 checkpoints. |

Total expected wall-clock from "now" to "recommendation written":
~8h (current run) + ~2.5h (sweep) = **~10.5h**.

---

## 7. Reference Anchor (filled in after step 1)

| Metric | Value |
|--------|-------|
| reference run dir | `outputs/long-train-2026-05-14-diffusion-dreamerv3-4h/` |
| reference policy ckpt | `…/policy-diffusion/checkpoints/0024000/pretrained_model` (24,000 steps) |
| reference raw_loss (final) | **0.015** (vs 0.040 at 30 min — 2.7× lower) |
| reference pc_success (eval on 17/18/19) | **0.0003410836** |
| reference mse | 2930.83 |
| reference n_train_eps | 17 (training pool 0..16; 17/18/19 held out) |
| reference n_frames_evaluated | 1306 |
| training wall-clock | 14,400 s (4 h) |
| GPU avg | 35.7% util / 9.3 GB / 62.2 °C / 174 W |

"Good" threshold: `pc_success ≥ 0.9 × 0.0003410836 = 0.000307`.
NOTE: proxy is saturated → expect every trial within ±1% of reference.
Closed-loop hardware eval on extremes (N=2 and N=17) is the actual
discriminator (deferred to step 5 of the execution checklist).

---

## 8. Why This Plan Exists

The standard ML wisdom for SO-101 from the LeRobot docs is "20-50 demos
for a single task". This sweep produces a workspace-specific answer
grounded in our open-loop MSE proxy + ideally a closed-loop hardware
double-check. Future tasks should re-run this sweep when the recorder
changes (new camera, new operator, new task) — the answer depends on
the demonstrator's skill, not just the task.
