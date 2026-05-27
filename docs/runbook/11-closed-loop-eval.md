# Runbook 11 — Closed-Loop Hardware Evaluation

**Audience:** anyone who has finished the open-loop deploy walkthrough
(runbook 10) and now wants a `pc_success` number from N real-arm rollouts.
**Outcome:** `outputs/eval/<run_id>.json` populates the dashboard's
Evaluation tab with closed-loop success rate, per-episode trace, and
intervention rate.
**CLI:** `robot-data-run-eval` (installed by `robot-data-runner`).
**Plan reference:** [`plans/2026-05-15-closed-loop-eval.md`](../../plans/2026-05-15-closed-loop-eval.md).

---

## TL;DR

```bash
# 1. ALWAYS finish runbook 10 first (dry-run + bench execute clamp test).

# 2. One-time consent (reads safety preconditions; stores ~/.config/robot-data-runner/safety_ack):
robot-data-run-eval --policy-path /tmp/_  --i-have-read-the-safety-runbook --help

# 3. 10-episode manual-scored eval:
robot-data-run-eval \
    --policy-path outputs/.../checkpoints/last/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --task-spec prompt_user_observer \
    --n-episodes 10 \
    --duration-per-episode-s 10 \
    --max-relative-target 3.0 \
    --home-on-exit \
    --output-json outputs/eval/closed-loop-N17.json \
    --run-id closed-loop-2026-05-15-N17 \
    --task-label so101-pickplace1-closed-loop \
    --n-train-eps 17
```

`--task-spec prompt_user_observer` means the CLI asks `success? (y/n/abort)`
after each episode. Day-one path with zero extra hardware.

---

## 1. Why Closed-Loop Eval Matters

The open-loop action-MSE proxy (`scripts/_open_loop_eval.py`) is saturated
on the SO-101 dataset (see `plans/2026-05-14-post-ar-next-steps.md`).
Trainings from 30 min to 4 h cluster at the same `pc_success ≈ 0.000341`
even though raw loss drops 2.7×. The open-loop metric reads "the policy
predicted the recorded action well per-frame"; it does NOT read "the
policy completed the task".

Closed-loop eval = run the policy on the real motors, reset between
episodes, count how many times the arm actually does the thing.
**This is the only signal that ranks checkpoints meaningfully.**

---

## 2. Pre-flight (every session)

1. Finish runbook 10's bench dry-run + execute clamp test on the
   checkpoint you intend to eval. If the open-loop dry-run looked
   broken, do NOT proceed to closed-loop.
2. Re-calibrate motors if anything physical changed:
   `pixi run -e train-policy lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0`.
3. Verify the dashboard's Evaluation tab loads. Start the live dashboard:
   `pixi run -e dashboard dashboard` → http://localhost:8501.
4. Place the source object at the demo's start pose (reset jig
   recommended — see plan §7).

---

## 3. Task Specs

| Type | Description | Hardware extras |
|---|---|---|
| `prompt_user_observer` | Step loop just times out after `--duration-per-episode-s`; after each episode CLI asks `y/n/abort` on stdin. | none — works day one |
| `gripper_at_target_pose` | Step loop ends EARLY when joint state matches `--target-joint-pos` (within `--tolerance-per-joint`) AND `--require-gripper-closed`. Auto-scored. | none — reads joint state only |
| `camera_object_detection` | Reserved for future. | object detector |

Selection rule of thumb:

- First N=3 sanity check → `prompt_user_observer` (you watch + judge).
- Standard 25-episode batches where you trust the success predicate →
  `gripper_at_target_pose`.

**`gripper_at_target_pose` example.** Target = SO-101 home + gripper
closed:

```bash
robot-data-run-eval ... \
    --task-spec gripper_at_target_pose \
    --target-joint-pos 0,0,0,0,0,60 \
    --tolerance-per-joint 3.0 \
    --require-gripper-closed
```

Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper` (canonical `SO101_JOINT_ORDER`). Units match
`--use-degrees`.

---

## 4. Episode Budget Recommendations

| Budget | n_episodes | Use when |
|---|---|---|
| **Sanity** | 3 | First-time on a checkpoint or after recalibration |
| **Smoke** | 10 | Minimum for a meaningful `pc_success` number |
| **Standard** | 25 | Cross-checkpoint comparison, ±2 ep noise band |
| **Rigorous** | 50 | Paper-grade / publish-grade results |

Reset between episodes takes ~10 s of human attention. A 25-episode
manual-scored run is ~25 × (10 s reset + 10 s rollout + 5 s scoring) ≈
10 min. Auto-scored (gripper_at_target_pose) is faster.

---

## 5. Output JSON

```
outputs/eval/<run_id>.json
```

Schema matches the dashboard's `EVAL_SCHEMA`:

```json
{
  "run_id": "closed-loop-2026-05-15-N17",
  "task":   "so101-pickplace1-closed-loop",
  "ts":     "2026-05-15T07:00:00+00:00",
  "pc_success":        0.30,
  "n_episodes":        10,
  "intervention_rate": 0.10,
  "mean_ep_len":       4.2,
  "_metadata": {
    "source":             "closed_loop_hardware",
    "policy_path":        "...",
    "n_train_eps":        17,
    "task_spec_type":     "prompt_user_observer",
    "max_relative_target": 3.0,
    "rate_hz":             30,
    "duration_per_episode_s": 10,
    "host":                "koen-Legion-T7-...",
    "per_episode": [
      {"index": 0, "success": true,  "ep_len_s": 5.1, "intervention": false},
      ...
    ]
  }
}
```

`pc_success` / `mean_ep_len` / `intervention_rate` light up the Evaluation
tab automatically.

---

## 6. Comparing Open- vs Closed-Loop

After running both, save a labelled snapshot and use the dashboard's
2-way compare:

```bash
pixi run -e dashboard snapshot save --label closed-loop-N17
pixi run -e dashboard compare \
    --snapshots <open-loop-snapshot-id> <closed-loop-snapshot-id> \
    --output-dir outputs/eval/compare-open-vs-closed
```

Expected: pc_success absolute values will be VERY different (closed-loop
is a real success rate; open-loop is an MSE proxy near 3e-4). Look at
**ranking** across checkpoints, not absolute values.

---

## 7. Safety Recap

All six layers from runbook 10 still apply. Closed-loop adds:

7. **`--i-have-read-the-safety-runbook` consent gate.** The CLI refuses
   to start without an explicit one-time read receipt
   (`~/.config/robot-data-runner/safety_ack`). Don't bypass with
   `touch`; read the runbook.
8. **Default `--max-relative-target=3.0` deg.** LOWER than open-loop
   deploy's default 5 deg. Tighter clamp because the loop is longer
   and resets are imperfect.
9. **`--duration-per-episode-s` hard cap.** Even if the success predicate
   never fires, the episode ends at this time. Default 10 s.
10. **Observer abort.** Typing `abort` at the y/n prompt raises
    KeyboardInterrupt, runs `--home-on-exit`, exits cleanly.

If anything looks wrong mid-episode, hit the physical power switch.
Always. Software safety is never sufficient.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `First closed-loop eval on this machine.` consent prompt | Add `--i-have-read-the-safety-runbook` once. |
| All episodes scored 0/N success | Wrong task spec target, wrong camera resolution, or under-trained checkpoint. Re-check open-loop dry-run first. |
| Stuck-action watchdog fires every episode | Policy NaN — retrain with different seed. |
| Reset prompt loops without continuing | stdin closed (background process). Run in foreground terminal. |
| Per-episode JSON length wrong (some episodes missing) | Operator aborted mid-run. Output JSON still valid; `n_episodes` matches actual count. |

---

## 9. What This Does NOT Do (yet)

- **Camera-object-detection success.** Sketched as a third task spec in
  the plan; not implemented.
- **Automatic reset.** Resets are still manual. A 3D-printed object jig +
  workspace fiducials makes this consistent but is hardware-side work.
- **Closed-loop feedback into autoresearch.** The output JSON is consumed
  by the dashboard but not by the AR proposer. Tracked as an open
  question in the post-AR plan.

---

## 10. Related Files

- Plan: [`plans/2026-05-15-closed-loop-eval.md`](../../plans/2026-05-15-closed-loop-eval.md)
- Sibling runbook: [`docs/runbook/10-deploy-to-hardware.md`](10-deploy-to-hardware.md)
- Standalone package: `src/robot-data-runner/` ([github.com/kvgork/robot-data-runner](https://github.com/kvgork/robot-data-runner))
- Task specs: `src/robot_data_runner/task_specs.py`
- Episode runner: `src/robot_data_runner/episode_runner.py`
- CLI: `src/robot_data_runner/cli_eval.py`
- Output ingestion: `lerobot_isaac_dashboard.loaders.eval_results.EVAL_SCHEMA`
