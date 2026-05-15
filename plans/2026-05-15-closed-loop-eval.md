# Closed-Loop Hardware Eval — Design & Requirements

**Created:** 2026-05-15
**Trigger:** open-loop action-MSE proxy saturated. The only remaining signal
is whether the policy actually completes the task on the real SO-101.
**Status:** plan + implementation skeleton committed; hands-on hardware
runs deferred until the user has the arm + workspace available.
**Sibling docs:**
[`plans/2026-05-14-post-ar-next-steps.md`](2026-05-14-post-ar-next-steps.md)
§A | [`docs/runbook/10-deploy-to-hardware.md`](../docs/runbook/10-deploy-to-hardware.md)
| [`plans/2026-05-14-episode-sample-complexity.md`](2026-05-14-episode-sample-complexity.md)
step 5.

---

## 1. What "Closed Loop" Means Here

| Loop  | Definition                                | Currently exists |
|-------|-------------------------------------------|-----------------|
| **Open-loop**   | Read recorded teleop frames → predict actions → score against the recorded actions. No arm motion. | yes (`scripts/_open_loop_eval.py`, used by the autoresearch + sweep paths) |
| **Closed-loop (sim)** | Reset Isaac Lab env, step policy through env until terminate, score with the env's reward function. | NO — `SO101RewardsCfg(success=None, progress=None)`. The sim env exists but has no closed-loop reward yet. |
| **Closed-loop (hardware)** | Reset the real arm + object pose, step the policy on the real motors until time-out or success criterion fires, repeat N episodes, report success rate. | **THIS PLAN** |

This document covers only the hardware path. The sim path is a separate
piece of work tracked as "What This Does NOT Do" in the deploy runbook §8.

---

## 2. What's Needed (Checklist)

### 2.1 Hardware

| Item | Spec | Have? |
|---|---|---|
| SO-101 follower arm | 6 DYNAMIXEL motors, calibrated | ☐ user-provided |
| U2D2 USB↔DYNAMIXEL adapter | or built-in USB | ☐ |
| 12 V power supply | ≥3 A for the motors | ☐ |
| Camera | RealSense D435 or USB webcam matching dataset's `observation.images.<name>` | ☐ |
| Workspace | ≥0.7 m radius clear of obstacles | ☐ |
| Source object | Same as recorded — `dex_cube` or whatever was teleop'd | ☐ |
| Target zone marker | Optional but easier for visual success-check | ☐ |
| Reset jig (optional) | A 3D-printed fixture that holds the source object at the demo's starting pose | ☐ |
| Physical e-stop / power switch | Within reach | ☐ |

### 2.2 Software (already implemented unless ☐)

| Component | Where |
|---|---|
| Policy CLI (open-loop deploy) | `robot-data-run` (this commit chain) |
| Robot driver | `lerobot.robots.so_follower.SO101Follower` (upstream) |
| Calibration | `lerobot-calibrate --robot.type=so101_follower` (upstream) |
| ☐ Episode loop with success scoring | **THIS PLAN** — `robot_data_runner.episode_runner` |
| ☐ Pluggable task-success specs | **THIS PLAN** — `robot_data_runner.task_specs` |
| ☐ Closed-loop CLI | **THIS PLAN** — `robot-data-run-eval` |
| ☐ Dashboard tab integration | already works (Evaluation tab reads `outputs/eval/*.json`) |
| ☐ Hardware-eval runbook | **THIS PLAN** — `docs/runbook/11-closed-loop-eval.md` |

### 2.3 Task definition

Each closed-loop run needs a **task spec** that answers three questions:

1. **When does the episode start?** ("user presses ENTER" / "auto after 2 s of stillness" / "after reset routine completes")
2. **When does it end?** ("after T seconds" / "success fires" / "human aborts")
3. **What counts as success?** ("gripper closed within ε of target position" / "manual key press by observer" / "object detected in target zone via camera")

For SO-101 pick-and-place, the canonical spec is:

```yaml
task: pick_and_place
start: prompt_user_enter
end:
  - timeout_s: 10
  - success_check_hz: 5
success:
  type: gripper_state_at_target_pose
  target_pose: [0.35, 0.10, 0.05]   # workspace-frame meters
  position_tolerance_m: 0.03
  require_gripper_closed: true
```

Initial implementation supports two task-spec types:

- `prompt_user_observer` — after each episode the CLI asks
  "success? (y/n)". Manual scoring; works on day one with no extra
  hardware. Fastest path to any closed-loop number.
- `gripper_state_at_target_pose` — automated success from joint state
  alone (no extra camera/object detector needed). Requires the user to
  specify the target joint pose. Useful when the target object is at a
  fixed location.

A third (`camera_object_detection`) is in scope for a future task spec
once a detector is wired in — not implemented in this milestone.

---

## 3. Output Schema (matches dashboard `EVAL_SCHEMA`)

Each closed-loop run writes one JSON to `outputs/eval/<run_id>.json`:

```json
{
  "run_id":              "closed-loop-2026-05-15-N17",
  "task":                "so101-pickplace1-closed-loop",
  "ts":                  "2026-05-15T07:00:00+00:00",
  "pc_success":          0.30,            // # successes / # episodes
  "n_episodes":          10,
  "intervention_rate":   0.10,            // human-aborted / total
  "mean_ep_len":         4.2,             // seconds, mean across episodes

  "_metadata": {
    "source":            "closed_loop_hardware",
    "policy_path":       "/abs/path/to/pretrained_model",
    "n_train_eps":       17,
    "task_spec_type":    "prompt_user_observer",
    "max_relative_target": 3.0,
    "rate_hz":           30,
    "duration_per_episode_s": 10,
    "host":              "koen-Legion-T7-34IMZ5",
    "per_episode": [
      {"index": 0, "success": true,  "ep_len_s": 5.1, "intervention": false},
      {"index": 1, "success": false, "ep_len_s": 9.8, "intervention": false},
      ...
    ]
  }
}
```

Fields with names matching `EVAL_SCHEMA` (`run_id`, `task`, `ts`,
`pc_success`, `n_episodes`, `intervention_rate`, `mean_ep_len`) light up
the existing Evaluation tab. Everything else goes in `_metadata` so
the dashboard's strict-schema loader doesn't reject it.

---

## 4. Implementation Components

### 4.1 `robot_data_runner.task_specs` (new)

```python
@dataclass
class TaskSpec(Protocol):
    """Per-episode success/termination criterion."""
    type: str

    def start_episode(self, robot) -> None: ...
    def is_done(self, t: float, obs: dict) -> tuple[bool, bool]:
        """Return (done, success). Called every step at rate_hz."""
    def on_episode_end(self, idx: int, success: bool, ep_len: float) -> None: ...
```

Concrete implementations:

- `PromptUserObserverSpec(timeout_s, prompt_text)` — never auto-terminates
  during the step loop except on `timeout_s`. After the loop ends, asks
  the user `success? (y/n/abort)` on stdin.
- `GripperAtTargetPoseSpec(target_pose, tolerance_m, require_gripper_closed,
  timeout_s)` — terminates early when the success predicate fires; reads
  joint state from the observation dict and computes end-effector pose via
  the SO-101 forward kinematics (uses lerobot's
  `robots.so_follower.robot_kinematic_processor`).

### 4.2 `robot_data_runner.episode_runner` (new)

Module-level function:

```python
def run_episodes(
    cfg: RunnerConfig,
    *,
    task_spec: TaskSpec,
    n_episodes: int = 10,
    duration_per_episode_s: float = 10.0,
    reset_prompt: str = "[reset] press ENTER when arm + object are at start pose",
    output_json: Path,
) -> Path:
    """Run N closed-loop episodes; write outputs/eval/<run_id>.json."""
```

Behaviour:

1. Load policy + connect to robot (reuses :func:`robot_data_runner.runner.load_policy` + `_build_robot`).
2. For each episode `idx`:
   1. Print `reset_prompt`. Wait for user ENTER.
   2. `task_spec.start_episode(robot)`.
   3. Step loop at `cfg.rate_hz` until `task_spec.is_done()` returns
      `done=True` OR `t >= duration_per_episode_s`.
   4. `task_spec.on_episode_end(idx, success, ep_len)` — prompts user
      and stores result.
3. Aggregate `pc_success = sum(success) / n_episodes` etc.
4. Write the JSON to `output_json`.

### 4.3 CLI `robot-data-run-eval` (new console entry)

```bash
robot-data-run-eval \
    --policy-path outputs/.../pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --task-spec prompt_user_observer \
    --n-episodes 10 \
    --duration-per-episode-s 10 \
    --max-relative-target 3.0 \
    --home-on-exit \
    --output-json outputs/eval/closed-loop-N17.json
```

The CLI internally sets `cfg.execute = True` (closed-loop without motor
writes is meaningless) but the user must pass `--i-have-read-the-safety-runbook`
on the first invocation. Confirms read-receipt and stores a marker at
`~/.config/robot-data-runner/safety_ack` so it's a one-time prompt.

### 4.4 Runbook `docs/runbook/11-closed-loop-eval.md` (new)

Pairs with runbook 10. Adds the task-spec selection guide, the manual
observer protocol (when to score y/n), and a recommendation table for
how many episodes to run for a meaningful number.

---

## 5. Recommended Eval Protocol

| Budget | n_episodes | Notes |
|--------|-----------|-------|
| First test  | 3   | sanity; expect mostly failure unless dataset is trivial |
| Smoke       | 10  | minimum for a meaningful `pc_success` |
| Standard    | 25  | recommended default; ±2 ep noise band |
| Rigorous    | 50  | for paper-grade results |

Pre-flight:

1. Run open-loop eval first to confirm checkpoint loads + obs/action
   shapes match.
2. Run a 1-episode bench dry-run (`robot-data-run --duration-s 10` with
   the arm clamp-mounted) to verify the policy produces sane actions
   on live obs.
3. Calibrate any new motor.
4. Tighten `--max-relative-target` to 2–3 deg for the first 3 episodes.
5. Open the dashboard live (`http://localhost:8501`) so the Evaluation
   tab updates as the JSON writes.

---

## 6. Execution Checklist

| # | Step | Owner | Wait condition |
|---|------|-------|----------------|
| 0 | Plan committed | done | — |
| 1 | `task_specs.py` + `episode_runner.py` + CLI scaffolded in robot-data-runner | this commit | — |
| 2 | 4 smoke tests pass | this commit | step 1 done |
| 3 | Runbook 11 written | this commit | step 1 done |
| 4 | Sample-complexity sweep finishes (open-loop) | running (PID 848211) | — |
| 5 | Hardware setup: power, USB, calibration | user | — |
| 6 | Run smoke (3 episodes, N=17 reference ckpt) | user | step 5 done |
| 7 | Run standard 25-episode eval on best AR ckpt + each sweep extreme (N=2, N=17) | user | step 6 done |
| 8 | Compare snapshots (open-loop vs closed-loop) on dashboard | user | step 7 done |
| 9 | Append final pc_success ladder to this plan | user / agent | step 7 done |

Steps 1-3 land in this commit. Everything from step 5 onwards is
user-driven because it needs the hardware.

---

## 7. Risk + Mitigation

| Risk | Likely cause | Mitigation |
|------|--------------|------------|
| Arm crashes into table | `--max-relative-target` too loose, or `--home-on-exit` zero pose intersects fixture | Start at 2 deg, disable `--home-on-exit` until verified, physical e-stop within reach. |
| Policy returns NaN mid-episode | Untrained or under-trained checkpoint, or input shape mismatch | Stuck-action watchdog already in place (warns at 30 consecutive ε-identical actions). |
| User-scored success is biased | Single observer | Two observers score independently for the first 10 episodes; report inter-rater agreement. |
| Reset pose drift across episodes | Manual reset is sloppy | Use a 3D-printed jig for the source object; reset prompt explicitly mentions "match the demo's start pose". |
| Camera frame mismatch | Resolution / colour space differs from training | `robot-data-run-check` reports the policy's expected `observation.images.*` shape; match the camera spec to that. |

---

## 8. Out-of-Scope (this milestone)

- Closed-loop Isaac Lab sim rollout (depends on `SO101RewardsCfg` having
  real reward terms).
- Bimanual coordination (single follower only).
- Automated object detection success-check (`camera_object_detection`
  task spec — sketched in §2.3, not implemented here).
- Closed-loop pc_success feedback into autoresearch `history.jsonl`
  (deferred — listed in the post-AR plan's open questions §6).

---

## 9. Why a Separate Plan

The post-AR plan §A handles "run closed-loop hardware eval" as one of
five levers. This plan unpacks ONE of those levers into a concrete
implementation, output schema, and runbook so the user can act on it
without re-deriving the design. The two plans link in both directions.
