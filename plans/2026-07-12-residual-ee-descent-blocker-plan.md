# Plan — residual-RL next blocker: scripted base won't descend (ez≈0.30 vs grasp_z 0.106)

**Created:** 2026-07-12 · **Branch:** `feature/residual-rl` · **Blocks:** the residual-RL run actually
grasping (no-stall fix already landed + smoke-confirmed; see
`plans/2026-07-12-phase3-model-comparison-plan.md` sibling + memory `carryplace-cup0-warmstart-r4-result`).

## Symptom (from the 2026-07-12 smoke)
With the residual ENGAGED at `script_frac=1.0` (pure scripted driving the arm), `[script-dbg]` shows:
`phase=APPROACH ez=0.303`, `phase=DESCEND ez=0.295`, `oz=0.008` — the end-effector **hovers at
z≈0.30 and never descends** to `grasp_z=0.106`, so the die is never grasped/lifted. Phase advances
(APPROACH→DESCEND, no-stall fix holds) and reward climbs -61→-32 (reach only), but no grasp.

## Root cause (identified, not yet fixed)
The env action term is `mdp.JointPositionActionCfg(scale=0.5, use_default_offset=True)` →
`q_cmd = q_default + 0.5 · action`. The scripted IK inverts it as
`action = (q_des − q_default) / 0.5` (isaac_env.py `compute_scripted_action`, identical to the
WORKING `scripts/_gen_sim_demos.step_to`). So **any joint that must move >0.5 rad from default
requires |action|>1**. Descending from the ~0.30 m rest pose to grasp_z=0.106 needs large
shoulder/elbow moves → |action|>1.

- The **demo** passes the raw action to `env.step` (no clamp) → the big move executes → it grasps.
- The **residual** clamps in the blend: `a_blend = clamp(script_frac·a_scr + (1−script_frac)·a_pol,
  −1, 1)` (`_wm_isaac_entry.py:705`) → each joint capped at ±0.5 rad from default → the arm can't
  reach the descent pose → stuck at ez≈0.30.

The clamp exists on purpose (the tanh actor can only reproduce in-range actions, and the executed
action must equal the buffered action for the WM/BC handoff). So this is a real design tension, not a
one-line bug. Corollary: the recorded demos ALSO contain |action|>1 frames → their BC labels are
partly unreachable by the tanh actor too. **The action space is mis-scaled for the reachable joint
range** — everything (scripted base, BC labels, RL actor) should live in [-1,1].

Secondary interaction: my per-phase step caps (APPROACH 50 / DESCEND 60) were sized for the demo's
FAST (unclamped) moves. Under any in-range (slower) motion the caps may advance a phase before the
arm arrives — re-tune after the scale fix.

## Phase 0 — Measure (probe, no training, ~10 min)
Ground the fix before changing anything.
- Extend the demo/probe (`scripts/_gen_sim_demos.py` has the working controller; add a `--log_actions`
  path, or a small `scripts/_probe_action_scale.py`) to log, per step across a full pick→place:
  the raw `action[jid]` per arm joint + the resulting `q_des−q_default` (rad) + ee z.
- **Deliverables:** (a) the max |action| per joint during APPROACH/DESCEND (confirm >1, by how much);
  (b) the per-joint rad range actually used across the whole grasp. This sizes the new per-joint scale.
- **Gate:** confirm descent needs |action|>1 (expected). If NOT (|action|≤1 everywhere) → the clamp is
  innocent and the cause is elsewhere (IK convergence / caps too short) → jump to Phase 2 diagnostics.

## Phase 1 — Fix the action scale (primary: A; fallback: B)
**Option A (recommended) — rescale the env action so the full working joint range maps to [-1,1].**
- Set per-joint `_ACTIONS_SCALE_DICT` in `so101_env_cfg.py` so `scale_j ≈ max|q_used_j − q_default_j|`
  (from Phase 0), i.e. an action of ±1 spans each joint's actual working range. Keep the gripper joint
  as-is (already binary open/close).
- Update the scripted normalization to divide by the NEW per-joint scale (both `compute_scripted_action`
  and `_gen_sim_demos.step_to` — keep them identical).
- Consequence: scripted + demo + RL actions all fall in [-1,1]; the clamp becomes a no-op; the tanh
  actor can reproduce every scripted/demo action. **Regenerate the demo sets** (scripted → cheap) so
  their recorded actions use the new scale; old demos (`-op3`, `-cup0*`) are then stale for BC.
- Risk: any existing sim policy trained on scale=0.5 is invalidated (all failing anyway — acceptable).

**Option B (fallback, localized) — per-step target-delta limiting in the scripted controller.**
- Instead of commanding the full IK target each tick, command an intermediate joint target within
  ±0.5 rad of the CURRENT joint pos (so |action|≤1), stepping toward the goal over multiple ticks.
- Keeps scale=0.5, no demo regen; but the arm descends slower → **raise the per-phase caps**
  (APPROACH/DESCEND/LIFT/CARRY) proportionally so the arm arrives before the cap fires.
- Pick B only if A's demo-regen / policy-invalidation is undesirable.

Either way: re-tune `PHASE_STEP_CAP` (in `scripted_grasp_phases.py`) against the fixed motion speed —
the state gate should be the primary advance; the cap a generous safety, not the driver.

## Phase 2 — Re-smoke (GPU, learning_starts=200, ~15 min)
`scripts/launch_residual_rl.sh` with the smoke recipe (learning_starts=200, STEPS=700).
- **Gate (the fix works):** `[script-dbg]` shows `ez` DESCENDING to ≈0.106 in DESCEND, then
  `obj_lifted=True` (`oz`>0.07), phase reaching CLOSE→LIFT→CARRY, and reward climbing well above the
  -32 reach-only ceiling (toward the scripted ~67-80% place rate at script_frac=1.0).
- If ez still stuck → Phase 0 mis-diagnosed; instrument the IK output (`q_des` vs `q_default`) and the
  executed vs buffered action directly in `compute_scripted_action` + the blend.

## Phase 3 — Full residual run (GPU, detached, overnight)
Once the base grasps+places at script_frac=1.0: launch the real run (default learning_starts=1024,
STEPS=40000, w0=1.0, decay 30000) so the residual learns to improve on the working base and the WM
models the place. Hand ckpt to the Phase-3 real-arm bake-off is NOT the goal here (sim2real≈0); the
goal is a genuine WM-policy that solves carry-place in sim (Route C groundwork).

## Decision gates
| phase | gate | pass → | fail → |
|-------|------|--------|--------|
| 0 measure | descent needs \|action\|>1 | Phase 1 (scale fix) | Phase 2 diag (IK/caps) |
| 1 fix | scripted+demo+actor actions in [-1,1] | Phase 2 | try Option B / re-measure |
| 2 smoke | ez→0.106, oz>0.07, reward≫-32 | Phase 3 | instrument IK/action directly |
| 3 full | places at script_frac=1.0, RL retains as w0 decays | done | curriculum / more demos |

## Risks / notes
- **Demo regen (Option A)** invalidates `-op3`/`-cup0*` for BC. Cheap (scripted) but must rerun demo-gen
  + re-validate. Sim-only; does not touch the real-data candidates.
- **Actor exploration** changes if the action scale grows — bigger scale = coarser control; size per-joint
  to the ACTUAL used range, not the full joint limit, to avoid over-coarsening.
- **Keep `compute_scripted_action` and `_gen_sim_demos.step_to` byte-identical** on the IK + normalization
  — they diverging is what created this class of bug.
- The no-stall fix + caps are already correct for their job; this is a distinct action-authority fix.

## Files
- `src/lerobot-isaac-env/src/lerobot_isaac_env/so101_env_cfg.py` (`_ACTIONS_SCALE_DICT`, ~line 280-314)
- `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/sheeprl_plugin/isaac_env.py` (`compute_scripted_action` normalization)
- `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/scripted_grasp_phases.py` (`PHASE_STEP_CAP` re-tune)
- `scripts/_gen_sim_demos.py` (normalization + demo regen), `scripts/_wm_isaac_entry.py:705` (the clamp)
- `scripts/launch_residual_rl.sh` (smoke recipe)

## Related
- `plans/2026-07-12-phase3-model-comparison-plan.md` · memory `[[carryplace-cup0-warmstart-r4-result]]`
  (S3 diagnosis: "blended action doesn't drive IK to commanded approach pose; ee stuck at 0.30") ·
  `[[scripted-grasp-infeasible]]` · `[[sheeprl-action-override-buffer-seam]]` · `[[so101-sim-reach-envelope]]`
  (action-authority finding: gripper scale moved the reward ceiling).
