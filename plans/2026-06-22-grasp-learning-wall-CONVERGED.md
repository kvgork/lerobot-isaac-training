# Carry-place — CONVERGED finding: the RL agent can't learn the GRASP primitive (2026-06-22)

**Status: research/design decision point — needs a human steer. Autonomous knob-tuning is exhausted.**

## The converged finding (11 experiments, ~17 GPU-h, 2.5 days)
The DreamerV3 RL agent **does not learn to grip+lift the die** in this Isaac env — even when grip+lift
is the SOLE objective (grasp-first stage, `GRASP_STAGE=1`, success = lifted+held), with dense
`grasp_closure`(4) + `lift_shaping`(14) rewards, 25 seeded *successful* scripted demos in the buffer, and
a BC actor-loss pulling toward them. Across the grasp-first run: `ep_len` pinned at 300 (MIN-ever 300),
reward flat/declining (−14→−22), no grip+lift ever, well past the deterministic-replay divergence point.

Crucially: the **scripted controller grasps+lifts+places fine** (physics verified — memory
`scripted-grasp-infeasible` → SOLVED). So **the means exist; the RL agent isn't learning to use them.**
This is a contact-rich-manipulation RL *learning* failure, NOT a reward/geometry/tuning bug.

## How we got here (the diagnostic chain — the real deliverable)
1. "Plateau broken" (lever B, easier start) was a **SLIDE ARTIFACT** — `place_termination` was XY-only.
2. Objective favors placing (return −3 ≫ −25) → not a reward bug.
3. Lift-gate (`fd8e977`) made success require a carry → exposed the agent only ever SLID.
4. With sliding blocked, the chain breaks at **GRASP**: the agent reaches+pushes, never grips.
5. Grasp-first decomposition (isolate grip+lift) → STILL not learned → grasp is a genuine RL wall.

Levers exhausted (all hit the same wall): easier start, BC weight 0.3/0.6/1.5, PLACE_SUCCESS bonus,
lift-gate, demo-seeding, grasp-first isolation.

## Strategic fork (design-level — pick one; each is a real effort)
1. **Residual RL on the scripted grasp** (HIGH EV, my recommendation). The scripted controller grasps
   reliably; learn a *residual* policy on top of it (or use it as a high-prob action prior / action-space
   warm-start) instead of learning grasp from scratch. Sidesteps the contact-rich exploration wall by
   standing on the working scripted primitive. Effort: medium (wire the scripted grasp as a base action).
2. **Plan2Explore** (sheeprl `p2e_dv3`) — principled novelty-driven exploration to discover the grasp.
   Effort: real integration (adapter `exp=` + entry + two-phase). Uncertain it finds contact-rich grasp.
3. **Action-space / gripper-control review** — verify the agent has the *authority* to close+hold the
   gripper under the current `JointPositionActionCfg` (action clip, gripper-joint scaling). If grip is
   hard to command, no reward fixes it. Effort: low-medium (inspect action mapping + a hand-action probe).
   DO THIS FIRST as a cheap check before #1/#2.
4. **Stronger/curriculum grasp reward or contact-based grasp success** (e.g. reward actual fingertip
   contact + closure, not just proximity). Effort: medium.

## Recommendation
**(3) cheap action-authority check FIRST** (rule out a broken gripper command path), **then (1) residual
RL on the scripted grasp** (highest EV — reuse the working primitive rather than re-learn it). Avoid
pouring more GPU into from-scratch grasp RL — 11 runs say it won't spontaneously emerge here.

## State preserved
- All env machinery shipped + committed + pushed: BC actor-loss (GPU-validated), PLACE_SUCCESS, lift-gate,
  grasp-first stage, distance-curriculum driver, demo-gen state=13, RL+WM dual-mode recommendations.
- `ckpt_10000` (reach/lift base) preserved. GPU idle. No arm motion taken (hardware untouched).
- Memory: `carryplace-place-wall-plateau` (full chain), `dreamerv3-carryplace-launch-gotchas`.

## Action-authority check (2026-06-22, cheap diagnostic — done)
Action = `JointPositionActionCfg(scale=0.5, use_default_offset=True)` → ±0.5 rad DELTA from the rest pose.
Gripper rest = open. So action=0 (policy center) = gripper OPEN; a firm grip needs SUSTAINED extreme action
(demos use ±1.0). Authority EXISTS (demos grip via the same mapping) — but the parameterization BIASES the
gripper open and makes grip hard to explore/sustain. Rules out "broken action path"; adds a cheaper lever:
- **(0) Re-parameterize the gripper action** (NEW, cheapest targeted lever, do before residual-RL/P2E):
  give the gripper joint a larger action scale OR absolute-position control so a grip is commandable with
  MODERATE actions (not only the ±1.0 extreme). CAVEAT: changes the action interface → invalidates resuming
  ckpt_10000 (action semantics shift) → fresh run; it's a design change, so confirm direction before doing it.

## Updated recommendation
0. **Gripper-action re-parameterization** (cheapest; attacks the open-bias directly) — needs a design nod
   (alters the action interface).  THEN
1. **Residual RL on the scripted grasp** (reuse the working primitive).  Cheap action-authority check is DONE
   (#3 ruled out).  P2E (#2) remains a bigger bet.

**PAUSED for user direction on the fork — autonomous knob-tuning + cheap diagnostics are exhausted; the
next moves (action re-parameterization / residual-RL / P2E) are design choices. GPU idle, no run active.**

---

## 2026-06-23 — grasp-knob space EXHAUSTED + both pivots scoped

**Lever 0 (gripper re-param) verdict.** Swept `LEROBOT_ISAAC_GRIPPER_ACTION_SCALE` 0.5→3→5 with
`GRASP_STAGE=1` (grip+lift sole objective) + `LIFT_HOLD_STEPS` 10→3. Clean monotonic grip-signal climb
(rew peak −14 → −1 → −0.0 = firmer grip commandable), but at scale 5 + hold 3, past the deterministic
divergence (~step 13000), `ep_len` stayed PINNED at 300 (MIN-ever 300) — the grip FIRES but never SUSTAINS a
3-step held-lift. **Control failure (momentary bump, not a controlled raise), not firmness/discovery/reward.**
Gripper re-param broke the reward ceiling but did not solve the sustain → knob space done. (memory
[[carryplace-place-wall-plateau]].)

**Lever 2 (P2E) — VALIDATED RUNNABLE (not yet a grasp bet).** `p2e_dv3_exploration` now runs end-to-end on
the Isaac SO-101 env (intrinsic/ensemble/wm losses log, fits 10GB) after fixing 3 bugs (obs key `rgb`,
forced-compact model, `32-true` precision). Recipe + the bf16+fabric.backward follow-up that a full-size
grasp bet needs: `plans/2026-06-22-plan2explore-integration.md` + [[dreamerv3-carryplace-launch-gotchas]].
P2E targets DISCOVERY; the grasp wall is CONTROL → lever 1 is higher-EV for the sustain.

**Lever 1 (residual RL) — CORRECTED design (scoping found a model-based-RL footgun).** Naive mechanism
"blend inside `IsaacSO101Env.step`" is WRONG: sheeprl dreamer_v3 records the action to the replay buffer
at `rb.add` (dreamer_v3.py:587) BEFORE `envs.step` (:590), and stores the POLICY action (:577 `real_actions
= actions = player.get_actions`). Blending inside env.step → buffer-action (policy) ≠ executed-action
(blended) → **WM learns wrong dynamics → silently broken**. CORRECT injection:
1. Expose `IsaacSO101Env.compute_scripted_action()` — scene-side (robot articulation, object pose, IK from
   `_gen_sim_demos.py:step_to`), returns a `(6,)` scripted action for the CURRENT pre-step state. Sim-only;
   on hardware returns identity/zeros so weight→0 (dual-mode boundary lives here).
2. Patch the `player.get_actions` seam (dreamer_v3.py:577, BEFORE rb.add) — NOT env.step — so a single
   blended action feeds BOTH the buffer and the env: `a_applied = (1−p_t)·a_script + p_t·a_policy`.
3. **Decay direction: p_t (policy weight) RISES 0→1** over warmup (DAgger/residual handoff) — script-dominant
   early (WM+actor see successful carry-place), policy-dominant late (actor flies solo; WM converges to pure
   policy actions). The scoping agent's decay was inverted.
4. Gate `LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT` (default 0.0=OFF) + `_DECAY_STEPS`, mirroring `_patch_bc_actor_loss`
   / `_patch_seed_demo_buffer` in `scripts/_wm_isaac_entry.py`. Action format at the seam is a list-of-tensors
   (cat at :578) — blend must respect that shape. Verify get_actions is a clean monkeypatch seam (player obj).
This is the next focused BUILD (orchestration pipeline: plan→implement→grill→verify). NOT a monitoring-tick edit.

### Residual RL — BUILT + GRILLED (2026-06-23, default-OFF, GPU-validation pending)
Implemented per the corrected design:
- `IsaacSO101Env.compute_scripted_action()` (`sheeprl_plugin/isaac_env.py`) — reactive grasp
  state-machine (approach→descend→close→carry→place inferred from live state), reuses the IK
  math from `_gen_sim_demos.step_to`. Sim-only → None on hardware (residual auto-OFF).
- `_patch_residual_rl_action()` (`scripts/_wm_isaac_entry.py`) — blends at the `player.get_actions`
  seam (before rb.add), gated `LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT` (w0, default 0.0=OFF) +
  `_DECAY_STEPS` (default 50000); script_frac decays w0→0 (policy takes over).
- 7 CPU/torch tests pass; ruff clean.

**Grill (3 adversarial attackers) caught + FIXED:** (1) eval leak — sheeprl calls
`test(greedy=False)` so the greedy flag doesn't mark eval → now wrap dreamer_v3's `test()` with an
`in_eval` guard; (2) `self.actions`/action shape by broadcast-luck → reshape to `a_pol.shape`
exactly; (3) decay clock froze on skipped steps + sticky init-fail silently disabled residual →
step now advances every non-eval call, init retries 3× before latching, per-step warnings throttled;
(4) num_envs>1 would poison the buffer → guarded+warned; (5) `lifted` had no grasp-confirmation
(empty-gripper carry on a knocked-up die) → `holding` now also requires ee↔die co-location; (6)
`_LAST_WRAPPER` set for all runs → gated on the env var (true no-op when OFF); magic numbers named;
test sys.modules pollution → monkeypatch.setitem.

**OPEN efficacy risks for the GPU run (the approach's real uncertainty, not code bugs):**
- Actions clipped to [-1,1] (actor-reproducibility) → the REACTIVE controller rate-limits to ≤0.5 rad/step
  vs the open-loop demo's |a|>1 single moves; relies on multi-step convergence — VERIFY the clamped
  reactive controller still grasps within an episode.
- MUST run with `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1` (actor needs the object location to learn to
  reproduce the grasp) and RESUME a ckpt (`resume_from`) so residual fires from step 0 (fresh runs
  skip the ~1024-step random prefill where get_actions isn't called).
- The reactive phase thresholds are GPU-untested; first run should log a few `compute_scripted_action`
  outputs to sanity-check phase sequencing before a long run.
**Launch sketch (when GPU frees from P2E):** resume the reach/lift ckpt_10000, GRASP_STAGE=1 +
LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT=1.0 + _DECAY_STEPS≈30000 + INCLUDE_OBJECT_POSE=1 + BC + seed, watch
ep_len<300 (held-lift) emerging earlier than the from-scratch runs.
