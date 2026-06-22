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
