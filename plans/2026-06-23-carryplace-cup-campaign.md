# Carry-place-into-cup — campaign handoff (2026-06-23)

Session built the full real-place-into-a-cup pipeline + ran 2 warm-starts (both plateaued). Paused for a
dedicated curriculum campaign. Everything below is committed + ready to resume.

## What "place" means now (changed this session — see [[carryplace-real-place-cup]])
A REAL place: grasp → lift → carry → **lower + RELEASE** the 16mm die into a **9cm-diameter × 7cm-tall CUP**
(4 static collidable walls). Success = `is_placed` = die XY in cup (radius 0.05) **AND** ever-lifted (latch)
**AND** resting (z<0.04) **AND** gripper released (open). All default-ON, env-tunable.

## What's BUILT + verified (committed)
- **Grasp works ~80%** (retracted the "infeasible" call) — `scripts/_probe_lift_stats.py`, `_probe_carry_mechanism.py`.
- **Env real-place predicate** (latch + resting + release) — sibling `lerobot-isaac-env` commits f15988f, 4d23cc1.
  Helpers: `terminations.latch_ever_lifted`, `is_placed`, `_gripper_open`. Both place_termination +
  place_success_reward use `is_placed` with a shared 0.05 radius (env knob `LEROBOT_ISAAC_PLACE_SUCCESS_RADIUS`).
- **Cup walls** — `pick_and_place.py`, static `AssetBaseCfg`+collision (no kinematic sim-hang). Knobs:
  `LEROBOT_ISAAC_PLACE_CUP` (1), `_PLACE_CUP_RADIUS` (0.045), `_PLACE_CUP_HEIGHT` (0.07).
- **Demos** `datasets/local/so101-sim-pickplace-demos-op3` — 40 full pick-lift-carry-place-RELEASE demos (die 0.18),
  validated VALID-WITH-CAVEATS vs recorded human demos (orchestrated workflow; gripper/lift/arm/smoothness all
  pass; warns = scripted determinism). Demo-gen: `scripts/_gen_sim_demos.py` (workspace 9af0427, 28900a6).
- **SO-101 reach**: die hangs ~0.096 below gripper_link; carry height `LEROBOT_ISAAC_CARRY_Z=0.19` lifts the die
  to ~0.107 (clears the 7cm rim); 0.22 BREAKS the grasp (reach-capped).
- **Warm-start machinery** (all in `scripts/_wm_isaac_entry.py`, env-gated): demo-seed (`LEROBOT_ISAAC_DEMO_DATASET`),
  DreamerFD BC actor-loss (`LEROBOT_ISAAC_BC_WEIGHT`), residual RL (`LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT`).

## The 2 plateaus (this session, ~8h GPU)
| run | die / carry | place bonus | result |
|-----|-------------|-------------|--------|
| `cup-warmstart-v1` | 0.18 / 18cm (full) | OFF | ep_len=300 pinned, 0 places, rew flat ~−31 |
| `cup-warmstart-cur-v1` | (0.20,0.0) / ~13cm | ON (weight 1.0) | ep_len=300 pinned, 0 places, rew flat ~−27 (10k–14.5k) |
| `cup0-s0-r4` (2026-06-26) | (0.22,−0.06) / ~7cm, cup 0.03 low | ON (1.0) + **ent_coef 1e-3 + horizon 25 + R=4** | ep_len_avg=300@5k, **0 places**; rew −28→−20 (creeping, beats v1 band) but no break to −10.6. Cut @7825/4.1h by inner `LEROBOT_TRAIN_TIMEOUT=14400` default → **no ckpt** (ckpt_every=10000). Demo-gen unblocked via `DEMO_REST_Z` decouple; 37 demos `so101-sim-pickplace-demos-cup0`. |
Both: BC active + decaying, demos seeded (40 ep / 19400 transitions), reward climbs (reach/grasp/lift) then
FLAT — **the place is never DISCOVERED** (no positive reward spikes, no terminating episodes). Matches every prior
full-task run + the ACT-BC closed-loop failure ([[demo-warmstart-pipeline]]).

## Why (the obstacle)
The carry-place breakthrough (cur1, 2026-06-16: 30% places) needed a **short ~6.6cm carry on a FLAT target**.
The 7cm cup walls block that: die-near-cup → the gripper fingers (±2cm) hit the 7cm wall during grasp (die never
lifts); and the scripted release ejects the die ~5cm in +X, which lands on the rim/outside from non-tuned
positions. So the easiest CLEAN-grasp stage is ~13cm carry — still too far for RL to discover the place. The
place isn't an RL-control problem; it's a **sparse-reward DISCOVERY** problem made harder by the cup geometry.

## RECOMMENDED next: easy-cup curriculum (make place discoverable, then harden)
The lever that ever worked was an easy-enough start that exploration STUMBLES into a place. The cup must be made
easy first, then hardened on TWO axes (cup height + carry distance):
- **Stage 0** — `LEROBOT_ISAAC_PLACE_CUP_HEIGHT=0.03` (low cup, fingers clear it), die IN/at the cup
  (`OBJECT_X=0.22 OBJECT_Y=-0.13`) → trivial carry; agent discovers lift→release-in-cup. Regenerate matched demos
  at this config first (`_gen_sim_demos.py --obj_x 0.22 --obj_y -0.13` with env `LEROBOT_ISAAC_PLACE_CUP_HEIGHT=0.03` exported — cup height is an ENV knob read at scene build, NOT a script flag).
  > **GPU finding (2026-06-24, demo-gen smokes):** die-IN-cup geometry is **UNDEMOABLE** by the scripted
  > controller — it cannot lift a die from inside the cup (walls block the side-approach grasp; **8/8 attempts
  > `lifted=False`, maxz≈0.008**). The scripted grasp needs the die on the OPEN TABLE. Corrected easy-Stage-0 =
  > die ~7 cm from the low cup (`OBJECT_Y=-0.06`, tgt `(0.22,-0.13)`, cup 0.03 — replicates the cur1 ~6.6 cm
  > flat-carry that hit 30% places): grasp then LIFTS, but matched-demo gen is still **marginal** — carry-slip
  > (die lands far from cup) + **`released=False` finger-jam on the narrow 3 cm-cup release** (near-misses landed
  > die in-radius + resting but gripper read closed); **0/8 clean** in a quick smoke (`-op3` itself was only ~48%).
  > **To proceed:** either (a) tune demo-gen (lower `--jitter`, more `--max_attempts`, debug the release-jam),
  > (b) seed Stage-0 with the existing `-op3` demos (geometry-mismatched but valid place trajectories), or
  > (c) run **no-seed** (easy geometry + raised `actor.ent_coef`/`replay_ratio`/`horizon` alone may stumble into
  > the place, per cur1). GPU paused 2026-06-24 before committing the multi-hour run — pending this choice.
- **Stage 1** — cup 0.03, die ~6cm out (e.g. `OBJECT_Y=-0.05`) → short carry. Resume Stage-0 ckpt.
- **Stage 2..N** — raise `PLACE_CUP_HEIGHT` 0.03→0.07 and push the die out toward (0.18,0.05), resuming each ckpt
  (`EXTRA_HYDRA=checkpoint.resume_from=<ckpt>`, bump STEPS). cur1→cur2 resume pattern
  ([[dreamerv3-carryplace-launch-gotchas]]).
Keep: BC (`BC_WEIGHT=1.0`), seeding (matched demos per stage), place bonus (`PLACE_SUCCESS_WEIGHT=1.0`),
`INCLUDE_OBJECT_POSE=1` + `EXTRA_HYDRA=algo.mlp_keys.encoder=[state]`, batch 8, ckpt 5k,
`LEROBOT_TRAIN_TIMEOUT=46800`.
**ADD the exploration counter-levers BOTH v1 plateaus omitted** — the flat-reward / ep_len=300-pinned signature IS
entropy/exploration collapse (2026-06-16 wm-vla playbook): actor `ent_coef 1e-3` (5–10× the DreamerV3 default),
`replay_ratio 16`, `horizon 25`, `demo_ratio 0.5`. Treat the curriculum as ONE of TWO simultaneous levers (easier
start **AND** raised exploration entropy), not curriculum alone — both v1 plateaus ran without the raised actor_ent.
See `[[2026-06-16-wm-vla-training-playbook]]`.

## Other options (if curriculum stalls)
- **BC policy instead of RL**: `lerobot-isaac-train --target_arch act --dataset .../so101-sim-pickplace-demos-op3`
  then closed-loop eval — the plan's "quickest path"; sidesteps DreamerV3 discovery (earlier ACT-BC was 0% but
  on the broken task / narrow demos — worth a re-try on the correct task + -op3).
  - **RAN 2026-06-26 on cup0 (correct task + matched demos): ACT loss converged 0.037 but closed-loop
    `task_success=0/20`, mean_ep_len=300.** BC fails closed-loop via COMPOUNDING ERROR (narrow deterministic
    scripted manifold, no recovery data) — opposite failure mode to RL's discovery wall. Both standard
    approaches now exhausted; gap = closed-loop robustness from narrow scripted demos. Next: DAgger/noise-
    injected corrective demos, more+diverse demos, DreamerFD harder, or diffusion policy. (`_sim_eval.py` got a
    13-dim object-pose state fix — was hardcoded 12-dim.) See memory `[[carryplace-cup0-warmstart-r4-result]]`.
- **DreamerFD harder**: bc_weight 1→3, decay 20k→50k, prioritized place-transition replay, more steps.
- **Residual-RL / Plan2Explore** (vault IL-plateau ladder, `[[2026-06-16-wm-vla-training-playbook]]`): freeze a BC
  base + train a small residual head online via `LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT` (built, default OFF — see line 24),
  or add P2E ensemble-disagreement exploration. **Caveat** (`[[Plan2Explore]]`, 2026-06-21): DreamerV3-XP finds
  disagreement gains *modest* vs prediction-error replay prioritization; LeRobot has no native curiosity trainer;
  the clean `--expl_behavior` flag is DreamerV2-era — not one-flag-easy.

## Launch templates (MUST RECREATE — were in ephemeral `scratchpad/`, never committed; absent as of 2026-06-24)
- Full warm-start: `scratchpad/launch_warmstart_full.sh` (die 0.18, the v1 config) — **GONE; recreate from the Stage-2
  launch block in `plans/2026-06-23-world-model-training-plan.md` + the env knobs above.**
- Easy curriculum: `scratchpad/launch_warmstart_cur.sh` (edit OBJECT_X/Y + PLACE_CUP_HEIGHT per stage) — **GONE; recreate.**
- Pre-flight: kill stray GPU procs (`nvidia-smi --query-compute-apps`); batch 8 (not 16) on this 10GB GPU.

## Known caveats
- Eval `test()` crash (inference-mode tensor in Isaac DR reset) — **FIXED + GPU-verified 2026-06-24** (adapters
  commit `741eccc`: `IsaacSO101Env.reset/step` wrap the Isaac calls in `torch.inference_mode(False)`; a 600-step
  verify run reached end `test()` clean — `Test - Reward` emitted, rc=0, zero inference-tensor errors). Historically
  hit cur1/v5 (rc=1 at the very end; ckpts survived). Still read the place signal from the TB reward curve (rew>−10) +
  `Game/ep_len_avg` (<300 = real terminating places), NOT the test() eval. Proper ckpt eval =
  sheeprl `evaluate.py` (DreamerV3), not `_sim_eval.py` (lerobot only).
- **Prove the seeding actually helps** each stage with a seed-vs-no-seed A/B (`LEROBOT_ISAAC_DEMO_DATASET` set vs
  unset) — vault gate-on-place-rate principle (`[[2026-06-23-wm-training-plan-research]]`). Coverage from online
  failures matters more than demo purity; keep the messy seeded transitions, don't aggressively SAL/TED-filter.
- Bash safety classifier (Opus-backed) was intermittently unavailable this session — retry transient failures.
