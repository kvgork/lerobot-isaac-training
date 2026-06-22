# RL-for-World-Models — vault → pipeline recommendations (dual-mode: sim + HITL)

**Date:** 2026-06-21 · **Source:** vault `05-Wiki` RL+WM notes added 2026-06-21 (Plan2Explore, Director,
reward-coupled WM, hierarchical planning, cs4756, MACRL). **Method:** 8 parallel readers → 49 techniques
adversarially vetted for feasibility on this stack AND sim-vs-HITL dual-mode → ranked.
**Constraint honored:** every recommendation works BOTH in pure sim AND with a real SO-101 arm in the loop.

## Dual-mode principle (the rule that keeps everything HITL-valid)
Ground every reward / success / exploration signal on EITHER (a) the **shared binary `outcome_verifier`
predicate** (`object_in_bin` → identical in sim via the termination manager and on hardware via
`outcome_reader.read_object_in_bin`), OR (b) **WM-internal quantities** (ensemble disagreement, prediction
error, BC loss from demos) computed from obs available in both worlds. NEVER privileged sim state
(exact object_pose, instant resets, parallel rollouts, ground-truth dynamics). Result: **47/49 techniques
are dual-mode** (23 native, 24 adaptable); only 2 sim-only (rejected).

## The headline
The vault material does NOT require abandoning the current approach — it **completes** it. The carry-place
plateau is a sparse-reward exploration failure, and the research identifies the exact missing piece:
demo-seeding (lever A) loads place *dynamics* into the WM but the **BC actor-gradient is unwired**, so the
actor never converts to place *behaviour*. Two principled, dual-mode, small-code fixes dominate.

## ADOPT-NOW (highest EV, do these)
1. **Decaying BC actor loss (DAPG) + RLPD 50/50 demo/online replay** — `native-dual, small-code, EV=high, plateau=YES`.
   THE load-bearing fix. `demo_buffer.behavior_cloning_loss` is BUILT but has ZERO callers — wire it into the
   sheeprl actor update via the `_wm_isaac_entry.py` monkeypatch: 50/50 demo+online batch sampling + a
   decaying BC weight on the actor. Converts the already-seeded demo dynamics into actor place behaviour.
   **Sim+HITL:** identical code path; demo source differs (scripted vs teleop), signal is the same.
   *This is the queue's elevated next lever (was "hardest/last", research says small-code + highest-EV).*
2. **Intrinsic-reward sign/normalization guardrail** — `native-dual, small-code, EV=high`. Prereq for any
   intrinsic-reward lever: `r_int = +error(...)`, `assert beta>=0`, running mean/std normalization. Cheap insurance.
3. (Running) **Easier curriculum step-0 + demo-seed** = lever B, in flight.

## QUEUE (principled exploration — the next lever family if BC+curriculum insufficient)
- **Plan2Explore / ensemble-disagreement intrinsic reward** (`native-dual, small-medium-code, plateau=YES`) —
  the principled cure for sparse-reward exploration: intrinsic reward = disagreement across a WM-ensemble's
  latent predictions, computed in imagination. sheeprl ships a `p2e_dv3` algo (config-level start). Dual-mode
  native (disagreement needs no privileged state). **Strongest principled plateau lever after BC.**
- **RND / curiosity intrinsic reward in the RSSM latent** (`native-dual, small-code, EV=high, plateau=YES`) —
  lighter-weight Dreamer-native exploration bonus; cheap alternative to full P2E.
- **Plan2Explore reward-free pretrain → task fine-tune** (adaptable-dual) — explore-first to populate the WM,
  then attach the task reward; relabel via the shared verifier.
- Terminal place bonus grounded on `outcome_verifier` (env-var, low-EV alone — pairs with the above).

## AUTONOMY recommendations (make the pipeline self-directing — all dual-mode)
- **Distance/spawn-radius curriculum auto-relaunch DRIVER gated on the binary verifier** (`adaptable-dual, small-code, EV=high`) —
  the Phase-2 driver (deferred): loop DISTANCE_LADDER → set OBJECT_X/Y via `distance_step_env_values` → train →
  `_sim_eval` task_success → `advance_distance` → relaunch w/ resume. Sim: relaunch subprocess. **HITL: dispatch a
  `physical-reset-agent` staging request (safety-gated) instead of an instant reset; gate = same binary predicate
  (sim termination-manager OR hardware `read_object_in_bin`/`manual_confirm`).** *Caveat surfaced by research:
  DISTANCE_LADDER step-0 (0.16,-0.10)=7.2cm is NOT actually easier than the 6.6cm plateau, and success_radius is
  hardcoded 0.04 — to bootstrap, the ladder needs a genuinely easier step-0 (~2-3cm) and/or a `LEROBOT_ISAAC_SUCCESS_RADIUS` knob it widens. (Lever B already uses die 3.5cm to address this.)*
- **Affordance/precondition gating (SayCan-style)** as an autonomy controller signal (`native-dual, small-code`).

## RESEARCH (bigger lifts — spike later)
- **Director-style manager/worker latent-subgoal hierarchy in DreamerV3** (`native-dual, large-code, plateau=YES`) —
  the worker's dense intrinsic reward = latent-distance-to-subgoal (`Director`/`LEXA`), which decomposes the
  long carry-place horizon. High-EV but large-code; revisit if intrinsic-reward + BC don't suffice.
- **LEXA explorer/achiever split**, **CEM/MPC test-time planning over the DreamerV3 latent** (re-ranked by the
  binary verifier — dual-mode), goal-autoencoder subgoal compression.

## Rejected
2 sim-only techniques (relied on privileged sim state / instant parallel resets — no HITL signal). Several
redundant with the existing autoresearch loop / curriculum_controller / outcome_verifier (don't rebuild).

## IMPLEMENTED 2026-06-21 (adopt-now code tier — committed, GPU-validation pending)
- **BC actor-loss wiring** ✅ — `behavior_cloning_loss` + DAPG decay + RLPD demo sampling now wired into
  sheeprl `dreamer_v3.train` via `_wm_isaac_entry._patch_bc_actor_loss` (commits: adapters `606971b`,
  workspace `6e7e32e`). Default OFF (`LEROBOT_ISAAC_BC_WEIGHT=0.0`); enable with `BC_WEIGHT>0` +
  `BC_DECAY_STEPS`. 14 CPU tests; grill-reviewed (OFF-path no-op verified, 3 ON-path fixes applied).
  **GPU-VALIDATED 2026-06-21** — the `cp-bc-smoke` run crossed `learning_starts` and ran the `_bc_step`
  actor-update past step 1024 with NO Traceback (all 5 sheeprl-integration risks cleared: rssm.dynamic
  sig, actor() unpack, fabric.backward, zero_grad interplay, cfg attrs; `decoupled_rssm:false` path).
  One cosmetic logging bug found+fixed (`a2dad55`: register `Loss/bc_loss`/`Params/bc_weight` in the
  MetricAggregator so bc_loss reaches TB). **Full BC run LAUNCHED** `cp-stage1-bc-20260621`: resume
  ckpt_15000 (lever-B place experience) + `BC_WEIGHT=0.3 BC_DECAY_STEPS=20000` + seed + 3.5cm start —
  testing the hypothesis that the continuous imitation gradient gives STABLE placing (vs lever B's
  oscillation 43↔300).
- **Distance-curriculum auto-relaunch driver** ✅ — `lerobot_isaac_autoresearch.curriculum_campaign`
  (commit autoresearch `dc22eea`): loops DISTANCE_LADDER, derives place-success from TB `ep_len_avg`,
  `advance_distance`, chains `resume_from`. 36 tests + `--dry_run`. **GPU validation PENDING** (live chained run).
- **Deferred** (next): Plan2Explore/RND intrinsic reward + sign/normalization guardrail (queue); Director (research).

## Recommended sequence (folds into the plateau lever queue)
B (running) → **wire BC actor loss + RLPD (adopt-now #1)** → Plan2Explore/RND intrinsic reward → build the
autonomous distance-curriculum driver → (research) Director hierarchy. Every step dual-mode by the principle above.

## Regression mechanism analysis (2026-06-21) — why placing isn't retained
Placing is reliably DISCOVERED (easier start) but not RETAINED (lever B + BC-0.3 both regressed). Analysis:
`Rewards/rew_avg = ep["episode"]["r"]` (episode return); placing return ≈ −3 (incl +5 success_bonus on
place_termination) ≫ timeout return ≈ −25 → **the objective FAVORS placing** (not a reward-design bug).
Root cause = **WM-imagination instability**: DreamerV3 trains the actor in imagination, the WM under-models
the RARE place event → imagined return doesn't credit placing → actor drifts off → regression. Fix family
(make place learnable by the WM, ranked): BC actor-loss (keep place data flowing — testing 0.3→0.6);
**`LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT` terminal bonus is OFF (0.0) — a missed lever** (salient place reward →
WM models it); prioritized place replay; Plan2Explore. **Best next lever after BC-weight = BC + PLACE_SUCCESS_WEIGHT.**

## Retention ladder progress (2026-06-21, autonomous)
- Lever B (easier start, no BC): placed → regressed +3500.
- BC-0.3 (resume ckpt_15000): placed → regressed +1500.
- BC-0.6 (resume clean ckpt_10000): placed → regressed +1500 (SAME point). Key: bc_loss magnitude small (~0.3) → BC gradient weak regardless of weight 0.3 vs 0.6. **BC-weight alone EXHAUSTED.**
- → BC-0.6 + **PLACE_SUCCESS_WEIGHT=10** (cp-stage1-bcps): attacks the WM-under-modeling root directly — a salient dt-invariant terminal place bonus (placing return -3→~+7) so the WM reward head models place strongly. + slower BC decay (30000). IN FLIGHT.
- If BC+PLACE_SUCCESS also regresses → Plan2Explore intrinsic reward (sheeprl p2e_dv3) or prioritized place-replay (both attack WM coverage of the rare place event).

## Carry-place retention — full ladder + the slide revelation (2026-06-22)
- B / BC-0.3 / BC-0.6 / BC+PS: all "placed" (ep_len→43, rew→-3) then regressed. rew capped at -3.1 across ALL.
- ROOT CAUSE found: place_termination was XY-only → the agent SLID the die in (rew -3, no lift); place_success(+50) needs a lift → never fired. The "placing" was a SLIDE ARTIFACT — true carry-place never learned.
- FIX (fd8e977): place_termination now requires the lift (LEROBOT_ISAAC_PLACE_REQUIRE_LIFT). Slides stop terminating → ep_len back to 300, CONFIRMING the agent only slid. rew moved to ~-14 (lift_shaping engaging = partial grasp+lift) then declined → agent reaches+pushes, won't discover the grasp→lift→carry→place chain.
- Bottleneck is now EXPLORATION of the grasp→carry chain (not retention). Cheap levers exhausted (BC weight 0.3/0.6, PS, lift-gate, easy start, seed).
- LAST CHEAP LEVER: strong persistent BC (weight 1.5, ~no decay) to maximally inject the demos' full chain (cp-stage1-bcstrong). IN FLIGHT.
- If strong-BC fails → next is BIGGER: Plan2Explore integration (sheeprl ships p2e_dv3 but adapter hardcodes exp=dreamer_v3 + BC/seed monkeypatches target dreamer_v3 → real integration, not a config swap), OR a grasp-first sub-curriculum / more+better demos. Research-scale; consolidate before committing more GPU.
