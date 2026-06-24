Now I have everything. One critical viewpoint finding: the env docstring claims sim d435 is "wrist-mounted" but the real dataset uses key `observation.images.overhead`. So real = overhead view, sim demos = d435 wrist view — confirmed a **viewpoint mismatch**, not just resolution. This is exactly the hard incompatibility the critique flagged. The demo-warmstart pipeline (Stages 1-3 built, GPU-verified) is the genuinely viable path.

I have enough verified ground truth to write a recommend-don't-enumerate plan. Writing it now.

# World Model Training Plan for SO-101 — Recorded-First, Sim-Enhanced

**Status:** 2026-06-23. Verified against repo datasets, bridge skill, env obs terms, and the existing warm-start pipeline. This plan supersedes the broad multi-architecture landscape with one verified, scoped path.

> **Provenance:** produced via the `/orchestrate` master-project-orchestrator pipeline (rigor=agentic; planning workflow `wf_a4880af4-f74` — 4 repo+web gatherers → adversarial critic [verdict RECONSIDER, folded into §6] → synthesis). **Strategic backbone:** the user's own vault notes — `05-Wiki/research/2026-06-23-world-model-pipeline-practical-plan.md` (the recorded/generated/both build plan) + `05-Wiki/synthesis/2026-06-23-world-model-training-pipelines.md` (the 7-stage spine). This doc OPERATIONALIZES those for this exact repo + **reconciles two places where repo reality overrides the vault's general advice** — see §7.

---

## 0. Verified ground truth (read before trusting any prior plan)

These were confirmed by reading `meta/info.json` and the actual parquet columns/env code — they overturn several assumptions in the gathered research:

| Fact | Verified value | Source |
|------|----------------|--------|
| Real `so101-pickplace-new` state | **12-dim = joint_pos[6] + joint_vel[6]** (names present) | `datasets/local/so101-pickplace-new/meta/info.json` |
| Real image | key `observation.images.overhead`, **HWC (480,640,3)**, dtype `image` | same |
| Sim `so101-sim-pickplace-demos-op3` state | **13-dim = joint_pos_rel[6] + object_pose[7]** (NO velocity) — decoded from column ranges: dims 6–8 = object XYZ (dim 8 = height 0.008–0.117), dims 9–12 = quaternion | parquet column read |
| Sim image | key `observation.images.d435_rgb`, **CHW (3,64,64)**, dtype `image` | `meta/info.json` |
| Camera **viewpoint** | real = **overhead** fixed cam; sim demos = **d435 wrist-mounted** (env docstring `observations.py:19`, `so101_env_cfg.py:166`) | code |
| Bridge `lerobot_to_worldmodel` | **DOES** accept `state_keys=` / `image_keys=` overrides; `image_keys=[]` = state-only | `~/.claude/skills/lerobot_world_model_bridge/operations.py:111-213` |
| Adapter `wm_dreamerv3._convert_dataset` | **hardcodes** `image_size=(64,64)`, auto-picks first camera, **never passes state_keys** — overrides unreachable via `lerobot-isaac-train` | `targets/wm_dreamerv3.py:139-152` |
| `le_world_model` real training | BLOCKED (lerobot 0.5.x has no `train_world_model` CLI); in-process `_lewm_minimal` is a toy CNN | CLAUDE.md + `targets/wm_leworldmodel.py` |
| Demo-warmstart pipeline | Stages 1–3 **built + GPU-verified** ("SEEDED 38 demo episodes", rc=0); Stage 4 = launch command ready | memory `demo-warmstart-pipeline`, `plans/2026-06-11-demo-warmstart-plan.md` |

**The two consequences that reshape this plan:**

1. **Real and sim cannot be co-trained into one offline corpus.** They share only `joint_pos[6]`. The non-overlapping halves are *velocity* (real-only) vs *object pose* (sim-only), and the cameras are different *viewpoints* (overhead vs wrist) — unfixable by resize. `merge_datasets` validates every frame shape against one `info.json` and will reject every cross-schema frame; its dedup even hashes a `wrist` key neither dataset has. **A naive merge does not run.** Any plan whose centerpiece is "merge recorded + sim → one HDF5" is dead on arrival.

2. **An offline reconstruction WM on success-only real data is already known useless for control** (`plans/2026-06-07-good-world-model-plan.md`: recon_loss dropped fine, but the offline `hdf5_env` never reads proprio, uses only the first 16 frames/episode, and reward≡0). Re-running it as the headline deliverable optimizes a metric with no link to the actual goal.

---

## 1. Goal + the architecture decision

**Goal:** Produce a *controllable* world model for SO-101 pick-place — one whose imagined rollouts are good enough to train (or unstick) a policy that discovers the **place** step. A WM that only reconstructs expert pixels does not serve this goal.

**Architecture decision: DreamerV3 (RSSM via sheeprl), trained ONLINE in Isaac sim, warm-started from the recorded + demo data.** Recommended over the alternatives:

- **HF LeWorldModel — rejected (blocked).** lerobot 0.5.x ships no `train_world_model` CLI; the in-process `_lewm_minimal` is a CNN+linear toy with no actor-critic, no reward, no imagination. It cannot break a policy plateau. Revisit only if upstream lands the CLI.
- **V-JEPA 2 / Cosmos / DreamerV4 — rejected (not wired, OOM, or research-grade).** V-JEPA 2-AC needs a ViT-L (OOM on 10 GB) and ~16 s/action planning; Cosmos 2B OOMs even quantized; DreamerV4 has only an unofficial impl and no bridge. None are reachable this quarter on an RTX 3080.
- **DreamerV3 — chosen.** It is the only WM in this repo that is (a) fully wired (`env=isaac_so101`), (b) fits 10 GB at `image_size=64`, batch 8, num_envs=1, (c) trains a WM *and* a policy by imagination from a reward, and (d) has a **built, GPU-verified warm-start path** that injects the recorded/demo data as replay-buffer seeds — exactly the "recorded-first" mechanism, without the impossible merge.

The phrase "recorded-first, then enhance with generated" maps onto DreamerV3 as: **recorded/demo data seeds the replay buffer (the prior), online sim rollouts supply failures + exploration + reward (the enhancement)** — *not* as two corpora merged into one HDF5.

---

## 2. Data strategy — how "recorded-first, sim-enhance" actually works here

### 2.1 The schema reality (and why merge is the wrong frame)

```
REAL  so101-pickplace-new : state[12]=jpos6+jvel6   img=overhead  HWC 480x640   reward in HDF5 sidecar, success-only
SIM   demos-op3 / demos    : state[13]=jpos6+objpose7 img=d435      CHW  64x64    scripted demos, ~48% success
ONLINE env=isaac_so101     : state=jpos6 (+objpose7 opt) img=d435   CHW  64x64    LIVE reward, failures FREE
```

Real and sim agree only on `joint_pos[6]` and on the *task*. They disagree on the other 6 state dims, the camera viewpoint, and the resolution. So we do **not** unify them into one training file. Instead:

- **Online sim is the WM's training distribution** (single consistent schema, reward-bearing, generates failures).
- **Recorded + scripted-demo trajectories are injected as warm-start priors** via the existing `_patch_seed_demo_buffer()`, which already adapts demo `state[12]→[6]` to the online schema (memory `demo-warmstart-pipeline`). This is the proven seam.

### 2.2 The one place recorded data trains a WM directly (Stage 1, scoped)

A recorded-only DreamerV3 WM is still worth one short run — **but only as a dynamics/visual sanity baseline and warm-start candidate, not as the deliverable**, and only if we fix the offline env's known defects (proprio never read; first-16-frames-only). Use the bridge **directly** (not `lerobot-isaac-train`, which hardcodes the wrong args) so we can pass `state_keys` and the `overhead` camera explicitly:

```python
from skills.lerobot_world_model_bridge.operations import lerobot_to_worldmodel
lerobot_to_worldmodel(
    dataset_path="datasets/local/so101-pickplace-new",
    output_path="outputs/hdf5/real_overhead_dreamerv3.hdf5",
    output_format="hdf5", image_size=(64, 64),
    image_keys=["observation.images.overhead"],   # explicit — the only camera
    state_keys=["observation.state"],              # full 12-dim, NOT auto
    normalize_actions=True,
)
```

This is the boundary of what recorded data does on its own. It cannot model where the die goes (state[12] has no object pose; the overhead image must carry it), and it has zero failures — hence it is a baseline, not a controller.

### 2.3 Mixing verdict: warm-start (sequential), NOT co-train

Given the schema/viewpoint incompatibility, **pretrain→finetune (warm-start)** is the only viable mix. Co-training a single RSSM on both manifolds would, per the latent-separation failure mode (arXiv:2506.12735) and the viewpoint gap, learn two disjoint manifolds anchored to neither. The warm-start path sidesteps this entirely: recorded/demo transitions seed env-0's buffer, online rollouts dominate thereafter.

---

## 3. Staged roadmap

### Stage 0 — Sanity (½ day, no GPU) — RUNNABLE NOW
**Objective:** Lock the verified schema into the bridge call and confirm frame decode.
- Run the §2.2 bridge call on **5 episodes** (`episodes=[0..4]` if the API supports it, else full).
- Inspect output: `h5py` → assert `obs/image (T,64,64,3) uint8`, `obs/state (T,12) float32`, `actions (T,6)`.
- **Go/No-Go:** state width == 12 (not 1, not 6); image non-empty (guards the cv2-vs-PIL decode path — real data is dtype `image`/PNG, so the PIL path `_load_episode_frames_from_parquet` must fire, not the MP4/cv2 path).

### Stage 1 — Recorded-only WM baseline (1 GPU run, ~2 h) — RUNNABLE NOW (after a small adapter fix)
**Objective:** A visual+dynamics baseline and a candidate warm-start checkpoint. **Not** the deliverable.
**Needs-building (small):** make `wm_dreamerv3._convert_dataset` honor `--camera_key` / `--state_keys` (or just call the bridge offline per §2.2 and pass the resulting `.hdf5` to `--dataset`, which the adapter already accepts).
```bash
pixi run -e train-dreamer python -m lerobot_isaac_adapters.train \
  --target_arch dreamerv3 \
  --dataset outputs/hdf5/real_overhead_dreamerv3.hdf5 \
  --output_dir outputs/wm_real_baseline \
  --steps 50000 --batch_size 8 --lr 1e-4
```
**Inputs:** `real_overhead_dreamerv3.hdf5`. **Outputs:** checkpoint + `train.log`.
**Go/No-Go (measurable):** `recon_loss` decreases monotonically and plateaus < ~0.05 by 50k steps (the *training-time* sheeprl metric, parsed by `_RECON_LOSS_RE`). This proves the visual encoder learns the overhead distribution. **Do not interpret a low recon_loss as control quality** — it is a pipeline/encoder check only. If recon_loss never drops, the decode path or HDF5 is wrong — go back to Stage 0.

### Stage 2 — Online DreamerV3 in Isaac, demo+real warm-started (THE deliverable, multi-hour) — RUNNABLE NOW
**Objective:** A controllable WM with failure + exploration coverage and a live reward. This is where "sim enhances" happens.
**Mechanism:** `env=isaac_so101` generates the training distribution; `_patch_seed_demo_buffer()` (built, GPU-verified) seeds env-0's replay buffer from the scripted sim demos (and, optionally, the recorded trajectories adapted 12→6). Launch via the existing Stage-4 command:
```bash
STEPS=50000 BATCH_SIZE=8 SESSION_ID=wm-warmstart-v1 \
  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 \
  LEROBOT_ISAAC_OBJECT_X=0.18 LEROBOT_ISAAC_OBJECT_Y=0.05 \
  LEROBOT_ISAAC_STAGED_REWARD=1 \
  LEROBOT_ISAAC_DEMO_DATASET=datasets/local/so101-sim-pickplace-demos \
  bash scripts/_run_wm_isaac_overnight.sh
```
Optionally warm-start the WM weights from the Stage-1 checkpoint. **Caveat (verified 2026-06-24):** the `checkpoint.exploration_ckpt_path=` hook in `wm_dreamerv3.run` is appended ONLY when the sheeprl exp name ends with `_finetuning` (the P2E finetuning path, `wm_dreamerv3.py:216`) — it is NOT wired for the default `dreamer_v3` exp used in the launch command above. To resume a plain dreamer_v3 WM, use sheeprl's native `checkpoint.resume_from=` (confirm semantics) rather than this flag.
**Knobs (from the 2026-06-16 wm-vla playbook — Stage 2's failure surface IS entropy-collapse + sparse-credit, exactly what these target):** actor `ent_coef 1e-3` (5–10× the sheeprl default — beats the reach+grip entropy-collapse), `horizon 25`, `replay_ratio 16`, `kl_free 1.0`, `demo_ratio 0.5` (seed lift transitions), `batch 16` if VRAM allows (else keep 8 on the 10 GB 3080). Earlier repo runs used `3e-4 / horizon 15 / replay_ratio 2` — undershoots; raise them. See `[[2026-06-16-wm-vla-training-playbook]]`.
**Constraints:** num_envs=1 (the `is_first` bug at >1 is unfixed — see memory; note the `IsaacSO101VectorEnv` + "Fix 2" vector patch ARE built and invoked at `_wm_isaac_entry.py:114,721`, so >1 is attempted-but-crashing, not absent); ~0.5–1.2 steps/s (camera-bound); batch 8 fits 10 GB, drop to 4 on OOM. learning_starts ≥ **num_envs×seq_len** (=64 at num_envs=1, seq_len=64; would be 256 at 4×64 — do NOT state the rule as merely ≥ seq_len).
**Inputs:** sim env + demo dataset (+ optional Stage-1 ckpt). **Outputs:** WM+actor checkpoint, episode-return curve.
**Go/No-Go (measurable, the real one):** episode reward climbs **past the −10.6 carry-place plateau** toward a place-bearing return, AND `scripts/_sim_eval.py` reports a nonzero **place/task-success rate** (not just reach). This is the metric that matters; recon_loss is secondary here.

### Stage 3 — Sim scale-up for coverage (DR / more demos) — PARTIALLY RUNNABLE
**Objective:** If Stage 2 reward moves but stalls below success, expand the *initial-state* and *failure* coverage the WM sees.
- **Generate more scripted demos** at varied object x/y to broaden the seed buffer: `scripts/_gen_sim_demos.py --episodes N` (built).
- **DR replay** (`lerobot-isaac-synthetic`) to vary lighting/pose on the *sim* (single-schema) data only — never mixed with the overhead real data.
- **MimicGen — deferred** (Phase 4b, gated by `LEROBOT_MIMICGEN_ENABLED=1`; joint→EE calibration for SO-101 not done). Do not start here.
- **Plan2Explore (if reward moves but exploration stalls / entropy collapses despite raised `ent_coef`)** — ensemble-disagreement intrinsic reward (sheeprl, K≈5), injected at `player.get_actions` before `rb.add` (memory: `sheeprl-action-override-buffer-seam`; the `p2e_dv3_finetuning` exp path exists). **Temper expectations** — DreamerV3-XP (2025) finds disagreement gains *modest* vs prediction-error replay prioritization; try the prioritization first. The clean `--expl_behavior` flag is DreamerV2-era; current `danijar/dreamerv3` restructured it. LeRobot has no native curiosity trainer. See `[[2026-06-21-rl-reward-coupled-world-model-training]]`, `[[Plan2Explore]]`.
**Go/No-Go:** seed-buffer success fraction and Stage-2 place-rate both rise vs the Stage-2 baseline; if reward regresses, the added sim data is poisoning — cut it.

---

## 4. Success metrics + proving the enhancement is a net win

| Stage | Primary metric | Where it comes from | "Net win" bar |
|-------|----------------|---------------------|---------------|
| 1 | `recon_loss` (training) | sheeprl stdout `recon_loss=` | plateau < ~0.05; pipeline/encoder OK only |
| 2 | **episode return + place-rate** | online reward curve + `scripts/_sim_eval.py` (uses termination-manager verdict, per memory `sim-eval-terminal-pose-autoreset`) | return passes −10.6 plateau; place-rate > 0 |
| 3 | place-rate vs Stage-2 | same sim eval | place-rate strictly improves; else revert |

**How to prove sim *enhances* rather than just adds noise:** run Stage 2 **with** demo seeding vs an identical **no-seed** control (`LEROBOT_ISAAC_DEMO_DATASET` unset). The warm-start is a net win iff seeded reaches a higher place-rate (or the same place-rate in fewer steps). This A/B is the honest test — not a recon_loss delta, which the repo has already shown is decoupled from control quality. Reject the "compare holdout recon_loss" gate from the eval-track research: it optimizes a metric with no demonstrated link to place discovery.

---

## 5. Risks + mitigations; runnable-now vs needs-building

| Risk | Reality | Mitigation |
|------|---------|------------|
| **Schema mismatch real↔sim** | Confirmed hard: 12 vs 13 dims, overhead vs wrist view, 480×640 vs 64×64 | Never merge. Use warm-start seeding (12→6 adapter, built) + online single-schema training. |
| **Sim visual distribution shift** | sim 64×64 wrist render ≠ real overhead optics | Irrelevant to Stage 2 (sim-only training). Matters only if/when deploying to real — separate sim2real problem (memory: real-trained policy ≈0 in sim). |
| **le_world_model blocked** | lerobot 0.5.x has no CLI | Use DreamerV3. Don't burn compute on `_lewm_minimal`. |
| **GPU 10 GB** | batch 8 @ 64² fits; num_envs>1 crashes (`is_first` bug) | num_envs=1; batch→4 on OOM; image stays 64². |
| **Offline WM looks "done" via recon_loss** | recon_loss 0.000316 already hit, judged meaningless | Treat Stage 1 as a baseline only; gate the project on Stage-2 place-rate. |
| **`merge_datasets` / invented flags** | `--dataset.weights`, `wm_dreamerv3_isaac_online.yaml`, `lerobot_isaac_synthetic.merge` do **not** exist as written | Ignore them; use the verified §3 commands. |

**Runnable now:** Stage 0 (bridge call), Stage 1 (after the small adapter fix or via direct `.hdf5` input), Stage 2 (the Stage-4 warm-start command — fully built/verified), Stage 3 demo-gen + DR replay.
**Needs-building (small):** plumb `--camera_key`/`--state_keys` through `wm_dreamerv3._convert_dataset` (1 file, `targets/wm_dreamerv3.py:139`); optional: verify sheeprl WM-resume from a Stage-1 ckpt.
**Needs-building (large, defer):** MimicGen joint→EE calibration; num_envs>1 vectorization; any real-data co-training (not recommended at all).

---

## 6. Relation to the paused carry-place RL

**Does a better WM help the place-wall plateau? Partially — and only in combination with the reward (termination) fix.** The memory trail is explicit that the plateau is **not primarily a dynamics-model-quality problem**: it is (a) the success-termination REACH bug (episode ends on EE-to-object < 5 cm, so the agent is never rewarded for placing — `success-termination-reach-bug`), and (b) sparse-reward exploration from a cold start (`carryplace-place-wall-plateau`). **The earlier "grasp feasibility" framing is RETRACTED (2026-06-23 PM):** scripted grasp is CONFIRMED working ~80% pick-carry-place (ee↔die constant 0.096 thru lift+carry); the wall is **purely RL exploration**, not grasp control (`scripted-grasp-infeasible`, retraction note in `carryplace-place-wall-plateau`).

Therefore:

1. **Fix object-in-bin termination FIRST.** A perfect WM on a mis-specified reward (one that terminates on reach) will still never learn to place. This is the highest-leverage work and is independent of WM quality. (Grasp is NOT a blocker — scripted grasp works ~80%, see retraction above.)
2. **Then the warm-started DreamerV3 (Stage 2) directly attacks the exploration half** of the plateau: the demo seeding injects place-bearing trajectories the cold-start agent never reaches on its own, and imagination amplifies them. This is the legitimate WM contribution.
3. **Net:** the WM is a *complement* to, not a *substitute* for, the reward (termination) fix. Sequence them: object-in-bin termination fix → Stage-2 warm-start → measure place-rate. If place-rate is still 0 after both, the bottleneck is **RL exploration of the place step** (the scripted controller already grasps+carries ~80%) — escalate to more/varied demo seeding or Plan2Explore (Stage 3), **not** a grasp-mechanics rework.

**Files/paths referenced:** `datasets/local/so101-pickplace-new`, `datasets/local/so101-sim-pickplace-demos[-op3]`, `~/.claude/skills/lerobot_world_model_bridge/operations.py`, `src/lerobot-isaac-adapters/.../targets/wm_dreamerv3.py:139`, `scripts/_run_wm_isaac_overnight.sh`, `scripts/_gen_sim_demos.py`, `scripts/_sim_eval.py`, `plans/2026-06-07-good-world-model-plan.md`, `plans/2026-06-11-demo-warmstart-plan.md`.

---

## 7. Reconciliation with the vault practical-plan (where repo reality overrides general advice)

The vault notes (`world-model-pipeline-practical-plan`, `world-model-training-pipelines`) are the strategy; this
plan is its repo-grounded instantiation. Two vault recommendations are **overridden by verified repo facts**, and
several vault insights are **adopted**:

**OVERRIDDEN by repo ground truth:**
1. **Vault: "Plan C — co-train one WM on real+sim at ~5–10% real / 90% sim, `source`-tagged, sweep the ratio."**
   That assumes real+sim share a schema (one merged, source-tagged LeRobotDataset). **Repo reality (§0): they
   do NOT** — different camera *viewpoint* (overhead vs d435-wrist), state (joint_vel vs object_pose), and
   resolution; `merge_datasets` rejects cross-schema frames. So co-training is **not runnable as-is** → this plan
   uses **warm-start (sequential, §2.3)** instead, which IS built+GPU-verified. The vault's co-train becomes
   viable ONLY after a *needs-building* step: rebuild the sim obs to MATCH the real (overhead cam + matching
   state) per `2026-05-16-real-to-sim-isaac-from-lerobot-data` — a larger effort, deferred. **Your "recorded-first
   → enhance" intuition is the correct one for this repo** (warm-start), not the vault's ratio-co-train.
2. **Vault: "don't target Isaac on the 3080 — use MuJoCo/ManiSkill3."** Repo reality: Isaac Lab DreamerV3 **does
   run here** (~0.5–1.2 steps/s, camera-bound, num_envs=1). It's the *wired, verified* path → use it for Stage 2
   now. ManiSkill3 (128 envs = 3.5 GB, ~30k FPS) is a strong **future** option for *bulk sim coverage* (Stage 3)
   but is a port (needs-building); don't block Stage 2 on it.

**ADOPTED from the vault:**
- **"Coverage > demo purity for a dynamics WM"** → this is exactly why Stage 2 (online sim rollouts: failures +
  exploration + reward) is THE deliverable, not the success-only recon baseline (Stage 1). Do NOT aggressively
  SAL/TED-filter the seed demos; keep failure transitions.
- **"Eval on real held-out only"** applies to *sim2real deployment* (a later goal). For the *sim* carry-place
  task here, eval is sim place-rate (`_sim_eval.py`); add a real held-out probe only when targeting the real arm.
- **LeWM (15M) "genuinely fits the 3080"** — true via its **standalone repo** (le-wm.github.io), NOT the blocked
  `le_world_model` lerobot adapter. A cheap-latent-WM / probing option to revisit after Stage 2 if a
  non-control representation is wanted.
- **MimicGen 10→1000** — the vault's generated engine = this plan's deferred Stage 3 (needs SO-101 joint→EE calib).

## Related (vault)
- `[[2026-06-23-world-model-pipeline-practical-plan]]` — the recorded/generated/both build plan (backbone)
- `[[2026-06-23-world-model-training-pipelines]]` — the 7-stage-spine survey
- `[[2026-06-16-wm-vla-training-playbook]]` — knob-level DreamerV3/LeWM/SmolVLA recipes
- `[[2026-05-16-real-to-sim-isaac-from-lerobot-data]]` — real→sim rebuild (prerequisite for the co-train path)
- repo: `plans/2026-06-23-carryplace-cup-campaign.md` (the paused RL this WM complements; see §6)