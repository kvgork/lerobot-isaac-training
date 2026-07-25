# ACT-on-real campaign — extensive plan (current run → video_backend → after)

**2026-06-28.** Goal: a *working* SO-101 pick-place policy on the **real arm** (world-model is the
longer-term technique; "working" is the bar). Route A = BC (ACT) on real human demos → deploy on
hardware — the proven path (HF blog: ~90%). This plan sequences the current ACT run, the
video_backend speedup, and the branching runs after, with explicit go/no-go gates.

> Provenance: `/orchestrate`, rigor=agentic. Builds on this session's fixes — reward now flows
> real-data → parquet → bridge → WM; recorder/bridge/quality refactors landed + pushed.

## 0. Current state (what's true now)
- **Data:** `datasets/local/so101-pickplace-new` — 50 human-teleop eps, **48 success / 2 fail**
  (ep 0, 34), overhead cam **480×640** stored as **av1 video**, 12-dim state, 6-dim action,
  **reward now in parquet** (sparse_terminal_success, backfilled) + episode_labels sidecar.
- **Bottleneck:** training is **decode-bound** — av1 video decode dominates (`data_s≈1.7s/step`,
  GPU ~12% util). Can't `--cache_frames` (17 GB > free RAM).
- **Pipeline:** `lerobot-isaac-train --target_arch act|smolvla|diffusion` (uncached needs the env
  bin on PATH for the `lerobot-train` console script); `--successes_only` reads the sidecar.
- **Eval reality:** a real-data policy can only be judged on the **real arm** (`lerobot-rollout`);
  sim-eval ≈0 (sim2real gap — memory). No hardware in this loop ⇒ training produces a checkpoint;
  the human deploys/evaluates.

---

## RUN 1 — current: ACT on real (in progress)
- **Cmd:** ACT, `--successes_only` (48 eps), `--steps 15000 --batch_size 8 --lr 1e-5`,
  `--num_workers 12`, uncached, `video_backend=pyav`. Output `outputs/act_real_so101`.
- **Status:** stepping; loss 6.8→2.5; `data_s≈1.7`, ~0.53 step/s → **ETA ~7–8 h**.
- **Outputs:** ACT checkpoints (`checkpoints/{010000,015000}/pretrained_model`).
- **Gate (human, on hardware):** deploy last ckpt via `lerobot-rollout --policy.path=… --robot.type=so101_follower …`; success-rate over ~20 real episodes. `>0` = BC-on-real works; the blog's regime is 60–90% depending on data.
- **Caveats:** 50 eps is the blog's *marginal* size (their 90% needed 150+ diverse). Expect
  modest first-pass success; this run is the **baseline + pipeline proof**, not the final policy.
- **Decision:** if checkpoint trains cleanly (loss converges) → proceed to deploy + Run 2/3.

## RUN 2 — video_backend speedup (torchcodec)
- **Why:** Run 1 is decode-bound (`data_s` dominates, GPU idle). torchcodec 0.10.0 **is installed**
  (GPU-accelerated decode) → should cut `data_s` sharply, enabling more steps / faster iteration
  for all subsequent real-data runs (this is infrastructure, not a one-off).
- **What:** relaunch ACT with `-- --dataset.video_backend=torchcodec` (instead of pyav). Keep
  everything else identical so it's a clean A/B on throughput.
- **How / acceptance:**
  1. Short probe: 500 steps, compare `data_s` (pyav≈1.7 vs torchcodec). Expect a meaningful drop
     (target `data_s < 0.5`, GPU util up).
  2. If faster + loss curve matches Run 1 at equal steps → adopt torchcodec as the default backend
     for the adapter's uncached policy path (small adapter change: default `video_backend`).
  3. If torchcodec errors (codec/GPU support) or is slower → keep pyav; fall back to fewer steps.
- **Risk:** torchcodec av1 GPU-decode support varies by build; pyav is the safe fallback. Probe first.
- **Outcome:** either a faster default (re-run Run 1 quickly at more steps) or confirmation pyav is best.
- **RESULT (2026-06-28): REFUTED — torchcodec ≈ pyav.** `data_s 1.745` (torchcodec) vs `1.70` (pyav),
  GPU ~10% both. The bottleneck is **av1 random-frame access** (seek+decode per frame), not the codec
  lib — backend swap doesn't help. No libavutil crash (it ran), but no speedup. **Keep pyav default.**
  Real decode speedups need a different lever: **downsize images** (480×640 → ~128–256, re-encode the
  dataset; cuts decode + GPU) or **store frames as individual images** (not video) so workers stop
  seeking. The current torchcodec run was left to finish (equal speed, error-free).

---

## RUN 3+ — the branching tree (gated)

```
Run1 ACT(48ep) ──train ok──> [HW eval] ──success>0?──┐
                                                       ├─ YES → Run 3a: scale data + retrain
Run2 torchcodec ──faster?──> adopt backend            └─ NO  → Run 3b: diagnose + alternative policy
```

### Run 3a — scale + harden (if Run 1 shows promise on hardware)
- **More data (highest leverage):** collect **150+ diverse** real demos (phospho/lerobot recorder)
  — rotation ±45° yaw, multiple object/container, stratified bins (HF-blog recipe). Re-run ACT
  (fast now via torchcodec). Target the blog's 90% regime.
- **Eval discipline:** held-out OOD bin; track success during data growth.

### Run 3b — alternative policy (if ACT underperforms)
- **SmolVLA** on the same real data (`--target_arch smolvla --policy.load_vlm_weights=true`,
  `--cache_frames` feasible at its smaller footprint? still 480×640 — likely uncached+torchcodec).
  VLAs often beat ACT on multi-modal real data; lerobot-native.
- **Diffusion** revisit (now uncached+torchcodec viable; earlier blocked only by the cache-wrapper).
- **temporal ensembling** at eval for ACT (re-query every step; we proved the toggle works).

### Run 4 — world-model track (now that reward flows)
- **Recorded-first WM baseline:** bridge real data (reward-carrying HDF5, done) → DreamerV3 offline.
  Gate = *real* metric, NOT recon_loss (recon is decoupled). Treat as representation/dynamics check.
- **Route C (genuine WM policy):** DreamerV3 **online on the real arm** (real reward, HIL-SERL-style)
  — the only way to a *working world-model* policy that transfers. Needs hardware-in-loop + the
  existing `rollout-executor` + `physical-reset-agent` + safety. Research effort; scope after Run 3.

---

## Decision gates (summary)
| run | gate metric | pass → | fail → |
|-----|-------------|--------|--------|
| 1 ACT train | loss converges | deploy + Run 2 | debug data/config |
| 1 HW eval | real success-rate >0 | Run 3a (scale data) | Run 3b (alt policy) |
| 2 torchcodec | data_s drop + loss match | adopt default | keep pyav |
| 3a scaled | real success ≥ blog band (60–90%) | ship | iterate data/diversity |
| 4 WM offline | (representation only) | Route C scoping | — |

## Risks / open leads
- **Decode wall** — even at 12 workers `data_s≈1.7`; torchcodec (Run 2) is the fix; else downsize images.
- **50-ep marginal** — single biggest lever is more diverse real data (Run 3a).
- **ggando IK-frame / gripper-offset** — a lead for *sim* grasp, not this real-BC track; parked.
- **sim2real** — blocks the sim-WM route; only real-hardware RL (Route C) sidesteps it.
- **No hardware in this loop** — every eval/deploy gate needs the physical SO-101; training is all that runs here.

## Related
- `[[2026-06-28-working-policy-next-steps]]` (routes overview) · `[[act-so101-training-research]]` ·
  `[[so101-rl-lift-and-phospho-research]]` · `[[nvidia-sim-to-real-so101-research]]`
- memory: `[[carryplace-cup0-warmstart-r4-result]]`

---

## CONTINUATION — autonomous campaign (2026-07-04, `/orchestrate` fully-autonomous, GPU free)

**State discovered:** RUN 1 (`outputs/act_real_so101`) **completed cleanly (`rc=0`) but only at
5000 steps** (not the plan's 15000) — ~2.3 epochs over 48 eps, loss 6.8→**0.390**, undertrained.
`image_transforms.enable: False` (no aug). Checkpoint `005000` is complete + resumable. RUN 2
(torchcodec) already REFUTED. RUN 3a/3b + RUN 1 HW-eval need the physical arm → **not doable in
this loop**. So the autonomous forward motion = strengthen candidates + WM-offline baseline.

**Autonomous run queue (single GPU, sequential):**
1. **ACT-15k** (in progress) — re-run ACT fresh to **15000 steps**, same RUN-1 config (batch 8,
   lr 1e-5, seed 42, pyav, nw12, aug off, `--successes_only`). Output `outputs/act_real_so101_15k`.
   Chosen over fine-tune because the adapter forces `--policy.type=act` which conflicts with
   `--policy.pretrained_path` in lerobot 0.5.1. Clean + reproducible + matches plan's literal spec.
   *Primary deploy candidate.* Gate: loss converges → hand ckpt to human for `lerobot-rollout`.
2. **SmolVLA-real** — alt candidate (Run 3b hedge), same 48 eps, `load_vlm_weights=true`, uncached.
3. **WM-offline** — Run 4 representation baseline (bridge → DreamerV3 offline), lowest priority.

**Hard constraint:** every eval/deploy gate needs hardware; this loop only produces checkpoints.
Session `20260704-083706-act-real-campaign`.

### HW EVAL (2026-07-04, arm live) — ✅ FIRST REAL-ARM SUCCESS

RUN-1 gate **"real success-rate > 0" HIT**: ACT-15k did a full **successful pick-and-place** on the
physical SO-101 (reach→grasp→lift→carry→release). **BC-on-real validated.**

- **Deploy path reality:** `lerobot-rollout` does NOT exist — real path is the `robot-data-runner`
  family (`robot-data-run-check` / `robot-data-run` / `robot-data-run-eval`). Installed via
  `pixi run sync-runner` + `pip install -e src/robot-data-runner`; needs `lerobot[feetech]`.
- **Two latent bugs in `robot-data-runner/mappers.py` fixed** (never exercised before — first deploy):
  state was built from `sorted(.pos)` (PERMUTED vs canonical joint order) and was 6-dim vs the
  policies' 12-dim (pos+zero-vel). Fix = `resolve_state_mapping` (canonical order + std-verified
  zero-pad) applied to BOTH `run_policy` + `run_episodes`; 19/19 tests pass. Adversarially reviewed.
  NOT committed (src/robot-data-runner is its own repo).
- **Working real-arm cmd:** `robot-data-run --policy-path outputs/act_real_so101_15k/checkpoints/015000/pretrained_model
  --dataset-root datasets/local/so101-pickplace-new --port /dev/ttyACM0
  --camera overhead=/dev/video4,640,480 --id so101_follower --execute --max-relative-target 5.0
  --rate-hz 30 --duration-s 20 --home-on-exit` (D435 RGB=/dev/video4 = overhead; 30Hz = training fps).
- **Key finding:** ACT×50-demos = NARROW spatial tolerance — replays ~trained-mean trajectory;
  succeeds only with the die near its trained position + overhead framing matched to training
  (diagnosed by live-vs-training frame overlay). **Top lever = 150+ diverse demos** (HF-blog 90%
  regime). SmolVLA not yet HW-tested. Genuine WM policy = Route C (DreamerV3 online on real arm).

### CAMPAIGN RESULT (2026-07-04 16:25 — all 3 phases complete)

| phase | run | result | artifact |
|-------|-----|--------|----------|
| 1 | **ACT-15k** | ✅ rc=0, final loss **0.168** @ 15000 steps (6.89 ep; vs undertrained 0.39@5k) | `outputs/act_real_so101_15k/checkpoints/015000/pretrained_model` (206MB) |
| 2 | **SmolVLA** | ✅ rc=0, final loss **0.119** @ 20000 steps (4.59 ep, lr decayed, no OOM @batch4, ~4.5 step/s) | `outputs/smolvla_real_so101/checkpoints/020000/pretrained_model` (906MB) |
| 3 | **WM-offline** | ⚠️ baseline: bridge+offline DreamerV3 path **works** (grad updates, no crash) but full 30k impractical (~29h @ replay_ratio=1); stopped at policy_step 1800 | `outputs/wm_offline_real/dreamerv3_data.hdf5` (490MB, reward-carrying — reusable) |

**Two deploy candidates ready for the ONE gate that matters — real-arm eval** (`lerobot-rollout --policy.path=<ckpt> --robot.type=so101_follower …`, ~20 eps each). ACT vs SmolVLA on hardware decides Run 3a (scale data, blog 90% regime) vs 3b (already have the alt). Per plan: 50 eps is marginal — **more diverse real data (150+) is the top lever** for either.

**WM track:** reward-carrying HDF5 now exists (offline WM representation baseline path validated). A genuine WM *policy* still needs Route C (DreamerV3 online on the real arm, HIL-SERL) — hardware-in-loop, unchanged.

**Ops lessons (saved to memory):** long autonomous GPU jobs must be `setsid`-detached (harness `run_in_background` killed by `/remote-control` at step 3532); never `pkill -f` a pattern matching own shell (exit 144); verify via log file not self-matching pgrep. Warm page-cache (from ACT's run) made SmolVLA decode-free (data_s 0.045 vs 0.87). See `[[detach-long-training-jobs]]`.
