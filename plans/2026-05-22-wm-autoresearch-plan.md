# World-Model Autoresearch — Plan

**Date:** 2026-05-22
**Goal:** Make the bash autoresearcher (the deterministic, no-LLM pattern
established for LoRA in `scripts/_run_autoresearch_lora.sh`) also work for
world-model training (DreamerV3 first; LeWorldModel later when upstream
unblocks).
**Companion docs:**
- `plans/2026-05-19-lora-autoresearch-plan.md` (parent pattern)
- `plans/2026-05-22-lora-sweep-next-steps.md` (deploy roadmap)
- `programs/wm-dreamerv3.md` (target program schema)
- `scripts/_run_wm_session.sh` (single-trial generic runner)

---

## Current State

| Asset | Status |
|-------|--------|
| `programs/wm-dreamerv3.md` | Exists. Defines metric (`recon_loss` min), 8-trial sweep budget, HP search space, mutation hints. `path:` field points to stale archive path — fix needed. |
| `programs/wm-lewm.md` | Exists but BLOCKED — `lerobot 0.5.x` lacks `train_world_model` script. |
| `scripts/_run_wm_session.sh` | Single-trial generic WM runner. Handles bridge cache + ratchets metric. Arch-aware (dreamerv3 + le_world_model). |
| `scripts/_run_autoresearch_lora.sh` | Working multi-trial bash sweep template for LoRA. Pool of 16 pre-encoded configs, plateau-stop, ratchet best.json. |
| `scripts/_run_autoresearch_smoke.sh` | Deterministic multi-trial smoke for diffusion. 3 configs hardcoded. |
| `scripts/_open_loop_eval.py` | Action-MSE eval on held-out frames. Policy-only, NOT applicable to WM (WM predicts next-state, not actions). |
| `scripts/_merge_lora_ckpt.py` | LoRA-only — N/A for WM. |
| WM bridge skill | `~/tools/claude_code/skills/lerobot_world_model_bridge/operations.py` — already integrated by `_run_wm_session.sh`. |
| `lerobot_isaac_adapters.targets.wm_dreamerv3` | Working — invokes `sheeprl exp=dreamer_v3 env=custom_hdf5` via subprocess. Emits `recon_loss` to stdout via `metric_extractor.emit()`. |
| Dashboard | Autoresearch tab auto-discovers any `.agent-state/<session>/autoresearch/<slug>/` directory; schema-compatible with `best.json` / `plateau.json` / `program.json`. |

---

## Gaps (vs LoRA sweep)

1. **Multi-trial WM sweep script.** `_run_wm_session.sh` runs ONE config; no
   trial pool. Need a WM equivalent of `_run_autoresearch_lora.sh`.
2. **WM trial pool.** No pre-encoded HP combos. The wm-dreamerv3.md program
   names a search space but doesn't pin the explicit trial list a bash script
   would consume.
3. **Per-trial eval.** LoRA uses open-loop action MSE on held-out frames to
   produce a real `pc_success`. WM needs a DIFFERENT eval — `recon_loss` on a
   held-out HDF5 slice (image reconstruction error) and/or `pred_loss`
   (next-state prediction error). Training-loss-only ratchet risks overfit to
   train slice.
4. **Bridge cache vs trial diversity.** Trials that change `IMAGE_SIZE` or
   `WINDOW` require regenerating the HDF5 (~5–10 min each). Trials that only
   change algo Hydra knobs reuse cache. Sweep ordering must group by bridge
   params.
5. **Throughput unknown.** No baseline DreamerV3 wall-time numbers on this
   workspace's RTX 3080. CLAUDE.md SmolVLA table is irrelevant. Need a
   single-trial timing baseline before scheduling N-trial budgets.

---

## Goals

1. One generic multi-trial sweep script that runs DreamerV3 HP exploration on
   a fixed bridged HDF5, mirrors the LoRA sweep on-disk schema, and lets the
   dashboard auto-discover.
2. ≥ 6 distinct trials in the default pool covering lr / batch_size /
   sequence_length / replay_ratio / capacity (discrete_size, stochastic_size).
3. Real per-trial eval metric: open-loop next-state reconstruction MSE on the
   last 10 % of episodes (held out at bridge time).
4. Same env knobs as LoRA sweep (`MAX_TRIALS`, `STEPS`, `SECONDS_PER_EXP`,
   `PLATEAU_LIMIT`, `SKIP_TRIALS`) for budget elasticity.
5. Dashboard shows the WM sweep in the same Autoresearch tab without code
   changes.

---

## Phase 1 — Single-Trial Baseline (4–6 h elapsed; ~2 h compute)

**Goal:** unknown throughput is the biggest risk. Run a short single-trial
DreamerV3 train via `_run_wm_session.sh` to measure step/s, VRAM, and
final `recon_loss` at modest STEPS.

**Tasks:**
1. Pre-bridge once: `datasets/kvgork/so101-pickplace1` → `outputs/wm_data/so101-pickplace1_dreamerv3.hdf5` (64×64, window=16, stride=8). One-time ~10 min.
2. Run `bash scripts/_run_wm_session.sh --steps 50000 --timeout 3600`. Measure:
   - Wall-clock per 1000 sheeprl steps (compute step/s)
   - VRAM peak (nvidia-smi)
   - Initial vs final `recon_loss`
   - Convergence curve shape (use dashboard Autoresearch tab)
3. Record numbers in `docs/research/dreamerv3-reference.md` § "RTX 3080
   throughput (so101-pickplace1)" — new subsection.

**Acceptance:**
- Single-trial completes without OOM at batch_size=8, image_size=64.
- `recon_loss` decreases monotonically over the 50k step window.
- Step rate measured; one-trial wall time at 100k / 500k / 1M steps
  extrapolated.
- Dashboard's `wm-dreamerv3` slug shows the trial.

**Risks:**
- sheeprl 0.5.8.dev custom HDF5 env may have changed since last validation —
  bridge config or env registration may need fixes.
- Image-size 64 + batch 8 may OOM on RTX 3080 (10 GB) — fallback batch 4.

**Time:** ~3 h (10 min bridge + 1 h train + 1 h analysis & docs).

---

## Phase 2 — Define the WM Trial Pool (2 h, no compute)

**Goal:** pin an explicit, ordered trial array the sweep script can iterate
over. Mirror the LoRA sweep's pre-encoded design.

**Trial pool (provisional, 12 configs):**

| # | lr | bs | seq_len | replay_ratio | discrete | stochastic | total_steps | notes |
|---|----|----|---------|--------------|----------|------------|-------------|-------|
| 0 | 1e-4 | 8 | 16 | 1 | 32 | 32 | 200000 | baseline (sheeprl default) |
| 1 | 3e-5 | 8 | 16 | 1 | 32 | 32 | 200000 | lr lower |
| 2 | 3e-4 | 8 | 16 | 1 | 32 | 32 | 200000 | lr upper |
| 3 | 1e-4 | 4 | 16 | 1 | 32 | 32 | 200000 | smaller bs |
| 4 | 1e-4 | 16 | 16 | 1 | 32 | 32 | 200000 | larger bs (may OOM) |
| 5 | 1e-4 | 8 | 32 | 1 | 32 | 32 | 200000 | longer sequence |
| 6 | 1e-4 | 8 | 64 | 1 | 32 | 32 | 200000 | even longer sequence (VRAM risk) |
| 7 | 1e-4 | 8 | 16 | 2 | 32 | 32 | 200000 | replay_ratio=2 |
| 8 | 1e-4 | 8 | 16 | 4 | 32 | 32 | 200000 | replay_ratio=4 |
| 9 | 1e-4 | 8 | 16 | 1 | 64 | 32 | 200000 | wider discrete |
| 10 | 1e-4 | 8 | 16 | 1 | 32 | 64 | 200000 | wider stochastic |
| 11 | 1e-4 | 8 | 16 | 2 | 64 | 64 | 500000 | best-of-prior + longer train |

**Ordering rationale:**
- Trials 0–2 sweep lr at baseline (highest expected effect).
- Trials 3–4 sweep batch_size (VRAM-bounded).
- Trials 5–6 sweep sequence_length (compute + VRAM).
- Trials 7–8 sweep replay_ratio (cheap — same data, more updates).
- Trials 9–10 sweep capacity per axis.
- Trial 11 = best-guess combo + longer train (validates Phase 3 expansion).

**Acceptance:**
- 12 lines of `LR|BS|SEQ|RR|D|S|STEPS` in the script; one trial per line.
- Pool covers ≥ 3 axes (lr, capacity, compute).
- Order chosen so that early plateau-stop still produces useful coverage
  (lr trio first).

**Risks:**
- Search space defined in `programs/wm-dreamerv3.md` is broader than this
  12-trial subset; pool reflects RTX 3080 + 12 h reality, not the program's
  full hypothetical space. Document the subset explicitly.

**Time:** ~2 h to encode + cross-reference with the program doc.

---

## Phase 3 — Generic Sweep Script (1 day code, 0.5 day test)

**Deliverable:** `scripts/_run_autoresearch_wm.sh` (arch-aware multi-trial
runner).

**Design (mirror LoRA sweep structure):**

```bash
# Inputs (defaults shown):
ARCH=dreamerv3                      # only one supported until lerobot-train_world_model lands
DATASET=datasets/kvgork/so101-pickplace1
HDF5_CACHE=outputs/wm_data/so101-pickplace1_dreamerv3.hdf5
MAX_TRIALS=12
SKIP_TRIALS=0
SECONDS_PER_EXP=3000                # ~50 min/trial (Phase 1 will refine)
PLATEAU_LIMIT=4
IMAGE_SIZE=64
WINDOW=16
STRIDE=8
EVAL_ENABLED=1
EVAL_TIMEOUT=300
SESSION_ID=wm-bash-<ts>
```

**Per-trial pipeline:**
1. Read config from `TRIAL_POOL[$i]` (pipe-delimited).
2. Build sheeprl Hydra overrides from config.
3. Invoke `lerobot_isaac_autoresearch.train_wrapper --target_arch dreamerv3 ... -- <hydra-overrides>`.
4. Capture stdout to `trial_$i.log`.
5. Eval step:
   - Extract LAST `recon_loss` from the training log (always).
   - Run `scripts/_eval_wm_holdout.py` (Phase 4) → emits `eval_recon_loss` /
     `eval_pred_loss` on the held-out HDF5 slice. Real metric.
6. Append to `history.jsonl` with `metric_kind="eval_recon_loss"` (primary)
   and `metric_kind_fallback="train_recon_loss"`.
7. Ratchet best.json on LOWER metric (direction=minimize).
8. Plateau-stop on `PLATEAU_LIMIT` consecutive non-improvements.

**Bug to dodge (carried from LoRA sweep):**
- `best.json` not written when seeded best is never beaten. Fix here: always
  write a stub `best.json` from the seed on script start.

**Acceptance:**
- DRY_RUN=1 prints all 12 cmds, one per trial.
- Real run produces 12 history rows + best.json + plateau.json.
- Dashboard Autoresearch tab shows the new session with metric_kind set
  correctly.
- Script can resume mid-pool via `SKIP_TRIALS=N`.

**Risks:**
- sheeprl writes its own logs and may not emit `recon_loss` to stdout in
  every config — the adapter's `metric_extractor.emit()` is the safety net.
  Verify via Phase 1 grep.

**Time:** 1 day code (mostly copying / generalizing LoRA pattern) + 0.5 day
dry-run testing.

---

## Phase 4 — Held-Out WM Eval (REVISED 2026-05-22 — loader already exists)

**Deliverable:** thin shell wrapper `scripts/_eval_wm_holdout.sh` (or inline
in `_run_autoresearch_wm.sh`) that calls the existing rollout API.

**The hard part is done elsewhere.** The deploy workspace (separate session)
shipped a complete WM loader + offline rollout under
`~/workspaces/lerobot-isaac-deploy/src/lerobot_isaac_deploy/`:

- `wm_loader.load_dreamerv3(checkpoint_path)` — handles sheeprl
  `build_agent`, state restore, device placement, synthetic-marker
  short-circuit. Returns a `LoadedWMActor` with `.select_action(obs)` +
  `.reset()`.
- `wm_rollout.rollout(checkpoint_path, dataset_root, output_dir, kind="dreamerv3")`
  — loads ckpt, feeds LeRobotDataset obs sequences through
  `world_model.encoder` + `rssm` + `decoder`, writes:
   * `next_state_pred.npz` — per-timestep predicted next-state tensors
   * `rollout_summary.json` — `{mean_recon_loss, n_steps, n_episodes, ...}`
  CPU/GPU agnostic. No motors. Already handles synthetic test fixtures.

**Revised eval procedure (per trial):**

1. After training, find the latest sheeprl ckpt under
   `<OUTPUT_DIR>/checkpoints/` (dir layout per
   `docs/world-model-deploy.md`: `.hydra/config.yaml` + `ckpt_<step>.ckpt`).
2. Pre-flight: install the deploy package into `train-dreamer`:
   ```bash
   pixi run -e train-dreamer pip install -e ~/workspaces/lerobot-isaac-deploy
   ```
   (one-time; future workspaces should add it as a git+file:// dep
   alongside `lerobot-isaac-meta`.)
3. Run rollout against a held-out dataset slice. The bridge skill already
   supports `episode_filter`; pass the last 10 % of episode indices to
   `wm_rollout.rollout(dataset_root=..., episode_filter=...)`.
4. Parse `rollout_summary.json` → take `mean_recon_loss` as the trial metric.
5. Write `<AR_DIR>/trial_${i}_eval.json` mirroring the LoRA-sweep schema so
   the dashboard renders it uniformly.

**Acceptance:**
- Phase 4 wrapper completes in < 5 min per trial on the held-out slice.
- `rollout_summary.json` exists at `<OUTPUT_DIR>/rollout/`.
- `eval_recon_loss` ≠ `train_recon_loss` (split logic working).
- DreamerV3 synthetic-marker test ckpt round-trips through the wrapper
  without torch/sheeprl errors (validates fallback path).

**Risks (revised):**
- The deploy package's sheeprl version must match `train-dreamer` env's
  sheeprl version. Both should be ≥ 0.5.8.dev per current pin. Verify.
- Bridge `episode_filter` argument needs confirming (was an open question);
  if absent, fallback is to filter at rollout-load time via
  `LeRobotDataset(episodes=[...])` constructor.
- `wm_rollout.rollout()` signature may have changed since the deploy
  session shipped — re-read its docstring at integration time.

**Time:** ~3 h (collapsed from 1 day) — mostly install + glue, no new
loader code to write.

---

## Phase 5 — Bug Carry-Overs (0.5 day)

Bugs from the LoRA sweep that the WM sweep MUST fix at design time, not
discover at runtime:

1. **best.json not written on seeded-resume** — write stub from
   `RESUME_BEST_METRIC` env on script start.
2. **Plateau-stop too aggressive** — default `PLATEAU_LIMIT=4` (slightly
   higher than LoRA's eventual 6) since WM lr sweep is the highest-signal
   axis and 3 trials cover it.
3. **rm -rf before DRY_RUN check** — already fixed in LoRA; mirror the fix
   here from the start.
4. **Bridge cache invalidation** — sweep that changes IMAGE_SIZE or WINDOW
   needs a per-config HDF5 cache. Add `(image_size, window)` to the cache
   filename hash.

**Acceptance:** all four are validated by a DRY_RUN test before any compute
is burned.

---

## Phase 6 — 12 h DreamerV3 Sweep (12 h compute)

**Goal:** run the full 12-trial pool with eval wired.

**Per-trial budget (rough, Phase 1 will refine):**
- 200 000 sheeprl steps at ~30–100 step/s ≈ 30–110 min/trial.
- Worst case 110 min × 12 = 22 h — too long. Cap at SECONDS_PER_EXP=2700
  (~45 min) so most trials get truncated at step ~135 k. Accept the cap.
- Trial 11 (longer train, 500k steps) gets dedicated budget = 2 × normal.

**Adjusted schedule:**
- 11 normal trials × 45 min ≈ 8.25 h
- 1 long trial × 90 min ≈ 1.5 h
- Eval (~3 min × 12 trials) ≈ 0.6 h
- Total ≈ 10.5 h. Fits in 12 h with slack.

**Launch:**
```bash
SESSION_ID="wm-bash-$(date +%Y%m%d-%H%M%S)" \
MAX_TRIALS=12 \
SECONDS_PER_EXP=2700 \
PLATEAU_LIMIT=4 \
  bash scripts/_run_autoresearch_wm.sh
```

**Acceptance:**
- All 12 trials run (or plateau-stop after ≥ 6 non-improvements).
- best.json names a non-baseline config (i.e. exploration was useful).
- Dashboard Autoresearch tab visualizes the recon_loss curve.

**Risks:**
- 45-min cap may starve the larger-capacity trials of meaningful training.
  Plan B: drop them to recover budget for shorter, more-trials sweeps.
- If Phase 1 baseline reveals step/s is < 20 (slow), entire schedule needs
  rework — fewer trials, longer per trial.

---

## Phase 7 — Tech Debt + Documentation (0.5 day)

1. Fix stale `path:` in `programs/wm-dreamerv3.md` (points to
   `archive/packages/...`). Update to
   `src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py`.
2. Add `docs/runbook/04-train-world-model.md` § "Autoresearch sweep" with the
   launch command above.
3. Update CLAUDE.md "Common Pitfalls" with DreamerV3 throughput from Phase 1.
4. Cross-link this plan from `plans/2026-05-22-lora-sweep-next-steps.md`
   Phase 1b (closed-loop sim eval via WM).

---

## Critical Path

```
Phase 1 (single-trial baseline, 3 h)
     │
     ▼
Phase 2 (pin trial pool, 2 h) ─── reads Phase 1 timing
     │
     ▼
Phase 3 (generic sweep script, 1.5 d) ─┐
                                        │
Phase 4 (held-out eval script, 1 d) ────┤
                                        │
Phase 5 (bug fixes, 0.5 d)  ────────────┤
                                        │
                                        ▼
                       Phase 6 (12 h DreamerV3 sweep)
                                        │
                                        ▼
                       Phase 7 (docs + cleanup, 0.5 d)
```

Total active engineering time: **~3.5 working days** (was 4.5 before Phase 4
shrank — `lerobot-isaac-deploy.wm_loader` + `wm_rollout` already shipped in
the deploy session). Phase 6 is overnight / weekend compute, not engineer
time.

---

## Cross-Reference with LoRA Pattern

| Asset | LoRA equivalent | WM equivalent (this plan) |
|-------|-----------------|---------------------------|
| Sweep script | `_run_autoresearch_lora.sh` | `_run_autoresearch_wm.sh` (Phase 3) |
| Single-trial runner | (none — sweep IS the runner) | `_run_wm_session.sh` (already exists) |
| Eval script | `_open_loop_eval.py` (action MSE) | `lerobot_isaac_deploy.wm_rollout.rollout()` API call (shipped 2026-05-22) |
| Ckpt format fix | `_merge_lora_ckpt.py` (peft merge) | none needed — sheeprl ckpts already plain |
| Program file | `programs/lerobot-policy-smolvla-lora.md` | `programs/wm-dreamerv3.md` (fix `path:`) |
| Anchor | SmolVLA finetuned base | none — WM trained from scratch |
| Metric direction | maximize pc_success | minimize recon_loss |
| Metric source | open-loop eval JSON | held-out HDF5 eval JSON |

---

## Open Questions Before Implementing (updated 2026-05-22)

1. **CLOSED.** sheeprl's `eval_metrics` is irrelevant — Phase 4 uses
   `lerobot_isaac_deploy.wm_rollout.rollout()` instead (already shipped).
2. Is the bridge skill's `episode_filter` argument actually wired? If not,
   filter at `LeRobotDataset(episodes=[...])` load time inside
   `wm_rollout.rollout()`.
3. Should LeWorldModel sweep be planned alongside (deferred until upstream)
   or out of scope until lerobot ships `train_world_model`? — defer.
4. Bridge cache strategy: per-(image_size, window) pair, or single canonical
   bridge with eval HDF5 separate? Recommend: per-(image_size, window) hash
   in the cache filename, plus a `_eval.hdf5` sibling for the held-out slice.
5. **NEW.** sheeprl version pin in `train-dreamer` env must match the one
   `lerobot-isaac-deploy.wm_loader` was tested against. Cross-check before
   Phase 4 install.

Items 2, 4, 5 need ~30 min total before Phase 1 starts.

---

## Exit Criteria

WM autoresearch is "landed" when ALL hold:

- 12-trial sweep completes via single bash command.
- Dashboard shows the sweep with non-empty `eval_recon_loss` curve.
- best.json identifies a config different from the baseline (sweep extracted
  signal).
- DRY_RUN reproducibly prints the 12 trial commands.
- Resume (`SKIP_TRIALS=N`) works mid-pool.
- Docs (Phase 7) cross-link to the LoRA sweep pattern so future arches can
  follow the same template.
