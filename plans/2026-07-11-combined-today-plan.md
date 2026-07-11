# Combined today-plan — 0.6.0 WM-policy × ACT-real campaign (2026-07-11)

`/orchestrate`, rigor=agentic. Merges the two live parallel tracks now that **GPU is free**
(8.6 GB / 10 GB on the RTX 3080). Deliverable = a **`vla_jepa` real-data deploy candidate**
(world-model policy — the stated long-term goal) + clearing the commit debt both tracks left.

## The two tracks being combined
- **Track 1 — `plans/2026-07-08-lerobot-060-worldmodel-update.md`** (GPU-blocked). lerobot
  0.5.1→0.6.0 done + CPU-verified (216 adapter tests pass), UNCOMMITTED in workspace + adapter
  repos. `vla_jepa` wired (only WM policy that fits 10 GB; WM at train-only, dropped at inference).
- **Track 2 — `plans/2026-06-28-act-real-campaign-plan.md`** (hardware-blocked). ACT-15k
  (loss 0.168) + SmolVLA (loss 0.119) real-data deploy candidates trained; **first real-arm
  success 2026-07-04**. Remaining top items (150+ diverse demos, SmolVLA HW-eval) need the arm.
  `robot-data-runner/mappers.py` 12-dim state fix UNCOMMITTED (own repo). Reward-carrying HDF5
  (`outputs/wm_offline_real/dreamerv3_data.hdf5`, 490 MB) exists + reusable.

## Intersection (why these combine, not just run side-by-side)
Track 1 delivers the *capability* (a 10 GB-fit WM policy via `lerobot-train`); Track 2 delivers
the *data + deploy harness* (`so101-pickplace-new`, `robot-data-run`). Combined =
**train `vla_jepa` on `so101-pickplace-new`** → a 3rd, world-model-based candidate for the same
real-arm bake-off as ACT/SmolVLA. This is the first time the "working WM policy" goal is reachable
without the sim2real wall (Route A regime, WM technique).

## Constraints for "today"
- **Single GPU, sequential.** 8.6 GB free. `vla_jepa` (~2B, V-JEPA2 WM live at train) is **tight** —
  OOM ladder: batch 4 → 2 → 1 (+grad-accum). Smoke first (Phase 1) derisks fit before the long run.
- **Dataset is decode-bound** (av1 480×640, `data_s≈1.7`, torchcodec already REFUTED). A bigger model
  on the same data ⇒ the full run is an **overnight job**, not finishable in a work-day. So "today" =
  commit debt + flag guard + smoke + **launch full run detached** (`setsid`, per
  `[[detach-long-training-jobs]]`), hand ckpt to human.
- **No hardware in this loop.** Every eval/deploy gate needs the physical SO-101 → today produces
  checkpoints only.

---

## Phase 0 — Clear commit debt + 0.6.0 flag guard (NO GPU, ~15 min)
De-risks first: three repos hold uncommitted, already-verified work.
1. **Flag guard (blocking-cheap):** dry-run `lerobot-isaac-train --dry_run` for `act`, `smolvla`,
   `vla_jepa` — confirm 0.6.0 didn't shift the deploy-candidate CLI flags (Track-1 open item).
2. **Commit workspace** (`pixi.toml`, `install_train_deps.sh`, `CLAUDE.md`, 2 runbooks) — 0.6.0
   upgrade + `stable-worldmodel` removal.
3. **Commit + push adapter** inside `src/lerobot-isaac-adapters/` (own repo/PR): `train.py`,
   `policy_lerobot.py`, 2 test files. Run `pixi run -e default test` first (216 pass gate).
4. **Commit robot-data-runner** `mappers.py` 12-dim state fix inside `src/robot-data-runner/` (own
   repo, 19/19 tests) — Track-2 debt, needed for any future deploy.
- **Acceptance:** 3 clean commits (adapter + runner pushed), dry-runs emit correct `--policy.type`.
- **Review gate:** `/grill` staged diff before push (agentic default).

## Phase 1 — `vla_jepa` GPU smoke (GPU, ~30–60 min)
The Track-1 blocking item + Phase-2 fit derisk in one shot.
- **Cmd:** `lerobot-isaac-train --target_arch vla_jepa --dataset datasets/local/so101-pickplace-new
  --successes_only --batch_size 4 --steps 500 -- --policy.path=lerobot/VLA-JEPA-Pretrain`
- **Acceptance:** rc=0, no OOM, loss decreasing, `pc_success` metric extractor fires. Record
  train-time VRAM + `data_s`. If OOM → drop batch 4→2→1, note working batch for Phase 2.
- **Gate:** fits + steps cleanly → Phase 2. OOM at batch 1 → `vla_jepa` doesn't fit this box after
  all; STOP, report (fastwam/lingbot_va already known too big), fall back to Track-2-only day.

## Phase 2 — `vla_jepa` full real-data run (GPU, detached, overnight)
The deliverable: WM-policy candidate on the same demos as ACT/SmolVLA.
- **Cmd:** Phase-1 config, `--steps 20000`, output `outputs/vla_jepa_real_so101`, `setsid`-detached,
  log-poll heartbeat monitor (never `run_in_background` — killed by session events).
- **Acceptance:** launches clean, first checkpoint written, loss converging on heartbeat. Full
  convergence handed to human (many hours; decode-bound).
- **Gate:** loss converges → 3rd deploy candidate ready for the real-arm bake-off.

## Phase 3 — 3-way real-arm eval (HUMAN + HARDWARE, gated, not today)
- ACT-15k vs SmolVLA vs `vla_jepa` on the physical SO-101 via `robot-data-run --policy-path <ckpt>
  … --execute --rate-hz 30` (~20 eps each; the working cmd from Track 2's HW-eval).
- **Decides:** best technique → Run 3a (scale to 150+ diverse demos — the standing top lever) vs 3b.
- WM-policy question answered here: does `vla_jepa` beat BC on the real arm?

## Parked / lower priority (GPU available but low value today)
- **Resume WM-offline DreamerV3** — reward HDF5 exists; but offline+sparse+expert is weak for
  *control* and the gate needs a *real* metric, not recon_loss. ~29 h @ replay_ratio=1. Skip unless
  Phase 2 stalls and the GPU would otherwise idle.

## Decision gates (summary)
| phase | gate | pass → | fail → |
|-------|------|--------|--------|
| 0 flag guard | dry-runs emit right flags | commit/push | fix adapter |
| 1 smoke | fits 10 GB, rc=0 | Phase 2 | drop batch / STOP if OOM@1 |
| 2 full | loss converges | 3rd HW candidate | debug config |
| 3 HW eval | real success-rate | Run 3a scale-data | Run 3b |

## Risks
- **`vla_jepa` OOM** — 2B + train-time WM on 8.6 GB free is the top risk; Phase 1 smoke gates it.
- **Overnight decode wall** — same av1 bottleneck as ACT; full run won't finish in a work-day.
- **No hardware** — Phase 3 (the only gate that ranks techniques) is not doable in this loop.
- **3 uncommitted repos** — silent drift risk; Phase 0 clears it before any new run muddies the tree.

## Related
- `[[2026-07-08-lerobot-060-worldmodel-update]]` · `[[2026-06-28-act-real-campaign-plan]]` ·
  `[[2026-06-28-working-policy-next-steps]]`
- memory: `[[lerobot-060-worldmodel-policies]]` · `[[act-real-campaign-result]]` ·
  `[[detach-long-training-jobs]]`
