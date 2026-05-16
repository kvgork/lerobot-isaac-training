# Second-Brain → Workspace Audit — 2026-05-13

**Vault root:** `/home/koen/Documents/Vaults/Local`
**Workspace root:** `/home/koen/workspaces/lerobot-isaac-training`
**Cutoff for "recent" vault content:** 2026-04-01 (≈ 6 weeks).
**Audit produced by:** Phase 2 of the 3-phase orchestration run.
**Status:** AUDIT ONLY — no fixes applied.

This document compares **recent** Second-Brain knowledge against the current state
of the `lerobot-isaac-training` monorepo. Each entry names a single concrete
target file + change. Items are sorted by priority within their topic clusters.

A heavy fraction of the vault's new material (≈ 100 new concepts under
`05-Wiki/concepts/`) is generic software-engineering content (design patterns,
SOLID, distributed-systems references, etc.) — that material is in scope for the
**knowledge** layer of the brain, not for this workspace. Only items whose
content actively constrains, validates, or improves the existing pipeline
appear below.

---

## [Cross-Package On-Disk Contract Gaps]
**SB source:** `05-Wiki/synthesis/lerobot-isaac-contract-gaps.md`
**Date added:** 2026-05-12
**Workspace gap:** Dashboard loaders (`outputs/eval/*.json`, `outputs/curriculum_stage.json`, `outputs/curriculum_history.jsonl`) document `Contract (pending)` — none of these files are actually emitted by the producer agents. Three distinct mismatches identified in the synthesis page.
**Concrete fix:** Add a thin post-eval emitter (target: `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/eval_emitter.py` or similar) that writes `outputs/eval/{run_id}_{ts}.json` matching the dashboard `EVAL_SCHEMA`. Update `~/.claude/agents/orchestrators/lerobot-curriculum-agent.md` to also write `outputs/curriculum_stage.json` (translated view of the existing `curriculum_state.json`) and append-only `outputs/curriculum_history.jsonl`. Files affected: `packages/lerobot-isaac-dashboard/src/lerobot_isaac_dashboard/loaders/eval_results.py` (relax `_align_to_schema` as documented), `packages/lerobot-isaac-dashboard/src/lerobot_isaac_dashboard/loaders/curriculum.py` (drop "pending" docstring once producers exist).
**Priority:** high

## [LeRobotDataset v2.1 → v3.0 Schema Transition]
**SB source:** `05-Wiki/synthesis/lerobot-isaac-contract-gaps.md` §Gap 3
**Date added:** 2026-05-12
**Workspace gap:** Local install is already `CODEBASE_VERSION = "v3.0"` (`meta/episodes_stats.jsonl` replaces `meta/stats.json`). Dashboard `data_summary` loader does not detect dataset version. No documented migration path in `docs/internals/data-pipeline.md`.
**Concrete fix:** Add a `dataset_version` field to `packages/lerobot-isaac-dashboard/src/lerobot_isaac_dashboard/loaders/data_summary.py` (read `meta/info.json:codebase_version`, fall back to `"v2.1"` if `meta/stats.json` present, else `"unknown"`). Document migration in `docs/internals/data-pipeline.md` with a "When you have a v2.1 dataset" section pointing at `lerobot/scripts/convert_dataset_v21_to_v30.py`. Emit a dashboard badge per dataset showing the detected version.
**Priority:** high

## [Continual-Learning Metrics — BWT/FWT/FM not measured]
**SB source:** `05-Wiki/concepts/Continual-Learning-Metrics.md`
**Date added:** 2026-05-12
**Workspace gap:** `lerobot-evaluation-agent` only records per-stage `pc_success` (`R[i,i]`). To compute Backward Transfer, Forward Transfer, or Forgetting Measure, the evaluator must run a **regression pass over all prior curriculum stages** after each advancement. Current workspace does not have an `R` matrix anywhere.
**Concrete fix:** Add a `regression_eval` mode to `~/.claude/agents/workers/lerobot-evaluation-agent.md` that, after each stage advancement, evaluates the new checkpoint on all prior stages and writes `outputs/curriculum_R_matrix.jsonl` (one row per row-of-R). Update `packages/lerobot-isaac-dashboard/src/lerobot_isaac_dashboard/loaders/curriculum.py` (or sibling loader) to surface ACC, BWT, FWT, FM derived from this matrix. Add `docs/concepts/continual-learning-metrics.md` documenting the four numbers and the protocol requirements (frozen eval envs, fixed seeds, same protocol after every task).
**Priority:** high

## [World-Model Evaluation Axes — Prediction-Loss-Only Is The Wrong Metric]
**SB source:** `05-Wiki/concepts/World-Model-Evaluation.md`; `05-Wiki/research/2026-05-12-lerobot-leworldmodel-benchmarking.md`
**Date added:** 2026-05-12
**Workspace gap:** `docs/research/dreamerv3-reference.md` and `docs/research/leworldmodel-reference.md` describe the architectures but do not specify *how to evaluate them*. No notion of Axis 1 (prediction quality), Axis 2 (planning quality), Axis 3 (policy-from-rollouts), Axis 4 (physical executability — RoboWM-Bench). The autoresearch loop will optimize whatever metric we give it; absent a clear "pc_success on downstream policy trained on WM rollouts" metric, the loop may chase FVD/MSE — a known dead end.
**Concrete fix:** Add `docs/concepts/world-model-evaluation.md` adapting the four axes from the vault page. Update `docs/research/dreamerv3-reference.md` and `docs/research/leworldmodel-reference.md` with a `## Evaluation` section pointing at the new concept page. In `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/wm_dreamerv3.py` and `targets/wm_leworldmodel.py`, document which axis the emitted metric corresponds to (currently the metric extractor pulls training loss — Axis 1 only).
**Priority:** high

## [System Identification — Cheapest +30% Sim-to-Real Win]
**SB source:** `05-Wiki/concepts/Sim-to-Real-Transfer.md`; `05-Wiki/concepts/Actuator-System-Identification.md`; `05-Wiki/concepts/Sensor-Characterization-for-Sim.md`; `05-Wiki/concepts/Sim-Calibration-Loop.md`; `05-Wiki/concepts/Teleop-Mirror-Sysid.md`; `05-Wiki/concepts/AI-Driven-System-Identification.md`
**Date added:** 2026-05-12 (Sim-to-Real-Transfer.md updated)
**Workspace gap:** Workspace has no sysid story. The plan currently goes from "scaffolding" → "real data" → "DR replay". Vault notes call out 8 actuator parameters (torque constant, rotor inertia, viscous + Coulomb friction, gear ratio, control time constant, backlash, deadband) and 7 measurement protocols. Skipping sysid is the documented failure mode for SO-101 (HIL-SERL 2026: "robot just hit the ground").
**Concrete fix:** Add `docs/runbook/09-system-identification.md` adapting the Actuator-System-Identification + Sim-Calibration-Loop pages into a concrete SO-101 procedure. Document the dependency on calibrated MJCF in `docs/research/isaac-lab-reference.md` (Isaac Lab + USD does not by itself solve sysid — same parameter gap as MJCF). Optional: add a `lerobot-isaac-sysid` package skeleton or a `tools/sysid/` subdir with `measure_actuator.py`, `measure_sensor.py`, `calibration_loop.py` stubs.
**Priority:** med (cheap to scaffold; high payoff once real hardware arrives)

## [Domain Randomization: Spatial > Visual; Frame-Wise > Episode-Wise]
**SB source:** `05-Wiki/concepts/Sim-to-Real-Transfer.md` (Spatial DR > Visual DR; Frame-Wise > Episode-Wise tips); `05-Wiki/concepts/Domain-Randomization-(SO-101-MuJoCo).md`
**Date added:** Updated 2026-05-12
**Workspace gap:** `packages/lerobot-isaac-synthetic/` and `packages/lerobot-isaac-env/` do not encode the **ordering** of DR parameters (camera pose first, textures last) nor enforce frame-wise randomization. Current DR config is implicit / per-task with no explicit policy preference.
**Concrete fix:** Add `docs/concepts/domain-randomization-priority.md` summarizing the two ordering tips. Update the YAML configs in `packages/lerobot-isaac-configs/configs/` (DR-related templates) to put `camera_pose_jitter` and `camera_position_jitter` at the top of the randomization list, with comments referencing the vault note. Document the "progressive DR schedule" in `docs/runbook/05-augment-with-dr.md` (dynamics → widen dynamics → sensor noise → visual last).
**Priority:** med

## [V-JEPA-2 / Dreamer4 — Add Architectural Comparison Table]
**SB source:** `05-Wiki/entities/V-JEPA-2.md`; `05-Wiki/entities/Dreamer4.md`
**Date added:** 2026-05-12 (both new)
**Workspace gap:** `docs/research/` covers DreamerV3 and LeWM but does not mention V-JEPA-2 (validates internet-video → small-robot-data path) or Dreamer4 (token-transformer alternative to RSSM, shortcut forcing for real-time inference). The vault has a clean 3-way comparison table that would slot directly into a future workspace doc. V-JEPA-2 is MIT-licensed, ViT-S predictor ~22 M params — feasible on RTX 3080 if added as a `--target_arch` later.
**Concrete fix:** Add `docs/research/v-jepa-2-reference.md` and `docs/research/dreamer4-reference.md`, mirroring the structure of `docs/research/dreamerv3-reference.md`. Add a "Future targets" section to `docs/concepts/modular-training-adapter.md` listing both as potential `--target_arch` values. No code changes required yet.
**Priority:** low (informational; gates on whether you intend to add either as a target)

## [Data Quality Filtering (SAL + TED) — Workspace already has this]
**SB source:** `05-Wiki/concepts/Data-Quality-Filtering-(SAL-TED).md`
**Date added:** 2026-05-03
**Workspace gap:** Already implemented in `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/quality.py` and exposed via `lerobot-isaac quality-filter`. The vault note has +16% (SAL) / +20% (TED) success-rate numbers from the RINSE paper that should be cited in the workspace docs as the justification for the default thresholds.
**Concrete fix:** Add a "Why these thresholds?" subsection to the `quality-filter` `--help` output, and a paragraph in `docs/concepts/modular-training-adapter.md` (or a new `docs/concepts/data-quality-filtering.md`) citing RINSE arXiv 2604.23000 and the +16/+20 numbers. Add the vault page URL or wiki-link to the docstring of `quality.py:apply_quality_filter`.
**Priority:** low

## [SO-101 Camera Index Instability — Document in Recorder Runbook]
**SB source:** `05-Wiki/entities/SO-101.md` (Hardware Gotchas — "Camera Index Instability")
**Date added:** Updated 2026-05-08
**Workspace gap:** `docs/runbook/02-collect-data.md` does not call out that camera indices change every USB plug/unplug cycle. `packages/lerobot-isaac-recorder/` `check_hardware` test exists but the workflow does not force the user to run it before each session.
**Concrete fix:** Add a pre-flight checklist box at the top of `docs/runbook/02-collect-data.md` stating "Run `robot-data-check` before every session — camera indices are not stable across USB replug". Promote `robot-data-check` as a required step in the runbook flow chart. Also document the calibration uint16 overflow (Issue #1342) and the no-move-after-recording-starts rule.
**Priority:** med

## [Sim-to-Real Deployment Protocol — Bring the Checklist Inline]
**SB source:** `05-Wiki/concepts/Sim-to-Real-Deployment-Protocol.md`
**Date added:** 2026-05-03
**Workspace gap:** No deployment checklist exists in workspace. `docs/runbook/` covers data collection and training but not the deployment gate. The vault page has a complete pre-deployment checklist (workspace constraints, safety limits, supervised rollouts, torque monitoring, calibration zero convention).
**Concrete fix:** Add `docs/runbook/10-deploy-to-hardware.md` adapting the vault page. Mirror the "Reality Gap" table (contact dynamics 30%, sensor noise 25%, motor dynamics 20%, visual 15%, other 10%) in the workspace doc — these numbers are repeatedly cited in vault prose and orient users away from the visual-DR-first failure mode.
**Priority:** med

## [Curriculum-Learning Tier Ladder — Encode the 6-Stage SO-101 Curriculum]
**SB source:** `05-Wiki/concepts/Curriculum-Learning-(Robot-Manipulation).md` (SO-101 Practical Ladder); `05-Wiki/entities/SO-101.md` (Practical Curriculum Ladder)
**Date added:** 2026-05-12
**Workspace gap:** Plan §13 references curriculum but no machine-readable ladder file. `lerobot-curriculum-agent` consults `curriculum_state.json` which is per-run state, not the ladder definition. No central source of truth for "stage 1 = fixed pick, stage 4 = 5mm insertion".
**Concrete fix:** Add `packages/lerobot-isaac-configs/configs/curriculum_so101.yaml` declaring the 5–6 tiers (Pick fixed → Pick-and-place ±5cm → ±10cm → Insertion 5mm → Assembly), each with `task_name`, `tolerance`, `target_pc_success`, `min_episodes`. Update `lerobot-curriculum-agent.md` to load this file as the ladder definition. Document in `docs/concepts/curriculum-learning.md` (new).
**Priority:** med (blocked on Phase 3 having real data, but cheap to scaffold now)

## [Dual-Target Recording — Connect Vault Page to Recorder Package]
**SB source:** `05-Wiki/concepts/Dual-Target-Recording-(LeRobot-and-LeWorldModel).md`; `05-Wiki/entities/SO-101.md` (Dual-Target Recording section)
**Date added:** 2026-05-08
**Workspace gap:** `packages/lerobot-isaac-recorder/` (renamed to `robot-data-recorder`) implements dual-write but its README does not link to the canonical concept page in the vault. Reverse-discoverability is broken: a reader of `lerobot-isaac-recorder/README.md` cannot easily find the design rationale.
**Concrete fix:** Add a "Design rationale" section to `packages/lerobot-isaac-recorder/README.md` linking to the vault page (use `${VAULT_ROOT}/05-Wiki/concepts/Dual-Target-Recording-(LeRobot-and-LeWorldModel).md`). Mirror the load-bearing claims (stable-worldmodel HDF5 schema, why single-pass dual-write beats post-hoc conversion) inline so the file stands alone after spinout.
**Priority:** low

## [LeWorldModel Canonical Paper — Update Reference Doc]
**SB source:** `05-Wiki/sources/2026-05-12-lewm-paper.md`; `05-Wiki/entities/LeWorldModel.md`
**Date added:** 2026-05-12 paper, 2026-05-08 entity update
**Workspace gap:** `docs/research/leworldmodel-reference.md` calls the project "HF LeWorldModel (Alibert et al. 2025)" — the correct citation is Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv:2603.19312, March 2026. ~15 M params, two-loss training (prediction + SIGReg Gaussianity regularizer), ~48× faster planning than DINO-WM. Pretrained checkpoints exist at `huggingface.co/collections/quentinll/lewm`.
**Concrete fix:** Update header citation, parameter count, and planning-speed claim in `docs/research/leworldmodel-reference.md`. Add a "Pretrained checkpoints" subsection listing the 4 released task checkpoints (pusht, cube, tworooms, reacher). Cross-link to the new `docs/concepts/world-model-evaluation.md` once that exists.
**Priority:** med

## [Autonomous-ML-Training-Loop — Wire Up `final_metric=` Sentinel Pattern]
**SB source:** `05-Wiki/concepts/Autonomous-ML-Training-Loop.md`
**Date added:** Updated 2026-05-08
**Workspace gap:** Vault page documents the stdout-regex + sentinel-fallback pattern for metric capture: "if the backend never emits a metric line, the wrapper emits `<metric>=0.0` rather than crashing". OOM recovery ladder: "halve `batch_size`, retry once". Workspace `lerobot-isaac-autoresearch/train_wrapper.py` partially implements this but the sentinel value and the retry budget are not documented anywhere user-facing.
**Concrete fix:** Document the sentinel/retry contract in `docs/internals/autoresearch-integration.md` (`## Metric Capture Contract` section). Make the sentinel value (`0.0`) and retry budget configurable via `packages/lerobot-isaac-configs/configs/autoresearch.yaml` (new). Surface the OOM ladder in `CLAUDE.md` under "Common Pitfalls" as a confirmation that the existing pattern matches the documented one.
**Priority:** low

---

## Summary

| # | Topic | Priority |
|---|-------|----------|
| 1 | Cross-package on-disk contract gaps (3 sub-gaps) | high |
| 2 | LeRobotDataset v2.1 → v3.0 schema transition | high |
| 3 | Continual-learning metrics — BWT/FWT/FM not measured | high |
| 4 | World-Model evaluation axes documentation | high |
| 5 | System Identification scaffolding | med |
| 6 | DR ordering (spatial > visual; frame-wise > episode-wise) | med |
| 7 | V-JEPA-2 / Dreamer4 architectural references | low |
| 8 | SAL+TED threshold justification citations | low |
| 9 | SO-101 camera index instability pre-flight | med |
| 10 | Sim-to-real deployment runbook | med |
| 11 | 6-stage SO-101 curriculum ladder config | med |
| 12 | Dual-target recording rationale link | low |
| 13 | LeWorldModel paper citation update | med |
| 14 | Autoresearch metric-capture contract documentation | low |

**Total:** 14 items; **high:** 4, **med:** 6, **low:** 4.

**Recommended apply order:**
1. Items #1, #2, #3 — closes documented "Contract (pending)" gaps in the dashboard and enables continual-learning metrics.
2. Item #4 + #13 — world-model evaluation framing before any WM training run.
3. Item #11 — encodes the curriculum ladder so it's available when real data lands.
4. Items #5 + #9 + #10 — sim-to-real readiness; gates real-hardware deployment.
5. Remaining low-priority items — opportunistic, ride on the next docs PR.
