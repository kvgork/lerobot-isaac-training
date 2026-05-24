# Multi-Dataset Training — Plan

**Date:** 2026-05-23
**Branch:** TBD (`feature/multi-dataset` recommended)
**Scope:** Let `lerobot-isaac-adapters` accept multiple LeRobotDataset
sources in one training run — either by concatenating at load time OR
by pre-merging on disk into one dataset.

---

## Why

Today, every entrypoint takes exactly one `--dataset` path:

```python
# lerobot_isaac_adapters.train passes through to:
lerobot-train --dataset.repo_id=<repo> --dataset.root=<path> ...
```

This blocks several real workflows:

1. **SO-101 progressive curriculum.** Stage-1 demos + Stage-2 demos
   come from different recording sessions. To train on both, we
   currently have to:
   - Train sequentially (forget stage-1 between stages), OR
   - Manually export, concat, re-import via `lerobot-edit-dataset`
2. **Cross-task transfer.** Pick-place + push + rotate demos in
   separate datasets — combine them for a generalist policy.
3. **Real + DR synthetic mixing.** `lerobot-isaac-synthetic` emits
   `datasets/kvgork/so101-pickplace1-dr*` siblings of the real
   `so101-pickplace1`. The current pipeline trains on whichever ONE you
   point `--dataset` at. We want weighted mixing.
4. **Test/val splits in one run.** Held-out episodes today require a
   second dataset dir produced manually.

---

## Existing Building Blocks

| Asset | Path | What it gives |
|-------|------|---------------|
| LeRobotDataset (v3.0) | `lerobot==0.5.1` | Single-dataset loader. Schema = info.json + meta/episodes.jsonl + data/*.parquet + videos/<key>/*.mp4. |
| `MultiLeRobotDataset` | `lerobot.datasets.lerobot_dataset:MultiLeRobotDataset` | sheeprl-style ConcatDataset wrapper; lerobot already ships it, **gym branch only** (verify). |
| `lerobot_world_model_bridge` | skill — Parquet → HDF5 | One-input-dataset by design. |
| `lerobot_mimicgen_bridge` | skill — Parquet ↔ HDF5 | Already has merge utilities for synthetic vs real. |
| Adapter CLI `lerobot_isaac_adapters.train` | this workspace | argparse with `--dataset` (string). |

**Reuse rule**: pick MultiLeRobotDataset if it's stable in `lerobot
0.5.1`; otherwise pre-merge via `lerobot-edit-dataset` and treat the
merged repo like any other.

---

## Two Implementation Paths (build BOTH, pick per use case)

### Path A — Load-time concatenation (`MultiLeRobotDataset`)

Lightweight. No disk doubling. Train CLI accepts repeatable `--dataset`.

**Pros:**
- Zero disk overhead.
- Train + iterate quickly: edit list, no re-merge step.
- Per-dataset transforms still possible (different image norms etc.).

**Cons:**
- `MultiLeRobotDataset` API may not be stable in 0.5.x — verify column
  compatibility checks, episode_index uniqueness, video-backend
  multiplexing.
- Doesn't help the WM-bridge path (still needs single HDF5).

### Path B — Pre-merge on disk (single new LeRobotDataset)

Produces `datasets/<merged-name>/` that downstream treats as one
dataset.

**Pros:**
- Works for ALL pipelines (LeRobot, world-model bridge, MimicGen
  conversion) without per-pipeline changes.
- Reproducible: the merged dataset is a stable artifact you can sync,
  publish to HF Hub, etc.
- Each downstream consumer sees a normal LeRobotDataset.

**Cons:**
- Disk cost = sum of source datasets (~3 GB each for so101-pickplace1
  scale).
- Re-merge needed if a source changes.

---

## Phase Breakdown

### Phase 0 — Compatibility audit (0.5 day)

Tasks:
1. Read lerobot 0.5.1 source for `MultiLeRobotDataset`. Confirm:
   - Constructor signature accepts list[str | Path]
   - It re-indexes `episode_index` across the union
   - Video keys merge cleanly (or error explicitly on mismatch)
   - The lerobot-train CLI accepts list-style `--dataset.repo_id` arg
2. Read lerobot 0.5.1 source for `LeRobotDataset` info.json schema —
   list the required fields and confirm they MATCH across our existing
   `so101-pickplace*` datasets.
3. Identify which dataset-prep CLI tools ship with lerobot:
   - `lerobot-edit-dataset` (rename / split / merge?)
   - `lerobot-aggregate` (newer; verify)

**Acceptance:** a short table in `docs/research/lerobot-multidataset-reference.md` documenting the API surface for Path A AND Path B, plus any version caveats.

### Phase 1 — Path B: pre-merge utility (1 day)

**Deliverable:** `scripts/_merge_lerobot_datasets.py`

API:
```bash
python scripts/_merge_lerobot_datasets.py \
    --inputs datasets/kvgork/so101-pickplace1 \
             datasets/kvgork/so101-pickplace1-dr20 \
             datasets/kvgork/so101-pushtask \
    --output datasets/kvgork/so101-merged-pp-push \
    --weights 1.0 1.0 0.5         # optional per-dataset sample weight
                                    # (encoded as repeat counts in
                                    # the merged dataset)
    --episode-prefix-source        # optional: prefix episode_index
                                    # with source-shortname for forensics
```

Internal:
1. Read each input's `meta/info.json`; validate schema compatibility
   (same FPS, same camera keys, same action_dim, same state_dim).
   Hard-fail with a precise diff if a field disagrees.
2. Re-number `episode_index` to be globally unique across the union.
3. Copy `data/chunk-N.parquet` shards and `videos/<key>/chunk-N/...`
   into the output directory.
4. Write a merged `info.json` that lists the source paths, the merge
   timestamp, and the per-source episode-count breakdown so the merge
   is reversible (or at least diagnosable).
5. Apply `--weights` by oversampling: if weight is 2.0, the merged
   dataset re-references that source's episodes twice with distinct
   `episode_index`. Up-sampling factor capped at 5×.

**Acceptance:**
- Merging two ~20-episode datasets produces a 40-episode dataset that
  loads via `LeRobotDataset(repo_id, root=<merged>)` and returns valid
  frames + actions + obs.
- `lerobot-isaac-dashboard` Data Collection tab shows the merged
  dataset's episode breakdown (FPS, frames, action stats).
- The bridge skill (`lerobot_world_model_bridge`) successfully converts
  the merged dataset to HDF5 without schema errors.

### Phase 2 — Path A: adapter CLI accepts repeatable --dataset (0.5 day)

**Files modified:**

- `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py`:
  ```python
  parser.add_argument(
      "--dataset",
      action="append",            # changed: accepts multiple
      default=[],
      help="LeRobotDataset path(s). Repeat to train on a concatenation. "
           "Each path is forwarded as a separate --dataset.repo_id="
           "<repo>+--dataset.root=<path> pair to lerobot-train when "
           "lerobot 0.5+ supports list-style; otherwise the adapter "
           "pre-merges (via Path B) into a temp dir and trains on that.",
  )
  ```
- `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/policy_lerobot.py`:
  - Detect `len(args.dataset) == 1` → existing single-dataset behaviour
    unchanged (backwards compat).
  - Detect `len(args.dataset) > 1` → either:
    (i) emit `--dataset.repo_id=<repo>[,<repo>,...]` if lerobot 0.5+
        accepts that form (Phase 0 audit answers this), or
    (ii) call Phase 1 merge utility to a temp dir, forward as a single
         `--dataset.repo_id=<merged>`.

**Test:** the existing `tests/test_train_argparse.py` smoke covers
single-dataset; add a parameterised test that passes `--dataset a
--dataset b` and asserts the resulting subprocess cmd contains the
expected merged/list form.

**Acceptance:**
- Single-dataset CLI unchanged (no regression).
- `--dataset a --dataset b --dry_run` prints the resolved subprocess
  cmd showing both datasets passed through.
- `--dataset a --dataset b` actually runs (smoke against two tiny
  fixture datasets in `tests/fixtures/`).

### Phase 3 — Wire the WM bridge path (0.5 day)

`lerobot_world_model_bridge.lerobot_to_worldmodel(dataset_path=...)` takes
a single dataset. Two routes:

- Route 3a (recommended): the adapter `wm_dreamerv3` target pre-merges
  via Path B when given multiple `--dataset` args, then bridges the
  merged dataset to HDF5 as today. Zero changes to the bridge skill.
- Route 3b (alternative): teach the bridge skill to accept
  `dataset_paths: list[str]` and concatenate at HDF5 emission time.
  Cleaner but touches a shared skill.

**Acceptance for 3a:**
- `lerobot-isaac-train --target_arch dreamerv3 --dataset A --dataset B
  ... --dry_run` prints a single bridge cmd against the merged temp
  dir.

### Phase 4 — Tests + docs (0.5 day)

- Unit tests:
  - `tests/test_merge_datasets.py` — schema diff cases, episode renumbering, weight oversampling.
  - `tests/test_train_argparse.py::test_multi_dataset` — argparse + adapter dispatch.
- Doc updates:
  - `docs/runbook/02-collect-data.md` — section "Merging datasets".
  - `docs/runbook/03-train-policy.md` — section "Training on multiple datasets".
  - `docs/api-reference.md` — multi-dataset adapter API.
- `CLAUDE.md` "Common Pitfalls": add note about disk cost of Path B vs zero-overhead of Path A.

### Phase 5 — DR-synthetic + real-data validation run (1 day, optional)

Real end-to-end: take `datasets/kvgork/so101-pickplace1` + a
DR-augmented sibling, merge them via Path B, train a SmolVLA LoRA on
the merged set, eval open-loop on the held-out original-only episodes.

Acceptance: `pc_success` on real-only held-out ≥ pc_success when
trained on real-only (no regression from mixing in synthetic).

---

## Risks

| Risk | Mitigation |
|------|------------|
| MultiLeRobotDataset broken in 0.5.1 | Phase 0 audit catches this → fall back to Path B only |
| Schema mismatch (different FPS / camera resolutions) | Hard-error in merge utility with precise diff; suggest re-record |
| Disk balloon from oversampling | Cap `--weights` upsample factor at 5×; log final size before write |
| Episode index collisions | Merge utility re-numbers globally; tests assert uniqueness |
| Video re-encoding cost (MP4 chunks per camera key) | Just file copy; no re-encode unless camera keys mismatch |
| Bridge skill schema drift over time | Phase 3a keeps the bridge untouched; Phase 3b is opt-in |

---

## Effort Estimate

| Phase | Time |
|-------|------|
| 0 — compatibility audit | 0.5 day |
| 1 — merge utility | 1 day |
| 2 — adapter CLI | 0.5 day |
| 3 — WM bridge wiring | 0.5 day |
| 4 — tests + docs | 0.5 day |
| 5 — validation run (optional) | 1 day |
| **Total** | **~3 days** active eng + 1 day validation |

---

## Exit Criteria

Multi-dataset support is "landed" when ALL hold:

- `lerobot-isaac-train --dataset A --dataset B` trains successfully on
  the concatenation (no manual merge step needed by the user).
- `scripts/_merge_lerobot_datasets.py` produces a stable, dashboard-
  loadable merged LeRobotDataset that round-trips through the world-
  model bridge.
- Tests pass for single-dataset (unchanged), multi-dataset (new
  parameterised cases), and schema-mismatch error path.
- `docs/runbook/02-collect-data.md` documents the merge command + the
  CLI flag.
- A merged DR-synthetic + real-data SmolVLA LoRA training run completes
  end-to-end with non-empty pc_success.

---

## Out of Scope

- Stream-from-HuggingFace-Hub multi-dataset training (works in
  principle but introduces network dependency).
- Cross-task multi-task LR-per-task tuning (a separate sweep, not
  data-loader-level).
- Online dataset interleaving with curriculum staging (covered by
  `lerobot-curriculum-agent` orchestrator, not this plan).
- DR-augmentation pipeline changes (`lerobot-isaac-synthetic` already
  emits compatible LeRobotDataset format — just merge alongside real
  data).

---

## Cross-References

- `plans/2026-05-22-lora-sweep-next-steps.md` Phase 2 — Data Scaling
  (motivates this work)
- `plans/2026-05-23-wm-isaac-autoresearch-plan.md` — separate axis
- `scripts/_merge_lora_ckpt.py` — naming + structure template for the
  new dataset-merge utility
- `skills/lerobot_world_model_bridge/operations.py` — bridge skill
- `skills/lerobot_mimicgen_bridge` — has existing merge helpers
  (Parquet ↔ HDF5 + real/sim merge) worth referencing
