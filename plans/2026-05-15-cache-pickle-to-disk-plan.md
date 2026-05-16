# Cache Pickle-to-Disk — Conditional Plan (caveat #2 fix)

**Date:** 2026-05-15
**Owner:** TBD
**Status:** **COMPLETE.** Implemented + verified 2026-05-15:
- Smoke J (write): 940 s warmup, 4 s save → 6.94 GB on disk.
- Smoke K (load): 6.2 s load + "disk-cache HIT" + 9.04 step/s post-load.
- Savings vs warmup: **15.9 min per subsequent run**.
- Tests: 20/20 pass (4 new for disk cache).
- Wired through `LEROBOT_ISAAC_CACHE_DISK_DIR` env (default
  `outputs/cache_storage/`); signature key normalises `root` via
  `Path.resolve()` so relative + absolute paths produce identical keys.

**Parent plan:** `plans/2026-05-15-dataloader-gpu-decode-plan.md`
**Targets:** Phase 1 caveat #2 — "warmup is paid per-subprocess".

---

## Problem (one-paragraph recap)

`CachedDatasetWrapper` decodes 7491 frames in ~15 min via 4-worker
DataLoader. Each AR trial spawns a new subprocess → pays the 15 min
afresh. With 6 trials, **~90 min total = ~12% of a 12-hour overnight
budget** wasted re-decoding identical data.

---

## Fix shape

Pickle the cache (list of dicts of uint8 tensors) to disk after the
first warmup. On subsequent runs:

1. Compute a signature key from
   `(dataset_root, dataset_repo_id, n_rows, image_keys, lerobot_version)`.
2. If `cache_storage/<sha1(signature)>.pt` exists and the header
   matches, `torch.load` it (~20 s on SSD for 6.9 GB) and skip warmup.
3. Otherwise: warm up + dump to disk + proceed.

Expected gain: ~14 min/trial × (N − 1) trials = **84 min freed across
6 trials** when invalidation does not trigger.

Disk cost: 6.9 GB per (dataset × image_keys × n_rows) tuple. With one
dataset tonight: single 6.9 GB file under
`outputs/cache_storage/<hash>.pt`. Add to `.gitignore`.

---

## Trigger gate

Execute only if **both** hold:

1. AR plateau detector fires at trial ≤ 3 tonight (`.agent-state/<sess>/
   autoresearch/lerobot-policy-smolvla/plateau.json` shows
   `consecutive_non_improvements >= 3` before TRIAL=6 budget exhausts).
2. Remaining GPU budget is ≥ 2 h (enough to implement + test + redo
   trials with disk-cache enabled, leaving time for a final overnight
   loss curve).

If only one fires: park D until tomorrow. Sleep matters.

---

## Implementation outline (~90 min, single sitting)

1. **Disk format** (`cache_storage_io.py`, ~40 LOC)
   - `save_cache(cache, path)`: `torch.save({...}, path)` with header
     `{"version": 1, "signature": ..., "n_rows": ..., "compress_keys":
     ...}` + `cache` list payload.
   - `load_cache(path)`: read header, verify signature, return cache
     list. Raise `CacheSignatureMismatch` on mismatch.
2. **Signature** (~15 LOC)
   - `_signature(base) -> str`: sha1 of
     `f"{repo_id}|{root}|{len(base)}|{sorted(image_keys)}|{lerobot.__version__}"`.
3. **Wire into `CachedDatasetWrapper.__init__`** (~25 LOC)
   - New args: `cache_disk_path: Path | None = None`,
     `cache_disk_dir: Path | None = None`.
   - If `cache_disk_path` is None but `cache_disk_dir` is set, derive
     path = `cache_disk_dir / f"{signature}.pt"`.
   - If file exists and signature matches → `load_cache` and skip
     `_warmup`. Else warm up + `save_cache` at end.
4. **Adapter flag** in `train.py` + `cli_train_cached.py` (~10 LOC)
   - Read `LEROBOT_ISAAC_CACHE_DISK_DIR` env var; default
     `outputs/cache_storage/`.
   - Pass through to `CachedDatasetWrapper(... cache_disk_dir=...)`.
5. **Unit test** (~30 LOC)
   - Build a tiny fixture dataset.
   - Run wrapper once → confirm `.pt` file appears.
   - Build a new wrapper → confirm `_warmup` skipped (e.g. mock
     `base.__getitem__` to raise and verify it's never called).
   - Mutate signature components → confirm warmup re-runs.

Total: ~120 LOC + 1 test file.

---

## Acceptance criteria

- [ ] First wrapper instantiation with `cache_disk_dir` set warms up
      AND writes the `.pt` file.
- [ ] Second instantiation with matching signature loads in <30 s and
      returns the same rows (tensor-identical) as the first.
- [ ] Signature mismatch (e.g. add a new image key) triggers a fresh
      warmup, not a silent stale-cache hit.
- [ ] Smoke `bash scripts/_smoke_train.sh --arch smolvla --cache-frames
      --duration-s 600 --run-dir outputs/smoke-cache-disk-A` runs ONCE
      to populate the disk cache, then a second smoke at the same dir
      reports `[CachedDatasetWrapper] loaded cache from disk` in
      `train.log` and skips the preload progress lines.
- [ ] AR trial-to-trial warmup time drops from ~15 min to <1 min.

---

## Out of scope (defer until needed)

- Cross-dataset cache sharing (single dataset tonight).
- Compression of the on-disk cache (already uint8; further LZ4 won't
  help much at 6.9 GB).
- Distributed cache (single-machine training).
- Cache eviction policy (let the user `rm outputs/cache_storage/*` when
  disk pressure arrives).

---

## Rollback

If the disk-cache breaks anything: delete
`outputs/cache_storage/` and unset `LEROBOT_ISAAC_CACHE_DISK_DIR`.
Wrapper falls back to in-RAM warmup automatically.

---

## References

- Parent plan: `plans/2026-05-15-dataloader-gpu-decode-plan.md`
- DALI alternative: `plans/2026-05-15-dali-gpu-decode-plan.md`
- Wrapper code: `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/data/cached_dataset.py`
