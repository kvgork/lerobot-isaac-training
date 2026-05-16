# Dataloader Bottleneck Fix — Plan

**Date:** 2026-05-15
**Owner:** TBD
**Status:** **Phase 1 COMPLETE.** 7.0× throughput win confirmed via smoke G
(10.10 step/s post-warmup vs 1.44 baseline). Phase 2 (JPEG recompress)
deferred — current cap fits. Phase 3 (DALI) deferred per `plans/
2026-05-15-dali-gpu-decode-plan.md` decision gate.
**Trigger:** SmolVLA throughput smoke (`outputs/smoke-smolvla-*/`) showed
samples/s = 5.8 regardless of batch_size or num_workers. GPU is fully
underutilized (4.4 GB / 10 GB VRAM, ~25 % SM util). Bottleneck is CPU-side
PNG decode + image transforms.

## Phase 1 progress log (2026-05-15)

| Iteration | What changed | Result |
|-----------|--------------|--------|
| C — `outputs/smoke-smolvla-C-cached/` | First cache impl, default 8 GB cap | **FAIL.** `train.log` 0 bytes — subshell+SIGTERM dropped subprocess stdout. |
| D — `outputs/smoke-smolvla-D-cached-fix/` | Added stdbuf/tee pipeline | **FAIL.** Still 0 bytes — `tee \| tail -n 0` pipeline + `set -e` killed the script on the post-watchdog grep. |
| E — `outputs/smoke-smolvla-E-cached-fix2/` | Direct `>> file 2>&1` append, no subshell, no `set -e` | **partial.** Log writes correctly. Detection failed — cached float32 at 3.7 MB/row not uint8 at 0.92 MB/row. |
| F — `outputs/smoke-smolvla-F-debug/` | `_detect_image_keys` now matches `ndim in (3,4)`, schema dump | **detection PASS.** First-row schema reveals `(T, 3, H, W)` 4-D shape from delta_timestamps. uint8 compression engaged. But 180s smoke ran out before any preload tick. |
| G — `outputs/smoke-smolvla-G-cached-parallel/` | Added 4-worker parallel warmup via `torch.utils.data.DataLoader` | **PASS.** Warmup 925 s (15.4 min) for 7491 rows → 6.91 GB cached. Post-warmup steady state **10.10 step/s = 40.4 samples/s** (7.0× baseline). data_s 0.003 s (was 0.443 s), updt_s 0.095 s, peak VRAM 3.8 GB, loss 0.397 → 0.124 over 7613 steps. |

Permanent lessons captured (CLAUDE.md §Common Pitfalls):
- LeRobotDataset returns float32 (T,3,H,W) with `<key>_is_pad` masks.
- Watchdog-killed subshells lose stdout — use direct redirect + stdbuf.
- Parallel warmup is mandatory at ~7.5k row scale (serial would need ~50 min).

---

## Problem statement

`kvgork/so101-pickplace1` stores camera frames as **PNG bytes inline in the
LeRobotDataset Parquet** (`observation.images.d435_rgb.bytes`, format
`\x89PNG\r\n\x1a\n`, ~400 KB/frame, 7491 frames). At train time the
dataloader path is:

```
parquet row → bytes blob → PIL.Image.open(PNG) → np.asarray → tensor
                                                  → CPU resize/normalize
                                                       → pin_memory → cuda copy
```

Each step at `batch_size=4` waits ~0.44 s for data; the GPU update itself
takes 0.13 s. `num_workers=4` only drops `data_s` to 0.42 s — lerobot
already prefetches, the wall is the PNG decoder, not concurrency.

Quantified impact (RTX 3080, batch=4, smolvla, 2026-05-15 smoke):

| Component | Wall time / step |
|-----------|------------------|
| PNG decode + transform (data_s) | 0.42 s |
| forward + backward + step (updt_s) | 0.13 s |
| GPU SM util | ~25 % |
| Achieved samples/s | 5.8 |
| Theoretical samples/s if GPU-bound | ~30 |

20 k steps at current speed = 230 min. Same training **GPU-bound** would
take ~50 min — a 4–5× speedup is available.

---

## Goals

1. **Primary:** Lift `samples/s` from 5.8 → ≥ 20 (3.5× minimum). 4× and we
   stop being a tonight-problem.
2. **Secondary:** Reproducible measurement — extend
   `scripts/_smoke_train.sh` to emit a one-line metric that can be
   compared across approaches.
3. **Non-goals:** Re-record the dataset, change the LeRobotDataset schema
   upstream, refactor the lerobot CLI.

---

## Candidate approaches (ranked by ROI)

### A. In-RAM frame cache (LOW EFFORT, BIG WIN)

Wrap `LeRobotDataset` in a thin module that decodes every PNG once at
`__init__`, caches uint8 NHWC arrays in memory, and serves them
zero-decode from `__getitem__`.

- Dataset size: 7491 frames × 320 × 240 × 3 = **1.7 GB** uint8 — fits
  easily in 32 GB system RAM.
- Decode cost: 7491 × 40 ms ≈ **5 min** one-time at process start.
- Per-step cost after warmup: only resize/normalize (~50 ms with
  pillow-simd, ~10 ms with `torchvision.transforms.v2` on tensor).
- Expected throughput: **15–25 samples/s** (3–4×).

**Implementation surface:** one new file
`src/lerobot_isaac_adapters/data/cached_dataset.py` + a flag in the
adapter (`--cache_frames`) that swaps the dataset class. Lerobot's
internal `LeRobotDataset.__getitem__` is overridable.

**Risks:** memory pressure on larger datasets — gate by RAM threshold;
fall back to disk path if dataset > 8 GB.

---

### B. Pre-decode to JPEG (MEDIUM EFFORT, MODEST WIN)

Re-encode the PNG bytes as JPEG (q=95) in-place. JPEG decode is ~5–10×
faster than PNG via libjpeg-turbo, and the LeRobotDataset spec already
accepts JPEG bytes in the same column.

- Conversion job: one-shot script
  `scripts/_recompress_dataset_to_jpeg.py`, reads each parquet,
  re-encodes the image column, writes back with a `.jpeg` suffix dataset
  alongside the original.
- Dataset size: 7491 × ~30 KB JPEG = **220 MB** (vs current 3 GB PNG).
- Per-step decode: ~5 ms/frame × 4 = 20 ms ≈ **0.05 s data_s**.
- Expected throughput: **18–25 samples/s** (3–4×).

**Risks:** lossy compression. q=95 is visually indistinguishable but
loss is real. Document the conversion + checksum the alternate dataset
so it can be regenerated.

---

### C. Pre-decode to LeRobot videos/ MP4 layout (MEDIUM EFFORT, MODEST WIN)

LeRobot's canonical layout puts videos under `videos/chunk-XXX/<key>/
episode_NNNN.mp4` and parquet stores frame_index pointers. The dataloader
then uses `torchcodec` (default `pyav`) to seek. `torchcodec`
has a CUDA backend for hardware-decoded frames.

- Conversion: ffmpeg one-shot per episode, ~30 s/episode × 20 = 10 min.
- Per-step decode: pyav ~10 ms/frame batched, torchcodec-cuda ~3 ms.
- Expected throughput: **20–35 samples/s** (3.5–6×).

**Risks:** `dataset.video_backend=pyav` works today but `torchcodec` had
prior libavutil version skew on this workspace (see CLAUDE.md pitfalls).
Need a separate smoke probe before relying on it.

---

### D. NVIDIA DALI GPU decode (HIGH EFFORT, BIGGEST WIN, LONG TERM)

Replace the entire dataloader with a DALI pipeline: GPU-side JPEG/PNG
decode, GPU-side resize/normalize, zero copy into the train loop.

- Expected throughput: **40+ samples/s** (7×+, GPU becomes the wall).
- Effort: rewrite `LeRobotDataset.__getitem__` path or fork lerobot.

**Status:** parked. Revisit only after A or B/C lands and we still need
more throughput.

---

## Recommended sequence

1. **Phase 1 (this week):** Implement A. Measure. If samples/s ≥ 20,
   declare victory and unblock tonight-scale training. Estimated effort:
   **3–4 h** including a unit test and a comparative smoke entry.
2. **Phase 2 (next week, optional):** If A's RAM ceiling becomes a
   problem (multi-dataset training, full SO-101 corpus), add B as the
   on-disk fallback. **2–3 h**.
3. **Phase 3 (deferred):** D. Only if we end up training models that are
   GPU-bound at the A/B floor.

---

## Acceptance criteria

For Phase 1:

- [ ] `scripts/_smoke_train.sh --arch smolvla --cache-frames --batch 4
      --duration-s 300` reports `samples/s ≥ 20`.
- [ ] `scripts/_smoke_train.sh --arch smolvla --batch 4 --duration-s 300`
      (no cache) still reports ~5.8 samples/s — regression guard.
- [ ] Loss curve at step 1000 matches the non-cached baseline within
      ±1 % (deterministic seed) — confirms semantic equivalence.
- [ ] CLAUDE.md throughput note updated with the new measured floor.
- [ ] `plans/2026-05-15-dataloader-gpu-decode-plan.md` marked
      `Status: phase 1 complete`.

For Phase 2 (when triggered):

- [ ] `scripts/_recompress_dataset_to_jpeg.py` produces a parquet-only
      JPEG dataset at `datasets/kvgork/so101-pickplace1-jpeg/`.
- [ ] Smoke on the JPEG dataset reports `samples/s ≥ 15` without the
      RAM cache.
- [ ] Loss equivalence within ±2 % (lossy, slightly larger band).

---

## Tasks (Phase 1, ready to execute)

1. Add `--num-workers N` and `--cache-frames` plumbing to
   `lerobot_isaac_adapters.train` and the wrapper at
   `src/lerobot_isaac_adapters/targets/policy_lerobot.py`.
2. Implement `CachedFrameDataset` in
   `src/lerobot_isaac_adapters/data/cached_dataset.py`:
   - Subclass `LeRobotDataset`.
   - In `__init__`, iterate every parquet row once, decode the
     `observation.images.*` columns into a dict `{(ep_idx, frame_idx):
     np.ndarray uint8}`.
   - Override `_load_image_from_parquet` (or the equivalent helper) to
     hit the cache.
   - RAM ceiling: 8 GB hard cap; raise `MemoryError` with a clear hint
     to use approach B.
3. Add adapter flag wiring: when `--cache-frames` is passed, swap the
   dataset class via lerobot's policy factory hook (or monkey-patch).
   Document the patch surface in the adapter docstring.
4. Add `--cache-frames` flag to `scripts/_smoke_train.sh` and forward
   it to the adapter.
5. Unit test in
   `src/lerobot-isaac-adapters/tests/test_cached_dataset.py`:
   `len()`, `__getitem__` shapes match the un-cached dataset on a
   miniature 3-frame fixture.
6. Smoke A+B comparison entry in this plan's "Phase 1 complete" section
   (paste the two `smoke_report.txt` outputs).

---

## World-model dataset applicability

WM training (`--target_arch dreamerv3` / `--target_arch lewm`) reads
**HDF5** produced by the `lerobot_world_model_bridge` skill, NOT the
parquet+PNG path. Bottleneck profile differs:

| Pipeline stage | LeRobot policy (this plan) | DreamerV3 HDF5 | LeWM HDF5 |
|----------------|---------------------------|---------------|-----------|
| Codec decode | PNG/JPEG, ~30 ms/frame | none (uint8 raw) | none (uint8 raw) |
| Disk read | parquet rows, slow on cold cache | `h5py` slice, fast | `h5py` slice, fast |
| Transform | resize + normalize on CPU | resize 64×64, on CPU | resize 96×96, on CPU |
| **Bottleneck after A** | ~10 ms transforms | ~3 ms slice + transforms | ~5 ms slice + transforms |

**A applies to WM datasets, with two changes:**

1. Design `CachedDatasetWrapper(base, cache_keys)` as a **generic torch
   `Dataset` wrapper**, not a LeRobotDataset subclass. Accept any base
   that exposes `__len__` + `__getitem__`. The HDF5 path then becomes
   `CachedDatasetWrapper(HDF5SequenceDataset(...), cache_keys=["image",
   "action"])`.
2. Skip the PNG-decode-once warmup for HDF5 (data is already raw uint8).
   The wrapper just memcpys h5 slices into a contiguous numpy buffer at
   init.

Acceptance for WM applicability:

- [ ] `CachedDatasetWrapper` unit test against both a LeRobotDataset
      fixture AND a tiny HDF5 fixture (uint8 frames + actions).
- [ ] WM training smoke (`bash scripts/run_full_pipeline.sh
      --target-arch dreamerv3 --skip-policy --skip-eval`) with
      `--cache-frames` reports ≥ 1.3× steps/s vs without.

## Out of scope

- Multi-GPU sharding (single RTX 3080 box).
- Different policy archs — fix benefits diffusion + ACT identically.
- Re-recording the dataset with on-disk JPEG from `robot-data-recorder`
  (consider for v2; out of scope for the fix).

---

## References

- `outputs/smoke-smolvla-2026-05-15-103452/smoke_report.txt` —
  baseline 1.44 step/s
- `outputs/smoke-smolvla-A-nw4-b4/smoke_report.txt` — nw=4 1.45 step/s
- `outputs/smoke-smolvla-B-nw4-b8/smoke_report.txt` — batch=8 0.73 step/s
- `CLAUDE.md §Common Pitfalls` — SmolVLA throughput entry
- `docs/internals/data-pipeline.md` — current data lifecycle
- LeRobot dataset docs: video_backend, frame loading paths
- NVIDIA DALI: https://github.com/NVIDIA/DALI (phase 3 ref)
