# DALI GPU Decode — Future Plan (Approach D)

**Date:** 2026-05-15
**Owner:** TBD
**Status:** draft / deferred
**Trigger:** After Phase 1 (in-RAM cache, plan
`2026-05-15-dataloader-gpu-decode-plan.md`) lands and `samples/s` is still
under the GPU's theoretical ceiling, OR multi-dataset training kills the
RAM-cache approach. **Do not start D before A has shipped and been
measured.**

---

## Why D, not just A forever

A (in-RAM cache) is a one-time decode + memcpy. It removes the PNG cost
but leaves on the CPU:

| Per-step cost (after A) | Time |
|-------------------------|------|
| `numpy → torch.tensor` copy | ~5 ms |
| `torchvision.transforms.v2` resize + normalize | ~10 ms |
| `pin_memory` + `to(cuda)` | ~5 ms |
| Cross-process pickle (DataLoader workers) | ~5 ms |

These add ~25 ms of per-batch CPU/host work that scales linearly with
batch size and with the number of camera streams. SO-101 currently has
one camera; future hardware (wrist + overhead) doubles the cost. At
batch=8 + two cameras we expect A to floor near 12–15 samples/s.

D removes that floor by doing decode + resize + normalize + tensor copy
**on the GPU** in a fused pipeline, leaving only a zero-copy handoff to
the train loop.

---

## Architecture

```
┌─────────────────────────┐
│  parquet rows (CPU)     │  PNG bytes blob, state, action
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  DALI ExternalSource    │  pass bytes blob into DALI
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  fn.decoders.image      │  nvJPEG / nvDecodeJpeg2k → uint8 HWC on GPU
│    (device="mixed")     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  fn.resize / fn.crop    │  GPU resize to policy-required HxW
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  fn.normalize / cast    │  uint8 → float32, /255, mean/std
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  DALI → PyTorch         │  DALIGenericIterator → ready-on-GPU tensors
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  lerobot train_step()   │  no per-step .to(cuda) needed
└─────────────────────────┘
```

DALI handles only the image columns. State + action are tiny tensors —
keep them on the CPU side and `.to(cuda, non_blocking=True)` in the
collate fn.

---

## Required compat

- **CUDA:** DALI 1.40+ supports CUDA 12.x. Current train-policy env
  needs verification (`nvidia-smi` reports the runtime).
- **GPU compute capability:** RTX 3080 = sm_86, supported.
- **JPEG vs PNG decoders:** DALI's `fn.decoders.image` supports both via
  nvJPEG and a CPU-fallback PNG path. PNG-on-GPU was added in DALI 1.32
  via a software decoder running on the GPU SIMT lanes; throughput is
  ~5× CPU but slower than JPEG. **Recommend: pair D with the JPEG
  recompress step from approach B** so the GPU decoder runs in nvJPEG
  fast path.
- **Dependency weight:** `nvidia-dali-cuda120` wheel ≈ 700 MB on disk.
  Adds 5–10 s of env-resolve time. Gate behind a feature flag.

---

## Phases

### Phase 1: Spike (1 day)

- Install DALI in a scratch venv (NOT the train-policy env yet)
- Write a 50-line standalone benchmark:
  - Load 1024 PNG/JPEG blobs from the parquet
  - Run them through a DALI pipeline that decodes + resizes to 224×224
  - Measure throughput on the RTX 3080
- Acceptance: ≥ 200 images/sec single-threaded. (Skip phase 2 if DALI
  can't beat A's effective rate.)

### Phase 2: LeRobotDataset adapter (2-3 days)

- Implement `DALILeRobotDataset` wrapper in
  `lerobot-isaac-adapters/src/lerobot_isaac_adapters/data/dali_dataset.py`
- API: same `__len__` + `__getitem__` contract as
  `LeRobotDataset`, but `__getitem__` returns GPU tensors directly
- Internal: build a per-batch DALI pipeline; route image columns
  through DALI and pass non-image columns through unchanged
- Handle frame indexing: DALI's ExternalSource needs a Python callable
  that yields the next batch's PNG bytes
- Unit test: tensor shapes + dtype + device match the in-RAM-cache
  baseline within ±1 numeric ulp after normalization

### Phase 3: Lerobot integration (1-2 days)

- The lerobot `lerobot-train` CLI imports `LeRobotDataset` from
  `lerobot.datasets.factory.make_dataset`. **Three integration
  options**, pick at implementation time:
  1. **Monkey-patch via sitecustomize.py** — drop a custom
     `sitecustomize.py` on `PYTHONPATH` that replaces
     `lerobot.datasets.factory.make_dataset` before lerobot imports.
     Least invasive. Hardest to debug.
  2. **Fork the CLI** — copy lerobot's `train.py` to
     `lerobot_isaac_adapters/dali_train.py` and import the modified
     dataset directly. Easy to debug. Drifts from upstream over time.
  3. **Upstream a hook** — PR a `dataset_cls_override` arg to
     lerobot. Long-term win. Cuts our patching cost but slow to land.
- Recommend (1) for the first cut; revisit (3) after we know the
  performance numbers warrant a real upstream change.

### Phase 4: Bench + tune (1 day)

- Smoke `_smoke_train.sh --arch smolvla --dali --batch 8 --duration-s 300`
- Compare against:
  - baseline (no cache)
  - A (in-RAM cache)
  - B (JPEG recompress)
- Tune DALI knobs: `prefetch_queue_depth`, `device_memory_padding`,
  `host_memory_padding`. Document settings used.

### Phase 5: Cleanup / docs (½ day)

- Add `--dali` flag to `lerobot_isaac_adapters.train` + propagate
  through `run_full_pipeline.sh` and `_run_smolvla_tonight.sh`
- Update `docs/internals/data-pipeline.md` with the DALI path
- Update CLAUDE.md throughput entry with new numbers

---

## Acceptance criteria

- [ ] `scripts/_smoke_train.sh --arch smolvla --dali --batch 8
      --duration-s 300` reports `samples/s ≥ 40`.
- [ ] Loss curve at step 1000 matches the in-RAM-cache baseline within
      ±2 % (deterministic seed). Tighter band than B because DALI
      normalization is identical maths, just on GPU.
- [ ] VRAM peak under the same train run stays ≤ 9 GB on RTX 3080
      (leaves headroom for SmolVLA model + activations).
- [ ] env install: `pixi install -e dali` cleanly resolves DALI without
      blowing up the existing train-policy env.

---

## World-model dataset applicability

WM training pipelines read HDF5 with **raw uint8 arrays** (no codec). DALI
gain is reduced because the expensive nvJPEG path doesn't apply. Remaining
DALI ops still useful:

| Stage | Policy (PNG/JPEG) | WM (HDF5 raw uint8) |
|-------|-------------------|---------------------|
| `fn.decoders.image` | nvJPEG fast path (huge win) | not used |
| `fn.resize` (GPU) | ~5 ms saved/batch | ~5 ms saved/batch |
| `fn.normalize` / `cast` | ~5 ms saved/batch | ~5 ms saved/batch |
| host→device copy | eliminated | eliminated |

Net WM gain from D: estimated 1.2–1.5× steps/s — smaller than the 4–7× for
the policy path. Worth doing only as a side effect of the policy
implementation; do not justify D solely by WM speedup.

**Reusable DALI pipeline shape:**

- Provide a `build_dali_pipeline(source, image_keys, h_w, mean_std,
  decode=True/False)` factory in `lerobot_isaac_adapters/data/dali.py`.
- Policy path: `decode=True`, `source=parquet_external_source(...)`.
- WM path: `decode=False`, `source=hdf5_external_source(...)`.

Acceptance for WM applicability (if D is implemented):

- [ ] Same `build_dali_pipeline` powers both LeRobotDataset and
      HDF5SequenceDataset.
- [ ] DreamerV3 smoke (`--target-arch dreamerv3 --dali`) reports
      ≥ 1.2× steps/s vs A-only.

## Risk register

| Risk | Mitigation |
|------|-----------|
| DALI compute-capability mismatch on RTX 3080 (sm_86) | Phase 1 spike catches this in 1 hour |
| nvJPEG PNG fallback slower than expected | Pair with B (JPEG recompress) |
| Lerobot CLI factory change breaks monkey-patch | Pin lerobot version in `pixi.toml`; fall back to forking |
| Loss equivalence fails — DALI normalize float order != torch | Use DALI `fn.normalize` with explicit mean/std matching torchvision |
| 700 MB DALI wheel inflates CI image | Add a separate pixi feature `dali`; default envs skip it |

---

## Decision gate

Run D only if **all** of these hold after A ships:

1. A's measured `samples/s` is < 25 — i.e. we still need more speed.
2. Phase 1 spike shows DALI ≥ 200 images/sec on this GPU.
3. We have a planned use case for ≥ 2 camera streams (otherwise A is
   enough for SO-101 tonight).

If any fails: park D, revisit after upstream lerobot adds a dataset
hook (option 3 in phase 3).

---

## References

- `plans/2026-05-15-dataloader-gpu-decode-plan.md` — parent plan (A is
  Phase 1, D is Phase 3)
- NVIDIA DALI docs: https://docs.nvidia.com/deeplearning/dali/user-guide/
- DALI LeRobot proof-of-concept (community, not first-party): search
  github.com/topics/dali-loader
- LeRobot dataset factory:
  `.pixi/envs/train-policy/lib/python3.12/site-packages/lerobot/datasets/factory.py`
