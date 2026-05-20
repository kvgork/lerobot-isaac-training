# Autoresearch Domain Knowledge — LeRobot + Isaac Lab Stack

**Purpose:** Give the autoresearch ML proposer worker stack-specific facts so its
mutations are grounded in our reality (RTX 3080 10 GB, lerobot 0.5+, sheeprl
0.5.8.dev, Isaac Sim 6.0, SO-101 6-DOF action). Without this reference the
proposer falls back to generic Adam-1e-3 defaults that misfire for pretrained
SmolVLA fine-tunes, sheeprl-namespaced hyperparams, and Hydra `+` overrides.

**Read this file when:** the program target involves any of
`lerobot-isaac-adapters`, `target_arch`, `lerobot-train`, `sheeprl`, `dreamer_v3`,
`le_world_model`, `pc_success`, `recon_loss`, `pred_loss`.

---

## 1. Hardware Constraints (RTX 3080 10 GB)

| Constraint | Value | Hard / soft |
|------------|-------|-------------|
| VRAM ceiling                | 10 GB                            | hard — proposer must never raise batch_size above limits below |
| Diffusion policy batch      | ≤ 16 (8 typical)                 | hard |
| SmolVLA fine-tune batch     | ≤ 8 (4 typical, image inputs)    | hard |
| ACT batch                   | ≤ 8 (chunked attention)          | hard |
| DreamerV3 `per_rank_batch_size` | ≤ 16 (8 typical, 64×64 images) | hard |
| LeWM minimal-trainer batch  | ≤ 16 (96×96 images, 4-conv encoder) | hard |
| Isaac DR replay `--num_envs` | 4 (1 for record/replay)         | hard |

OOM recovery: `train_wrapper.py` halves batch_size once and retries. Two
consecutive OOMs in `failure_hints` → proposer MUST halve batch_size, not
just mutate other hyperparams.

---

## 2. Architectures We Dispatch (target_arch values)

`lerobot_isaac_adapters.train --target_arch X`:

| Arch              | Backend           | Metric (stdout)  | Direction | Notes |
|-------------------|-------------------|------------------|-----------|-------|
| `smolvla`         | `lerobot-train`   | `pc_success`     | maximize  | Pretrained on SO-101 data; fine-tune lr range 1e-5 → 1e-4. |
| `act`             | `lerobot-train`   | `pc_success`     | maximize  | ACT with `--policy.chunk_size`; lr 1e-4 → 1e-3. |
| `diffusion`       | `lerobot-train`   | `pc_success`     | maximize  | Best lr we observed: 5e-5 → 1e-4. Diffusion needs more steps than ACT. |
| `dreamerv3`       | `sheeprl` subprocess | `recon_loss`  | minimize  | Hydra config knobs under `algo.*` not flat. |
| `le_world_model`  | in-process `_lewm_minimal` | `pred_loss` | minimize | Embeddings predictor; tiny model. |

**Architecture swap as an operator:** if metric plateaus on one arch, swap to the
next in this priority order (lerobot-policy programs only):
`smolvla → diffusion → act` (SmolVLA pretrained gives best zero-shot;
diffusion is the strongest from-scratch baseline; ACT is fastest).

---

## 3. LeRobot 0.5+ CLI Flag Shape

`policy_lerobot.run()` emits exactly this flag set:

```
lerobot-train
  --policy.type=<smolvla|act|diffusion|...>
  --dataset.repo_id=<org/name>          # derived from path tail when local
  --dataset.root=<absolute path>        # added when --dataset is a directory
  --dataset.video_backend=pyav          # NEVER torchcodec — libavutil mismatch
  --batch_size=<N>                      # was --training.batch_size in lerobot <0.5
  --steps=<N>                           # was --training.num_steps
  --optimizer.lr=<F>                    # was --training.lr (flat)
  --seed=<N>
  --output_dir=<run_dir>
  --policy.push_to_hub=false            # local-only default; override via remainder
  --save_freq=<N>                       # auto-scaled by run_full_pipeline.sh
  --log_freq=50
```

If the proposer wants to mutate optimizer type, the correct flag is
`--optimizer.type=adamw` (passed through `args.remainder`). Do NOT emit
`--training.*` — those were removed in 0.5.

**Common mutations the proposer can make through remainder args:**

```
-- --optimizer.type=adamw --optimizer.weight_decay=1e-4
-- --optimizer.type=sgd --optimizer.lr=1e-2 --optimizer.momentum=0.9
-- --policy.chunk_size=100   # ACT only
-- --policy.n_action_steps=8 # diffusion only
-- --scheduler.type=cosine_annealing
-- --eval.n_episodes=10      # only if env is registered
```

---

## 4. sheeprl DreamerV3 Hydra Path

`wm_dreamerv3.run()` builds this command:

```
python -m sheeprl                                        # NOT sheeprl.cli
  --config-dir=<plugin>/configs                          # ships custom_hdf5.yaml
  exp=dreamer_v3
  env=custom_hdf5
  +env.dataset_path=<bridge>/dreamerv3_data.hdf5         # `+` REQUIRED for hydra append
  algo.per_rank_batch_size=<N>                           # NOT algo.batch_size
  algo.world_model.optimizer.lr=<F>                      # nested under world_model
  algo.total_steps=<N>                                   # NOT total_steps (root)
  seed=<N>
  hydra.run.dir=<output_dir>                             # NOT checkpoint.save_dir
```

Knob ranges:
- `algo.world_model.optimizer.lr` — 1e-5 to 3e-4 (1e-4 default is good)
- `algo.per_rank_batch_size` — {4, 8, 16}
- `algo.per_rank_sequence_length` — {32, 64} (64 default)
- `algo.world_model.discrete_size` — {16, 32, 64} (32 default)
- `algo.world_model.stochastic_size` — {16, 32, 64} (32 default)
- `algo.replay_ratio` — {1, 2, 4} (higher = more updates per env step)
- `algo.gamma` — 0.997 default; rarely worth changing
- `env.capture_video=False` — REQUIRED for our custom_hdf5 env (no render mode)

---

## 5. LeWM (in-process) Knobs

`_lewm_minimal` is a 790K-param model. Accepts plain `--lr`, `--batch_size`,
`--steps`, `--log_every`. Internal knobs only changed by editing
`_lewm_minimal.py`:

- `embed_dim = 128` — bump to 256 for capacity
- Encoder channels: 32 / 64 / 128 / 256 — proposer can multiply by 1.5 for
  capacity increase (modify_data_pipeline operator domain).
- Image size pinned at the bridge output (96×96 default).

---

## 6. Isaac DR Replay Knobs (modify_data_pipeline operator)

`replay_runner.py` exposes:

- `--n_variants` — synthetic variants per source episode (default 5)
- `--max_episodes` — cap source episodes
- DR config object passed in-process to `_apply_dr_config()`:
  - `object_pose` — {pos_jitter_xy_m, rot_jitter_rad}
  - `lighting` — {intensity_jitter_factor, direction_jitter_deg}
  - `friction` — {static_min, static_max, dynamic_min, dynamic_max}
  - `camera_fov` — {fov_jitter_deg}

`add_regularization` for our domain means "increase DR variance" — push
the friction/pose/lighting ranges wider until the metric stops improving.

---

## 7. Dataset Path Convention

- Real teleop: `datasets/<org>/<repo_id>/` (e.g. `datasets/kvgork/so101-pickplace1`)
- Synthetic DR replay: `outputs/<run>/synthetic/` (autolinked under `datasets/synthetic/`)
- Bridged HDF5: `outputs/<run>/bridge/{dreamerv3_data,leworldmodel_data}.hdf5`
- Merged (real + sim_dr): use `lerobot_isaac_synthetic.merge_utilities.merge_datasets`

The proposer should NEVER hardcode `lerobot/pusht` style hub repo_ids in our
programs — we use local SO-101 datasets. Adapter handles repo_id derivation
automatically when `--dataset` points at a directory.

---

## 8. Metric Stdout Contract

Every backend MUST emit a final stdout line:
```
<metric_name>=<float>
```

Autoresearch executor regex: `(\w+)[=:\s]+([0-9.eE+\-]+)` — captures the LAST
match in stdout. `train_wrapper.py` guarantees a fallback `pc_success=0.0`
sentinel if the subprocess emits nothing parseable.

| Arch                     | Primary metric        | Stdout line example          |
|--------------------------|-----------------------|------------------------------|
| smolvla / act / diffusion | `pc_success`         | `pc_success=0.741`           |
| dreamerv3                | `recon_loss`          | `recon_loss=0.0317`          |
| le_world_model           | `pred_loss`           | `pred_loss=0.0214`           |

**Special case — open-loop eval:** When the policy has no closed-loop env
(SO-101 doesn't), `scripts/_open_loop_eval.py` computes `pc_success =
1 / (1 + action_mse)`. The metric is a proxy, not closed-loop success.

---

## 9. Operator Priority (refined for our stack)

In **explore** mode, cycle operators in this order:

1. `tune_hyperparams` — learning rate first (×3 / ÷3 from baseline),
   then batch size, then weight_decay.
2. `change_scheduler` — try cosine_annealing if not present.
3. `modify_data_pipeline` — increase DR variance for sim2real robustness
   (only if synthetic dataset is in use).
4. `add_regularization` — dropout 0.1 / 0.2 / 0.3 in policy heads;
   weight_decay 1e-4 / 1e-3 for AdamW; gradient clip 1.0.
5. `change_optimizer` — Adam → AdamW; AdamW → SGD+momentum if
   generalization gap.
6. **`change_target_arch`** (new operator) — when metric plateaus on
   one arch, swap `--target_arch` to next in priority order. Only
   applies to policy programs (not WM).

In **refine** mode: prefer operators with positive Δmetric in history.
Use the `metric_history.json` produced by the executor.

---

## 10. Known Failure Hints (proposer must recognize)

When `failure_hints` contains any of these, take the prescribed action:

| Hint pattern                                     | Required action |
|--------------------------------------------------|-----------------|
| `OOM`, `cuda out of memory`, ×2 in a row         | halve batch_size |
| `policy.repo_id argument missing`                | append `--policy.push_to_hub=false` (already default; means user override broke it) |
| `Could not find rigid body when resolving`       | scene cfg issue — proposer should NOT touch; flag for human |
| `episode_data_index` AttributeError              | lerobot version drift — abort run, do not retry |
| `Map: 0%...` then crash                          | parquet write — set `LEROBOT_TRAIN_TIMEOUT` higher, do not mutate |
| `Failed to acquire interface: Urdf`              | Isaac Sim URDF importer not initialized; not a hyperparam issue |

---

## 11. Budget Heuristics

For RTX 3080 on the SO-101 dataset (20 eps × 7491 frames):

| Arch        | Steps to plateau (typical) | Wall-clock @ 2-3 sps |
|-------------|----------------------------|----------------------|
| diffusion   | 30k–80k steps              | 4-8 hours            |
| smolvla     | 10k–30k steps (pretrained) | 2-5 hours            |
| act         | 50k–150k steps             | 6-15 hours           |
| dreamerv3   | 100k–500k policy_steps     | 4-20 hours           |

For autoresearch budget: per-experiment 1-2 hours, max_experiments 10,
plateau_limit 3. For tight smoke runs use `programs/lerobot-policy-short.md`
with 8-min per-exp, 3 exps, plateau 2.

---

## 12. References

- Pipeline overview: [`docs/pipeline-overview.md`](../docs/pipeline-overview.md)
- Adapter internals: [`docs/internals/training-dispatch.md`](../docs/internals/training-dispatch.md)
- Autoresearch internals: [`docs/internals/autoresearch-integration.md`](../docs/internals/autoresearch-integration.md)
- LeRobot ref: [`docs/research/dreamerv3-reference.md`](../docs/research/dreamerv3-reference.md), [`docs/research/leworldmodel-reference.md`](../docs/research/leworldmodel-reference.md)
- Upstream proposer worker: `${CLAUDE_CODE_ROOT}/agents/workers/autoresearch-ml-proposer-worker.md`
- Upstream skill: `${CLAUDE_CODE_ROOT}/skills/autoresearch/SKILL.md`

---

## 13. LoRA Fine-tuning (SmolVLA only)

When `program_config.use_lora: true` OR `args.use_lora=1`, the adapter
constructs a `peft.LoraConfig` and wraps `SmolVLAPolicy.model` via
`peft.get_peft_model()` before `lerobot-train` starts. ONLY supported for
`target_arch=smolvla`.

### 13.1 Flag shape

The adapter accepts five LoRA flags on `lerobot_isaac_adapters.train`:

```
--use_lora
--lora_rank <int>            # default 8
--lora_alpha <int>           # default 16   (= 2 * default_rank)
--lora_dropout <float>       # default 0.0
--lora_target_modules <spec> # preset name OR csv of layer suffixes
```

Forwarded verbatim through `train_wrapper._build_cmd`. No remainder-arg
injection needed.

### 13.2 Target-module presets

| Preset       | Modules                                           | LoRA params (r=64) | Use when |
|--------------|---------------------------------------------------|--------------------|----------|
| `attn_qv`    | q_proj, v_proj (all attention layers)             | ~7.9 M (1.7%)      | Default. Paper recipe. |
| `attn_qkvo`  | q_proj, k_proj, v_proj, o_proj                    | ~15.7 M (3.5%)     | Broader coverage; HF VLM default. |
| `expert_only`| q_proj, v_proj — only in `lm_expert` submodule    | ~1.6 M             | Freeze VLM entirely; lowest VRAM. |

Raw csv (e.g. `"q_proj,v_proj,gate_proj"`) also accepted — applied as suffix
matches across the entire policy.model module tree.

### 13.3 Rank sweep range (RTX 3080, SmolVLA, SO-101 datasets) — VLA SPECIFIC

**LoRA ranks for VLAs are MUCH HIGHER than for LLMs/VLMs.** Sweep
`lora_rank ∈ {16, 32, 64, 128}`. Saturation expected at r=64–128.

Evidence (see plan §2.6 correction + `project-context/papers/vla-lora-rank-research-2026-05-20.md`):

- **LoRA-SP (arXiv:2603.07404)** ran the exact SmolVLA ablation across
  r ∈ {8, 16, 32, 64, 128}. r=8 → 0% success on most tasks; r=128 →
  40-93% per task. Single-task performance rose monotonically through
  the full range with no saturation below r=128.
- **HuggingFace LeRobot PEFT docs** default to `--peft.r=64
  --peft.lora_alpha=64` for SmolVLA.
- **OpenVLA (arXiv:2406.09246)** uses r=32 (single-task LIBERO).
- **Reasoning:** VLAs map vision → continuous actions, OOD vs VLM
  text-token pretraining. The rank-r subspace must span new feature
  directions absent from pretrained weights → wider subspace needed.

`{4, 8}` are null conditions for manipulation — skip them.

Tie `lora_alpha` to rank: sweep `alpha ∈ {r, 2*r}`. Effective scale
`alpha/r ∈ {1.0, 2.0}`. At r=128, k=1 is the safer starting point.

### 13.4 VRAM budget at LoRA enabled (VLA ranks)

| r   | LoRA params (attn_qv) | Extra VRAM (opt+grad) | Total at batch=4 |
|-----|------------------------|------------------------|------------------|
| 16  | 1.96 M                 | ~20 MB                 | ~7.4 GB |
| 32  | 3.93 M                 | ~40 MB                 | ~7.5 GB |
| 64  | 7.86 M                 | ~80 MB                 | ~7.5 GB |
| 128 | 15.72 M                | ~160 MB                | ~7.6 GB |

`attn_qkvo` doubles the LoRA-params column (still <320 MB optimizer
state at r=128). Headroom vs full-FT SmolVLA: LoRA frees ~3-4 GB by
skipping optimizer state on the 450M base. Keep `batch_size_max=8` to
leave slack at r=128 + attn_qkvo.

### 13.5 `tune_lora` operator (proposer)

New operator added to `autoresearch-ml-proposer-worker.md` §Mutation
Operators. Mutates exactly one of `{lora_rank, lora_alpha, lora_dropout,
lora_target_modules}` per call. See `plans/2026-05-19-lora-autoresearch-plan.md`
§3.5 for the rules.

**Auto-extension past ladder top** (opt-in via program flags):

```yaml
allow_rank_extension: true       # gates extension
rank_extension_cap: 512          # hard upper bound
rank_rising_threshold: 0.05      # delta to count as "still rising"
```

When enabled and pc_success(r_top) − pc_success(r_top/2) ≥ threshold,
proposer doubles rank (e.g. 128 → 256 → 512). Halts at cap OR when
curve plateaus. Use this when an exploratory sweep finishes the default
`{16,32,64,128}` ladder with the curve still climbing — typical for
multi-task or very long-horizon tasks where intrinsic rank is high.

### 13.6 Known LoRA failure modes

| Symptom | Cause | Action |
|---------|-------|--------|
| `pc_success` stuck at baseline value | LoRA scale `alpha/r` too small | Try `alpha = 2*r` |
| Loss diverges in first 500 steps | LoRA scale too large or lr too high for adapter path | Halve lr to 1e-5; keep r unchanged |
| `ValueError: Target modules ... not found` from peft | Wrong preset for this policy version | Use `attn_qkvo`; fall back to raw `q_proj,v_proj` |
| Trainable param count is the full 450 M | LoRA wrap did NOT happen (env var not set) | Verify `LEROBOT_ISAAC_USE_LORA=1` reached the subprocess; check cli_train_cached patched make_policy |
