# ADR-0005: Single train.py with --target_arch over Per-Arch Entrypoints

**Status:** Accepted
**Date:** 2026-05-06
**Deciders:** Project team

---

## Context

This workspace supports multiple training backends:

| Backend | Framework | Entrypoint style considered |
|---------|-----------|---------------------------|
| LeRobot ACT | lerobot | `python -m lerobot.train` wrapper |
| LeRobot SmolVLA | lerobot | Same |
| LeRobot Diffusion | lerobot | Same |
| DreamerV3 | sheeprl | `python -m sheeprl` wrapper |
| HF LeWorldModel | leworldmodel | Custom HF Trainer loop |

Two entrypoint architectures were considered:

**Option A — Per-arch entrypoints:**
```
scripts/train_act.py
scripts/train_smolvla.py
scripts/train_dreamerv3.py
scripts/train_leworldmodel.py
```

**Option B — Single modular entrypoint:**
```
packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py
# invoked as:
python -m lerobot_isaac_adapters.train --target_arch dreamerv3 --config ...
```

---

## Decision

Use **Option B: a single `train.py` with `--target_arch`** plus a **metric extractor pattern**
for per-arch output normalisation.

Architecture:

```
train.py
  --target_arch {act, smolvla, diffusion, dreamerv3, leworldmodel}
  --config <yaml>
  --dataset <path>
  --output <path>

  → dispatches to:
    backends/act_backend.py
    backends/dreamerv3_backend.py
    backends/lewm_backend.py
    ...

  → each backend produces raw metric dicts
  → MetricExtractor normalises to: {step, loss, eval_reward, success_rate, ...}
  → MetricExtractor output feeds autoresearch loop (see ADR-0004)
```

---

## Rationale

### Unified invocation surface

The autoresearch loop (`lerobot-isaac-autoresearch`) needs to launch training and collect
metrics without knowing which architecture it is training. A single `train.py` gives the
autoresearch orchestrator a stable interface: it always calls `train.py --target_arch <x>`,
reads metrics from stdout/JSON, and never needs arch-specific logic.

Per-arch entrypoints would require the orchestrator to maintain a dispatch table — essentially
reimplementing the backend registry in a second place.

### Metric extractor pattern

Different frameworks emit metrics in incompatible formats:

- lerobot: TensorBoard events + stdout lines
- sheeprl (DreamerV3): wandb + stdout
- HF Trainer (LeWM): HF `TrainingArguments` callbacks

The `MetricExtractor` protocol (defined in `lerobot-isaac-meta`) provides a consistent
interface:

```python
class MetricExtractor(Protocol):
    def extract(self, raw_output: str | dict) -> TrainingMetrics: ...
```

Each backend ships its own extractor. The autoresearch loop calls `extractor.extract()`
without caring which backend produced the output. Adding a new architecture requires:
1. A new `backends/<name>_backend.py`
2. A new `extractors/<name>_extractor.py` implementing `MetricExtractor`
3. One entry in the backend registry dict

No changes to `train.py` itself, no changes to the autoresearch loop.

### OOM retry ladder

The RTX 3080 (10 GB) is borderline for some architectures. `train.py` implements a common
OOM retry ladder that works regardless of architecture:

```
num_envs=8  → OOM → num_envs=4 → OOM → num_envs=2 → FAIL
batch_size=64 → OOM → batch_size=32 → OOM → batch_size=16 → FAIL
```

This logic lives once in `train.py` instead of being replicated in each arch script.

### Config unification

All architectures share a common config schema base (in `lerobot-isaac-configs`) with
arch-specific sections. A single `train.py` can validate the common base before
dispatching, giving a single place for config validation errors.

---

## Consequences

**Positive:**
- Autoresearch loop has a single stable interface
- OOM retry logic written once
- Common config validation in one place
- Adding a new arch is additive (backend + extractor), not structural

**Negative:**
- `train.py` is a dispatch layer; errors may be harder to attribute to the backend
- Developers must understand the backend/extractor split
- Each backend must implement `MetricExtractor` — adds a small per-arch burden

**Mitigation:**
- `docs/concepts/modular-training-adapter.md` explains how to add a new arch step-by-step
- Integration tests exercise each backend's extractor with canned stdout fixtures
- `MetricExtractor` is a simple Protocol (3 methods); low implementation cost

---

## Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| Per-arch entrypoints | Duplicates OOM ladder, config validation; breaks autoresearch interface |
| Makefile targets per arch | Build-system indirection adds complexity; no Python-level dispatch |
| Single monolithic train.py (no backends) | 1000+ line god-file; impossible to test arches in isolation |
| Plugin system (entry_points) | Over-engineered for 3–5 arches; adds packaging complexity |
