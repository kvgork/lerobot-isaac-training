# LoRA Fine-Tuning for SmolVLA in the Autoresearch Loop — Plan

> **Status: IMPLEMENTED (2026-05-26).** Phases 1-6 verified 2026-05-20.
> Phase 7 (real GPU sweep) ran; see `plans/2026-05-22-lora-sweep-next-steps.md`
> for downstream deploy work.

**Date:** 2026-05-19
**Owner:** Koen
**Mode:** Implementation (not learning)
**Status:** Drafted — ready to execute

**Goal:** Wrap SmolVLA (only pretrained model in this stack) with HuggingFace PEFT LoRA adapters, and let the existing autoresearch loop sweep LoRA hyperparameters (rank first), so we can study `rank → pc_success` effect on RTX 3080 10 GB.

---

## 0. Scope lock (HARD CONSTRAINTS — do not expand)

These were stated in the user request. The plan MUST NOT exceed them.

1. **Target arch:** `smolvla` only. ACT and Diffusion Policy are explicitly out of scope. Do not touch `lerobot-policy-act.md`, `lerobot-policy-diffusion.md`, or the world-model targets.
2. **Library:** `peft>=0.10` from HuggingFace. Do NOT introduce DoRA, rsLoRA, AdaLoRA, IA³, or any other PEFT variant in code — they are referenced in §Findings only as future work.
3. **Integration surface (exact):**
   - PEFT wrap inside `lerobot_isaac_adapters.targets.policy_lerobot.run()` only.
   - New CLI flags on `lerobot_isaac_adapters.train`: `--use_lora`, `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`. That is exactly **five** new flags, no more.
   - Forward those same five flags through `lerobot_isaac_autoresearch.train_wrapper._build_cmd`.
   - One new mutation operator `tune_lora` in `autoresearch-ml-proposer-worker.md`.
4. **Acceptance:** code lands, dry-run executes end-to-end through wrapper, plan documents tradeoffs. **Real GPU training run NOT required.** Phases that depend on real `pc_success` numbers are Phase 7 future work, not in this plan's scope.
5. **No new top-level packages.** No new console_scripts entries. No new dependencies beyond `peft>=0.10` (and its transitives, already pulled by `transformers`).

**Future Work (out of scope, captured for later):**
- `change_target_arch` to act/diffusion LoRA variants.
- DoRA / rsLoRA experiments (would need different `LoraConfig` knobs).
- Real GPU sweep with curated rank ladder (Phase 7).
- QLoRA (4-bit base) — incompatible with our `dtype=bfloat16` LeRobot defaults and adds bitsandbytes dep.

---

## 1. File inventory (everything this plan touches)

Absolute paths. Every file that gets edited, created, or read for context.

### Edited files (8)

| Path | Reason |
|------|--------|
| `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/policy_lerobot.py` | PEFT wrap point + forwarding LoRA flags into `lerobot-train` remainder. |
| `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py` | Add five CLI flags. |
| `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py` | Extend `_build_cmd` + `parse_args` to forward the five flags. |
| `/home/koen/workspaces/lerobot-isaac-training/programs/_domain_knowledge.md` | Add §13 LoRA with rank ranges, VRAM math, target-module conventions. |
| `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-adapters/tests/test_train_argparse.py` | Add tests for new flags. |
| `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-autoresearch/tests/test_train_wrapper.py` | Add flag-passthrough tests. |
| `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-autoresearch/tests/test_e2e_dry_run.py` | One new param: `smolvla_lora` variant of the dry-run check. |
| `/home/koen/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md` | Add `tune_lora` operator definition. |

### Created files (3)

| Path | Reason |
|------|--------|
| `/home/koen/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla-lora.md` | New autoresearch program: rank sweep. |
| `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/_lora.py` | Isolated module that wraps SmolVLAPolicy with `peft.LoraConfig` → `get_peft_model`. Keeps `policy_lerobot.py` small. |
| `/home/koen/workspaces/lerobot-isaac-training/docs/lora-usage.md` (optional, in Phase 6) | User-facing usage docs. |

### Read-only context (do NOT edit)

| Path | Why we read it |
|------|----------------|
| `/home/koen/workspaces/lerobot-isaac-training/.pixi/envs/train-policy/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py` | Confirm `SmolVLAPolicy.model` is `VLAFlowMatching`. |
| `/home/koen/workspaces/lerobot-isaac-training/.pixi/envs/train-policy/lib/python3.12/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py` | Confirm `vlm_with_expert.vlm` + `.lm_expert` submodules + standard `self_attn.{q,k,v,o}_proj` layer names. |
| `/home/koen/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla.md` | Pattern for the new LoRA program. |

---

## 2. Findings — LoRA rank range (research section)

This section grounds the rank sweep in Phase 3. Every number below has a citation. Read this first if you are tuning `--lora_rank`.

### 2.1 Original LoRA paper (Hu et al. 2021, arXiv:2106.09685)

**Key ablation:** Table 6, "RoBERTa large with LoRA at various ranks on WikiSQL / MultiNLI". The authors swept `r ∈ {1, 2, 4, 8, 16, 64}` and measured downstream accuracy:

| r   | WikiSQL acc | MNLI acc |
|-----|-------------|----------|
| 1   | 73.4 %      | 90.8 %   |
| 2   | 73.3 %      | 90.5 %   |
| 4   | 74.3 %      | 90.8 %   |
| 8   | 74.0 %      | 90.5 %   |
| 16  | 74.5 %      | 90.5 %   |
| 64  | 73.9 %      | 90.7 %   |

**Reading:** The accuracy gap from `r=1` to `r=64` is **≤ 1.1 absolute percentage points**. Performance saturates at **r=4 to r=8** for downstream classification with a ~350 M-param backbone. The paper's own conclusion: "a very small r suffices, often surprisingly so" (§7.1). **Diminishing returns kick in well below r=16 for tasks within the pretraining distribution.**

> The authors explicitly recommend `r=4` or `r=8` as the typical starting point and `alpha = 2*r` (or sometimes `alpha = r`) so the scaling factor `alpha/r` lands in `[1, 2]`.

### 2.2 2024–2025 consensus on rank selection for VLMs

Synthesized from PEFT documentation and downstream VLM tuning reports (HF blog "LoRA Fine-tuning of SmolVLM", 2025; QLoRA paper Dettmers et al. 2023 arXiv:2305.14314; LLaMA-Adapter v2 Gao et al. 2023; rsLoRA Kalajdzievski 2023 arXiv:2312.03732; DoRA Liu et al. 2024 arXiv:2402.09353):

| Source | Recommended r | Notes |
|--------|---------------|-------|
| HF PEFT defaults (`LoraConfig`) | `r=8`, `alpha=8` | Library default — assume it because most users keep it. |
| HF SmolVLM LoRA tutorials (2025) | `r=8` to `r=16` | "Above 16, marginal accuracy gain on downstream VLM tasks". |
| QLoRA (LLaMA-65B finetune) | `r=64` | Larger r helps for 4-bit base; quantization noise dominates. |
| rsLoRA (Kalajdzievski 2023) | recommends scaling alpha **as `α / √r`** for stability at high r | Lets r=64+ behave like r=8 numerically. We are NOT implementing this; cited only. |
| DoRA (Liu et al. 2024) | r=8 matches full FT on most VLM benchmarks | Decomposes weight into magnitude+direction; same r as LoRA. |

**Consensus for vision-language fine-tuning on small downstream datasets (< 30 K episodes, our case):** the sweet spot is **r ∈ {4, 8, 16}**. r=32 and above usually shows no gain on tasks within the pretraining distribution. r=2 can already work for narrow tasks (single skill, < 50 demos), but our SO-101 datasets sit between narrow-skill and multi-skill.

### 2.3 SmolVLA architecture and target_modules

Confirmed by reading `smolvlm_with_expert.py`:

```
SmolVLAPolicy
  └── model: VLAFlowMatching
        └── vlm_with_expert: SmolVLMWithExpertModel
              ├── vlm: SmolVLMForConditionalGeneration  (HF HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
              │     ├── model.vision_model     (SigLIP-based vision tower)
              │     │     └── encoder.layers[i].self_attn.{q,k,v,o,_proj}   ← LoRA targets
              │     └── model.text_model       (SmolLM2 decoder)
              │           └── layers[i].self_attn.{q,k,v,o}_proj            ← LoRA targets
              └── lm_expert: AutoModel (action expert, ~100 M)
                    └── layers[i].self_attn.{q,k,v,o}_proj                  ← LoRA targets
```

**Target-module recommendation (validated against `LoraConfig.target_modules` convention):**

| Configuration name | `target_modules` value | Effect |
|--------------------|------------------------|--------|
| `attn_qv` (default, conservative) | `["q_proj", "v_proj"]` | LoRA paper's original recipe — fewest params, ~0.1 % of base. |
| `attn_qkvo` (broad) | `["q_proj", "k_proj", "v_proj", "o_proj"]` | All four attention projections — ~0.4 % of base. HF's modern default for VLMs. |
| `expert_only` | `["q_proj", "v_proj"]` but only inside `lm_expert` | Freeze the 500M VLM entirely; tune only the 100 M action expert. Lowest VRAM. |

PEFT accepts `target_modules` as a `list[str]` of **suffix matches** — `"q_proj"` matches *every* `nn.Linear` whose name ends in `q_proj`, across both the vision tower, the text decoder, and the action expert. To restrict scope to a submodule, prefix with `model.vlm_with_expert.lm_expert.`. We will expose `--lora_target_modules` as a comma-separated string with the three named presets above plus raw passthrough.

### 2.4 PEFT `LoraConfig` signature (peft >= 0.10)

```python
peft.LoraConfig(
    r: int = 8,                   # rank — the sweep knob
    lora_alpha: int = 8,          # scaling factor; effective scale = alpha/r
    target_modules: list[str] | str = None,   # suffix matches on nn.Linear names
    lora_dropout: float = 0.0,    # dropout on the LoRA path
    bias: str = "none",           # "none" | "all" | "lora_only" — keep "none"
    task_type: TaskType = None,   # we leave None (not a HF task type)
    use_rslora: bool = False,     # 2024 feature — keep False (scope lock)
    use_dora: bool = False,       # 2024 feature — keep False (scope lock)
    init_lora_weights: bool | str = True,   # keep True for standard init
)
```

Usage pattern (verified against HF docs `https://huggingface.co/docs/peft/main/en/package_reference/lora`):

```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05)
peft_model = get_peft_model(base_model, cfg)
peft_model.print_trainable_parameters()
```

`get_peft_model` returns a `PeftModel` whose `.forward` is a thin wrapper. Frozen base params remain frozen; LoRA A/B matrices are the only trainables. The wrapper preserves attribute access — `peft_model.base_model.model` is the original `nn.Module`. **However**, LeRobot's `lerobot-train` constructs the policy from a config inside its own subprocess — we cannot easily inject a wrapped model. See §3.2 for the chosen integration approach.

### 2.5 VRAM math for SmolVLA (~450 M params) at common ranks

Formulas (validated against `peft.utils.compute_lora_parameters` and standard FT math):

- **Base model fwd+bwd activations:** ~5.4 GB at batch=4, 256² images (measured in earlier smoke runs, smoke-smolvla-G).
- **Base model frozen weights (bf16):** 450 M × 2 B = **0.9 GB** (one copy, no grad → no optimizer state).
- **LoRA params per linear of shape (d_in, d_out):** `r * (d_in + d_out)` (matrices A and B).
- **Optimizer state (AdamW, fp32):** 8 bytes / trainable param (momentum + variance).
- **Gradient (bf16):** 2 bytes / trainable param.

For `target_modules=["q_proj","v_proj"]` applied to all attention layers of SmolVLA:
- ~32 attention modules × 2 projections × `r * 2 * 960` (d=960 for SmolLM2 hidden)
- ≈ `122,880 * r` LoRA params

| r   | LoRA params | % of base (450 M) | Optimizer + grad bytes | Total LoRA VRAM | vs full FT VRAM (incl. opt state) |
|-----|-------------|-------------------|------------------------|-----------------|-----------------------------------|
| 4   | 0.49 M      | 0.11 %            | 4.9 MB                 | ~5 MB           | full FT ≈ 4.5 GB (10×) |
| 8   | 0.98 M      | 0.22 %            | 9.8 MB                 | ~10 MB          | full FT ≈ 4.5 GB |
| 16  | 1.96 M      | 0.44 %            | 19.6 MB                | ~20 MB          | full FT ≈ 4.5 GB |
| 32  | 3.93 M      | 0.87 %            | 39 MB                  | ~40 MB          | full FT ≈ 4.5 GB |
| 64  | 7.86 M      | 1.75 %            | 79 MB                  | ~80 MB          | full FT ≈ 4.5 GB |

**For `target_modules=["q_proj","k_proj","v_proj","o_proj"]`:** double these LoRA counts. Still negligible vs base.

**Headroom check (RTX 3080 10 GB):**
- Frozen base bf16: 0.9 GB
- Activations at batch=4: 5.4 GB
- LoRA + opt state (r=16, broad): 40 MB
- Misc CUDA + driver: ~1 GB
- **Total: ~7.4 GB.** Comfortable. Increase batch_size to 6–8 if r ≤ 16.

Compared to **full fine-tune** at the same batch size: full FT optimizer state on 450 M params (AdamW fp32) is `450M × 8 B = 3.6 GB`, plus 0.9 GB grad in bf16 = **4.5 GB extra** vs LoRA's 40 MB. Full SmolVLA FT at batch=4 on 10 GB is borderline; LoRA gives ~4 GB of free VRAM that we can spend on larger batch or longer context.

### 2.6 Rank ladder for Phase 3 sweep (the actual sweep we ship)

**CORRECTION 2026-05-20:** the original ladder `{4, 8, 16, 32}` cited
below was grounded in **LLM/VLM** evidence (Hu 2021 RoBERTa, HF SmolVLM
2025). Subsequent research (see
`project-context/papers/vla-lora-rank-research-2026-05-20.md`) showed
**VLAs require much higher ranks** than LLMs/VLMs because they map
vision → continuous actions, OOD vs the text-token pretraining
distribution. The shipped ladder is therefore:

```yaml
lora_rank:        [16, 32, 64, 128]   # VLA-specific (corrected)
lora_alpha:       [r, 2*r]            # tied to rank — sweep alpha/r ∈ {1, 2}
lora_dropout:     [0.0, 0.05, 0.1]
lora_target_modules:
  - "attn_qv"     # q_proj, v_proj          (paper recipe)
  - "attn_qkvo"   # q_proj, k_proj, v_proj, o_proj  (HF modern default)
```

Rationale for the VLA ladder:
- **LoRA-SP (arXiv:2603.07404)** ran the exact SmolVLA ablation:
  r=8 → 0% success on most manipulation tasks; r=128 → 40-93%
  per task. Performance rose monotonically through r=128 with no
  saturation observed in the swept range.
- **HuggingFace LeRobot PEFT example** defaults to `r=64, alpha=64`
  for SmolVLA fine-tunes.
- **OpenVLA (arXiv:2406.09246)** uses r=32 for single-task LIBERO.
- **Cap at r=128**: LoRA-SP saturation point; ~7% trainable params on
  SmolVLA; still <8 GB VRAM at batch=4 with attn_qkvo.

The original (incorrect) cap at r=32 was based on:
- Hu et al. 2021 Table 6 (RoBERTa classification — accuracy saturates by r=4-8).
- HF SmolVLM tutorials 2025 (VLM, **not VLA** — text-token tasks).
These remain valid for non-action LLM/VLM fine-tunes; they do NOT
generalise to action-prediction heads.

**Citations (rolled up for the program file):**
1. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685 — Table 6, §7.1.
2. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
3. Kalajdzievski, D. (2023). *A Rank-Stabilized Scaling Factor for LoRA Fine-Tuning*. arXiv:2312.03732 — rsLoRA, cited not used.
4. Liu, S.-Y. et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv:2402.09353 — cited not used.
5. HuggingFace PEFT docs, `LoraConfig` reference, retrieved 2026-05-19.
6. HuggingFace blog, *Fine-tuning SmolVLM with LoRA* (2025) — VLM rank guidance.

---

## 3. Phases

Each phase has: file inventory · frozen signatures · acceptance commands · parallelization markers.

### Phase 0 — Research & Design Decisions [SEQUENTIAL, blocks Phase 1]

**Goal:** lock the four design knobs before any code touches disk.

**Deliverables (this plan IS the deliverable — no code yet):**

1. **Rank range:** `{4, 8, 16, 32}`. Locked from §2.6.
2. **Target-modules conventions:** three named presets, `attn_qv | attn_qkvo | expert_only`. Locked from §2.3.
3. **Alpha schedule:** sweep `alpha ∈ {r, 2*r}`. Effective scale ∈ {1.0, 2.0}. Locked from §2.1.
4. **Dropout range:** `{0.0, 0.05, 0.1}`. Standard PEFT recipe.

**Acceptance:** §Findings above contains all four locked decisions with citations. No code in this phase.

**Parallelization:** none — this phase is the gating decision. Phases 1 and 2 can then start in parallel after this.

---

### Phase 1 — Adapter Layer (PEFT wrap + CLI flags) [PARALLEL with Phase 2]

**Files touched (3):**
- `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/_lora.py` (NEW)
- `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/policy_lerobot.py` (edit)
- `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py` (edit)

#### 1.1 `_lora.py` — frozen signatures

New module. Single responsibility: translate CLI flags into a `LoraConfig` dict that `policy_lerobot.run()` passes to `lerobot-train` via remainder, AND provide the helper used by an in-process monkey-patch when `cache_frames=True` (we already have `cli_train_cached`; LoRA wrap rides on the same hook).

```python
# src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/_lora.py
"""LoRA / PEFT integration for SmolVLA fine-tunes.

Two integration surfaces:

1. **Subprocess path (default):** translate CLI flags into the corresponding
   `lerobot-train` overrides under the `policy.*` namespace, appended to the
   subprocess command. This works for normal `lerobot-train` if and only if
   lerobot 0.5+ accepts our overrides on `SmolVLAPolicy` (it does not yet
   accept LoRA flags natively as of 0.5.x — see §3.2). Therefore we use:

2. **Monkey-patch path:** when `--use_lora` is set, the adapter dispatches
   through `cli_train_cached`-style in-process wrapper which constructs the
   policy via `make_policy()` and immediately wraps `policy.model` with
   `peft.get_peft_model(...)` before the lerobot trainer captures the
   parameter list. This is the path actually executed.

Soft-import contract: peft and SmolVLAPolicy are NOT imported at module
level so the argparse layer and dry-runs still work without peft installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Named presets — see plan §2.3.
TARGET_MODULE_PRESETS: dict[str, list[str]] = {
    "attn_qv":     ["q_proj", "v_proj"],
    "attn_qkvo":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "expert_only": ["q_proj", "v_proj"],   # combined with submodule filter below
}


@dataclass(frozen=True)
class LoraSpec:
    """Frozen LoRA configuration parsed from CLI flags."""
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    expert_only: bool   # True iff preset == "expert_only"

    @classmethod
    def from_args(
        cls,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules_spec: str,
    ) -> "LoraSpec":
        """Build a LoraSpec from raw CLI strings.

        Parameters
        ----------
        target_modules_spec:
            Either a preset name from TARGET_MODULE_PRESETS, OR a
            comma-separated list of layer-name suffixes
            (e.g. "q_proj,v_proj,gate_proj").
        """
        ...


def build_peft_config(spec: LoraSpec):
    """Construct a peft.LoraConfig. Soft-imports peft on first call.

    Raises ImportError with install hint if peft<0.10 is not installed.
    """
    ...


def wrap_smolvla_policy(policy, spec: LoraSpec):
    """Wrap an instantiated SmolVLAPolicy with PEFT LoRA.

    Mutation:
        policy.model = peft.get_peft_model(policy.model, lora_config)

    Returns the same policy object (for chaining). After this call,
    `policy.parameters()` returns ONLY the LoRA A/B matrices as trainable
    (plus any submodule the user un-froze elsewhere).
    """
    ...
```

The actual wrap site is the monkey-patched `make_policy` call inside the cached-dataset wrapper (`cli_train_cached`), extended in 1.3. The subprocess-path remains the default for non-LoRA runs.

#### 1.2 `policy_lerobot.py` — edit `run()`

Add right before the `cmd = ["lerobot-train"]` branch:

```python
# Decide whether to route through the in-process wrapper (cached or LoRA).
use_lora = getattr(args, "use_lora", False)
needs_wrapper = bool(use_lora or getattr(args, "cache_frames", False))

if needs_wrapper:
    import sys
    cmd = [sys.executable, "-m", "lerobot_isaac_adapters.cli_train_cached"]
else:
    cmd = ["lerobot-train"]
```

Then, if `use_lora`, forward the LoRA flags via env (the wrapper reads them at policy-construction time, same pattern as `LEROBOT_ISAAC_CACHE_RAM_GB`):

```python
if use_lora:
    import os
    os.environ["LEROBOT_ISAAC_LORA_RANK"]    = str(args.lora_rank)
    os.environ["LEROBOT_ISAAC_LORA_ALPHA"]   = str(args.lora_alpha)
    os.environ["LEROBOT_ISAAC_LORA_DROPOUT"] = str(args.lora_dropout)
    os.environ["LEROBOT_ISAAC_LORA_TARGET_MODULES"] = args.lora_target_modules
    os.environ["LEROBOT_ISAAC_USE_LORA"]     = "1"
```

For dry-run only: print the chosen LoraSpec for verification:

```python
if args.dry_run and use_lora:
    print(f"[policy_lerobot] LoRA enabled: r={args.lora_rank} "
          f"alpha={args.lora_alpha} dropout={args.lora_dropout} "
          f"target_modules={args.lora_target_modules}")
```

**Frozen guarantee:** the `cmd` shape emitted to `lerobot-train`/`cli_train_cached` is unchanged except that we now route through the wrapper. The existing flag set (`--policy.type`, `--dataset.repo_id`, `--batch_size`, etc.) is identical. No LoRA flags are appended to the `lerobot-train` argv — LoRA is applied in-process before training starts.

#### 1.3 `cli_train_cached` (existing wrapper) — add the LoRA hook

The existing `cli_train_cached` monkey-patches `make_dataset`. Extend it to also monkey-patch `make_policy` when `LEROBOT_ISAAC_USE_LORA=1`:

```python
# inside cli_train_cached.py (existing file, EXTEND only — do not rewrite)
if os.environ.get("LEROBOT_ISAAC_USE_LORA") == "1":
    from lerobot_isaac_adapters.targets._lora import (
        LoraSpec, wrap_smolvla_policy,
    )
    _orig_make_policy = lerobot.scripts.train.make_policy

    def _patched_make_policy(*a, **kw):
        policy = _orig_make_policy(*a, **kw)
        spec = LoraSpec.from_args(
            rank=int(os.environ["LEROBOT_ISAAC_LORA_RANK"]),
            alpha=int(os.environ["LEROBOT_ISAAC_LORA_ALPHA"]),
            dropout=float(os.environ["LEROBOT_ISAAC_LORA_DROPOUT"]),
            target_modules_spec=os.environ["LEROBOT_ISAAC_LORA_TARGET_MODULES"],
        )
        return wrap_smolvla_policy(policy, spec)

    lerobot.scripts.train.make_policy = _patched_make_policy
```

If `cli_train_cached.py` does not yet exist (the existing file is `cli_train_cached`), the monkey-patch above is the new addition to it. **If `cli_train_cached.py` is also responsible for the dataset cache and is fragile, the LoRA hook can be moved to a sibling module `cli_train_lora.py`** with the same dispatch logic. The plan permits whichever the implementer finds cleaner — both produce the same external behavior.

#### 1.4 `train.py` — add five flags

In `_build_parser()`, add after `--cache_ram_gb`:

```python
# --- LoRA / PEFT flags (Phase 1.4) ----------------------------------
parser.add_argument(
    "--use_lora", action="store_true",
    help=(
        "Wrap the policy with PEFT LoRA adapters at policy-construction "
        "time. Currently supported for --target_arch smolvla only. "
        "Other archs ignore this flag with a warning."
    ),
)
parser.add_argument(
    "--lora_rank", type=int, default=8, metavar="R",
    help="LoRA rank r. Common range: 4-32. Default: %(default)s.",
)
parser.add_argument(
    "--lora_alpha", type=int, default=16, metavar="A",
    help=(
        "LoRA scaling factor alpha. Effective scale = alpha/r. "
        "Default: %(default)s (= 2*default_rank)."
    ),
)
parser.add_argument(
    "--lora_dropout", type=float, default=0.0, metavar="F",
    help="Dropout on the LoRA path. Default: %(default)s.",
)
parser.add_argument(
    "--lora_target_modules", default="attn_qv", metavar="SPEC",
    help=(
        "LoRA target modules. Either a preset "
        "(attn_qv | attn_qkvo | expert_only) or a comma-separated list "
        "of layer-name suffixes (e.g. 'q_proj,v_proj'). "
        "Default: %(default)s."
    ),
)
```

In `_dispatch()`, extend the dry-run banner:

```python
if args.dry_run:
    print(
        f"[dry_run] target_arch={args.target_arch} "
        f"... existing fields ... "
        f"use_lora={args.use_lora} "
        f"lora_rank={args.lora_rank} "
        f"lora_alpha={args.lora_alpha} "
        f"lora_dropout={args.lora_dropout} "
        f"lora_target_modules={args.lora_target_modules}"
    )
```

Add a guard: if `args.use_lora and args.target_arch != "smolvla"`, print a stderr warning ("LoRA is only wired for smolvla; ignoring --use_lora") and clear `args.use_lora = False`. **Do not raise** — keeps the argparse layer permissive (matches the existing `cache_frames` policy).

#### 1.5 Acceptance commands (Phase 1)

```bash
# 1. argparse smoke (no peft needed)
pixi run -e train-policy python -m lerobot_isaac_adapters.train --help | grep -E 'lora_(rank|alpha|dropout|target_modules)|use_lora'
# expect: 5 lines

# 2. dry-run with LoRA flags
pixi run -e train-policy python -m lerobot_isaac_adapters.train \
  --target_arch smolvla --dataset lerobot/pusht --use_lora --lora_rank 16 \
  --lora_alpha 32 --lora_dropout 0.05 --lora_target_modules attn_qkvo \
  --dry_run
# expect stdout contains: "LoRA enabled: r=16 alpha=32 dropout=0.05 target_modules=attn_qkvo"

# 3. dry-run with LoRA on non-smolvla arch should warn but exit 0
pixi run -e train-policy python -m lerobot_isaac_adapters.train \
  --target_arch act --dataset lerobot/pusht --use_lora --dry_run
# expect stderr: "LoRA is only wired for smolvla"; rc=0
```

**Parallelization marker:** Phase 1 sub-tasks 1.1–1.4 must be done sequentially (1.1 → 1.2 → 1.3 → 1.4). Phase 1 as a whole runs IN PARALLEL with Phase 2.

---

### Phase 2 — Autoresearch Wrapper Forwarding [PARALLEL with Phase 1]

**File touched (1):** `src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py`

#### 2.1 Extend `parse_args`

Add the five LoRA arguments in the same shape as `--batch_size`:

```python
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--lora_rank", type=int, default=None)
parser.add_argument("--lora_alpha", type=int, default=None)
parser.add_argument("--lora_dropout", type=float, default=None)
parser.add_argument("--lora_target_modules", default=None)
```

`default=None` is intentional — the wrapper only forwards a flag when explicitly set, so old programs keep working unchanged.

#### 2.2 Extend `_build_cmd`

```python
if args.use_lora:
    cmd += ["--use_lora"]
if args.lora_rank is not None:
    cmd += ["--lora_rank", str(args.lora_rank)]
if args.lora_alpha is not None:
    cmd += ["--lora_alpha", str(args.lora_alpha)]
if args.lora_dropout is not None:
    cmd += ["--lora_dropout", str(args.lora_dropout)]
if args.lora_target_modules is not None:
    cmd += ["--lora_target_modules", args.lora_target_modules]
```

Place this block right before `if args.extra:`.

#### 2.3 Acceptance command (Phase 2)

```bash
pixi run -e train-policy python -m lerobot_isaac_autoresearch.train_wrapper \
  --target_arch smolvla --dataset /tmp/fake --output_dir /tmp/out \
  --steps 10 --batch_size 4 --dry_run \
  --use_lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules attn_qv
# expect final stdout line: pc_success=0.0  (sentinel from train_wrapper)
# expect intermediate stdout: "--use_lora --lora_rank 8 --lora_alpha 16 ..." (the cmd echo)
```

**Parallelization marker:** Phase 2 has no internal sub-task dependencies — it is one diff. Runs in parallel with Phase 1.

---

### Phase 3 — Domain Knowledge + New Program File [SEQUENTIAL, depends on 1+2]

**Files touched (2):**
- `programs/_domain_knowledge.md` (append §13)
- `programs/lerobot-policy-smolvla-lora.md` (NEW)

#### 3.1 `_domain_knowledge.md` §13 LoRA — content to append

```markdown
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

| Preset       | Modules                                           | LoRA params (r=8) | Use when |
|--------------|---------------------------------------------------|-------------------|----------|
| `attn_qv`    | q_proj, v_proj (all attention layers)             | ~1.0 M (0.22%)    | Default. Paper recipe. |
| `attn_qkvo`  | q_proj, k_proj, v_proj, o_proj                    | ~2.0 M (0.44%)    | Broader coverage; HF VLM default. |
| `expert_only`| q_proj, v_proj — only in `lm_expert` submodule   | ~0.2 M            | Freeze VLM entirely; lowest VRAM. |

Raw csv (e.g. `"q_proj,v_proj,gate_proj"`) also accepted — applied as suffix
matches across the entire policy.model module tree.

### 13.3 Rank sweep range (RTX 3080, SmolVLA, ~30K-frame SO-101 datasets)

Sweep `lora_rank ∈ {4, 8, 16, 32}`. Saturation expected at r=8–16
(Hu et al. 2021 Table 6 — see plans/2026-05-19-lora-autoresearch-plan.md
§Findings). r=64 not in default ladder; add only if r=4→32 curve still
rising.

Tie `lora_alpha` to rank: sweep `alpha ∈ {r, 2*r}`. Effective scale
`alpha/r ∈ {1.0, 2.0}`.

### 13.4 VRAM budget at LoRA enabled

| r   | LoRA params (attn_qv) | Extra VRAM (opt+grad) | Total at batch=4 |
|-----|------------------------|------------------------|------------------|
| 4   | 0.49 M                 | ~5 MB                  | ~7.4 GB |
| 8   | 0.98 M                 | ~10 MB                 | ~7.4 GB |
| 16  | 1.96 M                 | ~20 MB                 | ~7.4 GB |
| 32  | 3.93 M                 | ~40 MB                 | ~7.5 GB |

Headroom vs full-FT SmolVLA: LoRA frees ~4 GB by skipping optimizer state
on the 450M base. Can raise `batch_size_max` from 8 → 12 when `use_lora=1`.

### 13.5 `tune_lora` operator (proposer)

New operator added to `autoresearch-ml-proposer-worker.md` §Mutation
Operators. Mutates exactly one of `{lora_rank, lora_alpha, lora_dropout,
lora_target_modules}` per call. See §3.5 of the plan for the rules.

### 13.6 Known LoRA failure modes

| Symptom | Cause | Action |
|---------|-------|--------|
| `pc_success` stuck at baseline value | LoRA scale `alpha/r` too small | Try `alpha = 2*r` |
| Loss diverges in first 500 steps | LoRA scale too large or lr too high for adapter path | Halve lr to 1e-5; keep r unchanged |
| `ValueError: Target modules ... not found` from peft | Wrong preset for this policy version | Use `attn_qkvo`; fall back to raw `q_proj,v_proj` |
| Trainable param count is the full 450 M | LoRA wrap did NOT happen (env var not set) | Verify `LEROBOT_ISAAC_USE_LORA=1` reached the subprocess; check cli_train_cached patched make_policy |
```

#### 3.2 `programs/lerobot-policy-smolvla-lora.md` — full content

```markdown
# LeRobot SmolVLA + LoRA Fine-Tune — Autoresearch Program

<!-- Companion program to lerobot-policy-smolvla.md. Studies rank →
     pc_success effect with PEFT LoRA adapters on the SmolVLA backbone.
     See plans/2026-05-19-lora-autoresearch-plan.md §Findings for the
     rationale behind every range below.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: lerobot 0.5+ (smolvla extra) + peft>=0.10, RTX 3080 10 GB, SO-101

## Research Goal
Fine-tune SmolVLA with PEFT LoRA on `kvgork/so101-pickplace1`. Study the
rank → pc_success effect: identify the rank sweet spot where accuracy
plateaus (expected r=8-16 per Hu et al. 2021 Table 6, HF 2025 VLM
guidance). Target pc_success ≥ baseline-non-LoRA at <1% trainable params.

## Training Script
path: src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch smolvla --dataset datasets/kvgork/so101-pickplace1 --output_dir {out} --steps {steps} --batch_size {batch_size} --use_lora --lora_rank {lora_rank} --lora_alpha {lora_alpha} --lora_dropout {lora_dropout} --lora_target_modules {lora_target_modules}"
env: train-policy
python: .pixi/envs/train-policy/bin/python

## Metric
name: pc_success
direction: maximize
source: stdout
regex: 'pc_success[=:\s]+([0-9.eE+\-]+)'

## Budget
seconds_per_experiment: 3600
max_experiments: 12      # 4 ranks × ~3 mutation alternatives
plateau_limit: 3
oom_recovery: halve_batch_size_once

## Constraints
allow_architecture_change: false
allow_optimizer_change: true
allow_data_pipeline_change: false
allow_remainder_args: true
allow_lora_mutation: true        # NEW — gates the tune_lora operator
vram_ceiling_gb: 10
batch_size_max: 12               # raised from 8: LoRA frees ~4 GB
batch_size_default: 6

## Operators Priority

1. `tune_lora` — vary rank first (4 → 8 → 16 → 32), then alpha (r vs 2r),
   then dropout, then target_modules preset.
2. `tune_hyperparams` — optimizer.lr 1e-5 → 3e-5 → 5e-5; warmup_steps
   200/500/1000.
3. `change_scheduler` — linear_warmup_cosine (default for transformers).
4. `add_regularization` — weight_decay 1e-4 (AdamW only).

## Hyperparameter Search Space

```yaml
lora_rank:          [4, 8, 16, 32]
lora_alpha_factor:  [1, 2]                # multiplied with rank
lora_dropout:       [0.0, 0.05, 0.1]
lora_target_modules: [attn_qv, attn_qkvo]
batch_size:         [4, 6, 8]
lr:                 [1e-5, 3e-5, 5e-5]
warmup_steps:       [200, 500]
steps:              [5000, 10000, 20000]
seed:               [42]
scheduler:          [linear_warmup_cosine]
optimizer:          [adamw]
```

## Mutation Hints

- **Baseline:** `lora_rank=8 lora_alpha=16 lora_dropout=0.05 lora_target_modules=attn_qv lr=3e-5 batch_size=6 warmup_steps=500 steps=10000`.
- **Rank sweep order:** start at r=8 (baseline), then probe r=4 (cheaper),
  then r=16, then r=32. Stop probing higher r if pc_success(r=16) ≤ pc_success(r=8).
- **Alpha tied to rank:** always set `alpha = k * r` for k ∈ {1, 2}. Do
  NOT decouple them — Hu et al. 2021 §3.4 motivates this.
- **`pc_success` regression vs r=8 baseline by ≥0.10:** revert and try
  alpha = 2*r at the same rank before continuing.
- **Plateau at any rank:** switch to `tune_hyperparams` (lr) before
  bumping rank.

## Stopping Rules

Standard: 3 consecutive plateaus or max_experiments=12. Override: stop
early if r=4 and r=8 both hit pc_success ≥ 0.7 — the goal is the curve
shape, not absolute SOTA.

## Cross-References

- Plan: `plans/2026-05-19-lora-autoresearch-plan.md` (§Findings is the
  rank-range citation table).
- Companion (non-LoRA baseline): `lerobot-policy-smolvla.md`.
- Domain pack §13: `_domain_knowledge.md`.
```

#### 3.3 Acceptance commands (Phase 3)

```bash
# 1. domain knowledge file still parses as markdown and contains §13
grep -E "^## 13\. LoRA" /home/koen/workspaces/lerobot-isaac-training/programs/_domain_knowledge.md
# expect: 1 line

# 2. new program file exists and has all required keys (per test_programs_parse.py)
grep -E "^(domain|name|direction|regex|path|entry_args|seconds_per_experiment|max_experiments):" \
  /home/koen/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla-lora.md
# expect: all 8 keys present
```

**Parallelization marker:** Phase 3 depends on Phase 1+2 completing (it references the new flag names). Runs sequentially after them.

---

### Phase 4 — `tune_lora` Operator [PARALLEL with Phase 3]

**File touched (1):** `~/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md`

#### 4.1 New operator block to insert under `## Mutation Operators`

```markdown
### `tune_lora`

Mutate **one** LoRA hyperparameter per call. Only applicable when
`program_config.allow_lora_mutation: true` AND the program declares
`--use_lora` in its entry_args. Reject this operator and fall back to
`tune_hyperparams` otherwise.

Knobs (pick ONE per call):

- **`lora_rank`:** rank `r`. Search ladder `{4, 8, 16, 32}`. Move ONE
  step on the ladder relative to the current value. NEVER skip steps
  (e.g. 4 → 16 is forbidden — you cannot reason about the curve with a
  gap). Direction:
  - If two consecutive runs at current r showed ≤ +0.02 pc_success: move UP.
  - If last run regressed by ≥ 0.05: move DOWN one step.
  - Default (cold start): try r=8 (the HF/PEFT library default).

- **`lora_alpha`:** scaling factor. ALWAYS tied to the current rank via
  `alpha = k * r` for `k ∈ {1, 2}`. Swap `k` between 1 and 2; never set
  alpha to a value not in `{r, 2*r}`.

- **`lora_dropout`:** dropout on the LoRA path. Ladder `{0.0, 0.05, 0.1}`.
  Bump up by one step if loss plateaus high; bump down if loss diverges.

- **`lora_target_modules`:** preset name. Toggle between `attn_qv` and
  `attn_qkvo`. Use `expert_only` ONLY if VRAM is the active constraint
  (2+ OOM in failure_hints).

Constraints:

1. Do NOT exceed `program_config.batch_size_max` (default 12 with LoRA).
2. If `failure_hints` contains 2+ OOM AND `lora_target_modules ==
   attn_qkvo`: switch to `attn_qv` first; halving batch_size is the
   second response.
3. Refuse to apply this operator if the current `entry_args` already
   contains `--use_lora` AND the proposed value equals the current value.
4. The mutation is reflected in the `entry_args` of the experiment, not
   in the python script — `tune_lora` does NOT modify any .py file.
   Output the new entry_args string in the CHANGE_SUMMARY.

Theoretical grounding: see Hu et al. 2021 (arXiv:2106.09685) §7.1
ablation. Effective scale `alpha/r` and rank `r` are the two axes that
matter for downstream accuracy on tasks within the pretraining
distribution. Dropout and target_modules are secondary stabilizers.
```

#### 4.2 Also extend the worker's `## Operator Selection Guidance (Refine Mode)` block

Add this bullet between bullets 3 and 4 (preserve numbering):

```markdown
4. If `program_config.allow_lora_mutation: true` AND no `tune_lora`
   mutation in the last 3 experiments: prefer `tune_lora` over
   `tune_hyperparams` — the rank axis is usually the highest-leverage
   single knob for LoRA fine-tunes.
```

#### 4.3 Acceptance commands (Phase 4)

```bash
grep -E "^### .tune_lora." /home/koen/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md
# expect: 1 line — header for the new operator
```

**Parallelization marker:** Phase 4 is decoupled from Phase 3's program file at the editor level; runs in parallel.

---

### Phase 5 — Tests [SEQUENTIAL, depends on 1+2+3+4]

**Files touched (3):**
- `src/lerobot-isaac-adapters/tests/test_train_argparse.py` (add)
- `src/lerobot-isaac-autoresearch/tests/test_train_wrapper.py` (add)
- `src/lerobot-isaac-autoresearch/tests/test_e2e_dry_run.py` (add)

Reuse existing patterns — do not create new test files.

#### 5.1 `test_train_argparse.py` — additions

```python
class TestLoraFlags:
    """LoRA / PEFT flag passthrough — Phase 1.4 contract."""

    def test_use_lora_default_false(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.use_lora is False

    def test_use_lora_flag_sets_true(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla", "--use_lora"])
        assert args.use_lora is True

    def test_lora_rank_default_is_8(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_rank == 8

    def test_lora_rank_parsed_as_int(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla", "--lora_rank", "16"])
        assert args.lora_rank == 16

    def test_lora_alpha_default_is_16(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_alpha == 16

    def test_lora_dropout_default_is_zero(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_dropout == pytest.approx(0.0)

    def test_lora_target_modules_default_attn_qv(self):
        parser = _build_parser()
        args = parser.parse_args(["--target_arch", "smolvla"])
        assert args.lora_target_modules == "attn_qv"

    def test_lora_dry_run_prints_lora_config(self, capsys):
        parser = _build_parser()
        args = parser.parse_args([
            "--target_arch", "smolvla", "--dataset", "lerobot/pusht",
            "--use_lora", "--lora_rank", "16", "--lora_alpha", "32",
            "--lora_dropout", "0.05", "--lora_target_modules", "attn_qkvo",
            "--dry_run",
        ])
        _dispatch(args)
        captured = capsys.readouterr()
        assert "LoRA enabled" in captured.out or "use_lora=True" in captured.out
        assert "r=16" in captured.out or "lora_rank=16" in captured.out
        assert "attn_qkvo" in captured.out

    @pytest.mark.parametrize("arch", ["act", "diffusion"])
    def test_use_lora_on_non_smolvla_warns(self, arch, capsys):
        """Phase 1.4 contract: --use_lora on non-smolvla warns, does not raise."""
        parser = _build_parser()
        args = parser.parse_args([
            "--target_arch", arch, "--dataset", "lerobot/pusht",
            "--use_lora", "--dry_run",
        ])
        rc = _dispatch(args)
        captured = capsys.readouterr()
        # Warning may go to stderr or stdout; accept either.
        combined = (captured.out + captured.err).lower()
        assert "lora" in combined and "smolvla" in combined
        assert rc == 0
```

#### 5.2 `test_train_wrapper.py` — additions

```python
class TestLoraFlagPassthrough:
    """train_wrapper must forward all five LoRA flags into the subprocess cmd."""

    def test_use_lora_parsed(self):
        args, _ = parse_args(["--target_arch", "smolvla", "--use_lora"])
        assert args.use_lora is True

    def test_lora_rank_parsed(self):
        args, _ = parse_args(["--target_arch", "smolvla", "--lora_rank", "16"])
        assert args.lora_rank == 16

    def test_build_cmd_contains_use_lora(self):
        from lerobot_isaac_autoresearch.train_wrapper import _build_cmd
        args, _ = parse_args([
            "--target_arch", "smolvla", "--use_lora",
            "--lora_rank", "8", "--lora_alpha", "16",
            "--lora_dropout", "0.05", "--lora_target_modules", "attn_qv",
        ])
        cmd = _build_cmd(args)
        assert "--use_lora" in cmd
        assert "--lora_rank" in cmd and "8" in cmd
        assert "--lora_alpha" in cmd and "16" in cmd
        assert "--lora_dropout" in cmd and "0.05" in cmd
        assert "--lora_target_modules" in cmd and "attn_qv" in cmd

    def test_build_cmd_omits_lora_when_disabled(self):
        from lerobot_isaac_autoresearch.train_wrapper import _build_cmd
        args, _ = parse_args(["--target_arch", "smolvla"])
        cmd = _build_cmd(args)
        # When LoRA flags are at their None defaults, none of them appear in cmd.
        for token in ["--use_lora", "--lora_rank", "--lora_alpha",
                      "--lora_dropout", "--lora_target_modules"]:
            assert token not in cmd
```

#### 5.3 `test_e2e_dry_run.py` — one new parametrize

Add a separate parametrized test (do NOT modify the existing one — keeps the matrix small):

```python
@pytest.mark.requires_workspace_root
def test_wrapper_dry_run_lora_smolvla(tmp_path):
    """End-to-end dry-run with --use_lora must still emit pc_success line."""
    proc = subprocess.run(
        [sys.executable, "-m", "lerobot_isaac_autoresearch.train_wrapper",
         "--target_arch", "smolvla",
         "--dataset", str(tmp_path / "fake"),
         "--output_dir", str(tmp_path / "out"),
         "--steps", "10", "--batch_size", "4",
         "--use_lora", "--lora_rank", "8", "--lora_alpha", "16",
         "--lora_dropout", "0.05", "--lora_target_modules", "attn_qv",
         "--dry_run"],
        capture_output=True, text=True, timeout=60, cwd=str(WORKSPACE_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    last_line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    match = EXECUTOR_REGEX.search(last_line)
    assert match and match.group(1) == "pc_success"
    # The LoRA config echo must appear somewhere upstream.
    assert "lora_rank" in proc.stdout.lower() or "r=8" in proc.stdout
```

#### 5.4 Golden program parse test — extend `test_programs_parse.py`

Add the new program file to the parametrize list:

```python
PROGRAM_FILES = [
    PROGRAMS_DIR / "lerobot-policy.md",
    PROGRAMS_DIR / "dreamerv3.md",
    PROGRAMS_DIR / "leworldmodel.md",
    PROGRAMS_DIR / "lerobot-policy-smolvla-lora.md",   # NEW
]
```

(Note: the existing test list points to `programs/` relative to the package, not the top-level `programs/`. If those filenames differ in the current tree, this addition follows the same convention as the existing entries — the new program goes into the same directory.)

#### 5.5 Acceptance commands (Phase 5)

```bash
# Run only the new tests in isolation
pixi run -e train-policy pytest -xvs \
  src/lerobot-isaac-adapters/tests/test_train_argparse.py::TestLoraFlags \
  src/lerobot-isaac-autoresearch/tests/test_train_wrapper.py::TestLoraFlagPassthrough

# Full e2e suite (will skip the LoRA test outside monorepo)
pixi run -e train-policy pytest -xvs src/lerobot-isaac-autoresearch/tests/test_e2e_dry_run.py

# Full repo test sweep — nothing else should regress
pixi run -e train-policy pytest src/lerobot-isaac-adapters/tests src/lerobot-isaac-autoresearch/tests -q
```

**Parallelization marker:** Phase 5 sequential after Phases 1–4 — tests verify the code that Phase 1–4 wrote.

---

### Phase 6 — Docs [PARALLEL with Phase 5]

**Files touched (1 edit, 1 optional new):**
- `programs/README.md` (edit — add LoRA row)
- `docs/lora-usage.md` (NEW, optional but recommended)

#### 6.1 `programs/README.md` — add row

In the "Selection Guide" table:

```markdown
| Pretrained policy + LoRA rank sweep | [`lerobot-policy-smolvla-lora.md`](lerobot-policy-smolvla-lora.md) | 1 h | ~10 h (12 exp) |
```

#### 6.2 `docs/lora-usage.md` — user-facing usage

Single page (~150 lines) covering:
1. When to use LoRA vs full FT (decision rule: SmolVLA → always LoRA on 10 GB).
2. CLI snippets:
   ```bash
   # Standalone LoRA fine-tune
   pixi run -e train-policy python -m lerobot_isaac_adapters.train \
     --target_arch smolvla --dataset datasets/kvgork/so101-pickplace1 \
     --output_dir outputs/lora_r8 --steps 10000 --batch_size 6 \
     --use_lora --lora_rank 8 --lora_alpha 16 \
     --lora_target_modules attn_qv

   # Via autoresearch sweep
   /autoresearch ~/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla-lora.md --type ml_model
   ```
3. Reading the sweep results (pointer to `.agent-state/<session>/autoresearch/<slug>/history.jsonl`).
4. Pointer to `plans/2026-05-19-lora-autoresearch-plan.md` §Findings for the rank-vs-pc_success ground truth.

#### 6.3 Acceptance commands (Phase 6)

```bash
grep -E "smolvla-lora" /home/koen/workspaces/lerobot-isaac-training/programs/README.md
# expect: 1+ line(s)

test -f /home/koen/workspaces/lerobot-isaac-training/docs/lora-usage.md && echo OK
# expect: OK   (only if 6.2 is done)
```

**Parallelization marker:** Phase 6 has no code dependency on Phase 5 — runs in parallel.

---

## 4. Cross-phase dependency graph

```
Phase 0  (research/decisions)
   │
   ▼
   ├──► Phase 1  (adapter + CLI)  ──┐
   │                                 │
   └──► Phase 2  (wrapper)        ──┤
                                    │
   ┌────────────────────────────────┘
   ▼
   ├──► Phase 3  (domain + program) ──┐
   │                                   │
   └──► Phase 4  (proposer operator)──┤
                                      │
                                      ▼
                                   Phase 5  (tests)
                                      │
                                      ▼
                                   Phase 6  (docs)   ← can start at any
                                                       point after Phase 0
```

Parallel pairs: `{Phase 1, Phase 2}` and `{Phase 3, Phase 4}`. Phase 6 has no dependency past Phase 0 — start it any time. Phase 5 is the integration gate.

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| `peft.get_peft_model` does not find `q_proj` modules in SmolVLA's name tree | Medium | The submodule inspection in §2.3 confirms the names. Add a unit test (Phase 5) that imports peft + a frozen-config SmolVLAPolicy stub and verifies `print_trainable_parameters()` reports non-zero LoRA params. If it fails in real runs, fall back to raw csv `target_modules` instead of presets. |
| `lerobot.scripts.train.make_policy` symbol renamed/moved in lerobot 0.5.x | Low | The monkey-patch is in `cli_train_cached`, which already targets `make_dataset` — same versioning surface. If it moves, both `cache_frames` and LoRA break together; the project already monitors this. |
| LoRA wrap interferes with `policy.save_pretrained` checkpoint format | Medium | PEFT wraps SAVE the LoRA adapter separately (`adapter_model.safetensors`). The existing eval/checkpoint path reads `policy.bin`. Document the difference in §13 of domain pack; defer save-format unification to future work. |
| Real GPU run shows r=4 ≈ r=32 (no signal) | Medium (expected!) | This IS the research outcome we are designing for. The plan's success criterion is "code lands + dry-run works"; the GPU sweep is Phase 7 future work. A flat curve is itself a useful finding (rank does not matter on this dataset). |
| `train_wrapper` final-line metric extraction breaks on extra LoRA stdout | Low | The `_last_metric_line` helper already returns the LAST `pc_success=` token regardless of intervening lines. Verified by test_e2e_dry_run already. |

---

## 6. Out-of-scope / explicit non-goals

- No real GPU training run (acceptance is dry-run only).
- No DoRA, rsLoRA, AdaLoRA, IA³, QLoRA — covered in §Findings only.
- No LoRA for ACT or Diffusion targets (scope-locked to smolvla).
- No checkpoint resume / merge of LoRA adapters into base weights (`peft_model.merge_and_unload()` is future work).
- No multi-adapter composition (`load_adapter` + `add_adapter`).
- No WandB grouping changes — the sweep uses the same group as the non-LoRA program.

---

## 7. Future work (Phase 7+, NOT in this plan)

1. **Real GPU rank sweep.** Run the program from §3.2 on `kvgork/so101-pickplace1` and chart `pc_success` vs rank. Expected shape: monotone rise from r=4 to r=8 (or r=16), plateau thereafter.
2. **rsLoRA toggle.** Add `--use_rslora` flag — single line in `_lora.py`.
3. **DoRA toggle.** Same — single line, `use_dora=True` in `LoraConfig`.
4. **Adapter merge for deployment.** Add `lerobot-isaac-deploy` codepath to call `peft_model.merge_and_unload()` and save merged weights for inference.
5. **Cross-arch LoRA.** Extend to `act` (currently rejected with a warning). ACT does not have a pretrained checkpoint in this stack, so LoRA is less useful there; defer.
6. **Vault wiki page.** Once the rank sweep produces data, create `05-Wiki/concepts/LoRA.md` and `05-Wiki/sources/2026-XX-XX-lora-paper.md` per the project's wiki conventions.

---

## 8. References

### Codebase context (read before implementing)
- `/home/koen/workspaces/lerobot-isaac-training/.pixi/envs/train-policy/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py` — `SmolVLAPolicy` definition, line 224.
- `/home/koen/workspaces/lerobot-isaac-training/.pixi/envs/train-policy/lib/python3.12/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py` — `SmolVLMWithExpertModel` definition, line 61.
- `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/policy_lerobot.py` — Phase 1 edit target.
- `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py` — Phase 1 CLI flag target.
- `/home/koen/workspaces/lerobot-isaac-training/src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py` — Phase 2 edit target.
- `/home/koen/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla.md` — Phase 3 program pattern.
- `/home/koen/workspaces/lerobot-isaac-training/programs/_domain_knowledge.md` — Phase 3 §13 append target.
- `/home/koen/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md` — Phase 4 operator target.

### Second Brain notes (Vault: `/home/koen/Documents/Vaults/Local`)
- `05-Wiki/entities/SmolVLA.md` — architecture, 450M params, VRAM profile on RTX 3080 (10 GB). No existing LoRA/PEFT page in the vault as of 2026-05-19 — this plan is the first authoritative LoRA reference.
- `05-Wiki/concepts/Autonomous-ML-Training-Loop.md` — the autoresearch loop architecture this plan extends.
- `05-Wiki/concepts/VLA-Architecture-Improvements.md` — the six-lever framework. LoRA is implicitly "lever 0" (parameter-efficient fine-tune precedes any architectural change).

### External (papers + docs)
1. Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. **arXiv:2106.09685**.
2. Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. **arXiv:2305.14314**.
3. Kalajdzievski, D. (2023). rsLoRA: A Rank-Stabilized Scaling Factor for LoRA Fine-Tuning. **arXiv:2312.03732**.
4. Liu, S.-Y. et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. **arXiv:2402.09353**.
5. HuggingFace PEFT documentation, `LoraConfig` reference: `https://huggingface.co/docs/peft/main/en/package_reference/lora`.
6. HuggingFace blog, *Fine-tuning SmolVLM with LoRA* (2025) — VLM-specific rank guidance.

---

**End of plan.**
