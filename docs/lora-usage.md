# LoRA Fine-Tuning Usage Guide (SmolVLA)

**Status:** Phase 1–6 of `plans/2026-05-19-lora-autoresearch-plan.md` landed (2026-05-20).
**Scope:** PEFT LoRA adapters on top of `SmolVLAPolicy` only. ACT / Diffusion / DreamerV3 / LeWorldModel ignore LoRA flags.
**Stack:** lerobot 0.5+ (smolvla extra) + `peft>=0.10`, RTX 3080 10 GB, SO-101 6-DOF.

---

## When to use LoRA vs full fine-tune

| Scenario | Recommendation |
|----------|----------------|
| SmolVLA on RTX 3080 10 GB | **Always LoRA at r=64.** Full FT optimizer state (3.6 GB) eats the headroom you need for batch size 4-8 at 256² images. |
| Strong overfit signal on small dataset (<20 episodes) | LoRA r=32 with dropout=0.1. Mid-ladder, more regularization. |
| Curriculum advance / saturation pressure | LoRA r=128 with `attn_qkvo` preset. Broader coverage, ~7% trainable. |
| Multi-task / domain-shift | LoRA r=128 minimum (LoRA-SP showed multi-task collapse below r=64). |
| Anything other than SmolVLA | LoRA wrap is **not wired**; the adapter prints a warning and runs the unwrapped policy. |

**VLA rank reference:** LoRA-SP (arXiv:2603.07404) ran the exact SmolVLA ablation across r ∈ {8, 16, 32, 64, 128} on multi-task SO-101-style manipulation. Result: r=8 → 0% success, r=128 → 40-93% per task. HuggingFace's official LeRobot PEFT example defaults to `r=64, alpha=64`. OpenVLA (arXiv:2406.09246) uses r=32 for single-task LIBERO. **Do NOT cite Hu 2021 / HF SmolVLM r=8 guidance for action-prediction heads — VLAs map vision → continuous actions, OOD vs text-token pretraining.**

---

## CLI: standalone LoRA fine-tune

```bash
pixi run -e train-policy python -m lerobot_isaac_adapters.train \
  --target_arch smolvla \
  --dataset datasets/kvgork/so101-pickplace1 \
  --output_dir outputs/lora_r64 \
  --steps 10000 --batch_size 6 \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.0 \
  --lora_target_modules attn_qv \
  -- --policy.load_vlm_weights=true
```

Five LoRA flags. `--lora_target_modules` accepts:
- Preset name: `attn_qv` (q+v projections, paper recipe), `attn_qkvo` (q+k+v+o, HF default), `expert_only` (action expert only, lowest VRAM).
- CSV of suffix matches: `"q_proj,v_proj,gate_proj"`.

The `-- --policy.load_vlm_weights=true` remainder is mandatory the first time you train SmolVLA — without it, the SmolVLM2-500M backbone stays at random init and LoRA wraps nothing useful.

---

## CLI: autoresearch rank sweep

```bash
cd ~/tools/claude_code
/autoresearch ~/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla-lora.md --type ml_model
```

Program file: `programs/lerobot-policy-smolvla-lora.md`. Default sweep (VLA-specific):

```yaml
lora_rank:           [16, 32, 64, 128]   # VLA ladder — see plan §2.6
lora_alpha_factor:   [1, 2]              # multiplied with rank
lora_dropout:        [0.0, 0.05, 0.1]
lora_target_modules: [attn_qv, attn_qkvo]
```

Budget: 12 experiments × 1 h = ~10 h on RTX 3080. Plateau limit: 3.

The proposer worker has a `tune_lora` operator (mutates ONE LoRA knob per call, ladder-stepwise) that fires automatically when the program declares `allow_lora_mutation: true`.

**Auto-extension past r=128.** Program declares:

```yaml
allow_rank_extension: true
rank_extension_cap: 512
rank_rising_threshold: 0.05
```

When the curve still rises at the ladder top (default r=128), proposer auto-extends to r=256, then r=512. Stops when curve plateaus OR cap reached. Useful for multi-task / long-horizon tasks where intrinsic rank exceeds typical single-skill needs. Disable by setting `allow_rank_extension: false` if you want a strict ladder sweep without exploration.

---

## Reading the sweep results

Per-trial artefacts land under:

```
.agent-state/<session>/autoresearch/lerobot-policy-smolvla-lora/
  history.jsonl    # one record per trial: config + metric
  best.json        # current-best LoRA config
  plateau.json     # consecutive_non_improvements counter
  trial_<N>.log    # raw stdout (LoRA banner + pc_success line)
```

Plot rank → pc_success:

```bash
python -c "
import json, sys
from pathlib import Path
hist = Path(sys.argv[1])
for line in hist.read_text().splitlines():
    rec = json.loads(line)
    cfg = rec.get('config', {})
    print(rec['experiment'], cfg.get('lora_rank'), rec['metric'])
" .agent-state/<session>/autoresearch/lerobot-policy-smolvla-lora/history.jsonl
```

For richer visualisation use the dashboard (`pixi run -e dashboard dashboard`); the Autoresearch tab auto-detects the slug.

---

## Checkpoint format gotcha

PEFT saves the adapter as **`adapter_model.safetensors`** — NOT the full state dict. Treat the LoRA checkpoint as a delta:

```python
from peft import PeftModel
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

base = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
wrapped = PeftModel.from_pretrained(base, "outputs/lora_r8/checkpoint-10000")
# wrapped.merge_and_unload() flattens LoRA into the base weights — irreversible.
```

For deployment (`lerobot-isaac-deploy`), merge once with `merge_and_unload()` and save the result. Do NOT try to load `adapter_model.safetensors` as a full policy — it will fail with missing-key errors.

---

## VRAM budget table

Measured on RTX 3080 10 GB with batch_size=4, 256² images. VLA ladder:

| r   | LoRA params (attn_qv) | Extra VRAM (opt+grad) | Total |
|-----|------------------------|------------------------|-------|
| 16  | 1.96 M                 | ~20 MB                 | ~7.4 GB |
| 32  | 3.93 M                 | ~40 MB                 | ~7.5 GB |
| 64  | 7.86 M                 | ~80 MB                 | ~7.5 GB |
| 128 | 15.72 M                | ~160 MB                | ~7.6 GB |

`attn_qkvo` doubles the LoRA-params column. At r=128 + attn_qkvo: ~31M trainable + ~320 MB opt state — still fits 10 GB at batch=6.

Full SmolVLA FT at the same batch lands at ~9.8 GB (borderline OOM). LoRA buys ~3-4 GB of free VRAM — useful headroom for larger batch sizes or longer context windows.

---

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `pc_success` stuck at baseline | LoRA scale `alpha/r` too small | Try `alpha = 2 * r` |
| Loss diverges first 500 steps | LoRA scale too large or lr too high | Halve lr to 1e-5; keep r |
| `ValueError: Target modules ... not found` | Wrong preset for policy version | Try `attn_qkvo`; fall back to raw `q_proj,v_proj` |
| Trainable param count is full 450 M | Wrap did not happen | Check `LEROBOT_ISAAC_USE_LORA=1` reached subprocess; verify `cli_train_cached` patched `make_policy` |
| OOM with `attn_qkvo` | Broad target set + batch too high | Switch to `attn_qv` first, then halve batch |

---

## References

- Plan: [`plans/2026-05-19-lora-autoresearch-plan.md`](../plans/2026-05-19-lora-autoresearch-plan.md) §2.6 — VLA rank correction note (2026-05-20).
- VLA-specific rank research: [`project-context/papers/vla-lora-rank-research-2026-05-20.md`](../project-context/papers/vla-lora-rank-research-2026-05-20.md) — LoRA-SP table, OpenVLA, HF SmolVLA PEFT defaults.
- Domain pack §13: [`programs/_domain_knowledge.md`](../programs/_domain_knowledge.md) — proposer-side knowledge.
- Companion program (non-LoRA): [`programs/lerobot-policy-smolvla.md`](../programs/lerobot-policy-smolvla.md).
- Upstream proposer worker (with `tune_lora` operator): `${CLAUDE_CODE_ROOT}/agents/workers/autoresearch-ml-proposer-worker.md`.
- LoRA-SP paper: [arXiv:2603.07404](https://arxiv.org/abs/2603.07404) — exact SmolVLA rank ablation.
- OpenVLA paper: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) — r=32 for single-task LIBERO.
