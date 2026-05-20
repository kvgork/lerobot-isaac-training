# LoRA Autoresearch — Next Steps

**Parent plan:** [`2026-05-19-lora-autoresearch-plan.md`](2026-05-19-lora-autoresearch-plan.md)
**Date:** 2026-05-19 (updated 2026-05-20)
**Status:** Phases 1–6 landed and verified. Real GPU sweep (Phase 7 future work) remains.

---

## Done (verified 2026-05-20)

| Phase | Files | Acceptance |
|-------|-------|------------|
| **1** — PEFT wrap + CLI flags | `targets/_lora.py` (NEW), `targets/policy_lerobot.py`, `cli_train_cached.py`, `train.py` | ast.parse OK; `LoraSpec.from_args` accepts presets + comma-separated; smolvla guard warns on `act`/`diffusion`; dry-run echoes LoRA config |
| **2** — Wrapper forwarding | `src/lerobot-isaac-autoresearch/.../train_wrapper.py` | positive passthrough (all 5 tokens in cmd) + negative omission (no LoRA tokens when flags absent) |
| **3** — Domain pack §13 + new program | `programs/_domain_knowledge.md` (§13 appended), `programs/lerobot-policy-smolvla-lora.md` (NEW) | `^## 13\. LoRA` grep hit; all 8 schema keys present; `test_programs_parse.py` extends to workspace-level path, 11 new param tests green |
| **4** — `tune_lora` operator | `~/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md` (operator block + selection bullet) + `install.sh` mirrored to `~/.claude/agents/` | `^### .tune_lora.` grep hit on both upstream + mirror |
| **5** — Tests | `test_train_argparse.py::TestLoraFlags` (10 tests), `test_train_wrapper.py::TestLoraFlagPassthrough` (4 tests), `test_e2e_dry_run.py::test_wrapper_dry_run_lora_smolvla`, `test_programs_parse.py` extended | 14 LoRA tests + 10 program-parse param tests all PASS via `PYTHONPATH=src/.../src pixi run -e train-policy pytest ...`; pre-existing 6 `wm_leworldmodel` failures unrelated (backend changed in commit d87d677 without test sync) |
| **6** — Docs | `programs/README.md` row added, `docs/lora-usage.md` (NEW, ~140 lines) | both files present + row visible in selection table |

**Smoke run (no pixi env needed):**
```bash
cd ~/workspaces/lerobot-isaac-training
python3 -c "import sys; sys.path.insert(0,'src/lerobot-isaac-autoresearch/src'); from lerobot_isaac_autoresearch.train_wrapper import parse_args,_build_cmd; ns,_=parse_args(['--target_arch','smolvla','--use_lora','--lora_rank','8','--lora_alpha','16','--lora_dropout','0.05','--lora_target_modules','attn_qv']); print(_build_cmd(ns))"
```

---

## Pending (Phase 7 future work)

Real GPU rank sweep — out of scope for Phases 1–6. Run on RTX 3080:

```bash
/autoresearch ~/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla-lora.md --type ml_model
```

Budget ~10h (12 exp × 1h). Pre-flight checklist:
- [ ] `~/workspaces/spinouts/` populated (currently MISSING — see note below)
- [ ] Datasets: `datasets/kvgork/so101-pickplace1` present (~7491 frames)
- [ ] SmolVLM2-500M weights prefetched: `bash scripts/_run_smolvla_tonight.sh --prefetch-weights`
- [ ] `peft>=0.10` installed in `train-policy` env

---

## Resolved decisions (Phases 3–6 execution)

1. **Phase 4 location.** Applied upstream edit + `install.sh` mirror. Both
   `~/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md`
   and `~/.claude/agents/workers/autoresearch-ml-proposer-worker.md` carry
   the `tune_lora` operator + selection bullet.
2. **`docs/lora-usage.md`.** Created. 140 lines covering CLI snippets,
   sweep invocation, checkpoint format caveat, VRAM table, failure modes.
3. **`test_programs_parse.py` PROGRAM_FILES.** Extended with a workspace-
   level path constant `WORKSPACE_PROGRAMS_DIR` (4 parents up from the
   test file), so the new top-level program file is checked without
   duplicating it into the package-local `programs/` dir.
4. **Spinouts dir absence.** `~/workspaces/spinouts/` does NOT exist on
   this machine, so the `train-policy` env's git+file:// installs of
   `lerobot-isaac-adapters` / `lerobot-isaac-autoresearch` are stale.
   Tests were run via `PYTHONPATH=src/.../src pixi run -e train-policy
   pytest ...` which shadows the installed copy with the live source
   tree. Real LoRA training will need the spinouts populated OR a
   working `pixi install -e editable` for the train env.

---

## Risk reminders (from parent plan §5)

- PEFT module name resolution — `lerobot.scripts.train.make_policy` may not exist; cli_train_cached falls back via `getattr` + warn (already implemented Phase 1).
- LoRA flat curve = valid finding. Plan §2 cites Hu 2021 ≤1.1pp range. Stopping rule: kill sweep if r=4 and r=8 both ≥0.7.
- Checkpoint format: PEFT writes `adapter_model.safetensors`, NOT full state dict. Document in `docs/lora-usage.md` so users don't try to load LoRA checkpoints as full models.

---

## Resume command (Phase 7 GPU sweep)

```bash
# 1. Populate spinouts (if missing) — see plans/2026-05-13-clone-to-src-workspace-plan.md
# 2. Prefetch SmolVLM weights
bash scripts/_run_smolvla_tonight.sh --prefetch-weights

# 3. Kick off the rank sweep
cd ~/tools/claude_code
/autoresearch ~/workspaces/lerobot-isaac-training/programs/lerobot-policy-smolvla-lora.md --type ml_model
```

Expected curve shape (per `plans/2026-05-19-lora-autoresearch-plan.md`
§Findings, Hu et al. 2021 Table 6): monotone rise from r=4 to r=8 or r=16,
plateau thereafter. Flat curve is a valid finding — log it and stop.
