# lerobot 0.6.0 upgrade + world-model-policy wiring (2026-07-08)

`/orchestrate` deliverable. Updates the training pipeline from **lerobot 0.5.1 → 0.6.0**
and wires lerobot 0.6.0's incorporated **world models** so the pipeline can train them.

## What lerobot 0.6.0 changed (researched, source-verified)

- Release **v0.6.0** "Imagine, Evaluate, Improve" (2026-07-07). Ships world models as
  **policy types** — not a separate `train_world_model` script:
  - **`vla_jepa`** (~2B, Qwen3-VL + flow-matching DiT; V-JEPA2 world model used at
    *train only*, dropped at inference; pretrained ckpts e.g. `lerobot/VLA-JEPA-Pretrain`).
    **The only one that fits an RTX 3080 10GB.**
  - **`fastwam`** (~5B Wan2.2 video expert) — >>10 GB VRAM.
  - **`lingbot_va`** (~5B DiT + ~20 GB frozen VAE/text components; WM at train **and**
    inference) — big HW only.
- All three train via the **same `lerobot-train` CLI** as ACT/SmolVLA (only `--policy.type`
  / `--policy.path` differ) and report `eval/pc_success`.
- Breaking change that mattered: 0.6.0 split installs into **extras** — need
  `lerobot[training]` + per-policy extras (`vla_jepa` etc.). torch 2.10 / py 3.12 already
  satisfied 0.6.0's `torch>=2.7,<2.12` + `py>=3.12`, so **no torch/py bump needed**.
- `stable-worldmodel` (a dangling pixi dep since May) is an **independent** MBRL research
  harness (Galilai group), unrelated to lerobot's WM — **removed**.

## Decisions (user, 2026-07-08)

1. Wire **all three** WM policies (`vla_jepa` primary; `fastwam`/`lingbot_va` registered but
   need bigger HW). 2. **Remove** the dangling `stable-worldmodel` dep. 3. **Upgrade envs now**.

## Changes

**Workspace repo:**
- `pixi.toml` — removed `[feature.stable-worldmodel]` + its refs in `record`/`full` envs.
- `scripts/install_train_deps.sh` — version-aware: `LEROBOT_MIN_VERSION=0.6.0`, default
  `LEROBOT_EXTRAS=training,smolvla,feetech,vla_jepa`; `_lerobot_current_ok` helper installs/
  upgrades (`pip install -U "lerobot[EXTRAS]>=MIN"`) only when below min (no more blind
  short-circuit that never upgraded).
- `CLAUDE.md`, `docs/runbook/00-install.md`, `docs/runbook/03-train-policy.md` — WM-policy
  usage + corrected the stale "LeWorldModel BLOCKED" note.

**Adapter repo (`src/lerobot-isaac-adapters`, separate git checkout/PR):**
- `train.py` — new `_WM_POLICY_ARCHS=("vla_jepa","fastwam","lingbot_va")`; they dispatch as
  policies (`policy_lerobot`, metric `pc_success`), distinct from predictive `_WM_ARCHS`
  (`dreamerv3`, `le_world_model`).
- `targets/policy_lerobot.py` — `_lerobot_policy_type` map extended (identity); **guard**:
  a pretrained `-- --policy.path=…` omits the auto `--policy.type` (draccus conflict).
- `tests/` — arch count 5→8, new WM-policy tests (incl. the `--policy.path` guard); the 6
  pre-existing `le_world_model` `train_world_model` tests gated behind
  `LEROBOT_ISAAC_LEWM_BACKEND=hf` (the HF subprocess path is opt-in since 2026-05-14).

## Verification (this session, no GPU)

- Both `train-policy` + `train-lewm` envs upgraded **lerobot 0.5.1 → 0.6.0** (rc=0; pulled
  `transformers 5.5.4` + `qwen-vl-utils` for the vla_jepa backbone; no dep conflicts).
- `vla_jepa`/`fastwam`/`lingbot_va` subpackages + `factory.py` registration confirmed in the
  installed 0.6.0; `lerobot-train` entrypoint present.
- Adapter dry-runs emit correct `--policy.type`; `--policy.path` omits `--policy.type`.
- **216 adapter tests pass** (was 210 pass / 6 pre-existing fail → 0 fail).
- `install_train_deps.sh --policy/--lewm` short-circuit "lerobot 0.6.0 OK (>= 0.6.0)"; pixi
  manifest parses; 0 `stable-worldmodel` refs remain.

## Legacy `le_world_model` note

Still NOT served upstream (0.6.0 did not resurrect `lerobot.scripts.train_world_model`).
Defaults to the in-process `_lewm_minimal` trainer; `LEROBOT_ISAAC_LEWM_BACKEND=hf` opts into
the subprocess path (fork only). `dreamerv3` (sheeprl) remains the predictive-WM / MBRL route.

## Next steps (not done here — no GPU)

- Real GPU smoke: `lerobot-isaac-train --target_arch vla_jepa --dataset datasets/so101-pickplace-new
  --batch_size 4 --steps 20000 -- --policy.path=lerobot/VLA-JEPA-Pretrain`.
- Commit + push the adapter changes inside `src/lerobot-isaac-adapters/` (separate repo).
- Verify 0.6.0 didn't shift the ACT/SmolVLA deploy-candidate CLI flags before re-training them.
