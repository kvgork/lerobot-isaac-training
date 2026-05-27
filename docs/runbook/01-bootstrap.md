# Runbook 01: Bootstrap — Install and Environment Setup

**Prerequisites:** Linux (Ubuntu 22.04), NVIDIA GPU (RTX 3080 or better), CUDA 12+
**Expected outcome:** Workspace packages importable, pixi env active, USD asset script ready

---

## Step 1: Enter Workspace

```bash
cd ~/workspaces/lerobot-isaac-training
```

---

## Step 2: Install Pixi Environment

```bash
# Install pixi if not present:
curl -fsSL https://pixi.sh/install.sh | bash

# Install workspace environment:
pixi install

# Activate:
pixi shell
```

Expected: Python 3.11+, PyTorch, h5py, pyarrow, pandas, wandb, hydra available.

---

## Step 3: Install Workspace Packages

**Thin-meta-repo (post-spinout, 2026-05-13):** only `lerobot-isaac-meta` lives in
`packages/`. The 7 siblings (env, adapters, autoresearch, synthetic, configs,
live in public GitHub repos at `https://github.com/kvgork/<name>`.
They are installed as editable path deps from `src/<name>/` (cloned by `bash scripts/setup.sh`).

Prerequisite: run `bash scripts/setup.sh` first to clone the siblings from GitHub. See
`docs/runbook/00-install.md` if you need to (re-)create them.

Verify after `pixi install`:
```bash
pixi run -e default python -c "import lerobot_isaac_meta; print('meta OK')"
pixi run -e default python -c "import lerobot_isaac_adapters; print('adapters OK')"
pixi run -e default python -c "import lerobot_isaac_synthetic; print('synthetic OK')"
pixi run -e default python -c "import lerobot_isaac_configs; print('configs OK')"
pixi run -e default python -c "import lerobot_isaac_dashboard; print('dashboard OK')"
pixi run -e default python -c "import lerobot_isaac_env; print('env OK')"
pixi run -e default python -c "import lerobot_isaac_autoresearch; print('autoresearch OK')"
pixi run -e default python -c "import robot_data_recorder; print('recorder OK')"
```

**Post-spinout standalone install** (no monorepo, no pixi):
```bash
bash scripts/install.sh
# or:
pip install "packages/lerobot-isaac-meta[post-spinout]"
pip install git+https://github.com/kvgork/robot-data-recorder.git@main   # standalone
```

---

## Step 4: Install Isaac Lab (System-Level)

**[Phase 1 impl required for full env functionality]**

Follow: https://isaac-sim.github.io/IsaacLab/

Quick summary:
```bash
# 1. Install Isaac Sim via Omniverse Launcher (or pip):
pip install isaacsim==<version> --extra-index-url https://pypi.nvidia.com

# 2. Install Isaac Lab:
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
./isaaclab.sh --install

# 3. Verify:
python -c "import isaaclab; print('Isaac Lab OK')"
```

Note: Without Isaac Lab, all `lerobot_isaac_env` imports gracefully fall back to stubs.

---

## Step 5: Set Workspace Environment Variable

```bash
export LEROBOT_ISAAC_WORKSPACE=~/workspaces/lerobot-isaac-training
# Add to ~/.bashrc for persistence
echo 'export LEROBOT_ISAAC_WORKSPACE=~/workspaces/lerobot-isaac-training' >> ~/.bashrc
```

---

## Step 6: Download SO-101 USD Asset

**[Phase 1 impl required to actually use the env]**

```bash
# Run the provided conversion script:
bash packages/lerobot-isaac-env/assets/usd/download_so101_urdf.sh

# Expected output: packages/lerobot-isaac-env/assets/usd/so101.usd
# If the script fails, see README in that directory for manual steps.
```

The script fetches the URDF from `TheRobotStudio/SO-ARM100` and converts via Isaac Lab's URDF importer.

---

## Step 7: Smoke Test — Dry Run Training

```bash
# Policy dry run (scaffolding only — no LeRobot needed):
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dry_run

# World model dry run:
lerobot-isaac-train \
  --target_arch dreamerv3 \
  --config packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml \
  --dry_run
```

Expected: prints "DRY RUN" message, exits 0, no errors.

---

## Step 8: Run Workspace Tests

```bash
pytest
```

Expected: all tests pass. Tests are designed to pass without Isaac Lab or LeRobot installed.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `pixi install` fails | Check network; try `pixi install --verbose` |
| `import isaaclab` fails | Isaac Lab not installed — stubs still work |
| `lerobot-isaac-train: command not found` | Run `uv sync` or `pip install -e packages/lerobot-isaac-meta` |
| CUDA out of memory | Reduce `--num_envs` to 1 in config |
| USD conversion fails | See `packages/lerobot-isaac-env/assets/usd/README.md` for manual fallback |
