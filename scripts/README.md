# scripts/

This directory contains installation and verification scripts for the LeRobot + Isaac Lab training workspace. The scripts are idempotent: they can be re-run safely and will skip steps already completed. Run them in the order shown below — each script prints clear PASS/FAIL output and exits with a non-zero code if something goes wrong.

---

## Execution Order

1. **`pixi install`** — Install workspace Python packages (lerobot-isaac-env, lerobot-isaac-adapters, etc.) into the pixi-managed environment. Must run first; the later scripts import these packages.

2. **`bash scripts/install_isaac_lab.sh`** — Clone and install Isaac Lab. _Isaac Sim must already be installed on the system before this step_ (see Troubleshooting).

3. **`bash scripts/install_lerobot.sh`** — Install the LeRobot library from PyPI (`pip install lerobot[all]`). Can run in parallel with step 2.

4. **`bash scripts/download_so101_usd.sh`** — Clone `TheRobotStudio/SO-ARM100`, convert the SO-101 URDF to a USD asset using Isaac Lab's `convert_urdf.py`, and place the result at `packages/lerobot-isaac-env/assets/usd/so101.usd`. Requires Isaac Lab (step 2).

5. **`bash scripts/verify_install.sh`** — Run 6 smoke tests covering Isaac Lab, LeRobot, all workspace packages, and the USD asset. Exit code = number of failures.

---

## Script Reference

| Script | Purpose | Idempotent | Depends on |
|---|---|---|---|
| `install_isaac_lab.sh` | Clone Isaac Lab `v2.1.0`, run `./isaaclab.sh --install` | Yes — skips if `import isaaclab` succeeds | Isaac Sim pre-installed |
| `install_lerobot.sh` | `pip install lerobot[all]` from PyPI or editable from source | Yes — skips if `import lerobot` succeeds | Python ≥ 3.10, pip |
| `download_so101_usd.sh` | Clone SO-ARM100, convert URDF → USD via Isaac Lab | Yes — skips if `so101.usd` exists | Isaac Lab installed |
| `verify_install.sh` | 6-check smoke test: isaaclab, lerobot, env, adapters, synthetic, USD | N/A (read-only) | All of the above |

---

## Environment Variable Overrides

| Variable | Default | Used by |
|---|---|---|
| `ISAAC_LAB_DIR` | `~/IsaacLab` | `install_isaac_lab.sh`, `download_so101_usd.sh` |
| `SO_ARM100_DIR` | `/tmp/SO-ARM100` | `download_so101_usd.sh` |
| `LEROBOT_SRC` | (PyPI install) | `install_lerobot.sh` (editable mode only) |
| `LEROBOT_EXTRAS` | `all` | `install_lerobot.sh` |
| `ISAAC_LAB_TAG` | `v2.1.0` | `install_isaac_lab.sh` |

---

## pixi run Shortcuts

After running `pixi install`, the following shortcuts are available (defined in root `pixi.toml`):

```bash
pixi run install-isaac-lab      # equivalent to bash scripts/install_isaac_lab.sh
pixi run install-lerobot        # equivalent to bash scripts/install_lerobot.sh
pixi run download-usd           # equivalent to bash scripts/download_so101_usd.sh
pixi run verify                 # equivalent to bash scripts/verify_install.sh
```

---

## Troubleshooting

**Isaac Sim not installed (`exit code 1` from `install_isaac_lab.sh`):**
Isaac Sim is a prerequisite that must be installed manually via the NVIDIA Omniverse Launcher or standalone installer before running any of these scripts. See: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html. Isaac Lab v2.1.0 requires Isaac Sim 4.2.x.

**GPU driver mismatch:**
Isaac Sim and Isaac Lab require a compatible NVIDIA driver (typically ≥ 535 for CUDA 12.x). Check with `nvidia-smi`. If the driver version is too old, Isaac Sim will either fail to start or crash during USD conversion. Update the driver via `sudo apt install nvidia-driver-535` (or the version matching your CUDA target).

**Python version mismatch:**
LeRobot requires Python ≥ 3.10. Isaac Lab ships its own Python environment (managed by `isaaclab.sh`). If you use a system Python older than 3.10 for the workspace packages, `import lerobot` will fail. Use `pixi` (which pins the Python version) or create a virtual environment with Python 3.10+.

**`convert_urdf.py` not found after Isaac Lab install:**
The script path may differ between Isaac Lab versions. Set `ISAAC_LAB_DIR` to your actual install location and verify that `${ISAAC_LAB_DIR}/scripts/tools/convert_urdf.py` exists. If the path has changed, report it as a workspace issue.

**`import lerobot_isaac_env` fails after `pixi install`:**
Make sure you are running inside the pixi environment: `pixi shell` then retry, or prefix commands with `pixi run python3 ...`.
