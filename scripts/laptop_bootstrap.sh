#!/usr/bin/env bash
# laptop_bootstrap.sh — one-time setup on the deploy laptop.
#
# Mirrors the recorder host's software stack PLUS robot-data-runner.
# Idempotent: re-running it is safe; skips already-installed pieces.
#
# Prereqs on the laptop:
#   - Ubuntu 22.04+ or similar (the recorder host's distro).
#   - pixi installed (https://pixi.sh/install).
#   - NVIDIA driver loaded (the 6 GB GPU). `nvidia-smi` returns a row.
#   - U2D2 / SO-101 USB plugged in OR ready to plug in for calibration.
#
# Usage (on the laptop, NOT the desktop):
#   bash laptop_bootstrap.sh
#
# Pins lerobot to the version the desktop trained against to avoid
# policy-loading drift. Override with LEROBOT_VERSION=... if you know
# what you're doing.
set -uo pipefail

LEROBOT_VERSION="${LEROBOT_VERSION:-0.5.1}"
RUNNER_REPO="${RUNNER_REPO:-https://github.com/kvgork/robot-data-runner.git}"
# Local fallback when the GitHub repo isn't published yet — set to the
# rsync target the desktop pushes the spinout to.
RUNNER_LOCAL="${RUNNER_LOCAL:-$HOME/workspaces/spinouts/robot_data_runner}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/workspaces/lerobot-isaac-deploy}"

G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; R='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${C}[INFO]${NC} $*"; }
ok()    { echo -e "${G}[OK]${NC}  $*"; }
warn()  { echo -e "${Y}[WARN]${NC} $*"; }
err()   { echo -e "${R}[ERR]${NC} $*" >&2; }

# 1. Pre-flight ---------------------------------------------------------
info "checking pixi..."
command -v pixi >/dev/null 2>&1 || { err "install pixi first: https://pixi.sh/install"; exit 2; }
ok "pixi $(pixi --version)"

info "checking NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    warn "nvidia-smi not found — runner will use CPU + diffusion n_action_steps decimation"
fi

# 2. Workspace skeleton --------------------------------------------------
info "preparing workspace at $WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR/"{outputs/eval,checkpoints,datasets}
cd "$WORKSPACE_DIR"

if [ ! -f pixi.toml ]; then
    info "writing minimal pixi.toml..."
    cat > pixi.toml <<EOF
[workspace]
name = "lerobot-isaac-deploy-laptop"
version = "0.1.0"
channels = ["conda-forge", "nvidia"]
platforms = ["linux-64"]

[dependencies]
python = ">=3.10,<3.13"
pip = "*"

[pypi-dependencies]
numpy = "*"

[feature.lerobot]
# pinned manually — match desktop's lerobot version

[environments]
deploy = ["lerobot"]
EOF
fi

pixi install -e deploy 2>&1 | tail -3

# 3. Install lerobot pinned ---------------------------------------------
ENV_PY="$(pwd)/.pixi/envs/deploy/bin/python"
info "pinning lerobot==$LEROBOT_VERSION ..."
"$ENV_PY" -m pip install --quiet "lerobot==$LEROBOT_VERSION" || {
    err "lerobot pin install failed"
    exit 3
}
ok "lerobot $("$ENV_PY" -c 'import lerobot; print(lerobot.__version__)')"

# 4. Install robot-data-runner ------------------------------------------
RUNNER_SRC="$WORKSPACE_DIR/src/robot-data-runner"
mkdir -p "$WORKSPACE_DIR/src"
if [ -d "$RUNNER_SRC/.git" ]; then
    info "robot-data-runner clone exists; pulling..."
    git -C "$RUNNER_SRC" pull --ff-only 2>&1 | tail -2
elif [ -d "$RUNNER_LOCAL" ]; then
    info "cloning robot-data-runner from local: $RUNNER_LOCAL"
    git clone "$RUNNER_LOCAL" "$RUNNER_SRC" 2>&1 | tail -2
else
    info "cloning robot-data-runner from $RUNNER_REPO"
    git clone "$RUNNER_REPO" "$RUNNER_SRC" 2>&1 | tail -2
fi

"$ENV_PY" -m pip install --quiet -e "$RUNNER_SRC"
ok "$("$ENV_PY" -c 'import robot_data_runner; print(robot_data_runner.__version__)')"

# 5. Verify CLI entries -------------------------------------------------
info "verifying CLI..."
"$WORKSPACE_DIR/.pixi/envs/deploy/bin/robot-data-run --help" >/dev/null 2>&1 || true
"$WORKSPACE_DIR/.pixi/envs/deploy/bin/robot-data-run-check --help" >/dev/null 2>&1 || true
"$WORKSPACE_DIR/.pixi/envs/deploy/bin/robot-data-run-eval --help" >/dev/null 2>&1 || true
ok "CLI entries present"

# 6. USB + calibration hint --------------------------------------------
if [ -e /dev/ttyACM0 ] || [ -e /dev/ttyUSB0 ]; then
    info "USB device found:"
    ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head
    info "Next: run lerobot-find-port + lerobot-calibrate inside the env:"
    echo "    pixi shell -e deploy"
    echo "    lerobot-find-port"
    echo "    lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0"
else
    warn "no /dev/ttyACM* or /dev/ttyUSB* — plug the SO-101 + U2D2 in, then re-run calibration step"
fi

echo ""
ok "laptop bootstrap complete"
echo ""
echo "Next: rsync a checkpoint from the desktop:"
echo "    rsync -av desktop:.../checkpoints/0024000 $WORKSPACE_DIR/checkpoints/"
echo ""
echo "Then dry-run:"
echo "    pixi run -e deploy robot-data-run \\"
echo "        --policy-path $WORKSPACE_DIR/checkpoints/0024000/pretrained_model \\"
echo "        --port /dev/ttyACM0 \\"
echo "        --dataset-root $WORKSPACE_DIR/datasets/kvgork/so101-pickplace1 \\"
echo "        --camera d435_rgb=/dev/video0,640,480 --duration-s 30 -v"
