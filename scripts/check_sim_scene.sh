#!/usr/bin/env bash
# =============================================================================
# check_sim_scene.sh — Preflight a USD scene (output of isaac-auto-scene)
# for sim_deploy.sh.
#
# Validates:
#   1. <scene>.usd exists and parses via pxr.Usd.Stage.Open
#   2. Required prims present (SO101, object, basket, two cameras)
#   3. Sibling <scene>.meta.json present + schema fields complete
#
# Sim-deploy NEVER invokes isaac-auto-scene — the USD must already exist.
# This script confirms it's loadable + has the expected layout.
#
# Usage:
#   bash scripts/check_sim_scene.sh assets/sim_scenes/so101_workspace.usd
#   bash scripts/check_sim_scene.sh path/to/whatever.usd
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

USD_PATH="${1:?usage: $0 <usd-path>}"

if [ ! -f "$USD_PATH" ]; then
    echo "ERROR: USD file not found: $USD_PATH" >&2
    echo "Generate one via isaac-auto-scene on the laptop with the D435 attached." >&2
    exit 2
fi

META_PATH="${USD_PATH%.usd}.meta.json"
if [ ! -f "$META_PATH" ]; then
    echo "WARN: sibling meta file missing: $META_PATH" >&2
    echo "isaac-auto-scene should emit this alongside the USD." >&2
fi

PY="$WORKSPACE/.pixi/envs/sim/bin/python"
if [ ! -x "$PY" ]; then
    PY="$WORKSPACE/.pixi/envs/default/bin/python"
fi
if [ ! -x "$PY" ]; then
    echo "ERROR: no pixi env python found (tried sim, default)" >&2
    exit 3
fi

"$PY" - <<PY
import json, sys
from pathlib import Path

usd = Path("$USD_PATH").resolve()
meta = Path("$META_PATH").resolve()

REQUIRED_PRIMS = [
    "/World/SO101",
    "/World/object",
    "/World/basket",
    "/World/cameras/overhead",
    "/World/cameras/wrist",
]
REQUIRED_META_FIELDS = [
    "capture_ts",
    "source_workspace",
    "arm_pose",
    "camera_intrinsics",
    "basket_bounds",
    "usd_schema_version",
]

errors = []
warnings = []

# 1. Parse USD.
try:
    from pxr import Usd  # type: ignore
except ImportError as exc:
    print(f"[check_sim_scene] WARN: pxr.Usd not available in this env ({exc}). "
          f"Skipping prim presence check — install Isaac Sim or run inside .pixi/envs/sim.")
    warnings.append("pxr.Usd not importable — prim presence not verified")
else:
    stage = Usd.Stage.Open(str(usd))
    if stage is None:
        errors.append(f"failed to open USD stage at {usd}")
    else:
        present = {p.GetPath().pathString for p in stage.Traverse()}
        for path in REQUIRED_PRIMS:
            if path not in present:
                errors.append(f"missing prim: {path}")
        print(f"[check_sim_scene] USD loaded — {len(present)} prims, "
              f"{len([e for e in errors if 'missing prim' in e])} required missing")

# 2. Meta JSON.
if meta.is_file():
    try:
        meta_data = json.loads(meta.read_text())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"meta.json parse error: {exc}")
        meta_data = {}
    for field in REQUIRED_META_FIELDS:
        if field not in meta_data:
            errors.append(f"meta.json missing field: {field}")
    if "usd_schema_version" in meta_data:
        ver = meta_data["usd_schema_version"]
        if ver != 1:
            warnings.append(f"meta.usd_schema_version={ver} != 1 (this checker validates v1)")
else:
    warnings.append(f"meta.json absent at {meta}")

# 3. Report.
for w in warnings:
    print(f"[check_sim_scene] WARN: {w}")
for e in errors:
    print(f"[check_sim_scene] ERROR: {e}", file=sys.stderr)

if errors:
    print(f"[check_sim_scene] FAILED — {len(errors)} error(s)", file=sys.stderr)
    sys.exit(2)
print(f"[check_sim_scene] OK — {usd.name} ready for sim_deploy")
PY
