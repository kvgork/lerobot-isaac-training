"""Convert SO-101 URDF to USD with proper PhysicsDriveAPI + ArticulationRootAPI.

Uses Isaac Sim 5.1's URDFImporter directly (AppLauncher must boot first, then
the isaacsim.asset.importer.urdf extension is enabled and its Python path added).

Isaac Lab's UrdfConverter is NOT used because it attempts to pin-back to
isaacsim.asset.importer.urdf-2.4.31, whose native .so has a broken USD symbol
in this installation. The new Python-based URDFImporter in Isaac Sim 5.1
adds PhysicsDriveAPI + ArticulationRootAPI natively via convert_joints_attributes().

Usage:
    pixi run -e sim python scripts/_convert_so101_urdf.py \
        --input /tmp/so-arm100/Simulation/SO101/so101_new_calib.urdf \
        --output src/lerobot-isaac-env/assets/usd/so101.usd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    # 1. AppLauncher FIRST — carb must be initialised before any isaaclab.sim import.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="path to SO-101 URDF file")
    parser.add_argument("--output", required=True, help="destination USD file path")
    args = parser.parse_args()

    launcher = AppLauncher(headless=True, enable_cameras=False)
    for _ in range(2):
        launcher.app.update()

    # 2. Enable isaacsim.asset.importer.urdf extension and add its Python path.
    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()

    # Enable the extension (bundled with Isaac Sim 5.1, does NOT require old .so)
    ext_name = "isaacsim.asset.importer.urdf"
    if not ext_manager.is_extension_enabled(ext_name):
        ext_manager.set_extension_enabled_immediate(ext_name, True)
        for _ in range(3):
            launcher.app.update()

    ext_path = ext_manager.get_extension_path(ext_name)
    if ext_path and ext_path not in sys.path:
        sys.path.insert(0, ext_path)
    # Also add pip_prebundle path so urdf_usd_converter is importable
    pip_prebundle = Path(ext_path) / "pip_prebundle" if ext_path else None
    if pip_prebundle and pip_prebundle.exists() and str(pip_prebundle) not in sys.path:
        sys.path.insert(0, str(pip_prebundle))

    print(f"[convert] ext_path = {ext_path}")

    # 3. Now import URDFImporter
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"URDF not found at {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    robot_name = in_path.stem  # e.g. "so101_new_calib"

    print(f"[convert] input      = {in_path}")
    print(f"[convert] output     = {out_path}")
    print(f"[convert] robot_name = {robot_name}")

    cfg = URDFImporterConfig(
        urdf_path=str(in_path),
        usd_path=str(out_path.parent),
        merge_mesh=False,        # keep full mesh hierarchy
        collision_from_visuals=False,
        collision_type="Convex Hull",
        allow_self_collision=False,
        debug_mode=False,
    )

    importer = URDFImporter(config=cfg)
    final_path = importer.import_urdf()
    print(f"[convert] written    = {final_path}")

    # The importer places result at <usd_path>/<robot_name>/<robot_name>.usda.
    # Rename/copy it to the requested out_path if different.
    import shutil
    final = Path(final_path)
    if final.resolve() != out_path:
        if out_path.exists():
            out_path.unlink()
        shutil.copy2(str(final), str(out_path))
        print(f"[convert] copied to  = {out_path}")

    print("[convert] SUCCESS — closing app.")
    launcher.app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
