"""Build a placeholder merged real+synthetic LeRobot dataset for dashboard demo.

Real SO-101 dataset already lives under `datasets/kvgork/so101-pickplace1/`.
Full Isaac DR replay (`lerobot_isaac_synthetic.isaac_dr.replay_runner`) requires
Isaac Sim + Isaac Lab + `lerobot_isaac_env`, none of which are installed in this
workspace yet.

This script instead writes a single LeRobot v3-shaped `meta/episodes.parquet`
under `datasets/so101-merged/<repo_id>/meta/` that mirrors the real episode
counts but ALSO populates a `source` column with the three canonical labels
(`real`, `sim_dr`, `mimicgen`). It does NOT copy or fabricate frame data — the
file is purely metadata for the dashboard's Synthetic / Data Collection tabs.

The `_metadata.note` field in the parquet's schema metadata makes the synthetic
origin explicit so no downstream consumer mistakes this for a real merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--real_dataset",
        default="datasets/kvgork/so101-pickplace1",
        help="Path to the real LeRobotDataset to use as anchor.",
    )
    ap.add_argument(
        "--output_root",
        default="datasets/so101-merged",
        help="Where the synthetic-shape merged dataset will live.",
    )
    ap.add_argument(
        "--n_sim_dr",
        type=int,
        default=40,
        help="How many sim_dr-tagged placeholder episodes to inject.",
    )
    ap.add_argument(
        "--n_mimicgen",
        type=int,
        default=20,
        help="How many mimicgen-tagged placeholder episodes to inject.",
    )
    args = ap.parse_args(argv)

    import pandas as pd

    real_root = Path(args.real_dataset).resolve()
    real_info = json.loads((real_root / "meta" / "info.json").read_text())
    real_total_eps = int(real_info["total_episodes"])
    real_total_frames = int(real_info["total_frames"])
    real_fps = int(real_info.get("fps", 30))

    out_repo_root = Path(args.output_root).resolve() / "merged"
    (out_repo_root / "meta").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # Real episodes — copy episode-level rows from any existing meta/episodes.parquet
    # or just synthesise length-only rows so the dashboard sees them.
    # We synthesise lengths uniformly to total_frames / total_eps.
    real_avg_len = max(1, real_total_frames // max(real_total_eps, 1))
    for idx in range(real_total_eps):
        rows.append(
            {
                "episode_index": idx,
                "tasks_index": 0,
                "length": real_avg_len,
                "source": "real",
            }
        )

    next_idx = real_total_eps
    for i in range(args.n_sim_dr):
        rows.append(
            {
                "episode_index": next_idx + i,
                "tasks_index": 0,
                "length": real_avg_len,
                "source": "sim_dr",
            }
        )
    next_idx += args.n_sim_dr
    for i in range(args.n_mimicgen):
        rows.append(
            {
                "episode_index": next_idx + i,
                "tasks_index": 0,
                "length": real_avg_len,
                "source": "mimicgen",
            }
        )

    df = pd.DataFrame(rows)
    out_parquet = out_repo_root / "meta" / "episodes.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[synth] wrote {out_parquet} ({len(df)} rows)")

    # tasks.parquet — minimal single-task lookup matching the real task.
    tasks_path = real_root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        # Reuse real task table verbatim.
        out_tasks = out_repo_root / "meta" / "tasks.parquet"
        out_tasks.write_bytes(tasks_path.read_bytes())
        print(f"[synth] copied tasks.parquet from real dataset")

    # info.json mirroring v3 layout for the parquet_dataset loader.
    info_payload = {
        "codebase_version": "v3.0",
        "robot_type": None,
        "total_episodes": int(df.shape[0]),
        "total_frames": int(df["length"].sum()),
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": real_fps,
        "splits": {"train": f"0:{df.shape[0]}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "_metadata": {
            "note": (
                "Placeholder merged dataset for dashboard validation. Episode "
                "data is NOT present — only the meta/ summary. Real episodes "
                "remain in datasets/kvgork/so101-pickplace1/; sim_dr + mimicgen "
                "rows are placeholders pending full Isaac Lab + MimicGen wiring."
            ),
            "source_breakdown": {
                "real": real_total_eps,
                "sim_dr": args.n_sim_dr,
                "mimicgen": args.n_mimicgen,
            },
        },
        "features": real_info.get("features", {}),
    }
    info_out = out_repo_root / "meta" / "info.json"
    info_out.write_text(json.dumps(info_payload, indent=2), encoding="utf-8")
    print(f"[synth] wrote {info_out}")

    # Verify with the dashboard loaders.
    print(f"[synth] done — dataset root: {out_repo_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
