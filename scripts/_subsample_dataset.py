"""Subsample a LeRobotDataset v3 to first N episodes from the training pool.

Held-out episodes (the last `--eval-holdout`) are excluded from the
training pool so the sweep can evaluate every trial on the SAME 3
held-out episodes (matching the rest of this workspace's eval convention).

Output is a sibling directory that shares the underlying data parquet
(symlinked) but rewrites `meta/episodes/chunk-*/file-*.parquet` to only
reference the first N episode rows. `meta/info.json` is updated to
match.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Source LeRobotDataset root")
    ap.add_argument("--dst", required=True, help="Destination dataset root (created)")
    ap.add_argument("--n-episodes", type=int, required=True)
    ap.add_argument("--eval-holdout", type=int, default=3,
                    help="Last K episodes excluded from training pool (default 3)")
    args = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if dst.exists():
        shutil.rmtree(dst)

    # Symlink-heavy copy: keep data/ as symlinks (large), real-copy meta/
    dst.mkdir(parents=True)
    for child in src.iterdir():
        if child.name == "meta":
            shutil.copytree(child, dst / "meta")
        else:
            (dst / child.name).symlink_to(child)

    # Walk meta/episodes/chunk-XXX/file-XXX.parquet and trim each shard
    ep_shards = sorted((dst / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not ep_shards:
        print(f"WARN: no meta/episodes shards under {dst}", file=sys.stderr)

    total_kept = 0
    rows_remaining = args.n_episodes
    train_pool_cutoff = -args.eval_holdout if args.eval_holdout > 0 else None
    for shard in ep_shards:
        df = pq.read_table(shard).to_pandas()
        if train_pool_cutoff is not None:
            df = df.iloc[:train_pool_cutoff]
        kept = df.head(rows_remaining)
        rows_remaining -= len(kept)
        if kept.empty:
            shard.unlink()
            continue
        pq.write_table(pa.Table.from_pandas(kept), shard)
        total_kept += len(kept)
        if rows_remaining <= 0:
            break

    # Trim later shards we didn't reach (delete them)
    for shard in ep_shards:
        if shard.exists() and not pq.read_table(shard).num_rows:
            shard.unlink()

    info = json.loads((dst / "meta" / "info.json").read_text())
    info["total_episodes"] = total_kept
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"[subsample] {src.name} -> {dst.name} kept {total_kept} eps "
          f"(eval_holdout={args.eval_holdout})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
