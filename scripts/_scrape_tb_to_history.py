"""Scrape sheeprl TensorBoard event files → autoresearch history.jsonl backfill.

sheeprl writes per-trial losses to TensorBoard (`events.out.tfevents.*` files
under `logs/runs/dreamer_v3/<env>/<run_name>/version_*/`). The wm-dreamerv3
adapter's metric_extractor pipeline never captures these — it greps stdout for
`recon_loss=<float>` which sheeprl does NOT emit, so every trial ratchets a
`recon_loss=0.0` sentinel.

This script reads the TB events files for a given autoresearch session and
rewrites history.jsonl + best.json with real loss values.

Usage::

    python scripts/_scrape_tb_to_history.py \
        --session wm-bash-20260522-185502 \
        --slug wm-dreamerv3 \
        --metric Loss/observation_loss

Default metric = ``Loss/observation_loss`` (pixel reconstruction MSE — the
canonical "recon loss" for DreamerV3). Other interesting tags:

* ``Loss/world_model_loss`` — total WM loss = recon + reward + continue + KL.
* ``Loss/reward_loss`` — predicted vs target reward.
* ``Loss/state_loss`` — dyn vs repr KL.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def find_run_dirs(workspace: Path, session_id: str) -> list[tuple[int, Path]]:
    """Find per-trial sheeprl run dirs for a session.

    Returns list of (trial_index, run_dir) tuples. trial_index is parsed from
    the run_name pattern `trial_<i>_<session_id>` if present, else inferred
    from mtime ordering.
    """
    runs_root = workspace / "logs" / "runs" / "dreamer_v3" / "custom_hdf5"
    if not runs_root.is_dir():
        return []
    candidates = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        # Prefer the deterministic run_name format from the sweep script.
        m = re.search(r"trial_(\d+)_" + re.escape(session_id), child.name)
        if m:
            candidates.append((int(m.group(1)), child))
            continue
        # Fallback: timestamp-based dirs from the legacy sweep run. Match
        # by mtime against the session_id timestamp window. We don't try
        # to associate them with specific trials here — caller decides.
        candidates.append((-1, child))
    # Sort by trial index, then by mtime as tiebreaker.
    candidates.sort(key=lambda x: (x[0] if x[0] >= 0 else 99999, x[1].stat().st_mtime))
    return candidates


def read_metric_from_tb(run_dir: Path, tag: str) -> tuple[float | None, int | None]:
    """Return (last_value, last_step) for ``tag`` in the TB events under run_dir.

    Returns (None, None) if the tag is absent or no events file is present.
    """
    events_dir = next((p.parent for p in run_dir.glob("**/events.out.tfevents.*")), None)
    if events_dir is None:
        return None, None
    ea = event_accumulator.EventAccumulator(
        str(events_dir),
        size_guidance={event_accumulator.SCALARS: 0},  # load everything
    )
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    if not events:
        return None, None
    last = events[-1]
    return float(last.value), int(last.step)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default=".", help="workspace root")
    ap.add_argument(
        "--session", required=True, help="autoresearch session_id"
    )
    ap.add_argument("--slug", default="wm-dreamerv3", help="autoresearch slug")
    ap.add_argument(
        "--metric",
        default="Loss/observation_loss",
        help="TensorBoard tag to use as the primary metric (lower-is-better)",
    )
    ap.add_argument(
        "--legacy-runs-glob",
        default="",
        help="If the sweep used legacy timestamp-only run names (e.g. the "
        "2026-05-22_18-55-07_dreamer_v3_* dirs), pass a comma-separated list "
        "of dir names in trial order to map them to trial indices.",
    )
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    ar_dir = workspace / ".agent-state" / args.session / "autoresearch" / args.slug
    if not ar_dir.is_dir():
        print(f"ERROR: AR dir not found: {ar_dir}", file=sys.stderr)
        return 2

    history_path = ar_dir / "history.jsonl"
    if not history_path.is_file():
        print(f"ERROR: history.jsonl missing at {history_path}", file=sys.stderr)
        return 2

    # Load existing history to preserve config / status fields.
    rows = []
    with history_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Build trial_index → run_dir map.
    if args.legacy_runs_glob:
        legacy = [s.strip() for s in args.legacy_runs_glob.split(",") if s.strip()]
        runs_root = workspace / "logs" / "runs" / "dreamer_v3" / "custom_hdf5"
        trial_to_run = {i: runs_root / name for i, name in enumerate(legacy)}
    else:
        candidates = find_run_dirs(workspace, args.session)
        trial_to_run = {idx: d for idx, d in candidates if idx >= 0}

    print(f"[scrape] mapped {len(trial_to_run)} trials → run dirs")

    backup = history_path.with_suffix(".jsonl.bak")
    history_path.replace(backup)
    print(f"[scrape] backed up history → {backup.name}")

    new_rows = []
    best_metric = None
    best_row = None
    for row in rows:
        trial_idx = int(row.get("trial", row.get("trial_index", -1)))
        run_dir = trial_to_run.get(trial_idx)
        if run_dir is None:
            print(f"[scrape] trial={trial_idx}: no run dir match — keeping old row")
            new_rows.append(row)
            continue
        value, step = read_metric_from_tb(run_dir, args.metric)
        if value is None:
            print(
                f"[scrape] trial={trial_idx}: tag '{args.metric}' not in {run_dir.name}"
            )
            new_rows.append(row)
            continue
        # Rewrite the row with real metric.
        row = dict(row)
        row["metric_name"] = args.metric.replace("/", "_")
        row["metric_value"] = value
        row["metric_kind"] = f"tb_{args.metric.replace('/', '_')}"
        row["metric_source_run_dir"] = str(run_dir.relative_to(workspace))
        row["metric_source_step"] = step
        new_rows.append(row)
        print(f"[scrape] trial={trial_idx} step={step} {args.metric}={value:.6f}")

        if best_metric is None or value < best_metric:
            best_metric = value
            best_row = row

    with history_path.open("w") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")

    if best_row is not None:
        best_path = ar_dir / "best.json"
        best_obj = {
            "trial": best_row["trial"],
            "metric_value": best_row["metric_value"],
            "metric_kind": best_row["metric_kind"],
            "metric_name": best_row["metric_name"],
            "config": best_row.get("config", {}),
            "source": "scraped_from_tensorboard",
        }
        best_path.write_text(json.dumps(best_obj, indent=2))
        print(f"[scrape] wrote best.json → trial {best_row['trial']} = {best_metric:.6f}")

    plateau_path = ar_dir / "plateau.json"
    if best_metric is not None:
        plateau_path.write_text(
            json.dumps(
                {
                    "consecutive_non_improvements": 0,
                    "plateau_limit": 4,
                    "last_metric": new_rows[-1].get("metric_value"),
                    "best_metric": best_metric,
                    "completed_trials": len(new_rows),
                    "planned_trials": len(new_rows),
                    "status": "scraped",
                },
                indent=2,
            )
        )

    print(f"[scrape] done — rewrote {len(new_rows)} history rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
