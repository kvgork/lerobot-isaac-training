"""GPU monitor daemon — samples nvidia-smi at fixed interval, writes parquet.

Spawned in background by run_full_pipeline.sh during long training stages.
Run target: `outputs/system_metrics/<run_id>/gpu_metrics.parquet`.

Schema (one row per sample):
    ts                  datetime64[ns, UTC]
    elapsed_s           float64   (seconds since monitor start)
    stage               string    (e.g. "policy_train", "wm_train")
    run_id              string    (full-pipeline run dir basename)
    gpu_index           int       (0 by default)
    utilization_pct     float
    memory_used_mb      float
    memory_total_mb     float
    memory_pct          float
    temperature_c       float
    power_draw_w        float

Stops on SIGTERM/SIGINT, flushes parquet on exit.
"""
from __future__ import annotations

import argparse
import csv
import io
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUERY = (
    "timestamp,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw"
)


def _sample(gpu_index: int) -> dict[str, Any] | None:
    """One nvidia-smi snapshot. Returns None on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={QUERY}",
                "--format=csv,noheader,nounits",
                f"-i={gpu_index}",
            ],
            text=True,
            timeout=2,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    reader = csv.reader(io.StringIO(out))
    row = next(reader, None)
    if row is None or len(row) < 6:
        return None
    try:
        util = float(row[1].strip())
        mem_used = float(row[2].strip())
        mem_total = float(row[3].strip())
        temp = float(row[4].strip())
        power = float(row[5].strip())
    except (ValueError, IndexError):
        return None
    return {
        "ts": datetime.now(UTC),
        "gpu_index": gpu_index,
        "utilization_pct": util,
        "memory_used_mb": mem_used,
        "memory_total_mb": mem_total,
        "memory_pct": (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0,
        "temperature_c": temp,
        "power_draw_w": power,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="Output parquet path.")
    ap.add_argument("--stage", default="unknown", help="Training stage label.")
    ap.add_argument("--run-id", default="unknown", help="Run id label.")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--interval-s", type=float, default=2.0)
    args = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    stop = {"flag": False}

    def _flush() -> None:
        if not rows:
            return
        t0 = rows[0]["ts"]
        for r in rows:
            r["elapsed_s"] = (r["ts"] - t0).total_seconds()
            r["stage"] = args.stage
            r["run_id"] = args.run_id
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path)

    def _handle(_sig: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    print(f"[gpu_monitor] sampling gpu={args.gpu_index} -> {out_path}", flush=True)
    last_flush = time.monotonic()
    while not stop["flag"]:
        sample = _sample(args.gpu_index)
        if sample is not None:
            rows.append(sample)
        # Periodic flush every 30s so partial data survives crashes.
        if time.monotonic() - last_flush > 30:
            _flush()
            last_flush = time.monotonic()
        time.sleep(args.interval_s)

    _flush()
    print(f"[gpu_monitor] wrote {len(rows)} samples to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
