"""
train_wrapper.py

Thin shim invoked by autoresearch-ml-executor-worker.  Reads program.md,
sets env vars, invokes lerobot_isaac_adapters.train as a subprocess, captures
stdout, and guarantees a regex-friendly metric line is the final stdout line.

Usage (autoresearch-ml-executor-worker sets script_path to this file):
    python -m lerobot_isaac_autoresearch.train_wrapper \\
        --target_arch smolvla \\
        --dataset /path/to/dataset \\
        --output_dir /tmp/run_001 \\
        --steps 20000

OOM recovery: if the subprocess exits with a RuntimeError containing
"CUDA out of memory", batch_size is halved and the run retries once.
A hard timeout of TRAIN_TIMEOUT_SECONDS applies in addition to the
autoresearch executor's own budget_seconds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Optional

# Hard ceiling (seconds) independent of autoresearch budget.
# Set to 4 hours; executor_worker enforces its own budget_seconds on top.
TRAIN_TIMEOUT_SECONDS = int(os.environ.get("LEROBOT_TRAIN_TIMEOUT", str(4 * 3600)))

# Metric emitted by wm targets when main metric is unavailable.
FALLBACK_METRIC_LINE = "pc_success=0.0"


def _build_cmd(args: argparse.Namespace) -> list[str]:
    """Build the subprocess command from parsed args."""
    cmd = [
        sys.executable,
        "-m",
        "lerobot_isaac_adapters.train",
        "--target_arch",
        args.target_arch,
    ]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.output_dir:
        cmd += ["--output_dir", args.output_dir]
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    if args.config:
        cmd += ["--config", args.config]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.dry_run:
        cmd += ["--dry_run"]
    # Forward any extra unknown args verbatim.
    if args.extra:
        cmd += args.extra
    return cmd


def _run_subprocess(cmd: list[str], timeout: int) -> tuple[int, list[str]]:
    """
    Run cmd as a subprocess with a hard timeout.

    Returns (returncode, stdout_lines).
    stdout is captured and also mirrored to sys.stdout in real time.
    """
    stdout_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        start = time.monotonic()
        assert proc.stdout is not None
        for line in proc.stdout:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                proc.kill()
                sys.stdout.write(
                    f"[train_wrapper] TIMEOUT after {elapsed:.0f}s — killed\n"
                )
                sys.stdout.flush()
                return -1, stdout_lines
            sys.stdout.write(line)
            sys.stdout.flush()
            stdout_lines.append(line.rstrip())
        proc.wait()
        return proc.returncode, stdout_lines
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(f"[train_wrapper] subprocess error: {exc}\n")
        sys.stdout.flush()
        return -1, stdout_lines


def _detect_oom(stdout_lines: list[str]) -> bool:
    """Return True if any stdout line indicates a CUDA OOM error."""
    oom_signals = [
        "cuda out of memory",
        "out of memory",
        "cudaoutofmemoryerror",
        "runtimeerror: cuda",
    ]
    combined = " ".join(stdout_lines).lower()
    return any(sig in combined for sig in oom_signals)


def _last_metric_line(stdout_lines: list[str], metric_name: str) -> Optional[str]:
    """Return the last line that looks like <metric_name>=<float>, or None."""
    pattern = f"{metric_name}="
    for line in reversed(stdout_lines):
        if pattern in line:
            # Isolate just the 'name=value' token.
            for token in line.split():
                if token.startswith(pattern):
                    return token
    return None


def run(args: argparse.Namespace) -> int:
    """
    Execute the training run with optional OOM retry.

    Returns the process exit code (0 = success).
    """
    batch_size = args.batch_size
    retry_count = 0
    max_retries = 1

    while True:
        # Apply current batch_size to args for cmd construction.
        args.batch_size = batch_size
        cmd = _build_cmd(args)
        sys.stdout.write(f"[train_wrapper] running: {' '.join(cmd)}\n")
        sys.stdout.flush()

        returncode, stdout_lines = _run_subprocess(cmd, TRAIN_TIMEOUT_SECONDS)

        if returncode != 0 and _detect_oom(stdout_lines) and retry_count < max_retries:
            old_bs = batch_size if batch_size else 8
            new_bs = max(1, old_bs // 2)
            sys.stdout.write(
                f"[train_wrapper] CUDA OOM detected — halving batch_size "
                f"from {old_bs} to {new_bs} and retrying\n"
            )
            sys.stdout.flush()
            batch_size = new_bs
            retry_count += 1
            continue

        # Ensure the final stdout line is the metric in regex-parseable format.
        # Determine expected metric name based on target_arch.
        metric_map = {
            "smolvla": "pc_success",
            "act": "pc_success",
            "diffusion": "pc_success",
            "dreamerv3": "recon_loss",
            "le_world_model": "pred_loss",
        }
        metric_name = metric_map.get(args.target_arch, "pc_success")
        metric_line = _last_metric_line(stdout_lines, metric_name)

        if metric_line:
            # Always emit as the final line so executor regex matches it last.
            sys.stdout.write(f"{metric_line}\n")
            sys.stdout.flush()
        else:
            # Emit a sentinel so the executor does not crash on missing metric.
            fallback = f"{metric_name}=0.0"
            sys.stdout.write(
                f"[train_wrapper] WARNING: no {metric_name} line found in stdout; "
                f"emitting sentinel {fallback}\n"
            )
            sys.stdout.write(f"{fallback}\n")
            sys.stdout.flush()

        return returncode


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Autoresearch train wrapper: forwards args to lerobot_isaac_adapters.train "
            "and guarantees a metric line on stdout."
        )
    )
    parser.add_argument(
        "--target_arch",
        required=True,
        choices=["smolvla", "act", "diffusion", "dreamerv3", "le_world_model"],
        help="Which training target to invoke.",
    )
    parser.add_argument("--dataset", default=None, help="Path to dataset.")
    parser.add_argument("--output_dir", default=None, help="Output directory.")
    parser.add_argument("--steps", type=int, default=None, help="Training steps.")
    parser.add_argument("--config", default=None, help="Config YAML path.")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (OOM recovery may halve this)."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Pass --dry_run to the adapter; no training executed.",
    )
    known, extra = parser.parse_known_args(argv)
    known.extra = extra
    return known, extra


def main() -> None:
    args, _ = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
