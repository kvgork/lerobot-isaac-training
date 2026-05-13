"""
test_e2e_dry_run.py

End-to-end dry-run smoke test for the autoresearch ML loop wiring.

Invokes ``lerobot_isaac_autoresearch.train_wrapper`` as a real subprocess, which
in turn spawns ``lerobot_isaac_adapters.train --dry_run`` as a nested subprocess.
Verifies the full chain produces a regex-parseable metric line on stdout — the
contract that ``autoresearch-ml-executor-worker`` relies on.

This test is the single contract enforcement for the "wrapper -> train -> metric"
pipeline.  If it breaks, autoresearch loops will silently fail in production.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent

# Per-target_arch metric contract: arch -> expected last-line metric name.
ARCH_METRIC = {
    "smolvla": "pc_success",
    "act": "pc_success",
    "diffusion": "pc_success",
    "dreamerv3": "recon_loss",
    "le_world_model": "pred_loss",
}

# Regex used by autoresearch-ml-executor-worker.
EXECUTOR_REGEX = re.compile(r"(\w+)[=:\s]+([0-9.eE+\-]+)")


def _run_wrapper(arch: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run ``train_wrapper --dry_run`` and capture stdout."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot_isaac_autoresearch.train_wrapper",
            "--target_arch",
            arch,
            "--dataset",
            str(tmp_path / "fake_dataset"),
            "--output_dir",
            str(tmp_path / "run_out"),
            "--steps",
            "100",
            "--batch_size",
            "4",
            "--dry_run",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(WORKSPACE_ROOT),
    )


@pytest.mark.requires_workspace_root
@pytest.mark.parametrize("arch", list(ARCH_METRIC))
def test_wrapper_dry_run_emits_final_metric_line(arch: str, tmp_path: Path) -> None:
    """train_wrapper subprocess must terminate with a regex-parseable metric.

    Full chain:
      pytest -> train_wrapper.main() -> train.main() -> backend.run() with --dry_run
      (no real backend subprocess; everything terminates cleanly)
    """
    proc = _run_wrapper(arch, tmp_path)

    assert proc.returncode == 0, (
        f"train_wrapper failed for {arch} (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # Final stdout line must match the executor regex.
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert stdout_lines, "train_wrapper produced no stdout"

    last_line = stdout_lines[-1]
    match = EXECUTOR_REGEX.search(last_line)
    assert match is not None, (
        f"Final stdout line {last_line!r} does not match executor regex "
        f"{EXECUTOR_REGEX.pattern!r}"
    )

    metric_name, metric_value = match.group(1), match.group(2)
    expected_metric = ARCH_METRIC[arch]
    assert metric_name == expected_metric, (
        f"Wrong metric name on final line: got {metric_name!r}, "
        f"expected {expected_metric!r}.  Full stdout:\n{proc.stdout}"
    )

    # Sentinel value 0.0 is acceptable in dry-run (no real metric).
    float(metric_value)  # raises ValueError on bad float


@pytest.mark.requires_workspace_root
def test_wrapper_dry_run_does_not_invoke_heavy_backend(tmp_path: Path) -> None:
    """Confirm that --dry_run never reaches the heavy training subprocess.

    If sheeprl / lerobot-train were called for real, the test would either fail
    (FileNotFoundError → exit 127) or hang downloading deps.
    """
    proc = _run_wrapper("dreamerv3", tmp_path)
    assert proc.returncode == 0
    # The dry-run banner from the adapter must be present.
    assert "[wm_dreamerv3] Step 2 — train:" in proc.stdout, (
        f"Expected dry-run banner missing from stdout:\n{proc.stdout}"
    )
    # The actual sheeprl invocation must NOT have happened.
    assert "ERROR: 'sheeprl' not found" not in proc.stdout
    assert "Connection refused" not in proc.stdout
