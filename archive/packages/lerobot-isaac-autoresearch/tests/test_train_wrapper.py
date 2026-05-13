"""
test_train_wrapper.py

Argparse smoke tests for train_wrapper.py.

Tests:
  - All valid --target_arch choices are accepted without error.
  - Required argument --target_arch raises SystemExit when missing.
  - --dry_run flag is parsed correctly.
  - --batch_size, --steps, --dataset, --output_dir are passed through.
  - Unknown extra args are collected rather than causing parse failure.
  - _last_metric_line helper finds the correct token.
  - _detect_oom helper recognises common OOM patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the package src tree is importable without a full install.
PACKAGE_SRC = Path(__file__).parent.parent / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from lerobot_isaac_autoresearch.train_wrapper import (  # noqa: E402  — sys.path inserted above
    _detect_oom,
    _last_metric_line,
    parse_args,
)

# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------

VALID_ARCHES = ["smolvla", "act", "diffusion", "dreamerv3", "le_world_model"]


@pytest.mark.parametrize("arch", VALID_ARCHES)
def test_valid_target_arch(arch: str) -> None:
    args, _ = parse_args(["--target_arch", arch])
    assert args.target_arch == arch


def test_missing_target_arch_raises() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args([])
    assert exc_info.value.code != 0


def test_invalid_target_arch_raises() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--target_arch", "not_a_real_arch"])


def test_dry_run_default_false() -> None:
    args, _ = parse_args(["--target_arch", "smolvla"])
    assert args.dry_run is False


def test_dry_run_flag_sets_true() -> None:
    args, _ = parse_args(["--target_arch", "smolvla", "--dry_run"])
    assert args.dry_run is True


def test_steps_parsed_as_int() -> None:
    args, _ = parse_args(["--target_arch", "smolvla", "--steps", "20000"])
    assert args.steps == 20000


def test_batch_size_parsed_as_int() -> None:
    args, _ = parse_args(["--target_arch", "smolvla", "--batch_size", "8"])
    assert args.batch_size == 8


def test_dataset_passthrough() -> None:
    args, _ = parse_args(["--target_arch", "smolvla", "--dataset", "/data/so101"])
    assert args.dataset == "/data/so101"


def test_output_dir_passthrough() -> None:
    args, _ = parse_args(["--target_arch", "smolvla", "--output_dir", "/tmp/run_001"])
    assert args.output_dir == "/tmp/run_001"


def test_extra_unknown_args_collected() -> None:
    """Unknown args should not raise — they are forwarded verbatim to the adapter."""
    args, extra = parse_args(
        ["--target_arch", "smolvla", "--some_unknown_flag", "value"]
    )
    assert "--some_unknown_flag" in extra or "--some_unknown_flag" in (args.extra or [])


# ---------------------------------------------------------------------------
# _last_metric_line tests
# ---------------------------------------------------------------------------


def test_last_metric_line_basic() -> None:
    lines = [
        "step 100: loss=1.23",
        "pc_success=0.45",
        "some other output",
        "pc_success=0.67",
    ]
    result = _last_metric_line(lines, "pc_success")
    assert result == "pc_success=0.67"


def test_last_metric_line_returns_none_when_absent() -> None:
    lines = ["no metric here", "just logs"]
    assert _last_metric_line(lines, "pc_success") is None


def test_last_metric_line_prefers_last_occurrence() -> None:
    lines = [
        "recon_loss=0.99",
        "recon_loss=0.50",
        "recon_loss=0.03",
    ]
    result = _last_metric_line(lines, "recon_loss")
    assert result == "recon_loss=0.03"


def test_last_metric_line_embedded_in_longer_line() -> None:
    """Token extraction: should isolate 'pred_loss=0.12' from surrounding text."""
    lines = ["Epoch 5: pred_loss=0.12 other_metric=1.0"]
    result = _last_metric_line(lines, "pred_loss")
    assert result == "pred_loss=0.12"


# ---------------------------------------------------------------------------
# _detect_oom tests
# ---------------------------------------------------------------------------


def test_detect_oom_cuda_out_of_memory() -> None:
    lines = [
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "(malloc at /opt/conda/...)",
    ]
    assert _detect_oom(lines) is True


def test_detect_oom_negative() -> None:
    lines = ["Training completed successfully.", "pc_success=0.85"]
    assert _detect_oom(lines) is False


def test_detect_oom_case_insensitive() -> None:
    lines = ["runtimeerror: cuda out of memory"]
    assert _detect_oom(lines) is True
