"""
test_programs_parse.py

Validates that each program.md file in programs/ has all required metadata keys.

Required keys (checked via simple text scan + YAML block extraction):
  - goal          (Research Goal section or yaml key)
  - metric.name
  - metric.direction
  - metric.regex  (regex key in ## Metric section)
  - train_cmd_template equivalent (## Training Script section with path: and entry_args:)
  - operators_priority (## Operators Priority section or yaml key)
  - seconds_per_experiment
  - max_experiments
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Allow running from package root without installation.
PACKAGE_ROOT = Path(__file__).parent.parent
PROGRAMS_DIR = PACKAGE_ROOT / "programs"

PROGRAM_FILES = [
    PROGRAMS_DIR / "lerobot-policy.md",
    PROGRAMS_DIR / "dreamerv3.md",
    PROGRAMS_DIR / "leworldmodel.md",
]


def _read(path: Path) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_section(text: str, heading: str) -> bool:
    """Return True if a markdown heading matches (case-insensitive)."""
    pattern = re.compile(rf"^##\s+{re.escape(heading)}", re.MULTILINE | re.IGNORECASE)
    return bool(pattern.search(text))


def _extract_key_value(text: str, key: str) -> str | None:
    """
    Find 'key: value' in the raw text (handles leading whitespace).
    Returns stripped value string or None.
    """
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Per-file checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_program_file_exists(program_path: Path) -> None:
    assert program_path.exists(), f"Program file not found: {program_path}"


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_has_research_goal(program_path: Path) -> None:
    text = _read(program_path)
    assert _has_section(text, "Research Goal"), (
        f"{program_path.name}: missing '## Research Goal' section"
    )
    # Ensure it has non-trivial content (at least one non-blank line after heading)
    match = re.search(
        r"^##\s+Research Goal\s*\n(.*?)(?=^##|\Z)",
        text,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    assert match, f"{program_path.name}: '## Research Goal' section has no content"
    content = match.group(1).strip()
    assert len(content) > 10, (
        f"{program_path.name}: Research Goal content too short: {content!r}"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_has_metric_section(program_path: Path) -> None:
    text = _read(program_path)
    assert _has_section(text, "Metric"), (
        f"{program_path.name}: missing '## Metric' section"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_metric_name_key(program_path: Path) -> None:
    text = _read(program_path)
    value = _extract_key_value(text, "name")
    assert value and len(value) > 0, (
        f"{program_path.name}: 'name:' key missing or empty in ## Metric section"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_metric_direction_key(program_path: Path) -> None:
    text = _read(program_path)
    value = _extract_key_value(text, "direction")
    assert value in ("minimize", "maximize"), (
        f"{program_path.name}: 'direction:' must be 'minimize' or 'maximize', got {value!r}"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_metric_regex_key(program_path: Path) -> None:
    text = _read(program_path)
    value = _extract_key_value(text, "regex")
    assert value and len(value) > 0, (
        f"{program_path.name}: 'regex:' key missing or empty"
    )
    # Sanity-check: regex must compile.
    try:
        # Strip surrounding quotes if present.
        pattern = value.strip("'\"")
        re.compile(pattern)
    except re.error as exc:
        pytest.fail(f"{program_path.name}: 'regex:' value does not compile: {exc}")


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_has_training_script_section(program_path: Path) -> None:
    text = _read(program_path)
    assert _has_section(text, "Training Script"), (
        f"{program_path.name}: missing '## Training Script' section"
    )
    # Must have a path: key and an entry_args: key.
    path_val = _extract_key_value(text, "path")
    assert path_val, f"{program_path.name}: 'path:' key missing in Training Script"
    entry_val = _extract_key_value(text, "entry_args")
    assert entry_val is not None, (
        f"{program_path.name}: 'entry_args:' key missing in Training Script"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_has_operators_priority_section(program_path: Path) -> None:
    text = _read(program_path)
    assert _has_section(text, "Operators Priority"), (
        f"{program_path.name}: missing '## Operators Priority' section"
    )
    # Must list at least one operator.
    match = re.search(
        r"^##\s+Operators Priority\s*\n(.*?)(?=^##|\Z)",
        text,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    assert match, f"{program_path.name}: '## Operators Priority' has no content"
    content = match.group(1)
    operators = [
        ln.strip()
        for ln in content.splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert len(operators) >= 1, (
        f"{program_path.name}: Operators Priority lists no operators"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_seconds_per_experiment(program_path: Path) -> None:
    text = _read(program_path)
    value = _extract_key_value(text, "seconds_per_experiment")
    assert value is not None, (
        f"{program_path.name}: 'seconds_per_experiment:' key missing"
    )
    # Must be a positive integer (possibly with inline comment).
    numeric = value.split("#")[0].strip()
    assert numeric.isdigit() and int(numeric) > 0, (
        f"{program_path.name}: 'seconds_per_experiment' must be a positive integer, got {numeric!r}"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_max_experiments(program_path: Path) -> None:
    text = _read(program_path)
    value = _extract_key_value(text, "max_experiments")
    assert value is not None, f"{program_path.name}: 'max_experiments:' key missing"
    numeric = value.split("#")[0].strip()
    assert numeric.isdigit() and int(numeric) > 0, (
        f"{program_path.name}: 'max_experiments' must be a positive integer, got {numeric!r}"
    )


@pytest.mark.parametrize("program_path", PROGRAM_FILES, ids=lambda p: p.name)
def test_regex_matches_expected_metric(program_path: Path) -> None:
    """
    Smoke-test: the regex in the program.md must match a sample stdout line
    of the form '<metric_name>=<float>'.
    """
    text = _read(program_path)
    metric_name = _extract_key_value(text, "name")
    regex_raw = _extract_key_value(text, "regex")
    assert metric_name and regex_raw

    sample_line = f"{metric_name}=0.1234"
    pattern = re.compile(regex_raw.strip("'\""))
    m = pattern.search(sample_line)
    assert m, (
        f"{program_path.name}: regex {regex_raw!r} does not match sample line {sample_line!r}"
    )
    assert m.group(1) == "0.1234", (
        f"{program_path.name}: regex capture group did not extract '0.1234', got {m.group(1)!r}"
    )
