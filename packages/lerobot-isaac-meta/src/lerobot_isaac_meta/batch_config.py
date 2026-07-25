"""batch_config.py — YAML schema for sequential multi-run training batches.

A *batch* is an ordered list of training runs that share a dataset and a
post-batch comparison report.  Used by ``lerobot-isaac-batch`` to train e.g.
SmolVLA and LeWorldModel on the same dataset and compare the result in the
dashboard automatically.

Example
-------
::

    batch_id: smolvla-vs-lewm
    dataset: datasets/so101/pick-cube
    output_root: outputs/runs
    on_failure: continue
    compare:
      enabled: true
      mode: nway
    runs:
      - id: smolvla
        target_arch: smolvla
        config: packages/lerobot-isaac-configs/configs/policy_smolvla.yaml
        steps: 50000
        label: SmolVLA baseline
      - id: lewm
        target_arch: le_world_model
        config: packages/lerobot-isaac-configs/configs/wm_leworldmodel.yaml
        steps: 50000

Validation rules
----------------
* ``runs`` must be non-empty.
* Every ``run.id`` must be unique within a batch.
* ``run.target_arch`` must be one of the supported archs.
* ``compare.mode`` ∈ ``{"2way", "nway"}``; ``2way`` requires exactly two runs.
* ``on_failure`` ∈ ``{"continue", "abort"}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Mirrors lerobot_isaac_adapters.train._ALL_ARCHS but without importing it
# (keeps batch_config importable without the adapters package on hand).
# MANUAL SYNC REQUIRED: keep in lockstep with the adapter's _ALL_ARCHS
# (_POLICY_ARCHS + _WM_POLICY_ARCHS + _WM_ARCHS). The vla_jepa/fastwam/lingbot_va
# world-model policies were added in lerobot 0.6.0.
_VALID_ARCHS = (
    "smolvla", "act", "diffusion",         # plain policies
    "vla_jepa", "fastwam", "lingbot_va",   # lerobot 0.6.0 world-model policies
    "dreamerv3", "le_world_model",         # predictive world-model backends
)
_VALID_FAILURE_POLICIES = ("continue", "abort")
_VALID_COMPARE_MODES = ("2way", "nway")


class BatchConfigError(ValueError):
    """Raised when a batch YAML fails schema validation."""


@dataclass
class RunSpec:
    """Single training run inside a batch.

    Any field left as ``None`` falls back to the corresponding ``train.py``
    default.  ``extra_args`` is appended verbatim after a ``--`` separator and
    forwarded to the backend.
    """

    id: str
    target_arch: str
    config: str | None = None
    dataset: str | None = None
    steps: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    seed: int | None = None
    output_dir: str | None = None
    label: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise BatchConfigError("run.id must be a non-empty string")
        if self.target_arch not in _VALID_ARCHS:
            raise BatchConfigError(
                f"run {self.id!r}: target_arch={self.target_arch!r} not in "
                f"{_VALID_ARCHS}"
            )


@dataclass
class CompareSpec:
    """Post-batch comparison report configuration."""

    enabled: bool = True
    mode: str = "nway"
    output_dir: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_COMPARE_MODES:
            raise BatchConfigError(
                f"compare.mode={self.mode!r} not in {_VALID_COMPARE_MODES}"
            )


@dataclass
class BatchConfig:
    """Top-level batch description loaded from YAML."""

    batch_id: str
    dataset: str
    runs: list[RunSpec]
    output_root: str = "outputs/runs"
    on_failure: str = "continue"
    compare: CompareSpec = field(default_factory=CompareSpec)

    def __post_init__(self) -> None:
        if not self.batch_id or not self.batch_id.strip():
            raise BatchConfigError("batch_id must be a non-empty string")
        if not self.runs:
            raise BatchConfigError("runs must contain at least one entry")
        if self.on_failure not in _VALID_FAILURE_POLICIES:
            raise BatchConfigError(
                f"on_failure={self.on_failure!r} not in {_VALID_FAILURE_POLICIES}"
            )
        ids = [r.id for r in self.runs]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise BatchConfigError(f"duplicate run ids: {duplicates}")
        if self.compare.enabled and self.compare.mode == "2way" and len(self.runs) != 2:
            raise BatchConfigError(
                f"compare.mode='2way' requires exactly 2 runs, got {len(self.runs)}"
            )

    def resolved_dataset(self, run: RunSpec) -> str:
        """Return the dataset path for ``run`` (per-run override > batch default)."""
        return run.dataset or self.dataset


def load_batch_config(path: str | Path) -> BatchConfig:
    """Read and validate a batch YAML file.

    Parameters
    ----------
    path:
        Filesystem path to the YAML file.

    Returns
    -------
    BatchConfig
        Validated configuration ready to feed into ``run_batch``.

    Raises
    ------
    BatchConfigError
        On any schema violation.
    FileNotFoundError
        When the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Batch config not found: {path}")

    import yaml  # local import keeps module importable in stripped envs

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise BatchConfigError(
            f"Top-level YAML in {path} must be a mapping, got {type(raw).__name__}"
        )

    return _from_mapping(raw)


def _from_mapping(raw: dict[str, Any]) -> BatchConfig:
    runs_raw = raw.get("runs") or []
    if not isinstance(runs_raw, list):
        raise BatchConfigError(f"'runs' must be a list, got {type(runs_raw).__name__}")

    runs = []
    for i, r in enumerate(runs_raw):
        if not isinstance(r, dict):
            raise BatchConfigError(
                f"runs[{i}] must be a mapping, got {type(r).__name__}"
            )
        try:
            runs.append(RunSpec(**r))
        except TypeError as exc:
            raise BatchConfigError(f"runs[{i}]: {exc}") from exc

    compare_raw = raw.get("compare") or {}
    if not isinstance(compare_raw, dict):
        raise BatchConfigError(
            f"'compare' must be a mapping, got {type(compare_raw).__name__}"
        )

    try:
        compare = CompareSpec(**compare_raw)
    except TypeError as exc:
        raise BatchConfigError(f"compare: {exc}") from exc

    dataset = raw.get("dataset")
    if not dataset:
        raise BatchConfigError("'dataset' is required at the batch level")

    cfg_kwargs: dict[str, Any] = {
        "batch_id": raw.get("batch_id", ""),
        "dataset": dataset,
        "runs": runs,
        "output_root": raw.get("output_root", "outputs/runs"),
        "on_failure": raw.get("on_failure", "continue"),
        "compare": compare,
    }
    return BatchConfig(**cfg_kwargs)
