# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/setup_env.sh` and pixi activation hook — auto-export
  `LEROBOT_ISAAC_WORKSPACE`, `CLAUDE_CODE_ROOT`, `LEROBOT_CLAUDE_CODE_ROOT` and
  optional `VAULT_ROOT` whenever a pixi shell or task is started.
- `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, top-level badges and external-reader
  README — repository made publish-ready for GitHub.

### Changed
- Replaced hard-coded `/home/koen/...` absolute paths throughout docs and code
  with environment-variable placeholders (`${CLAUDE_CODE_ROOT}`,
  `${LEROBOT_ISAAC_WORKSPACE}`, `${VAULT_ROOT}`). Resolved at runtime via
  `os.path.expandvars`.
- `quality.py` and `bridge_invocation.py` now expand env-var placeholders at
  import time; missing env vars produce a clear runtime error instead of
  silently using a per-developer absolute path.
- Moved the orchestration internal log to `docs/internals/system-improvements.md`.

## [0.1.0] — 2026-05-08

Initial scaffold. Phases 0 – 5 plus dashboard package complete.

### Added

- **Workspace bootstrap** — pixi monorepo with 8 sibling packages, ruff config,
  pre-commit hooks, GitHub Actions matrix CI, conventional commits.
- **`lerobot-isaac-meta`** — umbrella CLI (`lerobot-isaac`) and workspace path
  resolver. Depends on all siblings.
- **`lerobot-isaac-env`** — Isaac Lab `ManagerBasedRLEnv` for SO-101 with
  domain randomization (soft-imported `isaaclab`); pick / pick-and-place /
  insertion (stub) tasks.
- **`lerobot-isaac-adapters`** — single `lerobot-isaac-train` entrypoint
  dispatches to LeRobot policies (SmolVLA / ACT / Diffusion), DreamerV3 and HF
  LeWorldModel via subprocess; canonical `name=value` metric stdout contract;
  Isaac data recorder.
- **`lerobot-isaac-autoresearch`** — `program.md` configs and `train_wrapper.py`
  shim consumed by the `autoresearch-loop-orchestrator` agent for automated
  hyperparameter search.
- **`lerobot-isaac-synthetic`** — Isaac Lab DR replay (priority path),
  parquet writer, dataset merge utilities, MimicGen bridge stub (deferred).
- **`lerobot-isaac-configs`** — six leaf YAML configs per `target_arch`.
- **`lerobot-isaac-recorder`** — RealSense D435 + SO-101 dual-write recorder
  producing LeRobotDataset Parquet and LeWorldModel HDF5 simultaneously.
- **`lerobot-isaac-dashboard`** — Streamlit + Plotly metrics dashboard with 8
  pipeline tabs, dual-render figures, static HTML export, snapshot save/load
  and 2-way / N-way run comparison (281 tests).
- **Documentation** — `ARCHITECTURE.md`, `USAGE.md`, `docs/runbook/`,
  `docs/internals/`, `docs/concepts/`, `docs/research/`, `docs/adr/0001..0006`,
  per-package `CLAUDE.md` orientation files and `docs/api-reference.md`.
- **End-to-end dry-run contract test** —
  `packages/lerobot-isaac-autoresearch/tests/test_e2e_dry_run.py` enforces the
  `train_wrapper → train → metric` chain across all five `target_arch`s.

[Unreleased]: https://github.com/kvgork/lerobot-isaac-training/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kvgork/lerobot-isaac-training/releases/tag/v0.1.0
