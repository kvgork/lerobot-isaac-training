# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the
`lerobot-isaac-training` workspace. Each ADR captures a significant
architectural decision: its context, the decision made, and the rationale.

ADR format: Context / Decision / Status / Consequences / Alternatives Considered.

---

## Index

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [0001](0001-isaac-lab-over-mujoco.md) | Isaac Lab over MuJoCo | Accepted | GPU parallelism, native DR, and USD ecosystem drove selection of Isaac Lab as sole simulator |
| [0002](0002-pixi-workspace.md) | Pixi Workspace | Accepted | Pixi manages conda+pip hybrid envs; per-package pixi.toml is dormant in monorepo and activates post-spinout |
| [0003](0003-soft-import-discipline.md) | Soft-Import Discipline | Accepted | Heavy deps (isaaclab, lerobot, dreamerv3, lewm) are lazy-imported inside functions so packages are importable with no GPU deps |
| [0004](0004-multi-package-monorepo.md) | Multi-Package Monorepo | Accepted | 6 packages with strict one-way coupling enable parallel CI, independent spinout, and fine-grained dep control |
| [0005](0005-modular-target-arch.md) | Modular Target Architecture | Accepted | Single train.py with --target_arch + MetricExtractor pattern gives autoresearch a stable interface and centralises OOM retry logic |

---

## How to Add an ADR

1. Copy the template below into `docs/adr/NNNN-short-title.md`
2. Fill in all sections
3. Set Status to `Proposed`, then update to `Accepted` or `Rejected` after review
4. Add a row to this index

### Template

```markdown
# ADR-NNNN: Title

**Status:** Proposed | Accepted | Rejected | Superseded by ADR-XXXX
**Date:** YYYY-MM-DD
**Deciders:** ...

## Context
...

## Decision
...

## Rationale
...

## Consequences

**Positive:**
...

**Negative:**
...

## Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| ... | ... |
```
