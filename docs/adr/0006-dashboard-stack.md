# ADR-0006: Streamlit + Plotly + Jinja2 Dashboard Stack

**Status:** Accepted
**Date:** 2026-05-08
**Deciders:** Project team

---

## Context

As the `lerobot-isaac-training` pipeline spans 8 packages and produces artefacts
across multiple directories (`outputs/`, `datasets/`, `.agent-state/`), visibility
into runs requires a unified metrics surface. Without it, debugging training failures
means grepping log files across three subsystems simultaneously.

The following requirements were locked by the user before implementation began:

1. **Streamlit + Plotly** — UI stack is non-negotiable; no FastAPI, no React.
2. **Local files only** — no external metric services (W&B, MLflow, InfluxDB).
3. **Dual-render** — same figures in live UI and static offline HTML; no separate
   template path.
4. **CPU-only** — dashboard runs in the base `default` pixi environment with the
   `dashboard` feature added; no GPU deps, no Isaac Lab.
5. **Snapshot + compare** — save workspace state at any point; reload and compare
   two or more snapshots without rerunning training.

The pipeline runs on a workstation with no internet access during training
(isolated lab network), which ruled out any cloud-first observability tool.

---

## Decision

Use **Streamlit + Plotly + Jinja2** with a **dual-render Tab contract** and
**Parquet + JSON snapshot format**.

Key sub-decisions:

### 1. Dual-render via `Tab.render(container=None)`

Every `Tab.render(ctx, *, container=None)` returns `list[go.Figure]` unconditionally.
When `container` is a Streamlit widget it also calls `container.plotly_chart(fig)`.
When `container=None` (static exporter, compare renderer) only the list is produced.

This means:
- Zero code duplication between live and static paths
- Static exporter calls the same render methods as the live app
- Compare renderer calls the same render methods and merges/overlays figures

### 2. Local-files-only metric source

All loaders read from:
- `datasets/` — LeRobot Parquet
- `outputs/` — checkpoints, eval JSON, curriculum JSON
- `.agent-state/` — autoresearch JSONL, event JSONL

No loader makes network calls. This is enforced by the `LoaderResult` contract
(no `requests`, `boto3`, or similar in any loader module).

### 3. Snapshot format: Parquet + JSON under `outputs/snapshots/<id>/`

DataFrames are persisted as Parquet (lossless dtypes via pyarrow). Dict members
(program configs, plateau state) are persisted as JSON. A `meta.json` header
records the schema version, git SHA, timestamp, and label.

Schema version `1` is the initial version. `load_snapshot` raises `ValueError`
when the on-disk version exceeds the installed version, giving a clear upgrade prompt.

### 4. Compare modes

Two compare modes are provided:
- **2-way side-by-side** — each tab split into two columns; delta KPI strip above
- **N-way overlay** — traces from all snapshots overlaid on shared axes

Both modes work in the live Streamlit UI (via the Mode radio) and in static HTML
via `export_compare_report`.

### 5. Static report: inline plotly.js default

By default `export_report` and `export_compare_report` embed `plotly.min.js` inline,
producing a self-contained ~5 MB HTML file that opens in any browser without internet.
The `--cdn` flag reduces the file to ~50 KB but requires an online viewer.

---

## Rationale

The dual-render contract is the core enabler: it lets the same tab code serve
three surfaces (live UI, static report, compare report) without any template
duplication. The only cost is that every `render` must return a figure list even
when running live — a minor discipline that is enforced by the `Tab` base class
raising `NotImplementedError`.

Parquet was chosen for snapshot persistence over JSON/CSV because:
- Preserves dtype fidelity (nullable Int64, float64, datetime64[ns, UTC])
- Handles large DataFrames (10k+ rows) without text-encoding overhead
- Already a hard dependency (pyarrow) for the loaders that read pipeline Parquet files

---

## Consequences

**Positive:**
- Fast Python-only development — no JS build step, no separate backend process
- Dual-render code reuse eliminates the most common dashboard bug class
  (live and static out of sync)
- Offline-safe HTML reports: useful on air-gapped training machines
- Snapshot + compare closes the comparison loop without W&B or MLflow

**Negative:**
- Inline plotly.js makes reports ~5 MB by default; use `--cdn` for large compare reports
- Streamlit's session state model requires careful `st.cache_data` keying on file mtimes
  to avoid stale data after training updates
- N-way overlay is only well-defined for time-series tabs; table-heavy tabs (Pipeline
  Health, Curriculum) produce empty overlay sections in static mode — acceptable trade-off
- Streamlit's single-threaded reruns mean the dashboard freezes for ~1–2 s during
  loader execution on first load of a large workspace

---

## Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| Grafana + InfluxDB | Requires a running InfluxDB instance; poor support for lineage tables and episode-level views; overkill for a local workstation |
| Dash (Plotly Dash) | Comparable capability to Streamlit but requires more boilerplate for layout; Streamlit's reactive model is faster for iteration |
| MLflow Tracking | Experiment tracker, not a pipeline visibility tool; would require wrapping every log write with `mlflow.log_metric`; doesn't cover autoresearch JSONL or event logs |
| Weights & Biases | External cloud service; out of scope for an air-gapped lab network; free tier rate limits |
| Panel (HoloViews) | Less community adoption; fewer Plotly-native integrations; harder onboarding |
| FastAPI + React | Production-grade but overkill; 3–5× more code for the same functionality; JS build step adds friction |
