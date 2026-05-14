# Autoresearch Programs — Index

Per-architecture autoresearch programs **tuned to this stack**
(lerobot 0.5+, sheeprl 0.5.8.dev, Isaac Sim 6.0, RTX 3080 10 GB,
SO-101 6-DOF arm). The upstream `autoresearch-ml-proposer-worker` is
domain-generic — it knows about Adam/SGD/dropout but nothing about
`--policy.push_to_hub`, sheeprl Hydra paths, our metric proxies, or
our VRAM ceilings. These programs encode the stack-specific reality.

**Always read first:** [`_domain_knowledge.md`](_domain_knowledge.md) —
single reference card the proposer should consult when reasoning about
hyperparameter ranges, CLI flag shape, and known failure modes.

---

## Selection Guide

| Goal | Program | Per-exp budget | Total wall-clock |
|------|---------|----------------|------------------|
| Strongest from-scratch policy | [`lerobot-policy-diffusion.md`](lerobot-policy-diffusion.md) | 30 min | ~5 h (10 exp) |
| Pretrained policy, fast convergence | [`lerobot-policy-smolvla.md`](lerobot-policy-smolvla.md) | 1 h | ~8 h (8 exp) |
| Fastest inference (chunked attention) | [`lerobot-policy-act.md`](lerobot-policy-act.md) | 45 min | ~6 h (8 exp) |
| World model with full Dreamer RSSM | [`wm-dreamerv3.md`](wm-dreamerv3.md) | 1 h | ~8 h (8 exp) |
| World-model smoke / cheap baseline | [`wm-lewm.md`](wm-lewm.md) | 10 min | ~1.5 h (8 exp) |
| Tight smoke run (3 trials, 8 min/exp) | [`lerobot-policy-short.md`](lerobot-policy-short.md) | 8 min | ~30 min |

---

## How to Invoke

### Via `/autoresearch` slash command (preferred)

```bash
cd ~/tools/claude_code
/autoresearch ~/workspaces/lerobot-isaac-training/programs/<program>.md --type ml_model
```

### Via workspace wrapper

```bash
bash scripts/run_autoresearch.sh --program diffusion
bash scripts/run_autoresearch.sh --program dreamerv3
bash scripts/run_autoresearch.sh --program lewm --max-experiments 3
```

### Via deterministic bash fallback (no LLM proposer)

```bash
bash scripts/_run_autoresearch_smoke.sh
```

Produces the same on-disk artefact shape (history.jsonl / best.json /
plateau.json / program.json) under `.agent-state/<session>/autoresearch/<slug>/`
so the dashboard's Autoresearch tab picks it up either way.

---

## Program Schema (extensions on top of upstream)

The upstream `autoresearch` skill defines the core schema. These
workspace programs add domain-specific fields:

| Field | Purpose |
|-------|---------|
| `domain: lerobot_isaac`         | Tells the proposer to consult `_domain_knowledge.md`. |
| `domain_knowledge: <path>`      | Explicit pointer to the domain reference card. |
| `stack: <one-line>`             | Quick stack version banner for the proposer's prompt. |
| `env: train-policy / train-dreamer / train-lewm` | Which pixi env runs the subprocess. |
| `python: <path>`                | Explicit interpreter path (avoids PATH order surprises). |
| `remainder: <args>`             | Pre-baked `--` remainder forwarded to the adapter (e.g. `env.capture_video=False`). |
| `oom_recovery: halve_batch_size_once` | Standard recovery contract. |
| `vram_ceiling_gb: 10`           | Hard cap the proposer must not exceed. |
| `batch_size_max: <N>`           | Per-arch hard cap (derived from VRAM ceiling). |

---

## Metric Direction Map

| Arch              | Metric           | Direction |
|-------------------|------------------|-----------|
| smolvla / act / diffusion | `pc_success` | maximize |
| dreamerv3         | `recon_loss`     | minimize |
| le_world_model    | `pred_loss`      | minimize |

`pc_success` for policies is the **open-loop action-MSE proxy** computed
by `scripts/_open_loop_eval.py` (`1 / (1 + mse)`). SO-101 has no
registered gym env so closed-loop rollout success is not measurable
in-workspace today. The proxy still tracks policy quality across runs.

---

## Where Histories Land

Each run writes to:
```
.agent-state/<session>/autoresearch/<slug>/
  program.json     # snapshot of the parsed program.md
  history.jsonl    # one JSON record per trial
  best.json        # current best config + metric
  plateau.json     # consecutive_non_improvements counter
  trial_<N>.log    # raw stdout per trial (for debugging)
```

The dashboard's `load_autoresearch` loader reads these directly —
no extra config needed. See
[`docs/pipeline-overview.md §Stage H`](../docs/pipeline-overview.md#stage-h--autoresearch-loop).
