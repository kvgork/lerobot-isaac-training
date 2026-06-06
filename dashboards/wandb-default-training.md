# W&B Default Training Dashboard (Bundle E Scaffold)

Status: **scaffold** — verify against real W&B run before promoting to canonical.

## Recommended Panels

### Row 1 — Training loss
- `train/loss` (line, smoothed window=100)
- `train/loss_action` (line)
- `train/loss_value` (line) — only if policy has value head
- `train/grad_norm` (line, log-scale)

### Row 2 — Validation / eval
- `eval/pc_success` (line, ladder over checkpoints)
- `eval/intervention_rate` (line)
- `eval/mean_episode_length` (line)
- `eval/per_stratum_easy_sr` (bar)
- `eval/per_stratum_hard_sr` (bar)

### Row 3 — System
- `system/gpu.0.memoryAllocated` (line, % of RTX 3080 10 GB)
- `system/gpu.0.gpu` (line, util %)
- `system/cpu` (line)
- `train/throughput_samples_per_sec` (line)

### Row 4 — Drift (eval-time, optional)
- `drift/visual_kl` (line)
- `drift/state_kl` (line)

## Variables (per-run filter)

- `$policy` — `smolvla | act | dreamerv3 | lewm | diffusion`
- `$dataset_hash` — short hash of LeRobotDataset
- `$task` — `pick_and_place | insertion | ...`

## Alerting Rules (Bundle E follow-up)

See `dashboards/wandb-alerts.yaml`.

## How to Apply

```bash
# Create the W&B report from this template (manual until automated):
1. wandb login
2. New report → "Default Training (lerobot-isaac-training)"
3. Add panels per "Recommended Panels" above
4. Save to org workspace as canonical default
```

## Status

- [x] Logger scaffold landed (`lerobot_isaac_adapters.loggers.wandb_logger`)
- [ ] Real training run produces ≥1 logged metric set
- [ ] Dashboard exported as JSON + committed
- [ ] Alerting rules validated
