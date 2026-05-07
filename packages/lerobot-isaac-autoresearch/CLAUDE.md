# lerobot-isaac-autoresearch — Package Orientation

**Role:** Autoresearch ML loop wiring. Holds `program.md` configs consumed by
`autoresearch-loop-orchestrator` and a thin `train_wrapper.py` shim.

## Package Map

```
packages/lerobot-isaac-autoresearch/
├── programs/
│   ├── lerobot-policy.md    # SmolVLA/ACT/Diffusion — metric: pc_success maximize
│   ├── dreamerv3.md         # DreamerV3 world model — metric: recon_loss minimize
│   └── leworldmodel.md      # HF LeWorldModel      — metric: pred_loss minimize
├── src/lerobot_isaac_autoresearch/
│   ├── __init__.py
│   └── train_wrapper.py     # shim → lerobot_isaac_adapters.train
└── tests/
    ├── test_programs_parse.py   # validates program.md YAML keys
    └── test_train_wrapper.py    # argparse smoke test
```

## Public API

`train_wrapper.main()` — CLI entrypoint, same args as `lerobot_isaac_adapters.train`.
Called by `autoresearch-ml-executor-worker` via `script_path`.

## Dependency on Sibling Packages

Only `lerobot-isaac-adapters` (invoked as subprocess). No circular deps.

## Agent Source of Truth (read-only references)

- `/home/koen/tools/claude_code/agents/orchestrators/autoresearch-loop-orchestrator.md`
- `/home/koen/tools/claude_code/agents/workers/autoresearch-ml-executor-worker.md`
- `/home/koen/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md`
- `/home/koen/tools/claude_code/skills/autoresearch/`

Installed copies live at `~/.claude/agents/`. Do NOT edit agents here.

## How to Run Autoresearch

```bash
cd ~/tools/claude_code
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md --type ml_model
```

## Metric Contract

Each program.md `regex` pattern must match the last metric line on stdout.
`train_wrapper.py` guarantees the final stdout line is `<metric>=<float>`.

## Tests

```bash
cd packages/lerobot-isaac-autoresearch
pytest tests/ -v
```

## Spinout

```bash
git subtree split -P packages/lerobot-isaac-autoresearch -b spinout-autoresearch
```
