# Autonomous fix ledger (2026-06-09 eve →)

Running log of things that broke + how I fixed them, while executing the 3-step autonomous
plan (`plans/2026-06-09-staged-reward-next-steps-plan.md`) unattended. Newest at bottom.

Mode: full autonomy — no input requests; fix breaks in place; record each here.

Format: `[time] SYMPTOM → ROOT CAUSE → FIX (commit/file)`.

---

- [19:2x] (baseline) run #3b healthy at −14.6 (3.9k steps), closure reward live. No breaks yet.
- [19:5x] STEP 1 done: `success_bonus` was a redundant proximity Gaussian + Isaac scales all
  reward terms by dt → negligible. Added dt-invariant `place_success_reward` (commit 892f5d5,
  lerobot-isaac-env). Not a break — a design gap fixed. 80 tests green.
- [19:5x] WATCH: run #3b GPU at 9.1 GB / 10 GB (desktop session :1 + replay buffer growth).
  Near OOM ceiling. Monitor greps for CUDA OOM; if it crashes → reduce batch_size or buffer.
  Branch trending A (reward climbing −18→−14.6).
