# Worktree Cleanup Plan (2026-04-10)

## Goal

Keep the repository easier to navigate without disrupting active AMLT runs, recovery scripts, or preserved checkpoints.

## Current Classification

### 1. Active / do not move during runs

1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `NODE_POLICY.md`
3. `configs/verl_gdpo_e21.yaml`
4. `configs/verl_gdpo_e21r.yaml`
5. `scripts/relaunch_verl_e21r_0410.sh`
6. `scripts/relaunch_verl_base_redirect_0410.sh`
7. `sitecustomize.py`
8. `src/training/rewards.py`
9. `src/training/verl_reward.py`

### 2. Preserve but classify later

1. `checkpoints_recovered/`
   - currently about 17G
   - likely important recovery artifact, not a cleanup target until backup provenance is confirmed
2. `analysis/behavior_uncertainty_lab/`
   - separate project lane
   - should remain isolated from main Meta-CoT execution work
3. `tmp/run_base_redirect_0410.sh`
4. `tmp/run_e21r_redirect_0410.sh`
   - generated launcher wrappers
   - can be archived only after confirming no active launcher references them

### 3. Safe documentation / review area

1. `results/codex_reviews/`
2. `docs/`
3. `tests/`

## Proposed Next Cleanup Pass

### A. Script organization

1. keep active recovery launchers in `scripts/`
2. move obsolete one-off launchers to `scripts/archive/` only after grep-based reference check
3. keep `tmp/` wrappers until the related AMLT runs are fully complete

### B. Config organization

1. keep active veRL configs in `configs/`
2. move legacy `verl07_*` configs into `configs/archive/verl07/` after confirming they are not used by active holders
3. separate TRL-only configs from veRL configs in a later non-running window

### C. Results organization

1. keep active plan and current alignment notes in `results/`
2. move stale review dumps and old ad hoc notes into `results/archive/` after indexing them
3. avoid moving large eval result folders until report references are checked

## Immediate Principle

No destructive cleanup while:
1. `metacot-eval-node-recovery-0402` is running
2. `metacot-train-b-recovery-0402` is running
3. their recovery launchers remain relevant
