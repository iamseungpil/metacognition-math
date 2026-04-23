# Worktree Layout

This document defines how to read the repository without confusing active, historical, and scratch artifacts.

## Active Source-of-Truth Paths

1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `configs/mainline_contract.yaml`
2. `NODE_POLICY.md`
3. `docs/mainline_registry_2026_04_13.md`
4. `results/codex_reviews/strict_alignment_checklist_2026_04_11.md`
5. `docs/pipeline_stages.md`
6. `docs/artifact_policy.md`

## Directory Roles

### `data/`

Authoritative datasets only.

Examples:

1. strict paired SFT data
2. RL-ready parquet files
3. see `data/README.md` for the current V8 inventory

### `configs/`

Launchable configs.

Interpretation rule:

1. configs referenced by the active plan are mainline candidates
2. older configs remain historical unless the plan reactivates them

### `scripts/`

Operational launchers, builders, validators, and preflight tools.

Interpretation rule:

1. launchers without an active-plan reference are not automatically mainline
2. `scripts/verify_mainline_alignment.py` is the preferred preflight check
3. validators and preflight tools are always preferred over ad hoc shell edits

### `src/`

Core implementation.

Subareas:

1. `src/training` for SFT/RL/reward logic
2. `src/eval` for benchmark evaluation
3. `src/curriculum` for downstream adaptation work

### `analysis/`

Interpretation layer.

Rules:

1. must consume saved artifacts
2. should not silently depend on scratch-only paths

### `results/`

Machine-readable outputs and review notes.

Rules:

1. keep claim-bearing outputs here
2. review notes belong under `results/codex_reviews/`

### `checkpoints/`

Local model outputs only.
Do not treat this directory as provenance by itself.

## Cleanup Policy

While active runs are in flight:

1. do not perform disruptive directory moves
2. prefer documentation and validation guards first
3. move or archive historical artifacts only after active runs complete
