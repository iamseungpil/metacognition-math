# Workspace Cleanup 2026-04-20

This folder stores dated materials moved out of the active workspace during the
2026-04-20 cleanup.

## Why these files were moved

The goal was to keep the active repository focused on the current mainline:

1. `control_v5` behavior data
2. `v8` strict-pair SFT
3. `rq3` self-distillation
4. `verl` GDPO / SDC work

Anything clearly tied to older `control_v4`, `v6`, `v7`, or superseded draft
plans was moved here instead of being deleted.

## Layout

1. `configs/` — superseded SFT configs for `control_v4`, `v6`, `v7`
2. `data/` — historical corpora, smoke datasets, and generation logs
3. `docs/` — old worktree cleanup notes
4. `results/` — superseded plan drafts and old study reports
5. `logs/` — historical E19 monitoring and analysis logs

## Policy

These files remain part of the project history and can be referenced for
reproducibility, but active scripts and documents should prefer the non-legacy
paths.
