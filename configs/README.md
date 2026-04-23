# Config Inventory

This directory contains active configs only.

## Current active families

1. `sft_control_v5_*`
2. `sft_v8_*`
3. `sft_self_distill_*`
4. `verl_*`
5. `contrastive_meta_rlsd.yaml`
6. infra configs such as `accelerate_*`, `ds_zero3*.json`, `eval_qwen3.yaml`

## Moved to legacy

The following superseded config families were moved on 2026-04-20 to
`legacy/2026_04_20_workspace_cleanup/configs/`:

1. `sft_control_v4_*`
2. `sft_v6_*`
3. `sft_v7_*`

These files are kept for reproducibility, but they are no longer part of the
active workspace.
