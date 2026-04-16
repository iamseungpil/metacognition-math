# Behavior-Uncertainty Lab

This directory is a sidecar analysis workspace.
It is not the authoritative execution plan for the main `metacognition` run.

Its role is to analyze saved eval artifacts from the active V8 strict pipeline using two lenses:

1. behavior taxonomy
2. uncertainty / information-allocation

## Current Intended Inputs

The primary inputs should be saved eval outputs from:

1. strict meta SFT
2. strict base-matched SFT
3. later paired RL checkpoints that satisfy the frozen contract

The analysis should use machine-readable eval bundles, not undocumented local scratch files.

## Current Intended Questions

1. Does the model emit `verify`, `redirect`, `diagnosis`, `subgoal`, or `backward chaining`?
2. Are those actions conditioned on confidence and anomaly signals?
3. On hard sets such as `aime2024`, does Meta-CoT create real route changes or only extra text?
4. Do entropy and confidence statistics change before and after meta interventions?

## Status

Historical notes in this directory may mention older V5/V6 experiments.
Those notes are context only.
The main claim-bearing pipeline is defined by:

1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `NODE_POLICY.md`
3. `results/codex_reviews/strict_alignment_checklist_2026_04_11.md`
