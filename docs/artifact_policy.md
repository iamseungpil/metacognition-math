# Artifact Policy

This document keeps HF, git, datasets, checkpoints, and reports from drifting into each other.

## 1. Data

Source-of-truth datasets live in `data/`.

Naming:

1. `v8_*_strict.parquet` for claim-bearing strict paired data
2. `verl_*` for RL-ready parquet files
3. do not overwrite strict data with exploratory transforms

## 2. Checkpoints

Local training outputs live in `checkpoints/`.

Rules:

1. strict SFT outputs must encode whether they are `meta` or `base_matched`
2. exploratory RL outputs must not be named as if they were `mainline`
3. do not silently reuse a previous checkpoint as if it were raw base

## 3. Results

Machine-readable eval outputs live in `results/`.

Required bundle:

1. JSON
2. metadata JSON
3. parquet
4. decoding parameters in metadata
5. evidence label in the surrounding report or manifest

Qualitative notes may also be saved under `results/` or linked from `analysis/`.

## 4. Analysis

Analysis code lives in `analysis/` or `src/eval/`.

Rules:

1. analysis docs must say whether they are `mainline` or `historical`
2. analysis must consume saved eval artifacts, not undocumented local state

## 5. HuggingFace

HF should contain:

1. final strict datasets
2. final strict SFT checkpoints
3. final claim-bearing RL checkpoints

HF should not be the only copy of provenance.
Every pushed model must still have a matching local config, launcher, and eval bundle in git.

## 6. Git

Git is for:

1. code
2. configs
3. plans
4. validation reports

Git is not for:

1. large temporary checkpoints
2. scratch logs
3. ad hoc local mirrors without provenance

## 7. Evidence Label

Every pushed artifact should carry one evidence label:

1. `mainline`
2. `side_evidence`
3. `historical`
4. `invalid_for_claim`
