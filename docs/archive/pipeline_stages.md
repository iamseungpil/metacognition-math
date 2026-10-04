# Mainline Pipeline Stages

This document defines the stage boundaries for the claim-bearing `metacognition` pipeline.

## Stage 0. Strict Paired Data

Inputs:

1. source V8 corpora

Outputs:

1. `data/v8_meta_inside_strict.parquet`
2. `data/v8_base_matched_strict.parquet`
3. strict validation summary

Advance only if:

1. row parity passes
2. prompt parity passes
3. boxed answer parity passes

## Stage 1. Strict Paired SFT

Inputs:

1. Stage 0 strict paired data
2. raw `Qwen/Qwen3-8B`

Outputs:

1. `checkpoints/v8_meta_inside_strict_sft`
2. `checkpoints/v8_base_matched_strict_sft`
3. W&B runs

Advance only if:

1. both lanes finished
2. correct initializer recorded
3. no duplicate launch or silent resume confusion

## Stage 2. Paired Eval

Inputs:

1. strict paired SFT checkpoints
2. checkpoints synced into the local worktree or another documented path used by the bundle

Outputs:

1. JSON results
2. metadata JSON
3. parquet results
4. qualitative sample bundle

Advance only if:

1. both lanes evaluated on the same benchmark slice
2. eval is deterministic (`do_sample=False`, fixed seed, recorded metadata)
3. outputs saved in machine-readable form

## Stage 3. Behavior / Confidence / Entropy Analysis

Inputs:

1. Stage 2 eval outputs

Outputs:

1. accuracy summary
2. confidence / ECE summary
3. behavior summary
4. AIME qualitative notes
5. entropy diagnostics for the meta lane when meta blocks are present

Advance only if:

1. required eval JSON and metadata exist for both lanes
2. meta parquet exists before entropy analysis is attempted
3. controller interpretation is documented
4. remaining behavior gap is identified

## Stage 4. Paired RL Anchor

Inputs:

1. strict SFT checkpoints
2. frozen paired RL hyperparameters
3. reward smoke pass

Outputs:

1. paired meta RL anchor
2. paired base RL anchor

Advance only if:

1. frozen shared keys are identical
2. evidence class is explicitly recorded

## Stage 5. Reward Comparison

Candidates:

1. `E21`
2. `E21R`
3. `E21S`

Outputs:

1. reward-family comparison tables
2. behavior deltas

Advance only if:

1. reward family remains interpretable
2. reward confusion smoke cases are covered

## Stage 6. RQ3 Inference Utilization

Inputs:

1. diagnosis-ready checkpoints
2. retrieved example bank

Outputs:

1. diagnosis-triggered retry evaluation
2. retry gain analysis with path logs
3. `E21M` multi-turn retry readout
4. optional confidence-bucket branching side-evidence readout
5. comparable `root -> intervention -> outcome` JSON traces
6. retrieval score breakdown plus `next meta` recovery readout

Advance only if:

1. diagnosis quality is already established in earlier stages
2. RQ2 reward-family comparison is already read out
3. retry and branching are evaluated as separate downstream uses of meta state
4. branching is explicitly labeled `side_evidence` unless promoted by a revised contract

Interpretation contract:

1. diagnosis-triggered retry asks whether meta state provides a useful failure description for retrieval / retry
2. confidence-bucket branching asks whether confidence is useful only as a selective expansion prior
3. branching is not evidence of full MCTS, learned value estimation, or training-time search
4. retrieval should be audited by solution/method alignment, not only question overlap
5. successful dynamic memory append must stay separated from the stable seed library

Reference implementation paths:

1. `src/curriculum/control_rag.py`
2. `src/curriculum/one_example_adapt.py`
3. `src/curriculum/mcts_lite.py`
4. `src/curriculum/rq3_pipeline.py`
5. `scripts/smoke_control_rag.py`
6. `scripts/smoke_mcts_lite.py`
7. `scripts/smoke_rq3_pipeline.py`
8. `scripts/build_control_rag_seed_library.py`
9. `scripts/build_control_rag_dynamic_library.py`
10. `scripts/audit_control_rag_real.py`
11. `scripts/run_rq3_side_eval.py`

Library workflow:

1. `stable_seed_library` is built from earlier correct typed artifacts and is reused across runs
2. `dynamic_success_library` is built only from successful repaired traces produced by RQ3 side evaluation
3. the two libraries must stay physically separable on disk even if both are passed together as `--example_bank`
4. audit and side-eval manifests should report which bank roles were active so retrieval provenance remains inspectable

Minimal smoke sequence:

1. build stable seed bank
   `python scripts/build_control_rag_seed_library.py --inputs ... --output /tmp/control_rag_seed_library.json --require_correct --require_study_need`
2. run retrieval smoke
   `python scripts/smoke_control_rag.py --skip-model --output /tmp/control_rag_smoke.json`
3. run RQ3 pipeline smoke
   `python scripts/smoke_rq3_pipeline.py --output /tmp/rq3_pipeline_smoke.json`
4. re-run side eval over the smoke trace bundle
   `python scripts/run_rq3_side_eval.py --cases /tmp/rq3_pipeline_smoke.json --example_bank /tmp/control_rag_seed_library.json --output_dir /tmp/rq3_side_eval_smoke`
5. build dynamic success bank from the saved traces
   `python scripts/build_control_rag_dynamic_library.py --traces /tmp/rq3_side_eval_smoke/rq3_traces.jsonl --output /tmp/control_rag_dynamic_library.json`
