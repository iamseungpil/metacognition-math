# Mainline Registry

This is the shortest route for locating the current claim-bearing files.
It complements the active plan by indexing the exact code, data, and result paths that implement it.

## Source Of Truth

1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `configs/mainline_contract.yaml`
3. `NODE_POLICY.md`
4. `docs/artifact_policy.md`
5. `docs/pipeline_stages.md`

## Active Data

Strict paired SFT anchor:

1. `data/v8_meta_inside_strict.parquet`
2. `data/v8_base_matched_strict.parquet`
3. `results/strict_data/v8_strict_validation_summary.json`

Paired RL redirect subset:

1. `data/verl_train_redirect.parquet`
2. `data/verl_val_redirect.parquet`
3. `data/verl_train_redirect_base.parquet`
4. `data/verl_val_redirect_base.parquet`

Historical but still useful:

1. `data/v8_meta_inside_think.parquet`
2. `data/v8_base_matched_clean.parquet`

## Active Code

Data build / validation:

1. `scripts/build_v8_strict_paired_data.py`
2. `scripts/validate_v8_strict_data.py`
3. `src/training/verl_gdpo_data.py`

SFT:

1. `configs/sft_v8_meta_inside_strict.yaml`
2. `configs/sft_v8_base_matched_strict.yaml`
3. `src/training/sft.py`
4. `scripts/launch_v8_strict_sft_nodes.sh`

RL:

1. `scripts/launch_e21_vs_base_matched_0410.sh`
2. `src/training/verl_reward.py`
3. `src/training/rewards.py`
4. `src/training/verl_gdpo.py`
5. `scripts/verify_mainline_alignment.py`

RQ3 self-distill mainline (current claim-bearing next stage):

Ladder:

1. `question_only_best_of_n`
2. `correctness_only`
3. `correct_then_meta`
4. `meta_only KL`

1. `scripts/run_self_distill_roundtrip.sh`
2. `scripts/run_self_distill_sft_h200.sh`
3. `scripts/build_teacher_topk_targets.py`
4. `src/training/self_distill/online.py`
5. `src/training/self_distill/teacher_query.py`
6. `src/training/meta_quality.py`
7. `configs/sft_self_distill_base_qonly_naive_h200_4gpu.yaml`
8. `configs/sft_self_distill_meta_qonly_epistemic_h200_4gpu.yaml`
9. `configs/sft_self_distill_meta_qonly_scored_h200_4gpu.yaml`
10. `configs/sft_self_distill_meta_qonly_epistemic_meta_kl_h200_4gpu.yaml`

Additive side-evidence RL smoke:

1. `scripts/launch_e21r_v4_commit_shape_0416.sh`
2. `src/training/verl_reward.py::compute_score_e21r_v4_smoke`

Feedback-conditioned side-evidence full loop:

1. `scripts/run_rq3_sdpo_regen_roundtrip.sh`
2. `scripts/prepare_self_distill_sft_config.py`
3. `src/training/self_distill/pipeline.py`

Guardrails:

1. `sdpo_regen` is side-evidence only and must not be marked claim-bearing
2. incorrect selected teachers are dropped before parquet / teacher-topk / KL
3. retrieval-conditioned runs require an explicit example bank; warning-only fallback is not considered valid RAG

Eval / analysis:

1. `src/eval/eval_hf.py`
2. `scripts/run_post_sft_bundle.py`
3. `scripts/run_strict_pair_analysis.py`

## Active Results

1. `results/eval_v8_meta_inside_strict_sft/`
2. `results/eval_v8_base_matched_strict_sft/`
3. `results/strict_pair_analysis_repro_2026_04_12/`
4. `results/self_distill/` (RQ3 mainline outputs as they are generated)

## Historical / Side-Evidence Areas

These areas may still be useful, but they are not active mainline by default:

1. `results/archive/`
2. `results/eval_1030_v5/`
3. `results/eval_v6_E19/`
4. `results/eval_v8_E20a/`
5. `checkpoints_recovered/`

## Cleanup Rule

While active runs are in flight:

1. do not move or delete directories that a live node may still reference
2. add validation and indexing first
3. archive or rename only after the active run completes
