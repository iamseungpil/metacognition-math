# Analysis Index

Updated: 2026-04-16

This file separates current claim-bearing analysis code from historical or ad hoc scripts.

## Current Mainline Analysis

1. `scripts/run_strict_pair_analysis.py`
   - Purpose: strict paired SFT behavior summary for base vs meta
   - Authoritative outputs:
     - `results/strict_pair_analysis_2026_04_15/meta_strict_behavior.json`
     - `results/strict_pair_analysis_2026_04_15/base_strict_behavior.json`
     - `results/strict_pair_analysis_2026_04_15/paired_strict_behavior.json`
     - `results/strict_pair_analysis_2026_04_15/meta_strict_aime_examples.txt`

2. `scripts/analyze_entropy_meta.py`
   - Purpose: token-level entropy / surprisal around `<|meta|>` blocks
   - Authoritative outputs:
     - `results/entropy_strict_meta/entropy_stats.json`
     - `results/entropy_strict_meta/entropy_per_block.csv`
     - `results/entropy_strict_meta/entropy_per_sample.csv`
   - Note: expensive HF forward-pass analysis, not a cheap smoke script

3. `scripts/analyze_self_distill_eval.py`
   - Purpose: compare eval bundles with self-distill-focused metrics
   - Authoritative output:
     - `results/strict_pair_analysis_2026_04_15/strict_self_distill_eval_compare.remote.json`

4. `scripts/analyze_rq3_side_eval.py`
   - Purpose: compare RQ3 side-eval summaries against RQ3-D success gates
   - Current smoke outputs:
     - `results/rq3_side_eval_smoke/rq3_summary.json`
     - `results/rq3_side_eval_smoke_rel/rq3_summary.json`

5. `scripts/audit_control_rag_real.py`
   - Purpose: audit real retrieval behavior on local eval artifacts
   - Current output:
     - `results/control_rag_real_audit.json`

## Current Mainline Reports

1. `results/study_2026_04_16_metacot_v8_status_report.md`
2. `results/plan_metacot_v8_active_2026_04_09.md`
3. `results/codex_reviews/rq3_mainline_review_2026_04_16.md`

## Side-Evidence / Smoke

1. `scripts/smoke_mcts_lite.py`
   - Output: `results/mcts_lite_smoke.json`

2. `scripts/smoke_rq3_pipeline.py`
   - Output: `results/rq3_pipeline_smoke.json`

3. `scripts/smoke_rq3_self_distill_path.py`
   - Purpose: end-to-end smoke for RQ3 -> self-distill row construction
   - Output: temporary JSON path printed at runtime

## Historical / Legacy

These scripts are not the current claim-bearing path for V8 and should not be cited without context.

1. `scripts/analyze_control_v5_eval.py`
2. `scripts/analyze_1030_eval.py`
3. `scripts/final_analysis.py`
4. older `results/eval_v6_*`, `results/eval_v7_*`, `results/eval_v8_E20*`

`scripts/final_analysis.py` is especially ad hoc: it uses hardcoded `/scratch` paths and should be treated as scratch analysis, not as the mainline report generator.
