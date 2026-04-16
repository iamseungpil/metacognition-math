# Analysis Index

Updated: 2026-04-16 (cleanup)

This file separates current claim-bearing analysis code from historical or ad hoc scripts.
See also: top-level `ANALYSIS_MAP.md` for data/code/reports navigation and `results/cleanup_audit_2026_04_16.md` for the full audit.

## Current Mainline Analysis

1. `scripts/run_strict_pair_analysis.py`
   - Purpose: strict paired SFT behavior summary for base vs meta
   - Authoritative outputs:
     - `results/strict_pair_analysis_2026_04_15/meta_strict_behavior.json`
     - `results/strict_pair_analysis_2026_04_15/base_strict_behavior.json`
     - `results/strict_pair_analysis_2026_04_15/paired_strict_behavior.json`
     - `results/strict_pair_analysis_2026_04_15/meta_strict_aime_examples.txt`

2. `scripts/analyze_entropy_meta.py`
   - Purpose: token-level entropy / surprisal around `<|meta|>` blocks (or free-text `Confidence:`)
   - Flag: `--marker_mode {meta,confidence}` (added 2026-04-16, commit d93be4b)
   - Authoritative outputs:
     - `results/entropy_strict_meta/entropy_stats.json`
     - `results/entropy_strict_meta/entropy_per_block.csv`
     - `results/entropy_strict_meta/entropy_per_sample.csv`
     - `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/` (confidence-mode, 16k re-eval; populated by remote)
     - `results/entropy_base_grpo_step300_16k/` (in-flight)
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

## Current Eval Tools

1. `scripts/eval_vllm_1030.py` (NEW 2026-04-16) — vLLM 1030-problem eval with configurable `--max_tokens`
   - Canonical use: GRPO 16k re-eval (step 300, E21R-v2)
   - Expected outputs: `results/eval_1030_meta_grpo_e21r_v2_step300_16k/`, `results/eval_1030_base_grpo_step300_16k/`
2. `src/eval/eval_hf.py` — HF generate eval (max_tokens=4096 default, used by bundle)
3. `scripts/run_post_sft_bundle.py` — mainline post-SFT bundle (calls `analyze_confidence_distribution.py` + `extract_aime_qualitative.py`)

## Current Mainline Reports

1. `results/study_2026_04_16_metacot_v8_status_report.md`
2. `results/plan_metacot_v8_active_2026_04_09.md`
3. `results/codex_reviews/rq3_mainline_review_2026_04_16.md`
4. `results/cleanup_audit_2026_04_16.md`

## Side-Evidence / Smoke

1. `scripts/smoke_mcts_lite.py`
   - Output: `results/mcts_lite_smoke.json`

2. `scripts/smoke_rq3_pipeline.py`
   - Output: `results/rq3_pipeline_smoke.json`

3. `scripts/smoke_rq3_self_distill_path.py`
   - Purpose: end-to-end smoke for RQ3 -> self-distill row construction
   - Output: temporary JSON path printed at runtime

## Historical / Legacy

The following were moved to `archive/2026_04_16_cleanup/scripts/` on 2026-04-16 (see `results/cleanup_audit_2026_04_16.md` for the complete list):

- `final_analysis.py`, `analyze_1030.py`, `analyze_1030_eval.py`
- All V3/V6/V7 data generators (`gen_v3*`, `gen_v6_switch*`, `gen_switch*`, `merge_v6_clean_10k`)
- All control-v4/v5 build / qc / launch / autoresearch scripts
- Legacy analyze scripts (`analyze_responses`, `analyze_wrong`, `analyze_meta_behavior`, etc.)
- Legacy launchers (`launch_v2*`, `launch_v6*`, `launch_v7*`, `launch_e11*`, `launch_slot*`, `launch_behavior*`, `launch_control_v4*`, `launch_control_v5*`)
- Legacy autoresearch shells (`autoresearch.sh`, `autoresearch_loop.py`, `full_pipeline.sh`, `sequential_pipeline.sh`)
- `patch_verl_all.py`, `patch_verl_compat.py`
- `sync_v2_to_nodes.sh`

Still in `scripts/` but marked historical (retained because of test dependencies):

1. `scripts/analyze_control_v5_eval.py` — imported by `tests/test_control_v5_eval_analysis.py`
2. `scripts/build_probe_rollouts_hf.py` — imported by `tests/test_probe_rollout_contract.py`

Historical result directories moved to `archive/2026_04_16_cleanup/results/`: all `eval_v6_*`, `eval_v7_*`, `eval_1030_v5`, `eval_E21_screen`, `eval_base_matched`, `strict_pair_analysis`, `strict_pair_analysis_2026_04_11`, `entropy_analysis`, `entropy_v8_E20a`, `local_mirror`, `hf_backup_2026_04_01`, `study_2026_04_01_tex`, `study_2026_04_04`, `rerouting_analysis`.

Retained in `results/` (referenced by active RAG seed library pipeline): `results/eval_v8_E20a/`, `results/eval_v8_E20b/`.
