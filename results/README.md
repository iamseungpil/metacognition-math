# Results Index

Updated: 2026-04-16 (post cleanup)

This directory contains current V8 mainline artifacts. Older superseded artifacts were moved to `archive/2026_04_16_cleanup/results/` on 2026-04-16.

See also: top-level `ANALYSIS_MAP.md` and `results/cleanup_audit_2026_04_16.md`.

## Current V8 Mainline

1. Strict paired SFT behavior:
   - `strict_pair_analysis_2026_04_15/`
   - `strict_pair_analysis_repro_2026_04_12/` (registry-active)

2. Entropy around `<|meta|>` (SFT) and `Confidence:` (RL):
   - `entropy_strict_meta/`
   - `entropy_analysis_step300/`

3. Step-300 RL deep analysis:
   - `step300_deep_analysis/`

4. Strict SFT eval bundles:
   - `eval_v8_meta_inside_strict_sft/`
   - `eval_v8_base_matched_strict_sft/`

5. Reports and plans:
   - `study_2026_04_16_metacot_v8_status_report.md`
   - `plan_metacot_v8_active_2026_04_09.md`
   - `metacot_v8_experiment_report.md`
   - `mainline_analysis_manifest_2026_04_16.json`
   - `codex_reviews/rq3_mainline_review_2026_04_16.md`
   - `cleanup_audit_2026_04_16.md`

6. RQ3 self-distill mainline outputs:
   - `self_distill/` (question-only artifact generation, scored teacher selection, teacher-topk KL targets, and SFT readouts)

## 16k Re-Eval (NEW 2026-04-16)

Produced by `scripts/eval_vllm_1030.py` (configurable max_tokens). In-flight on remote:

1. `eval_1030_meta_grpo_e21r_v2_step300_16k/` — 1030 problems, max_tokens=16384, E21R-v2 step 300
2. `entropy_meta_grpo_e21r_v2_step300_16k_conf/` — confidence-mode entropy on the above
3. `eval_1030_base_grpo_step300_16k/` — in-flight
4. `entropy_base_grpo_step300_16k/` — in-flight

## Smoke / Side-Evidence

1. `rq3_pipeline_smoke.json`
2. `mcts_lite_smoke.json`
3. `rq3_side_eval_smoke/`
4. `rq3_side_eval_smoke_rel/`
5. `control_rag_real_audit.json`, `control_rag_real_audit_with_seed.json`
6. `control_rag_smoke.json`
7. `pass_at_k/`, `probe/`
8. `post_sft_bundle/`, `strict_data/`, `behavior_strict_pair/`
9. `reward_smoke_2026_04_09.md`
10. `token_ablation_base_2048.json`, `token_ablation_meta_2048.json`

## RAG Seed Sources (kept in place — referenced by active pipeline)

- `eval_v8_E20a/`, `eval_v8_E20b/` — read by `scripts/build_control_rag_seed_library.py`

## Historical / Superseded

Most historical directories moved to `archive/2026_04_16_cleanup/results/` (see cleanup audit for full list). Remaining in `results/`:

1. `archive/` — older archive (pre-2026-04-16)
2. `autoresearch/` — notes from the v8 autoresearch loop (rq3 files; keep)
3. `autoresearch_e6e7_loop/results.tsv` — E6/E7 sweep results
4. `analysis_E19_behavioral_2026_04_07.md` — historical note
5. `study_metacot_v5_2026_04_05.md`, `study_metacot_v6_2026_04_07.md` — historical studies

When a result exists in both a dated mainline folder and an older undated folder, prefer the dated mainline folder.
