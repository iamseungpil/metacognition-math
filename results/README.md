# Results Index

Updated: 2026-04-21 (H200 SDC-shared session)

This directory contains current V8 mainline artifacts. Historical artifacts were moved to `archive/2026_04_16_cleanup/` (first cleanup) and `legacy/2026_04_20_workspace_cleanup/` (second cleanup).

**START HERE** for current state: `results/status_2026_04_21_session.md` — consolidates all HF-pushed experiments, baseline numbers, and open questions for the next SDC-shared launch.

See also: top-level `ANALYSIS_MAP.md`, `docs/mainline_registry_2026_04_13.md`, `results/cleanup_audit_2026_04_16.md`.

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

5. Reports and plans (chronological, newest first):
   - **`status_2026_04_21_session.md`** — ⭐ current session: HF audit + SDC-shared launch prep
   - **`plan_h200_2node_parallel_2026_04_21.md`** — ⭐ current plan: SDC-shared + possible M1/base ablation
   - `report_SDC_v6_2026_04_19.md` — SDC-split v6 failure analysis (-20pp)
   - `plan_SDC_v4_veRL_2026_04_19.md` — veRL port design
   - `plan_SDC_v3_2026_04_19.md` — shared-preserve design
   - `plan_meta_rlsd_v2_2026_04_17.md` — M1 / N3 hypothesis
   - `plan_EAD_unified_v3_2026_04_17.md` — EAD family paper plan
   - `plan_and_findings_consolidated_2026_04_16.md` — RQ1/2/3 consolidated
   - `study_2026_04_16_metacot_v8_final_report.md`
   - `study_2026_04_16_metacot_v8_status_report.md`
   - `plan_metacot_v8_active_2026_04_09.md` — V8 mainline activation plan
   - `metacot_v8_experiment_report.md` — legacy experiment report
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
4. older dated plans and study notes now live in `legacy/2026_04_20_workspace_cleanup/results/`

When a result exists in both a dated mainline folder and an older undated folder, prefer the dated mainline folder.
