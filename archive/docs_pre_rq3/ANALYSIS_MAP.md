> ⚠️ **DEPRECATED** (pre-rq3 V8/H200 세대 기록): 현행 실험은 **rq3 매치드 래더** — `README.md` 및 `docs/redesign/` 참조.

# Analysis Map — Meta-CoT V8

Newcomer-friendly navigation. Last updated 2026-04-16 after cleanup.

For deeper detail see:
- `scripts/ANALYSIS_INDEX.md` — code inventory
- `results/README.md` — result directory inventory
- `results/cleanup_audit_2026_04_16.md` — per-file audit (244 files reorganized)
- `docs/mainline_registry_2026_04_13.md` — canonical mainline registry
- `docs/artifact_policy.md` — artifact rules
- `docs/pipeline_stages.md` — pipeline stages
- `CLAUDE.md` — project goal and constraints

## Data

HuggingFace: `iamseungpil/metacot` (source of truth)
- `base_sft.parquet` — 4,996 base chains (meta stripped)

Local (strict paired SFT anchor):
- `data/v8_meta_inside_strict.parquet`
- `data/v8_base_matched_strict.parquet`
- `results/strict_data/v8_strict_validation_summary.json`

Local (paired RL redirect subset):
- `data/verl_train_redirect.parquet`, `data/verl_val_redirect.parquet`
- `data/verl_train_redirect_base.parquet`, `data/verl_val_redirect_base.parquet`

RAG seed library (active):
- `data/control_rag_seed_library.json` — referenced by `scripts/audit_control_rag_real.py` and `scripts/run_rq3_side_eval.py`

## Analysis Code (the 5 mainline scripts)

1. `scripts/run_strict_pair_analysis.py` — paired SFT behavior summary
2. `scripts/analyze_entropy_meta.py` — token-level entropy around `<|meta|>` or `confidence: 0.XX` (`--marker_mode {meta,confidence}`)
3. `scripts/analyze_self_distill_eval.py` — self-distill-focused eval comparison
4. `scripts/analyze_rq3_side_eval.py` — RQ3 success-gate comparison
5. `scripts/audit_control_rag_real.py` — real retrieval audit

Eval helpers:
- `scripts/eval_vllm_1030.py` (NEW 2026-04-16) — vLLM 1030-problem eval, configurable `--max_tokens` (used for 16k re-eval)
- `src/eval/eval_hf.py` — HF generate eval (legacy path, slow)
- `scripts/run_post_sft_bundle.py` — post-SFT bundle orchestrator

Training code (for reference):
- `src/training/sft.py` — SFT
- `src/training/grpo_v2.py` — GRPO with modular rewards (E1-E7)
- `src/training/rewards.py` — 7 reward functions
- `src/training/verl_gdpo.py`, `src/training/verl_reward.py` — VERL RL path
- `src/training/self_distill/` — self-distill pipeline (active WIP, vLLM-based)
- `src/curriculum/rag.py` — FAISS-based curriculum

Data build / validation:
- `scripts/build_v8_strict_paired_data.py`
- `scripts/validate_v8_strict_data.py`
- `scripts/build_control_rag_seed_library.py`
- `scripts/build_self_distill_dataset.py`

## Reports

Current V8 reports:
- `results/study_2026_04_16_metacot_v8_status_report.md` — **latest status**
- `results/plan_metacot_v8_active_2026_04_09.md` — active execution plan
- `results/metacot_v8_experiment_report.md`
- `results/mainline_analysis_manifest_2026_04_16.json` — analysis manifest
- `results/codex_reviews/rq3_mainline_review_2026_04_16.md`
- `results/cleanup_audit_2026_04_16.md` — today's cleanup audit
- `results/autoresearch/rq3_autoresearch_2026_04_14.md`, `rq3_d2b_dryrun_2026_04_15.md`

Historical reports (pre-V8):
- `results/archive/` — plans v1-v6, phase reports, study notes
- `results/study_metacot_v5_2026_04_05.md`, `results/study_metacot_v6_2026_04_07.md`

## Current Experiments

Checkpoints:
- `checkpoints/` — local training outputs (meta vs base_matched strict SFT, E21R-v2 RL)
- `checkpoints_recovered/` — archived models, 17 GB, DO NOT TOUCH

Launchers (active):
- `scripts/launch_v8_strict_sft_nodes.sh` — strict SFT
- `scripts/launch_v8_meta_inside_strict_remote.sh`, `launch_v8_base_matched_strict_remote.sh`
- `scripts/launch_e21_vs_base_matched_0410.sh` — E21 RL
- `scripts/relaunch_verl_e21_0410.sh`, `relaunch_verl_e21r_0410.sh`, `relaunch_verl_e21_mainline_0410.sh`, `relaunch_verl_base_redirect_0410.sh`

Step-300 RL analyses (reference):
- `results/step300_deep_analysis/` — full_analysis.json, e21r_v2_step300.json, base_step300.json, AIME qualitative, strict_pair
- `results/strict_pair_analysis_2026_04_15/` — paired SFT behavior snapshot (mainline)
- `results/entropy_strict_meta/` — entropy on strict meta SFT (n=120, delta +0.300 nats)
- `results/entropy_analysis_step300/rl_meta_confidence/` — confidence-mode entropy on RL step300 (n=200, delta -0.052)

16k re-eval (launched 2026-04-16):
- `results/eval_1030_meta_grpo_e21r_v2_step300_16k/` — DONE (max_tokens=16384, 1030 problems)
- `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/` — DONE (confidence-mode entropy, delta -0.0305 resolved)
- `results/eval_1030_base_grpo_step300_16k/` — in flight on train_b
- `results/entropy_base_grpo_step300_16k/` — will run after base eval

Eval result directories (strict SFT):
- `results/eval_v8_meta_inside_strict_sft/`
- `results/eval_v8_base_matched_strict_sft/`

RQ3 smoke:
- `results/rq3_side_eval_smoke/`, `results/rq3_side_eval_smoke_rel/`
- `results/rq3_pipeline_smoke.json`, `results/mcts_lite_smoke.json`

Compute notes:
- `NODE_POLICY.md` — node assignment rules
- `node_recovery_0415.yaml` — AMLT recovery config
- `metacognition_*.yaml` — AMLT submission configs

## Archive

- `archive/2026_04_16_cleanup/` — scripts and results archived during 2026-04-16 cleanup
  - `scripts/` — legacy V3-V7 generators, control-v4/v5, autoresearch shells, V2/V6/V7 launchers
  - `results/` — legacy `eval_v6_*`, `eval_v7_*`, `eval_1030_v5`, older strict/entropy snapshots
- `results/archive/` — pre-2026-04-16 archive of plans and study notes

## Cleanup stats (2026-04-16)

- Before: `scripts/` 145 files, `results/` 80 items
- After:  `scripts/`  67 files, `results/` 38 items
- Archived: 78 scripts + 21 result dirs
- Deleted transient: 44 files (autoresearch .log/.pid/.state from March)
- Broken references detected and fixed: 2
