# Cleanup Audit 2026-04-16

Authoritative inventory for `scripts/` (145) and `results/` (80 items). Each entry is tagged:

- `MAINLINE` — active V8 claim-bearing, keep in place
- `ACTIVE-DEP` — imported/called by a mainline script, keep in place
- `SIDE` — side-evidence / smoke, keep
- `ARCHIVE` — historical, move to `archive/2026_04_16_cleanup/`
- `DELETE` — transient logs/pids/state, safe to `git rm`
- `KEEP-UNKNOWN` — unclear, retained in place for user review

Basis:
- `scripts/ANALYSIS_INDEX.md`, `results/README.md`, `docs/mainline_registry_2026_04_13.md`, `docs/artifact_policy.md`
- grep-confirmed non-references from `src/`, `tests/`, `configs/`, mainline scripts

---

## scripts/ (145 items)

### MAINLINE (5) — per ANALYSIS_INDEX.md
- `scripts/run_strict_pair_analysis.py`
- `scripts/analyze_entropy_meta.py` (extended 2026-04-16 with `--marker_mode`)
- `scripts/analyze_self_distill_eval.py`
- `scripts/analyze_rq3_side_eval.py`
- `scripts/audit_control_rag_real.py`

### MAINLINE — V8 data/train/eval/RL (registry)
- `scripts/build_v8_strict_paired_data.py`
- `scripts/validate_v8_strict_data.py`
- `scripts/launch_v8_strict_sft_nodes.sh`
- `scripts/launch_v8_meta_inside_strict_remote.sh`
- `scripts/launch_v8_base_matched_strict_remote.sh`
- `scripts/launch_e21_vs_base_matched_0410.sh`
- `scripts/verify_mainline_alignment.py`
- `scripts/run_post_sft_bundle.py`
- `scripts/sync_strict_sft_and_run_bundle.sh`
- `scripts/run_online_sdpo_regen.py`
- `scripts/eval_vllm_1030.py` (NEW, 2026-04-16 — 16k re-eval)
- `scripts/run_eval_1030.sh`
- `scripts/run_eval.sh` (called from full_pipeline; keep for now)
- `scripts/run_rq3_side_eval.py`
- `scripts/build_control_rag_seed_library.py`
- `scripts/build_control_rag_dynamic_library.py`
- `scripts/build_self_distill_dataset.py`

### MAINLINE (side/smoke)
- `scripts/smoke_mcts_lite.py`
- `scripts/smoke_rq3_pipeline.py`
- `scripts/smoke_rq3_self_distill_path.py`
- `scripts/smoke_control_rag.py`
- `scripts/smoke_probe_pipeline.py`
- `scripts/smoke_self_distill_dataset.py`
- `scripts/smoke_test_rewards.py`

### ACTIVE-DEP — called from run_post_sft_bundle.py
- `scripts/analyze_confidence_distribution.py`
- `scripts/extract_aime_qualitative.py`

### ACTIVE-DEP — called from run_rq3_*_roundtrip.sh / relaunch_verl_*
- `scripts/run_rq3_sdpo_regen_roundtrip.sh`
- `scripts/run_rq3_teacher_query_roundtrip.sh`
- `scripts/run_fixed_k_self_distill_roundtrip.sh`
- `scripts/relaunch_verl_e21_0410.sh`
- `scripts/relaunch_verl_e21_mainline_0410.sh`
- `scripts/relaunch_verl_e21r_0410.sh`
- `scripts/relaunch_verl_base_redirect_0410.sh`

### ACTIVE-DEP — tests reference
- `scripts/analyze_control_v5_eval.py` (tests/test_control_v5_eval_analysis.py) — keep, mark historical
- `scripts/build_probe_rollouts_hf.py` (tests/test_probe_rollout_contract.py)

### KEEP — general infra / HF sync / env
- `scripts/common.sh`
- `scripts/setup_node.sh`
- `scripts/rebuild_eval_node.sh`
- `scripts/cleanup_compute.sh`
- `scripts/install_verl.sh`
- `scripts/hf_checkpoint_sync.sh`
- `scripts/hf_sync_latest.py`
- `scripts/push_models_hf.py`
- `scripts/ensure_hf_model.py`
- `scripts/sync_checkpoint_to_hf.py`
- `scripts/sync_v2_to_nodes.sh`
- `scripts/upload_dataset_artifacts.py`
- `scripts/get_active_proxy_endpoint.py`
- `scripts/launch_sft_remote.sh`
- `scripts/run_verl_gdpo.sh`
- `scripts/run_grpo_v2.sh`
- `scripts/run_base_sft.sh`
- `scripts/run_eval_all.sh`
- `scripts/run_eval_1030_eval_node.sh`
- `scripts/run_eval_1030_trainb_node.sh`
- `scripts/gen_metacot_v2.py` (source-of-truth data gen)
- `scripts/test_eval.py`
- `scripts/test_meta_tokens.py`
- `scripts/test_parsing.py`
- `scripts/check_runtime_env.py`

### ARCHIVE — legacy V3-V7 / control-v4/v5 / control_v4_v5
Move to `archive/2026_04_16_cleanup/scripts/`. Self-referenced only (within peer legacy scripts); NOT referenced from src/, tests/, or mainline scripts.

Generators:
- `scripts/gen_v3.py`
- `scripts/gen_v3_hard.py`
- `scripts/gen_local.py`
- `scripts/gen_v6_switch_data.py`
- `scripts/gen_switch_trajectory.py`
- `scripts/gen_control_v4_trapi.py`
- `scripts/gen_control_v5_trapi.py`
- `scripts/gen_behavior_trapi.py`
- `scripts/merge_v6_clean_10k.py`

Build variants:
- `scripts/build_behavior_sft_variants.py`
- `scripts/build_control_v4_sft_variants.py`
- `scripts/build_control_v5_sft_variants.py`
- `scripts/build_v7_think_meta_data.py`
- `scripts/build_teacher_topk_targets.py`

QC / side analysis:
- `scripts/qc_control_v4_samples.py`
- `scripts/qc_control_v5_samples.py`
- `scripts/analyze_control_v5_failure_modes.py`
- `scripts/analyze_e11_pilot.py`
- `scripts/analyze_responses.py`
- `scripts/analyze_wrong.py`
- `scripts/analyze_meta_behavior.py`
- `scripts/analyze_1030.py`
- `scripts/analyze_1030_eval.py`
- `scripts/compare_responses.py`
- `scripts/deep_compare.py`
- `scripts/rescore_eval.py`
- `scripts/compute_pass_at_k.py`
- `scripts/check_data.py`
- `scripts/check_direction_change.py`
- `scripts/check_rollout_data.py`
- `scripts/create_base_sft.py`
- `scripts/create_verifyonly_sft.py`
- `scripts/augment_sft_data.py`
- `scripts/filter_by_passrate.py`
- `scripts/meta_rerouting_experiment.py`
- `scripts/selective_abstention.py`
- `scripts/final_analysis.py` (declared legacy in ANALYSIS_INDEX.md)

Launchers (V2/V5/V6/V7/E9/E11 — superseded by v8 strict launchers):
- `scripts/launch_v2_experiments.sh`
- `scripts/launch_v2_nodes.sh`
- `scripts/launch_v2_on_eval_trainb.sh`
- `scripts/launch_v6_clean_sft_all_nodes.sh`
- `scripts/launch_v7_sft_all_nodes.sh`
- `scripts/launch_e11_sft.sh`
- `scripts/launch_e6_e7_on_e8.sh`
- `scripts/launch_slotB_rl.sh`
- `scripts/launch_slotB_rl_e9c.sh`
- `scripts/launch_slotC_sft.sh`
- `scripts/launch_behavior_all_sft_remote.sh`
- `scripts/launch_behavior_redirect_sft_remote.sh`
- `scripts/launch_control_v4_all_sft_remote.sh`
- `scripts/launch_control_v4_redirect_sft_remote.sh`
- `scripts/launch_control_v4_verify_sft_remote.sh`
- `scripts/launch_control_v5_all_sft_remote.sh`
- `scripts/launch_control_v5_redirect_sft_remote.sh`
- `scripts/launch_control_v5_rl_eval_lane_remote.sh`
- `scripts/launch_control_v5_rl_nodes.sh`
- `scripts/launch_control_v5_rl_probe_lane_remote.sh`
- `scripts/launch_control_v5_rl_train_b_lane_remote.sh`
- `scripts/launch_control_v5_sft_nodes.sh`
- `scripts/launch_control_v5_verify_sft_remote.sh`

Autoresearch (V3-V5 loops, superseded by v8 plan):
- `scripts/autoresearch.sh`
- `scripts/autoresearch_loop.py`
- `scripts/autoresearch_behavior_phase2.sh`
- `scripts/autoresearch_behavior_round1.sh`
- `scripts/autoresearch_control_v4.sh`
- `scripts/autoresearch_control_v4_train.sh`
- `scripts/autoresearch_control_v5_pipeline.sh`
- `scripts/autoresearch_followup_monitor.sh`
- `scripts/monitor_redirect_and_launch.sh`
- `scripts/monitor_session_2026_03_31.sh`
- `scripts/retry_probe_lane_after_e10.sh`
- `scripts/sequential_pipeline.sh`
- `scripts/full_pipeline.sh`
- `scripts/run_control_v5_eval_matrix.sh`
- `scripts/run_phase0.sh`
- `scripts/run_phase1.sh`

VERL patching (if not in active CI):
- `scripts/patch_verl_all.py`
- `scripts/patch_verl_compat.py`

---

## results/ (80 items)

### MAINLINE
- `results/README.md`
- `results/strict_pair_analysis_2026_04_15/` — current paired SFT behavior
- `results/entropy_strict_meta/` — current entropy around <|meta|>
- `results/step300_deep_analysis/` — Step-300 RL deep analysis
- `results/entropy_analysis_step300/` — RL confidence + SFT meta sync
- `results/study_2026_04_16_metacot_v8_status_report.md`
- `results/plan_metacot_v8_active_2026_04_09.md`
- `results/metacot_v8_experiment_report.md`
- `results/mainline_analysis_manifest_2026_04_16.json`
- `results/eval_v8_meta_inside_strict_sft/` — registry-active
- `results/eval_v8_base_matched_strict_sft/` — registry-active
- `results/strict_pair_analysis_repro_2026_04_12/` — registry-active
- `results/strict_data/`
- `results/behavior_strict_pair/`
- `results/post_sft_bundle/`
- `results/codex_reviews/` (contains today's rq3_mainline_review)
- `results/autoresearch/rq3_autoresearch_2026_04_14.md`, `rq3_d2b_dryrun_2026_04_15.md` (rq3 mainline notes)

### EXPECTED (in-flight, populated by remote) — referenced in new ANALYSIS_MAP.md
- `results/eval_1030_meta_grpo_e21r_v2_step300_16k/` (to be populated)
- `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/` (to be populated)
- `results/eval_1030_base_grpo_step300_16k/` (in-flight)
- `results/entropy_base_grpo_step300_16k/` (in-flight)

### SIDE / smoke
- `results/rq3_side_eval_smoke/`
- `results/rq3_side_eval_smoke_rel/`
- `results/rq3_pipeline_smoke.json`
- `results/mcts_lite_smoke.json`
- `results/control_rag_smoke.json`
- `results/control_rag_real_audit.json`
- `results/control_rag_real_audit_with_seed.json`
- `results/pass_at_k/`
- `results/probe/`
- `results/reward_smoke_2026_04_09.md`
- `results/token_ablation_base_2048.json`
- `results/token_ablation_meta_2048.json`
- `results/reroute*` — reserved

### HISTORICAL — mentioned in README.md as historical/superseded
- `results/strict_pair_analysis/`
- `results/strict_pair_analysis_2026_04_11/`
- `results/entropy_analysis/`
- `results/entropy_v8_E20a/`
- `results/archive/` (already the archive path — leave in place)
- `results/eval_1030_v5/` (117MB — large, historical per registry)
- `results/eval_v6_E11/`
- `results/eval_v6_E19/`
- `results/eval_v6_E19b/`
- `results/eval_v6_E19c/`
- `results/eval_v6_SlotB/`
- `results/eval_v6_SlotC/`
- `results/eval_v7_E19v2/`
- `results/eval_v7_E19v2b/`
- `results/eval_v7_E19v2c/`
- `results/eval_v8_E20a/` — seed source for seed_library, keep in place (referenced)
- `results/eval_v8_E20b/` — seed source for seed_library, keep in place (referenced)
- `results/eval_E21_screen/`
- `results/eval_base_matched/`
- `results/local_mirror/`
- `results/hf_backup_2026_04_01/`
- `results/study_2026_04_01_tex/`
- `results/study_2026_04_04/`
- `results/study_metacot_v5_2026_04_05.md`
- `results/study_metacot_v6_2026_04_07.md`
- `results/analysis_E19_behavioral_2026_04_07.md`
- `results/plan_metacot_v8_active_2026_04_09.md` — mainline, keep
- `results/rerouting_analysis/`
- `results/control_v5_probe_lane/`

Note: `results/eval_v8_E20a`, `eval_v8_E20b` remain in place since they are referenced by the active RAG seed library build in `docs/pipeline_stages.md`.

### DELETE — transient files (logs/pids/state/out from March-April 2026)
- `results/autoresearch_behavior_phase2_2026_04_01.log`
- `results/autoresearch_behavior_phase2_2026_04_01.pid`
- `results/autoresearch_behavior_phase2_2026_04_01.state`
- `results/autoresearch_behavior_phase2_launcher.out`
- `results/autoresearch_followup_2026_04_01.log`
- `results/autoresearch_followup_2026_04_01.pid`
- `results/autoresearch_followup_2026_04_01.state`
- `results/autoresearch_followup_launcher.out`
- `results/autoresearch_monitor_2026_03_31.log`
- `results/autoresearch_monitor_2026_03_31.pid`
- `results/autoresearch_monitor_2026_03_31.state`
- `results/autoresearch_monitor_2026_03_31.tsv`
- `results/autoresearch_monitor_launcher.log`
- `results/autoresearch_monitor_runner.out`
- `results/autoresearch_plan_alignment_2026_04_06.tsv`
- `results/critic_log.txt`
- `results/training_status.txt`
- `results/autoresearch_round1/` (3 files, all stale logs)
- `results/autoresearch_control_v4/` (5 transient files)
- `results/autoresearch_control_v5/` (6 transient files)
- `results/autoresearch_control_v5_rl/` (9 transient files)
- `results/autoresearch_v2_experiments/` (4 transient files)
- `results/autoresearch_e6e7_loop/results.tsv` — keep, references E6/E7 experiments mentioned in memory

### KEEP-UNKNOWN
- `results/autoresearch_e6e7_loop/` — small (24KB), referenced in project memory as historical e6e7 run

---

## tmp/ cleanup plan

- KEEP (moved 2026-04-16 to `data/control_rag_seed_library.json`): referenced by active RAG pipeline per pipeline_stages.md
- DELETE: `tmp/control_rag_real_bank.json`, `tmp/rag_all.json`, `tmp/rag_bank_from_meta_eval.json`, `tmp/rag_study.json` (duplicate/intermediate RAG banks — can be rebuilt)
- DELETE: `tmp/run_base_redirect_0410.sh`, `tmp/run_e21r_redirect_0410.sh` (scratch copies of relaunch scripts that exist in scripts/)
- DELETE: `tmp/self_distill_smoke/` (smoke run artifacts)

---

## logs/ cleanup plan

- KEEP in place (small, may contain provenance):
  - `logs/entropy_analysis_E19.log` (10KB)
  - `logs/monitor_redirect.log` (11KB)
- DELETE (zero-byte stubs):
  - `logs/hf_sync_v8_meta_inside_strict_sft.log` (0 bytes)
  - `logs/strict_meta_eval_local.log` (0 bytes)
  - `logs/strict_post_sft_bundle_2026_04_11.log` (34 bytes)

---

## analysis/behavior_uncertainty_lab/

Sub-project with its own README + PLAN. NOT touched during this cleanup. Untracked in git, so no `git rm` impact. Reviewed separately.

---

## Safety-checked references

- `scripts/analyze_control_v5_eval.py` — retained (test file imports it).
- `scripts/build_probe_rollouts_hf.py` — retained (test file imports it).
- `scripts/analyze_confidence_distribution.py`, `scripts/extract_aime_qualitative.py` — retained (run_post_sft_bundle.py calls them).
- `data/control_rag_seed_library.json` — retained and relocated from `tmp/` on 2026-04-16 (docs/pipeline_stages.md + plan references updated).
- All `scripts/relaunch_verl_*.sh` — retained (E21 0410 RL workflow references).
