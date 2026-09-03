# Dead code archived 2026-09-04

Move-only commit. No file content was edited — every entry below was relocated
with `git mv` so history follows. Confirmed zero importers in
`src/`, `scripts/`, `tests/`, `probes/` as of 2026-09-03 (grep for both plain
substring hits and actual `import ... <module>` / `from ... import <module>`
statements; only hits were self-references, docstrings, or historical
`results/*.md` prose — never a live code path).

## src/training/ (RL trainer lineage superseded by verl_sdc.py)
- `src/training/grpo_v2.py` — pre-veRL GRPO trainer; superseded by
  `verl_sdc.py`. Only mentioned in a docstring comment in
  `verl_gdpo_data.py`, never imported.
- `src/training/verl_gdpo.py` — GDPO trainer generation retired in favor of
  the VANILLA_GRPO / TRIOBJ_DCPO_V4 paths in `verl_sdc.py`. Zero importers
  outside this same retired group.
- `src/training/verl_gdpo_algos.py` — advantage-computation helper imported
  only by `verl_gdpo.py`, which is archived alongside it.

## src/curriculum/ (RAG-based curriculum, never wired into the mainline)
- `src/curriculum/rag.py` — `CurriculumRAG` (FAISS + sentence-transformers);
  only self-referenced in its own module docstring example.
- `src/curriculum/one_example_adapt.py` — one-example adaptation experiment;
  zero importers.

## src/metacot/ (prompt-builder generations superseded by prompt_redirect_verify.py)
- `src/metacot/generator.py` — zero importers.
- `src/metacot/prompt_behavior.py` — superseded; only mentioned in a
  docstring comment of `prompt_redirect_verify.py` ("mirrors
  `prompt_behavior.py`") and a comment in `scripts/pg0_yield_pilot.py`.
- `src/metacot/prompt_control_v4.py` — superseded; same docstring mention
  only, no import.
- `src/metacot/prompt_control_v5.py` — zero importers.
- `src/metacot/prompt_v3.py` — zero importers (only referenced from already
  archived `archive/2026_04_16_cleanup/scripts/gen_v3*.py`).

## src/rollout/ (whole directory)
- `src/rollout/__init__.py`, `src/rollout/hidden_cache.py`,
  `src/rollout/vllm_rollout.py` — zero references anywhere outside this
  directory.

## src/eval/ (counterfactual-difficulty / pass@k probes, no live callers)
- `src/eval/eval_counterfactual_difficulty.py` — invoked historically via
  `scripts/run_e190_cf_eval.sh`-style launchers that are themselves already
  archived; no live script or test calls it.
- `src/eval/eval_counterfactual_difficulty_summarize.py` — companion
  summarizer, imported only by the file above (archived together).
- `src/eval/eval_passk_headroom.py` — imports
  `eval_counterfactual_difficulty` (archived together); no external callers.

## scripts/ (orphan one-off scripts, zero references anywhere)
Each of the following had zero non-self, non-archive hits for its basename
across `*.py`, `*.sh`, `*.yaml`, `*.md` (excluding `scripts/ANALYSIS_INDEX.md`
/ `scripts/README.md`, which enumerate every script by design):

- `build_stuck_curriculum.py`
- `countdown_self_distill.py`
- `decoy_variants_test.py`
- `final_ruler_compare.py`
- `gaming_resample.py`
- `goldset_test.py`
- `inv_unify.py`
- `inv_witness_variance.py`
- `label_hygiene.py`
- `make_paper_figures.py`
- `math_sites_probe.py`
- `meta_oracle_probe.py`
- `meta_quality_gate.py`
- `osd_delta_ab.py`
- `pmi_aggregation_test.py`
- `pmi_ruler_probe.py`
- `prefix_state.py`
- `probe_hygiene_clean.py`
- `probe_monitor.py`
- `prompt_cells_probe.py`
- `ruler_arena.py`
- `ruler_bakeoff.py`
- `timepoint_test.py`
- `upper_bound_probe.py`
- `retired/` (whole directory: `reach_shift_probe.py`, `steerability_probe.py`,
  and its own `README.md`) — already a "retired" holding area with zero
  external references.

## Skipped (do NOT move — kept in place, reported instead)
The following were on the initial candidate list but re-verification found a
live or documentary reference outside the two exempted index files, so per
the audit rule they were left untouched:

- `scripts/cleanup_compute.sh`, `scripts/hf_checkpoint_sync.sh`,
  `scripts/hf_sync_latest.py`, `scripts/poll_h200_and_launch.sh`,
  `scripts/rebuild_eval_node.sh`, `scripts/run_base_sft.sh`,
  `scripts/run_eval.sh`, `scripts/run_eval_1030_eval_node.sh`,
  `scripts/run_eval_1030_trainb_node.sh`, `scripts/run_eval_all.sh`,
  `scripts/smoke_test_rewards.py`, `scripts/sync_strict_sft_and_run_bundle.sh`,
  `scripts/test_eval.py`, `scripts/test_meta_tokens.py`,
  `scripts/test_parsing.py` — each is named in `results/*.md` audit/status
  reports (e.g. `results/cleanup_audit_2026_04_16.md`,
  `results/plan_h200_2node_parallel_2026_04_21.md`,
  `results/status_2026_04_21_session.md`) outside the two exempted index
  files, so they were left in place rather than moved.
- `scripts/steer_prompts.py` — kept; it IS imported live
  (`scripts/t2_seed_advantage.py`, `scripts/countdown_self_distill.py`,
  `scripts/meta_worth_probe.py`, `scripts/prompt_cells_probe.py`).
- `configs/sft_v8_base_matched_strict.yaml`,
  `configs/sft_v8_meta_inside_strict.yaml` — referenced by live scripts
  (`scripts/launch_v8_strict_sft_nodes.sh`,
  `scripts/launch_v8_base_matched_strict_remote.sh`,
  `scripts/launch_v8_meta_inside_strict_remote.sh`), `configs/mainline_contract.yaml`,
  and `docs/mainline_registry_2026_04_13.md` — not referenced solely by files
  in this move, so left in place.
- `configs/accelerate_grpo.yaml` — referenced by `scripts/run_grpo_v2.sh` and
  `scripts/run_base_sft.sh` (neither moved), so left in place.
- `configs/phase1_base_sft.yaml` — referenced by `scripts/run_base_sft.sh`
  (not moved), so left in place.
