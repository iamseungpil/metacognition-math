# RQ3 Autoresearch Log

Date: 2026-04-14
Scope: `results/plan_metacot_v8_active_2026_04_09.md`, `docs/pipeline_stages.md`, `src/curriculum/*`, `scripts/smoke_*`, `tests/test_rq3_pipeline.py`
Goal: Strengthen RQ3 so plan and implementation align around auditable downstream use of structured meta state.
Metric: plan/code alignment, smoke pass, unit test pass
Guard: do not overstate MCTS or value-learning claims; keep RQ3 as side-evidence until RQ2 readout is complete

## Iteration 1

Change:

1. Split RQ3 into `RQ3-A` diagnosis-triggered retrieval retry and `RQ3-B` confidence-bucket selective branching
2. Added explicit RQ3 artifact contract to the active plan
3. Added `src/curriculum/rq3_pipeline.py` to orchestrate `root -> trigger -> intervention -> outcome`
4. Added `scripts/smoke_rq3_pipeline.py`
5. Added `tests/test_rq3_pipeline.py`

Verify:

1. `python scripts/smoke_rq3_pipeline.py`
2. `python -m py_compile src/curriculum/rq3_pipeline.py scripts/smoke_rq3_pipeline.py`
3. `pytest -q tests/test_control_rag.py tests/test_rq3_pipeline.py`

Result:

1. smoke: pass
2. py_compile: pass
3. tests: `2 passed`

Decision:

Keep.

Learned:

1. The biggest RQ3 gap was not prompt logic but missing orchestration and controls.
2. `plain_retry` control must remain first-class, otherwise retrieval gain is uninterpretable.
3. `MCTS-lite` is best kept as a selective branching side branch, not described as learned search.

## Next Iteration Candidate

1. Add dataset-backed RQ3 side-evidence evaluation script over saved root completions
2. Add per-case metrics: false-trigger rate, retrieved-example relevance labels, branch wasted-expansion rate
3. Resume RL mainline only after launcher/contract/runtime alignment is repaired

## Iteration 2

Change:

1. Added explicit `plain_retry` control and `winner` label to `RQ3CaseTrace`
2. Aligned provenance labels to allowed evidence classes (`side_evidence`)
3. Renamed trigger aggregates to avoid mislabeled “precision” claims
4. Added `retrieval_retry vs plain_retry` and `best_branch vs retrieval_retry` comparison metrics
5. Added offline batch entrypoint: `scripts/run_rq3_side_eval.py`

Verify:

1. `pytest -q tests/test_control_rag.py tests/test_rq3_pipeline.py`
2. `python scripts/smoke_rq3_pipeline.py`
3. `python scripts/run_rq3_side_eval.py --cases /tmp/rq3_cases.json --output_dir results/rq3_side_eval_smoke_rel`
4. `python -m py_compile src/curriculum/rq3_pipeline.py scripts/smoke_rq3_pipeline.py scripts/run_rq3_side_eval.py`

Result:

1. tests: `2 passed`
2. smoke: pass
3. offline side-eval: traces + summary emitted

Decision:

Keep.

Learned:

1. The real missing control was explicit `plain_retry`, not more search logic.
2. `winner` and provenance labels matter for keeping RQ3 interpretable.
3. Offline side-eval without an example bank is still useful: it exposes that curriculum and branching gains should fall to zero instead of silently inventing gains.

## Iteration 3

Change:

1. Strengthened the active plan's `RQ3-D. Epistemic Self-Distillation` section while preserving the overall document structure
2. Added a new normalization and dataset-building layer: `src/training/self_distill_data.py`
3. Added explicit `naive` and `epistemic` self-distill dataset builders via `scripts/build_self_distill_dataset.py`
4. Added self-distill-specific eval comparison script: `scripts/analyze_self_distill_eval.py`
5. Added smoke and unit coverage:
   - `scripts/smoke_self_distill_dataset.py`
   - `tests/test_self_distill_data.py`
6. Hardened `src/training/sft.py` so `messages` may be either JSON strings or already-materialized lists
7. Saved the converged design review at `results/codex_reviews/self_distill_design_review_2026_04_14.md`

Verify:

1. `python scripts/smoke_self_distill_dataset.py`
2. `pytest -q tests/test_self_distill_data.py`
3. `python -m py_compile src/training/self_distill_data.py src/training/sft.py scripts/build_self_distill_dataset.py scripts/smoke_self_distill_dataset.py scripts/analyze_self_distill_eval.py tests/test_self_distill_data.py`
4. End-to-end smoke:
   - build naive dataset from synthetic RQ3-case JSON
   - build epistemic dataset from synthetic RQ3-case JSON
   - compare synthetic eval bundles with `scripts/analyze_self_distill_eval.py`

Result:

1. smoke dataset build: pass
2. unit tests: `4 passed`
3. `py_compile`: pass
4. end-to-end synthetic build + analysis: pass
5. one environment limitation surfaced: a direct tokenizer-backed `prepare_sft_dataset` smoke could not run in the default shell because `transformers` is missing there

Decision:

Keep.

Learned:

1. The right first implementation is an additive offline branch with a stable teacher-trace IR, not an immediate SDPO port.
2. The most important silent mismatch to avoid is prompt-distribution drift: evaluation uses a single user prompt, so D2 must preserve the natural control-v5 output style rather than inventing schema-heavy prompt turns.
3. The analysis stack was already strong enough for collapse checks; the missing piece was self-distill-specific data normalization and artifact generation.

## Iteration 4

Change:

1. Fixed a correctness bug in `src/training/self_distill_data.py` so incorrect `root_completion` fallback is no longer admitted as a teacher trace unless `root_judgment.is_correct` is explicitly true
2. Added a fixed empty-data schema for self-distill builds and made `scripts/build_self_distill_dataset.py` fail fast on 0-row outputs
3. Hardened `src/training/sft.py`:
   - drops all-masked rows after truncation
   - raises if no trainable rows remain
   - handles tiny datasets without invalid `train_test_split`
   - delays `transformers`/`torch` dependency requirements so dataset preparation stays unit-testable
4. Tightened `scripts/analyze_self_distill_eval.py` JSON contract handling to reject malformed dict payloads
5. Expanded test coverage in `tests/test_self_distill_data.py` for:
   - incorrect-root exclusion
   - empty-schema preservation
   - JSON contract rejection
   - `prepare_sft_dataset` all-masked filtering

Verify:

1. `python scripts/smoke_self_distill_dataset.py`
2. `pytest -q tests/test_self_distill_data.py`
3. `python -m py_compile src/training/self_distill_data.py src/training/sft.py scripts/build_self_distill_dataset.py scripts/smoke_self_distill_dataset.py scripts/analyze_self_distill_eval.py tests/test_self_distill_data.py`

Result:

1. smoke: pass
2. tests: `9 passed`
3. `py_compile`: pass

Decision:

Keep.

Learned:

1. The most dangerous early bug was not style drift but teacher-label corruption: allowing wrong root traces into the distill set would have invalidated D1 immediately.
2. Empty artifact contracts need to fail at build time, not later inside SFT.
3. Making dataset prep testable without heavyweight runtime dependencies materially improves iteration speed for this lane.

## Iteration 5

Change:

1. tightened `src/curriculum/control_rag.py` retrieval scoring with:
   - `typed_strategy_bonus`
   - `generic_penalty`
   - narrower family matching for `other`
2. fixed `load_example_bank()` so JSON banks saved from `ExampleRecord.to_dict()` preserve nested metadata on reload
3. added `scripts/build_control_rag_seed_library.py` to build a reusable typed `stable_seed_library` from older correct eval artifacts
4. refined family classification:
   - `geometric sequence` now maps to `exponential_growth`
   - bare `power` no longer forces `geometry`
5. expanded `tests/test_control_rag.py` to cover:
   - generic-easy fallback suppression
   - metadata round-trip preservation
   - family-classification edge cases

Verify:

1. `pytest -q tests/test_control_rag.py tests/test_rq3_pipeline.py tests/test_self_distill_data.py`
2. `python -m py_compile src/curriculum/control_rag.py scripts/audit_control_rag_real.py scripts/build_control_rag_seed_library.py tests/test_control_rag.py`
3. `python scripts/build_control_rag_seed_library.py --inputs results/eval_v8_E20a/eval_v8_meta_inside_E20a.json results/eval_v8_E20b/eval_v8_meta_inside_E20b_5ep.json --output data/control_rag_seed_library.json --require_correct --require_study_need`
4. `python scripts/audit_control_rag_real.py --bank_paths data/control_rag_seed_library.json --output_json results/control_rag_real_audit_with_seed.json`

Result:

1. tests: `11 passed`
2. seed library build: `227` typed exemplars written
3. real-artifact audit with seed library:
   - mean top-1 typed-strategy bonus `0.935`
   - `exponential_growth` top-1 family match improved to `3/7`
   - `exponential_growth` top-1 positive study-need overlap improved to `2/7`
4. remaining weak point: `probability_counting` still has many top-1 hits without specific study-need overlap, so coverage/typing of that seed family is still insufficient

Decision:

Keep.

Learned:

1. the main RQ3 blocker had shifted from pure scorer logic to bank construction and metadata fidelity
2. without nested-metadata round-trip, both stable-seed reuse and future dynamic-success replay would silently degrade
3. `exponential_growth` benefited immediately from even a very small typed seed library, which supports the `stable_seed_library + dynamic_success_library` design rather than more score hacking

## Iteration 6

Change:

1. added `build_dynamic_library_from_trace_dicts()` in `src/curriculum/rq3_pipeline.py` so saved JSON/JSONL trace artifacts can be turned back into a reusable dynamic bank without reconstructing dataclasses
2. added `scripts/build_control_rag_dynamic_library.py` to materialize a `dynamic_success_library` from saved RQ3 traces
3. hardened `scripts/run_rq3_side_eval.py` so it can read retry completions from either:
   - flat fields such as `curriculum_retry_completion`
   - nested trace-bundle fields such as `curriculum_retry.retry_completion`
4. extended `run_rq3_side_eval.py` manifest output with `bank_summary` so stable/dynamic bank provenance is visible in machine-readable form
5. updated `docs/pipeline_stages.md` with the concrete smoke sequence for:
   - stable seed build
   - retrieval smoke
   - RQ3 smoke
   - side eval
   - dynamic library build

Verify:

1. `pytest -q tests/test_control_rag.py tests/test_rq3_pipeline.py tests/test_self_distill_data.py`
2. `python scripts/smoke_control_rag.py --skip-model --output /tmp/control_rag_smoke_stage2.json`
3. `python scripts/smoke_rq3_pipeline.py --output /tmp/rq3_pipeline_smoke_stage2.json`
4. `python scripts/build_control_rag_seed_library.py --inputs results/eval_v8_E20a/eval_v8_meta_inside_E20a.json results/eval_v8_E20b/eval_v8_meta_inside_E20b_5ep.json --output /tmp/control_rag_seed_library_stage2.json --require_correct --require_study_need`
5. `python scripts/run_rq3_side_eval.py --cases /tmp/rq3_pipeline_smoke_stage2.json --example_bank /tmp/control_rag_seed_library_stage2.json --output_dir /tmp/rq3_side_eval_stage2b`
6. `python scripts/build_control_rag_dynamic_library.py --traces /tmp/rq3_side_eval_stage2b/rq3_traces.jsonl --output /tmp/control_rag_dynamic_library_stage2b.json`

Result:

1. tests: `11 passed`
2. retrieval smoke: pass
3. RQ3 pipeline smoke: pass
4. side-eval manifest now records stable-bank role counts
5. dynamic library build from saved side-eval traces wrote `1` reusable `retrieval_retry` example in the smoke flow

Decision:

Keep.

Learned:

1. the missing glue was not in retrieval scoring anymore but in artifact reusability
2. making trace bundles re-playable through `run_rq3_side_eval.py` removes an avoidable source of schema drift between offline audit and later library construction
3. RQ3 now has a documented and smoke-tested `stable_seed_library -> side_eval -> dynamic_success_library` path
