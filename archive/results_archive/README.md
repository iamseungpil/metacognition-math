# Archived plans and study notes (moved here 2026-08-03; contents archived earlier)

This was `results/archive/`. It moved verbatim so the repo has one archive root instead
of three. Nothing inside changed and nothing was deleted; `git log --follow <file>`
reaches each file's history across the rename.

## Why these are here

26 pre-2026-04-16 planning and study documents, the generation before the V8/H200 work
and well before the RQ3 matched ladder:

- **the experiment-plan chain** — `experiment_plan.md` through `experiment_plan_v5.md`,
  then `plan_metacot_v6_2026_04_05.md` → `v6.1` → `v6.2_final` → `v6.3_final` →
  `v6.4_active_2026_04_06.md`, plus `plan_12gpu_experiments_2026_04_05.md` and
  `experiment_analysis_plan_2026_04_01{,_ko}.md`.
- **phase reports** — `phase0_report.md` through `phase4_final_report.md`, and
  `phase2_3way_report.md`, `phase3_e5_report.md`.
- **study notes and readouts** — `metacot_study.md`, `metacot_study_2026_04_01.md`,
  `autoresearch_behavior_2026_04_01.md`, `autoresearch_config.md`,
  `control_v4_aime_notes_2026_04_01.md`, `control_v5_eval_readout_2026_04_04.md`,
  `benchmark_survey.md`, `session_log_2026_03_31.md`.

This directory was already recognised as terminal archive: `results/cleanup_audit_2026_04_16.md:241`
recorded it as "already the archive path — leave in place". The 0803 move does not
reopen that decision, it only relocates the directory under the single archive root.

## Stale pointers left alone

`archive/docs_pre_rq3/ANALYSIS_MAP.md:72,117`, `docs/mainline_registry_2026_04_13.md:113`,
`results/cleanup_audit_2026_04_16.md:241` and
`legacy/2026_04_20_workspace_cleanup/docs/worktree_cleanup_2026_04_10.md:57` all still say
`results/archive/`. These are historical audit records and a frozen-vintage registry;
rewriting them to match a later layout would falsify what the paths were at the time.
The directory is now `archive/results_archive/`.
