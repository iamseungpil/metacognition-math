# Pre-RQ3 reports (archived 2026-08-03)

13 reports from the CTSD / DCPO / confidence-RV instruct generation, superseded by the RQ3
matched-base ladder (`docs/PREREGISTRATION_rq3v2_base_replication.md`, `docs/EXPERIMENT_PLAN.md`).
None of them carried a deprecation banner; all were given one in this commit.

## Why each thing is here

`2026-06-05-ctsd-calibration-experiments-report.md` — CTSD calibration battery.
`2026-06-11-dcpo-v3-v4-study.{md,pdf,tex}` + `-s3b.pdf` — the DCPO v3/v4 study and its S3B
appendix, kept as a set so the `.md` source and its built artifacts stay together.
`2026-06-18-redirect-priming-progress.md` — redirect-priming progress note.
`2026-06-20-confidence-rv-stage0-study.md` — RV Stage-0 yield study.
`2026-06-22-confidence-rv-metacognition-study.{md,pdf,tex}` — the confidence-RV metacognition study.
`2026-06-24-confidence-rv-directional-self-distill-study.md` — directional self-distillation study.
`2026-06-29-degeneration-cause-diagnosis.md` — the forensics D1–D6 + V1–V2 synthesis on
non-termination. Its prescription (overlong shaping + Clip-Higher) has since been absorbed
into the current recipe.
`2026-06-29-matched-base-implementation-readiness.md` — the CPU-only GO/NO-GO gate report that
authorised the matched-base build. Its verdict has been acted on and the launch it gated has
happened, so it is a completed gate record. It remains the provenance document for how the
meta-removed twin was constructed confound-free.

## figures/ moved with the reports — deliberately

All 14 PNGs in `figures/` were verified to be referenced **only** by the four reports in this
directory (`2026-06-11-dcpo-v3-v4-study.md`, `2026-06-20-confidence-rv-stage0-study.md`,
`2026-06-22-confidence-rv-metacognition-study.md`, `2026-06-24-confidence-rv-directional-self-distill-study.md`).
No report that stayed in `docs/reports/` uses them. The two `figures/` hits in
`docs/reports/2026-07-17-rq3-run-and-iteration-log-part1.md:1900,2184` are the **`paper`
submodule's own** `figures/` directory, not this one.

Moving the directory alongside the reports was therefore chosen over rewriting paths: every
`![...](figures/*.png)` link is relative and keeps resolving with zero edits.

## One inbound reference that CANNOT be repaired

`docs/superpowers/specs/2026-06-11-dcpo-v4-likelihood-rmeta-design.md:86` reads:

    Findings (report `docs/reports/2026-06-11-dcpo-v3-v4-study.md`, H5–H7): raw Δ

`docs/superpowers/specs/` is on the never-move/never-edit list, so this citation now dangles
and no `sed` is permitted. It is backticked prose, not a clickable link. The report is at
`archive/reports_pre_rq3/2026-06-11-dcpo-v3-v4-study.md`. Recorded here rather than fixed.

Two cross-group citations WERE repaired, at their new paths:
`2026-06-05-...:137` → `archive/docs_pre_rq3/plans/intent_and_plan_2026_06_07.md`, and
`2026-06-11-dcpo-v3-v4-study.md:4` → `archive/docs_pre_rq3/plans/dcpo_v3_design_and_status.md`.
