# Pre-RQ3 root docs and plan chain (archived 2026-08-03)

Historical record. Nothing here is current guidance.

## Why each thing is here

`ANALYSIS_MAP.md`, `PLAN.md` — both already carried a line-1 `⚠️ DEPRECATED` banner
pointing at `README.md` / `docs/redesign/` / `docs/CONSTITUTION.md`. Archiving them
matches the filesystem to a deprecation that was already declared in the text.

`REPORT_REFERENCES.md` — the cross-reference index for
`results/study_2026_04_16_metacot_v8_final_report.md`, a V8/H200-generation report. It had
no banner (last updated 2026-04-16); one was added in this commit before it was archived.

`plans/` — 26 docs: the 12-step `plan_pt_rlsd_v5*` chain, the 7 `plan_ctsd_*` docs,
`plan_meta_rod_v53`, two `report_meta_*`, `research_plan.md`, `dcpo_v3_design_and_status.md`,
`brainstorm_metacot_failure_2026_06_07.md` and `intent_and_plan_2026_06_07.md`. Supersession
is established by an explicit chain: each plan names its predecessor, the chain terminates at
`plan_pt_rlsd_v517_2026_05_09_FINAL.md` ("plan iteration 종료"), and the whole PT-RLSD/CTSD
generation is superseded by `docs/redesign/` + `docs/CONSTITUTION.md` +
`docs/EXPERIMENT_PLAN.md`. Only `research_plan.md` had a banner; the other 25 were given one
in this commit.

The intra-chain citations (v511→v510→v59→v58→v57→v56→plan_meta_rod, and v58→report_meta_cot_v57)
are bare basenames and all 26 files landed in the same directory, so those references still
resolve untouched.

## Four planned moves that did NOT happen

`NODE_POLICY.md` stayed at the repo root. `configs/mainline_contract.yaml:4` pins it by name
(`node_policy: "NODE_POLICY.md"`), and `configs/` is frozen while the RL arms are live, so the
reference could not be repaired. The reference is in fact already dead code — its only consumer,
`scripts/verify_mainline_alignment.py`, crashes at line 137 on a missing
`data/verl_train_redirect.parquet` long before `check_docs` runs — but repairing it properly
still requires editing a frozen file. Decide after the arms finish.

`docs/mainline_registry_2026_04_13.md`, `docs/pipeline_stages.md`, `docs/artifact_policy.md`
stayed in `docs/`. They are the same V8 vintage and were listed by this archive's index, but
they are hard-coded in `scripts/verify_mainline_alignment.py:198-200,263`, and `scripts/` is
likewise frozen. Their in/out status was never explicitly ruled on, and a move commit must not
contain a guess.

Because those four stayed put, `ANALYSIS_MAP.md:11-13` and `REPORT_REFERENCES.md:135` still
resolve — they cite repo-root-relative `docs/...` paths that did not change. Only
`ANALYSIS_MAP.md:108` was touched, to say explicitly that `NODE_POLICY.md` is at the repo root
and did not move, since a bare basename now reads as relative to this directory.

## Stale references that could NOT be repaired

`scripts/build_sdc_code_snapshot.sh:19,20` still names `ANALYSIS_MAP.md` and
`REPORT_REFERENCES.md` in its copy list. `scripts/` is frozen, so this was left alone. It
degrades **silently**, not loudly: the loop is guarded by `if [ -e "${path}" ]`, so a rebuilt
code tarball simply omits the two docs with no error. Live arms are unaffected — they bootstrap
from pinned GitHub release asset 490407111, not from the working tree.

`scripts/ANALYSIS_INDEX.md:6` still says "top-level `ANALYSIS_MAP.md`". Same reason, same
frozen directory. It is backticked prose, not a link.

`PLAN.md`'s section numbers are cited from frozen code and configs and cannot be repaired
either: `src/training/meta_inject.py:1` ("PLAN.md §3 H2"), `experiments/probes/a3_inject_causal.py:3`,
`configs/CTSD_NODE_INDEX.md:1`, `configs/archive/verl_ctsd_inject_C_h200_4x4k.yaml:1`,
`runs/archive/h200_ctsd_inject_C_smoke.yaml:1`. All are docstring/comment prose — nothing
executes them.

Likewise `plans/plan_ctsd_E4_selfdistill_rl_2026_06_03.md` and `plans/plan_meta_rod_v53_2026_05_07.md`
are cited by name from `configs/verl_e4_selfdistill_h200_4x4k.yaml:2,101`,
`configs/archive/verl_e4_baseline_h200_4x4k.yaml:2`, `src/training/verl_sdc.py:1503,4672`,
`scripts/package_e4_release.sh:4` and `configs/archive/meta_rod_R8_h100_4x4k.yaml:1` — all frozen,
all comments.
