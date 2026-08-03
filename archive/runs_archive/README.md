# Archived amlt launchers (moved here 2026-08-03; contents archived earlier)

This was `runs/archive/`. It moved verbatim so the repo has one archive root instead
of three. Nothing inside changed, and nothing was deleted — `git log --follow <file>`
still reaches each file's full history across the rename.

## Why these are here

75 retired amlt launchers, grouped by the line that produced them:

- **ROD / OPD / RLSD / GDPO** (`h100_meta_rod_*`, `h200_meta_opd_R7_0506.yaml`,
  `h100_rod_pt_R10_0507.yaml`, `h200_rod_pt_R10_v2_verl.yaml`) — the meta-ROD and
  PT-RLSD generation, whose plan chain is archived at `archive/docs_pre_rq3/plans/`.
- **e4–e9** (`h200_e4_*` … `h200_e9_*`) — the CTSD self-distill / steering / BCI-inject
  ladder.
- **triobj DCPO v2–v4** (`h200_triobj_*`, `h100std_triobj_dcpo_v3*`,
  `h100std_triobj_dcpo_v4_*`) — the intermediate stages superseded by the RQ3 ladder.
- **probes and one-off evals** (`h100std_decoy_*`, `h100std_asymcf_*`,
  `h100std_weight_soup.yaml`, `h100std_placement_probe.yaml`, `h100std_decode_sweep.yaml`,
  `h100std_*_1030_eval.yaml`, `h100std_rv_*_eval*.yaml`, `h100std_passk_headroom.yaml`).
- **early metacognition A100 launchers** (`metacognition_*.yaml`, `node_recovery_0415.yaml`,
  `h200_2nodes_0421.yaml`).

Their hydra configs remain in `configs/archive/`, which did not move: `configs/` is
frozen while the RL arms are live.

## Referenced from elsewhere, deliberately not rewritten

`docs/reports/2026-07-17-rq3-run-and-iteration-log.md:1141` cites
`runs/archive/h100std_basearm_1030_eval.yaml` as the launcher that produced the B0
baseline `eval/base_matched_1030_v2`. That run log is append-only history and was left
unedited; the file is now at `archive/runs_archive/h100std_basearm_1030_eval.yaml`.

`h200_ctsd_inject_C_smoke.yaml:1` cites `PLAN.md`, now at
`archive/docs_pre_rq3/PLAN.md`. That citation is a comment, not a link.
