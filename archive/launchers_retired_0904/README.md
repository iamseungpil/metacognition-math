# Launchers retired 2026-09-04

Move-only commit. Every root-level `h100std_*.yaml` amlt launcher EXCEPT the
six current-arm launchers was moved here with `git mv` (no content edits).

Kept at repo root (still live):
- `h100std_sft_b0p2_rvfull.yaml`
- `h100std_sft_b2p2_rvfull.yaml`
- `h100std_sft_b2p3_vunmask.yaml`
- `h100std_rq3v2f_b0p.yaml`
- `h100std_rq3v2f_b2p.yaml`
- `h100std_rq3v2f_b3p.yaml`
- `countdown_rl_6arm.yaml` (not a `h100std_*` launcher, unaffected by this move)

Moved here (27 files) — retired evals / superseded arm variants, no longer
part of the current RQ3v2f/RQ3v2g run:
- `h100std_b2p3init_1030_eval.yaml`
- `h100std_b2p4init_gs0_eval.yaml`
- `h100std_e190_cf_eval.yaml`
- `h100std_rq3v2f_b2p3v.yaml`
- `h100std_rq3v2f_b2p3v_gs100_eval.yaml`
- `h100std_rq3v2f_b2p3v_gs50_eval.yaml`
- `h100std_rq3v2f_b2p_1030_eval.yaml`
- `h100std_rq3v2f_b3nopmi.yaml`
- `h100std_rq3v2f_b3nopmi_1030_eval.yaml`
- `h100std_rq3v2f_b3null.yaml`
- `h100std_rq3v2f_b3null_1030_eval.yaml`
- `h100std_rq3v2f_b3p2.yaml`
- `h100std_rq3v2f_b3p3.yaml`
- `h100std_rq3v2f_b3p3g.yaml`
- `h100std_rq3v2f_b3p3g_gs50_eval.yaml`
- `h100std_rq3v2f_b3s.yaml`
- `h100std_rq3v2f_b3sh.yaml`
- `h100std_rq3v2f_b3sh_1030_eval.yaml`
- `h100std_rq3v2f_b3shf.yaml`
- `h100std_rq3v2f_pair_1030_eval.yaml`
- `h100std_rq3v2f_rq2_1030_eval.yaml`
- `h100std_rq3v2g_b4_gs50_eval.yaml`
- `h100std_rq3v2g_b4p2.yaml`
- `h100std_rq3v2g_b4p3g.yaml`
- `h100std_rq3v2g_b4v.yaml`
- `h100std_sft2init_1030_eval.yaml`
- `h100std_sft_b2p4_vclean.yaml`

`tests/test_launcher_yaml_lint.py` globs `ROOT.glob("*.yaml")` for every
root-level launcher declaring a `jobs:` list; after this move it lints the
7 remaining root launchers (the 6 current arms + `countdown_rl_6arm.yaml`)
instead of the 34 that existed before. The suite still passes because the
glob is dynamic and `test_there_are_launchers_to_lint` only requires the set
to be non-empty.
