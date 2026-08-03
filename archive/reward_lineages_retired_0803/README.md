# Retired meta-reward generations — provenance preserved 2026-08-03

`TRIOBJ_DCPO_V4` accumulated six mutually exclusive meta-reward generations behind
one `algorithm.dcpo_rmeta_source` selector. Only `pmi_shift` is run. The others were
removed from the working tree in a staged cleanup on 2026-08-03; this file keeps the
provenance that was worth more than the code.

Reachability argument, common to all of them: `src/training/verl_sdc.py` assigns
`_rmeta_src` once and dispatches through a single-level `if/elif` chain. Python
evaluates exactly one arm, so with `dcpo_rmeta_source=pmi_shift` every sibling arm
and every knob read inside it is unreachable. The removals are recoverable from git
history and from the launchers under `archive/runs_archive/`.

## Why this file exists

`core/KNOBS.yaml` flagged the dense-PMI block as *adjudicate, not delete*: all eight
knobs were set in the live config with sixteen lines of probe-provenance commentary —
AUC figures, placebo t-statistics, cross-shuffle retention. That commentary is
genuinely valuable and had to be preserved rather than discarded with the keys. It is
reproduced verbatim below.

## Dense PMI (`dcpo_rmeta_source: pmi`) — probe provenance, verbatim

From `configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:192-215` as it stood before removal:

```yaml
  # PMI knobs — frozen from the offline probe (2026-06-11, n=7877 guard-filt):
  # method=mean won (AUC 0.779 overall / 0.696 entangled; placebo t=17.9).
  # clip_gate = p95 of the T=1 mean-delta distribution = 0.342.
  # CAVEAT (cross-shuffle probe): raw mean-delta retains 52% under
  # cross-problem shuffle and placebo retains 86% — most of the raw signal is
  # generic text-presence. Stage 2 MUST ship the placebo-corrected delta
  # (delta' = delta - delta_placebo per row) before launch; these values are
  # the agg/clip baseline that correction builds on.
  dcpo_pmi_agg: mean                # probe-frozen (AUC winner all splits)
  dcpo_pmi_topk_frac: 0.25          # only read for topk_mean (unused)
  dcpo_pmi_clip_token: 2.0          # per-token clip, sum_clip only (unused)
  dcpo_pmi_clip_gate: 0.1085        # probe-frozen p95|delta'| (CORRECTED dist;
                                    # raw was 0.342). E-corr verdict PASS
                                    # 2026-06-12 under the signed shuffle
                                    # criterion: mean_gt0 t=17.9 / signed
                                    # retention -2.43 / corrected AUC_ent
                                    # 0.714. Still RE-FREEZE at the gs50 probe
                                    # (scorer lineage) before stage-2 launch.
  # Cross-shuffle amendment (report 2026-06-11 §4.1): subtract the placebo
  # aggregate per row (third scored arm, ref cost x1.5) so only the CONTENT
  # increment is rewarded. Gated on the corrected probe verdict (E-corr).
  dcpo_pmi_placebo_correct: true
  dcpo_pmi_ngram_n: 8               # C2 overlap guard: word n-gram order
  dcpo_pmi_ngram_threshold: 0.25    # C2 overlap guard: invalidation ratio
```

The single most reusable finding above, for anyone who builds a likelihood-delta
reward again: **the raw signal is mostly text-presence, not content.** Raw mean-delta
retained 52% of its discrimination under a cross-problem shuffle and the contentless
placebo retained 86%. Any future dense-PMI variant must ship the per-row
placebo-corrected delta before launch, and must re-freeze `clip_gate` against its own
scorer lineage rather than inheriting 0.1085.

Dense PMI was superseded by `pmi_shift` (two-position teacher-forcing) on 2026-06-25.
The live config had already recorded the succession: *"The PMI knobs below (dcpo_pmi_*)
are now DORMANT (kept, harmless)."* Accurate about the arithmetic; wrong about the
reader, which is why the block was removed rather than left in place.

## The other retired generations

| source | what it was | launchers |
| --- | --- | --- |
| `asym_cf` | asymmetric counterfactual gate, 2026-06-25: DERAIL penalised 2.5x SAVE, confidence-gated, whole-group centering | `archive/runs_archive/h100std_asymcf_a1.yaml`, `_a2.yaml`, `_v2.yaml` |
| `pmi` | dense likelihood-delta head, probe-frozen 2026-06-11 (n=7877) | see provenance above |
| `cf_group` | group-branch counterfactual with placebo without-arm, 2026-06-21/22 | design note `docs/superpowers/specs/2026-06-21-group-branch-counterfactual-rmeta-design.md` |
| `decoy_did_gm` | Gandhi-style directional contrast + AdaCoT over-emission penalty | `archive/runs_archive/h100std_decoy_gm_dcpo.yaml` |
| `decoy_did_rlsd` | RLSD multiplicative meta factor | `archive/runs_archive/h100std_decoy_gm_rlsd.yaml` |

`cf_group`'s `dcpo_cf_without_mode: placebo` carries one finding worth restating: the
earlier `ban` mode degenerated on the SFT init. Banning the meta-open token produced
empty `<think></think>`, which drove `acc_without` to ~0, which collapsed the
counterfactual delta into "always emit meta". A without-arm that cannot solve
on-distribution does not measure the counterfactual — it measures the ban.

## What survives, and must not be confused with the above

`src/training/dcpo_directional.py` and `src/training/dcpo_pmi.py` remain in the tree.
They are PMI-*named* but partly live: `boxed_answer_string`, `divergent_token_mask` and
`split_first_meta` are called by the `pmi_shift` scorer on every step. Likewise
`_build_pmi_score_batches`, `_dcpo_v4_ref_logprobs`, `_pmi_position_scalar` and
`_meta_body_token_jaccard` in `verl_sdc.py` are shared machinery, not dense-PMI code.
A future cleanup that greps for `pmi` and deletes what it finds will break the live
reward.
