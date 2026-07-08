# RQ2 (partial) — Priming effect: gandhi vs matched-base

**Date**: 2026-07-07. **Status**: PRELIMINARY (single seed). shiftonly arm still training (→ PMI-shift net effect pending).

## What this isolates

- **gandhi** = meta-SFT-2 init + VANILLA_GRPO (correctness only, **no meta reward**). Isolates the SFT *priming* (format/initialization) contribution.
- **matched-base** = base-SFT-2 init + VANILLA_GRPO. Byte-identical except init.
- **priming effect = gandhi − base**. (PMI-shift net effect = shiftonly − gandhi, pending.)

## Grader-fairness fix (prerequisite)

The robust `math_verify` grader scored an answer wrong when the completion had **no `\boxed{}`** wrapper, even when the prose stated the correct answer ("the verified answer is 540."). This penalized **gandhi selectively**: gandhi boxes only ~85% of GSM8K completions vs ~100% for base/pmishift. Uncorrected, gandhi's GSM8K robust accuracy read 77.8% (vs 92.2% runtime), which would have **inflated the PMI-shift net effect** (gandhi artificially low) and even flipped the GSM8K priming sign.

**Fix** (`experiments/analysis/analysis_common.py`, `Grader.grade` / `regrade_frame`): when no `\boxed{}` is present, grade the runtime `answer_extracted` through the SAME math_verify path (wrap in `\boxed{}`). Uniform across arms; arms that always box never reach the branch. Regression: base GSM8K 89.9%→89.9% (unchanged), base MATH500 62.7%→63.1% (negligible); gandhi GSM8K 77.8%→92.8%, MATH500 62.9%→72.1% (legitimate prose answers recovered).

## Priming result (format-fair robust re-grade, avg@8, held-out)

| benchmark | budget | base | gandhi | priming (gandhi−base) | boot p | McNemar p |
|---|---|---|---|---|---|---|
| GSM8K | 4k | 89.9 | 92.8 | +2.9pp | <.001 | .003 |
| GSM8K | 16k | 90.2 | 92.5 | +2.3pp | .001 | .064 |
| MATH500 | 4k | 63.1 | 72.1 | +9.0pp | <.001 | <.001 |
| MATH500 | 16k | 63.3 | 71.5 | +8.2pp | <.001 | <.001 |
| AIME | 4k | 5.0 | 21.2 | +16.2pp | <.001 | .031 |
| AIME | 16k | 4.8 | 20.0 | +15.2pp | <.001 | .062 |

gandhi meta emission: GSM8K ~1%, MATH500 ~14%, AIME ~71% (rises with difficulty).

## Interpretation (no over-claim)

Placed against the **full package** T1 (pmishift − base: GSM8K +4.0, MATH500 +18.8, AIME +14.2):

- **AIME**: priming alone (+16.2) ≥ full package (+14.2) → the PMI-shift reward adds **~0 or slightly negative** on AIME. The AIME win is essentially **all priming** (meta-SFT init), not the reward.
- **MATH500**: priming +9.0 vs package +18.8 → PMI-shift + package-rest adds **~+9.8pp**. Here the reward carries real weight.
- **GSM8K**: both small (priming +2.9, package +4.0); saturated benchmark.

So "PMI-shift drives the win" is **benchmark-dependent**: supported on MATH500, but on AIME the meta-SFT priming is the primary driver. The exact PMI-shift isolation (shiftonly − gandhi) is pending the shiftonly gs300 eval.

## Artifacts

- Eval: `eval/gandhi_1030_v2/` and `eval/base_matched_1030_v2/` on `iamseungpil/metacot-h200-triobj-dcpo-v3`.
- Analysis: `experiments/analysis/paired_eval.py --arms gandhi base` with the format-fair grader.
