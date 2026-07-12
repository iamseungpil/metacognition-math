# RQ2 (instruct) — Isolated PMI-shift net effect: shiftonly vs gandhi

**Date**: 2026-07-08. **Status**: instruct substrate (secondary/appendix in the base redesign), single seed, format-fair robust re-grade.

## What this isolates (the correct twin)

Both arms start from the SAME meta-SFT init (`v8_rv_functional_sft`) and share
byte-identical non-reward hyperparameters. They differ ONLY in the RL reward:

- **gandhi (B2-analog)** = meta-SFT + VANILLA_GRPO (correctness only).
- **shiftonly (B3-analog)** = meta-SFT + correctness + **PMI-shift only** (all other
  heads zeroed: cal/format/emit/len_cost/over = 0, cf_group split removed).

So `shiftonly − gandhi` = the **isolated net effect of the PMI-shift reward**,
with meta-SFT priming held constant. This is the clean twin comparison the earlier
`pmishift − base` (package vs base) and `gandhi − base` (priming) rungs could not
give.

## Result (held-out 1030, avg@8, format-fair math_verify re-grade, paired bootstrap)

Sign convention: PMI-shift net = shiftonly − gandhi.

| benchmark | budget | gandhi | shiftonly | PMI-shift net | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|
| GSM8K | 4k | 92.8 | 92.8 | ~0 | .97 | 1.0 |
| GSM8K | 16k | 92.5 | 93.3 | +0.8 | .15 | .42 |
| **MATH500** | 4k | 72.1 | 78.0 | **+5.9** | **<.001** | **<.001** |
| **MATH500** | 16k | 71.5 | 77.1 | **+5.6** | **<.001** | **.001** |
| AIME | 4k | 21.2 | 14.2 | −7.1 | <.001 | .25 (n=30) |
| AIME | 16k | 20.0 | 14.2 | −5.8 | .002 | 1.0 (n=30) |

## Interpretation

1. **PMI-shift has a real, significant net positive on solvable-hard problems**:
   MATH500 +5.6–5.9pp, bootstrap p<.001 and McNemar p<.001. This is the key
   finding — with priming removed, the PMI-shift REWARD alone adds ~+6pp on
   MATH500. It refutes the "the win is just the package/priming, not PMI-shift"
   concern by isolation.
2. **PMI-shift hurts on capability-bound AIME** (−5.8 to −7.1pp, bootstrap
   significant; McNemar n.s. at n=30). Mechanism (per-cell): shiftonly meta
   emission ~98.7%, AIME mean tokens 3065 vs gandhi 2467, truncation 38% vs 30%
   — PMI-shift inflates meta/length, and on problems the model largely cannot
   solve this extra length manifests as non-termination that lowers accuracy.
3. **GSM8K saturated** (~0, n.s.).

## Bearing on the base redesign

This validates the base experiment: there IS a real, significant PMI-shift effect
(+6pp on MATH500) worth reproducing cleanly on the Qwen3-8B-Base substrate. The
AIME negative is a length/termination side-effect (a gen-2 anti-degeneration
target), not a refutation of the mechanism. The base ladder (B0/B2/B3) reproduces
exactly this twin structure: PMI-shift net = B3 − B2.

## Artifacts

- Eval: `eval/shiftonly_1030_v2/` and `eval/gandhi_1030_v2/` on `iamseungpil/metacot-h200-triobj-dcpo-v3`.
- Analysis: `experiments/analysis/paired_eval.py --arms shiftonly gandhi` (format-fair grader).
