# Strict Pair Behavior Report

Date: 2026-04-11

Scope:

1. claim-bearing strict paired SFT comparison
2. behavior / epistemic / AIME analysis from remote original eval artifacts
3. interpretation against `RQ1` in `results/plan_metacot_v8_active_2026_04_09.md`

Inputs:

1. [meta_strict_behavior.json](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_11/meta_strict_behavior.json)
2. [base_strict_behavior.json](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_11/base_strict_behavior.json)
3. [meta_strict_aime_examples.txt](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_11/meta_strict_aime_examples.txt)
4. [plan_metacot_v8_active_2026_04_09.md](/home/v-seungplee/metacognition/results/plan_metacot_v8_active_2026_04_09.md)

## 1. Executive Summary

The strict paired SFT comparison is effectively tied on accuracy.

- Meta strict SFT: `1176 / 1560 = 75.38%`
- Base matched strict SFT: `1178 / 1560 = 75.51%`

However, the behavior analysis does not support the claim that "nothing changed."
The Meta-CoT model learned a strong confidence-conditioned controller:

1. `high confidence -> verify`
2. `low confidence -> redirect`
3. explicit diagnosis appears much more often than in the base model

What did **not** happen is conversion of that controller signal into reliable route repair.
The model knows when a route is weak, but usually emits only one meta intervention and rarely executes a multi-step recovery.

## 2. Historical Structure Check

The "single meta block" pattern is not new in the strict rerun.

Historical eval:

- `eval_v8_meta_inside_E20a.parquet`
  - `0 blocks`: 5
  - `1 block`: 1522
  - `2 blocks`: 3

Current strict eval:

- `eval_v8_meta_inside_strict_sft.parquet`
  - `0 blocks`: 1
  - `1 block`: 1559

Training data contained some two-step traces, but only as a small minority:

- `v8_meta_inside_think.parquet`
  - `1 meta`: 6083
  - `2 meta`: 246
  - `>1 meta rate`: `3.89%`
- `v8_meta_inside_strict.parquet`
  - `1 meta`: 4055
  - `2 meta`: 209
  - `>1 meta rate`: `4.90%`

Interpretation:

1. the model has long been collapsing to a single intervention at inference time
2. strict data cleanup did not create this issue; it exposed it more clearly
3. current Meta-CoT is better described as a `single-step router` than a `multi-step reviser`

## 3. Strict Paired Accuracy

Per benchmark:

| Model | GSM8K | MATH500 | AIME2024 | Overall |
|---|---:|---:|---:|---:|
| Meta strict | 88.54% | 51.8% | 16.7% | 75.38% |
| Base strict | 88.54% | 51.6% | 26.7% | 75.51% |

Interpretation:

1. GSM8K is exactly tied
2. Meta is only `+1` on MATH500
3. Base is `+3` on AIME, but AIME has only 30 problems
4. overall difference is `+2 / 1560`, which is not meaningful evidence of superiority either way

## 4. RQ1 Behavior Findings

### 4.1 Meta Emission

Meta strict:

- meta emission rate: `99.94%`
- average meta blocks: `0.999`
- almost every answer contains exactly one meta block

Base strict:

- meta emission rate: `0%`

This is enough to say the strict meta SFT did learn the formatting and controller slot.

### 4.2 Confidence-Conditioned Routing

Meta strict:

- average confidence: `0.625`
- ECE: `0.130`
- wrong-high-confidence rate at `conf >= 0.7`: `8.65%`
- wrong-high-confidence rate at `conf >= 0.8`: `3.59%`
- low-confidence redirect rate (`conf <= 0.5`): `89.68%`
- high-confidence verify rate (`conf >= 0.7`): `100%`

This is the strongest positive result in the strict run.
The learned policy is not random text insertion; action choice is strongly tied to reported confidence.

### 4.3 Difficulty-Conditioned Behavior

Meta strict by benchmark:

- GSM8K
  - verify: `99.32%`
  - redirect: `1.55%`
  - diagnosis: `0.29%`
  - epistemic: `0.10%`
  - avg confidence: `0.764`
- MATH500
  - verify: `33.4%`
  - redirect: `76.8%`
  - diagnosis: `14.2%`
  - epistemic: `3.2%`
  - avg confidence: `0.360`
- AIME2024
  - verify: `30.0%`
  - redirect: `96.7%`
  - diagnosis: `30.0%`
  - epistemic: `6.7%`
  - avg confidence: `0.240`

Hard subset (`math500 + aime2024`):

- accuracy: `49.81%`
- verify: `33.21%`
- redirect: `77.92%`
- diagnosis: `15.09%`
- epistemic: `3.40%`
- avg confidence: `0.353`

Interpretation:

1. on easy problems the model behaves like `high-conf verify`
2. on hard problems the model behaves like `low-conf redirect`
3. diagnosis and epistemic language do increase on hard problems, but remain much weaker than redirect

### 4.4 Marker Presence vs Accuracy

Meta strict overall:

- verify present: `1199` samples, `84.15%` accuracy
- verify absent: `361` samples, `46.26%` accuracy
- redirect present: `429` samples, `45.69%` accuracy
- redirect absent: `1131` samples, `86.65%` accuracy
- diagnosis present: `83` samples, `42.17%` accuracy
- diagnosis absent: `1477` samples, `77.25%` accuracy
- epistemic present: `19` samples, `31.58%` accuracy
- epistemic absent: `1541` samples, `75.92%` accuracy

Interpretation:

1. these raw deltas are mostly selection effects
2. verify is triggered on easier/high-confidence regions
3. redirect, diagnosis, and epistemic markers are triggered on genuinely harder failures
4. therefore lower accuracy under redirect is not evidence that redirect is harmful

Still, one point matters:

- hard subset verify-present accuracy is `57.39%`
- hard subset verify-absent accuracy is `46.05%`

So verify is at least plausibly useful on the hard subset, while redirect remains underpowered.

## 5. Epistemic Interpretation

The current strict model is **not** yet strongly epistemic in the deeper sense.

Evidence:

1. epistemic marker rate is only `3.40%` on the hard subset
2. diagnosis rate is only `15.09%` on the hard subset
3. almost all answers have exactly one meta block
4. confidence bins show a near-binary split:
   - `0.0-0.3`: 187 samples, redirect `97.86%`, accuracy `32.62%`
   - `0.3-0.5`: 278 samples, redirect `84.17%`, accuracy `56.83%`
   - `0.7-0.9`: 1087 samples, verify `100%`, accuracy `87.58%`

This means confidence is being used mostly as a routing switch, not as a gradual epistemic state that supports multiple revisions.

Current status:

1. `confidence-conditioned controller`: yes
2. `multi-step epistemic revision`: weak
3. `diagnosis-rich control`: weak to moderate

## 6. AIME Qualitative Pattern

Strict AIME summary:

- `5 / 30` correct
- average confidence: `0.240`
- all 30 responses use exactly one meta block
- redirect rate: `96.7%`
- diagnosis rate: `30.0%`
- epistemic rate: `6.7%`

Representative pattern from [meta_strict_aime_examples.txt](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_11/meta_strict_aime_examples.txt):

1. correct cases often say the first route is weak, name the missing structure, then switch once to a cleaner formulation
2. wrong cases usually do the same rhetorical move, but then proceed into a long single continuation without another checkpoint
3. many wrong cases still contain good local diagnosis text such as:
   - "What is missing is ..."
   - "I should stop ... and switch to ..."
4. but the post-meta continuation frequently remains a single uninterrupted solve attempt rather than:
   - diagnosis
   - subgoal split
   - second check
   - revised confidence
   - true route replacement

Interpretation:

1. AIME behavior is no longer mere blind continuation
2. but it is still mostly `low-confidence self-redirection`
3. it is not yet `diagnosis -> decomposition -> strategy replacement -> verification`

This aligns with the older note in [control_v4_aime_notes_2026_04_01.md](/home/v-seungplee/metacognition/results/archive/control_v4_aime_notes_2026_04_01.md): the model lowers confidence and interrupts itself, but usually does not complete a deeper repair loop.

## 7. Comparison Against Base Strict

Base strict has nearly identical accuracy but much weaker control behavior.

Hard subset comparison:

- Meta strict
  - verify rate: `33.21%`
  - redirect rate: `77.92%`
  - diagnosis rate: `15.09%`
  - epistemic rate: `3.40%`
- Base strict
  - verify rate: `15.09%`
  - redirect rate: `12.83%`
  - diagnosis rate: `3.58%`
  - epistemic rate: `1.70%`

Base strict emits occasional verify / redirect words, but they look more like residual phrasing than a controller.

This means:

1. Meta-CoT behavior is genuinely present
2. the failure is not "no controller learned"
3. the failure is "controller learned, but too shallow to improve end accuracy"

## 8. What Is Still Missing

The following are still missing for a full RQ1 closeout:

1. strict mainline entropy run for the strict meta checkpoint
2. a strict provenance behavior bundle saved from the current pair, not the older `eval_1030_v5` directory
3. explicit subgoal / backward-chaining extraction on the strict pair
4. paired example-level comparison (`meta_only`, `base_only`) for targeted qualitative audit

Important provenance note:

- existing entropy files in `results/entropy_analysis` and `results/entropy_v8_E20a` are not current strict-pair evidence
- they should be treated as historical / side evidence only

## 9. RQ1 Judgment

Current claim status:

Safe to claim:

1. strict Meta-CoT SFT learns a high-rate meta controller slot
2. strict Meta-CoT behavior is strongly conditioned on confidence
3. the model reliably maps low confidence to redirect and high confidence to verify
4. hard problems trigger substantially more redirect and diagnosis behavior than easy problems

Not yet safe to claim:

1. Meta-CoT SFT improves accuracy over a fair base matched strict baseline
2. the current controller performs strong multi-step repair
3. the current controller is sufficiently diagnosis-rich for curriculum / retrieval triggers

Best current interpretation:

1. `RQ1 partially supported`
2. the supported part is `controller acquisition`
3. the unsupported part is `controller depth / repair effectiveness`

## 10. Immediate Next Steps

Recommended next sequence:

1. run strict entropy analysis with the strict meta checkpoint and strict eval parquet
2. add a strict paired behavior-bundle runner so the current pair produces machine-readable behavior reports by default
3. move RL reward focus away from "meta phrase present" and toward:
   - wrong-high-confidence reduction
   - low-confidence redirect with actual correction
   - confidence revision across multiple meta steps
   - diagnosis / next-strategy quality
4. keep curriculum / RAG / search behind the RQ2 gate until diagnosis quality is stronger

Reward implication:

1. `E21 / E21R` are still directionally sensible
2. but the next useful branch is the one that breaks single-step collapse
3. the current evidence points toward stepwise confidence-revision reward rather than more single-block style reward
