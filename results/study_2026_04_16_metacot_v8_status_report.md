# Meta-CoT V8 Status Report

**Author**: Seungpil Lee  
**Date**: 2026-04-16  
**Project**: `metacognition`  
**Scope**: strict paired SFT, entropy audit, step-300 RL analysis, RQ3 self-distill mainline reset

## Executive Summary

이번 업데이트의 결론은 단순하다. Meta-CoT는 strict SFT 단계에서 구조화된 controller를 매우 안정적으로 학습했지만, 현재까지는 OOD 이득으로 이어지지 않았다. RL step-300은 전체 정확도를 끌어올렸지만, 그 이득은 metacognitive control 개선보다는 긴 추론 증가에서 나왔고, hard OOD인 AIME에서는 오히려 크게 악화되었다.

RQ3 mainline은 therefore RL-first가 아니라 self-distill-first로 재정렬해야 한다. 현재 claim-bearing path는 `strict SFT -> fixed_k_repair -> reward-ranked teacher selection -> CE/SFT readout -> optional control-span KL`이다. Retrieval은 코드상 지원되지만 example bank가 없으면 실제로는 꺼져 있으므로, 앞으로는 retrieval 사용 여부를 artifact에 명시적으로 기록해야 한다.

## 1. Verified Fact Base

| ID | Claim | Source |
|---|---|---|
| F1 | base strict SFT accuracy is 0.7551 on 1560 problems | [strict_self_distill_eval_compare.remote.json](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_15/strict_self_distill_eval_compare.remote.json) |
| F2 | meta strict SFT accuracy is 0.7538 and meta emission is 0.9994 | [strict_self_distill_eval_compare.remote.json](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_15/strict_self_distill_eval_compare.remote.json) |
| F3 | strict SFT OOD accuracy is 0.2667 for base and 0.1667 for meta | [strict_self_distill_eval_compare.remote.json](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_15/strict_self_distill_eval_compare.remote.json) |
| F4 | strict meta SFT ECE is 0.1298 and study-need rate is 0.3013 | [strict_self_distill_eval_compare.remote.json](/home/v-seungplee/metacognition/results/strict_pair_analysis_2026_04_15/strict_self_distill_eval_compare.remote.json) |
| F5 | entropy after meta increases by +0.2999 nats on average | [entropy_stats.json](/home/v-seungplee/metacognition/results/entropy_strict_meta/entropy_stats.json) |
| F6 | step-300 E21R-v2 accuracy is 0.7981 vs base 0.7592 | [full_analysis.json](/home/v-seungplee/metacognition/results/step300_deep_analysis/full_analysis.json) |
| F7 | step-300 AIME accuracy is 0.1333 for E21R-v2 vs 0.3333 for base | [full_analysis.json](/home/v-seungplee/metacognition/results/step300_deep_analysis/full_analysis.json) |
| F8 | step-300 meta wrapping rate is 0.0, while free-text confidence appears in 89.13% of outputs | [full_analysis.json](/home/v-seungplee/metacognition/results/step300_deep_analysis/full_analysis.json) |
| F9 | step-300 confidence AUROC is 0.5221, ECE is 0.0768, Brier is 0.1092 | [full_analysis.json](/home/v-seungplee/metacognition/results/step300_deep_analysis/full_analysis.json) |
| F10 | step-300 MATH500 gain is the main source of the overall improvement, while AIME is the main loss source | [analysis_report.md](/home/v-seungplee/metacognition/results/step300_deep_analysis/analysis_report.md), [full_analysis.json](/home/v-seungplee/metacognition/results/step300_deep_analysis/full_analysis.json) |

## 2. Intent, Hypotheses, and What the Results Actually Say

### 2.1 RQ1: Can SFT learn a controller?

**Intent**: learn a confidence-conditioned controller rather than a style template.

**What is supported**: yes, at the structural level. Strict meta SFT emits `<|meta|>` on 99.94% of examples and produces non-zero diagnosis and study-need fields. This is strong evidence that the controller format itself was learned.

**What is not supported**: OOD utility. The same strict rerun shows essentially tied overall accuracy with base SFT and a 10-point AIME gap against meta. The current evidence supports “controller acquisition,” not “OOD robustness.”

### 2.2 RQ2: Can RL improve the controller?

**Intent**: strengthen controller behavior with explicit reward.

**What is supported**: step-300 RL improved overall accuracy by 3.88 points, mostly through MATH500.

**What is not supported**: that the gain came from metacognitive control. The verified step-300 analysis shows `<|meta|>` wrapping collapsed to zero, free-text confidence/assessment collapsed to a near-single template, and AUROC remained near random. The reward path did shape output form, but not the intended structured controller.

### 2.3 RQ3: Can self-distill succeed on OOD?

**Intent**: use the controller to produce better repair teachers and preserve useful control at distillation time.

**Current status**: this is still open. The correct claim-bearing path is not “raw SFT data with meta preserved,” but `fixed_k_repair` artifacts with explicit selector provenance. Retrieval is not yet active by default, so current mainline should be described as repair-only self-distill unless an example bank is actually loaded.

## 3. Main Experimental Findings

### 3.1 Strict paired SFT

| Metric | Base strict SFT | Meta strict SFT | Delta |
|---|---:|---:|---:|
| Overall accuracy | 0.7551 | 0.7538 | -0.0013 |
| Hard accuracy | 0.5019 | 0.4981 | -0.0038 |
| OOD accuracy | 0.2667 | 0.1667 | -0.1000 |
| Meta emission rate | 0.0000 | 0.9994 | +0.9994 |
| Avg completion length | 248.79 | 323.17 | +74.38 |
| Wrong high confidence @0.7 | 0.0000 | 0.0865 | +0.0865 |

해석은 명확하다. Meta format preservation은 성공했지만, 같은 strict setting에서 OOD 이득은 없다. 따라서 “meta token = immediate accuracy gain”은 현재 data에서 지지되지 않는다.

### 3.2 Entropy audit

| Region | Mean entropy |
|---|---:|
| Before meta | 0.3306 |
| During meta | 0.2741 |
| After meta | 0.6305 |
| After - before | +0.2999 |

Meta block 이후 entropy가 낮아지지 않고 오히려 증가했다. 이는 meta block이 uncertainty를 바로 해소한다기보다, 새로운 탐색 구간을 열거나 reasoning mode를 다시 전개하는 쪽에 가깝다는 해석을 더 강하게 만든다.

### 3.3 RL step-300 pathology

| Metric | E21R-v2 step300 | Base step300 | Delta |
|---|---:|---:|---:|
| Overall accuracy | 0.7981 | 0.7592 | +0.0388 |
| MATH500 | 0.7160 | 0.6180 | +0.0980 |
| AIME | 0.1333 | 0.3333 | -0.2000 |
| Avg completion tokens | 723.33 | 452.03 | +271.30 |
| Meta wrap rate | 0.0000 | 0.0000 | 0.0000 |
| Free-text confidence rate | 0.8913 | 0.0000 | +0.8913 |
| AUROC | 0.5221 | N/A | N/A |

핵심은 reward misspecification이다. Current reward path let free-text confidence substitute for wrapped meta, so RL had no reason to preserve structural tokens. In parallel, the calibration reward encouraged constant high confidence once accuracy exceeded 50%, which explains the 0.96 collapse.

## 4. Why the Mainline Was Reset

기존 해석은 “meta RL이 controller를 강화하고 있다”에 가까웠다. 지금은 그 문장을 그대로 유지하면 안 된다. Verified analysis 기준으로는 다음과 같이 정리해야 한다.

1. strict SFT proved controller structure can be learned
2. RL step-300 improved accuracy, but not in the intended structural way
3. therefore the next clean claim is not RL superiority but whether self-distill can preserve or improve useful control under OOD stress

이 때문에 RQ3 mainline은 self-distill-first로 재설계되었다.

## 5. Updated Mainline Contract

### 5.1 Claim-bearing path

1. `strict_base_sft -> fixed_k_repair -> naive self-distill`
2. `strict_meta_sft -> fixed_k_repair -> claim-bearing epistemic self-distill`
3. optional extension: `teacher top-k -> control-span-weighted KL`

### 5.2 What counts as valid evidence

1. Base and meta lanes must share the same fixed repair budget and decode settings.
2. Claim-bearing meta lane must satisfy `synthetic_meta_injected_rate == 0`.
3. Retrieval can be claimed only when an example bank is supplied and the artifact reports non-zero retrieval usage.
4. If OOD does not improve, the result is still publishable as collapse analysis, but not as OOD self-distill success.

## 6. Code/Plan Misalignment That Was Fixed

이번 업데이트에서 바로 수정한 것은 “retrieval이 nominally 켜져 있지만 실제로는 꺼져 있는 상태를 문서상 구분하지 못하던 문제”다.

수정 내용:

1. `run_fixed_k_self_distill_roundtrip.sh` now disables retrieval explicitly when no example bank is passed.
2. `run_online_sdpo_regen.py` now warns when retrieval was requested but no retriever was loaded.
3. `write_online_sdpo_outputs()` now records whether retrieval was active, enabled, and non-empty.
4. Active plan now states that retrieval claims require actual example-bank-backed usage.

이 수정으로 인해, 이후 artifact는 `repair-only`와 `retrieval-backed repair`를 더 이상 혼동하지 않는다.

## 7. Immediate Next Experiments

### 7.1 Mainline

1. Build base `fixed_k_repair` artifacts and train the naive lane.
2. Build meta `fixed_k_repair` artifacts and train the claim-bearing epistemic lane.
3. Evaluate both against strict SFT baselines on controller retention, collapse metrics, and OOD accuracy.
4. Only if the meta lane shows preserved control and competitive OOD behavior, extend to control-span KL.

### 7.2 Side evidence

1. Keep RL reward redesign separate.
2. Replace the current calibration reward with a proper scoring objective and strict wrapped-only parsing.
3. Re-run RL only after the self-distill comparison is saved.

## 8. Limitations

현재 결과만으로는 다음 주장을 할 수 없다.

1. Meta-CoT improves OOD accuracy.
2. RL strengthened metacognitive control in a clean causal sense.
3. Retrieval is already part of the active self-distill evidence.

반대로, 현재 결과로는 다음 세 주장을 할 수 있다.

1. strict paired SFT can reliably teach the controller format
2. current RL reward design admits structural collapse
3. RQ3 should be tested with a fair repair teacher pipeline before any stronger paper claim
