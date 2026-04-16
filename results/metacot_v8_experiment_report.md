# Meta-CoT V8: Metacognitive Chain-of-Thought for Math Reasoning

> Verified update: the current claim-bearing status report is [study_2026_04_16_metacot_v8_status_report.md](/home/v-seungplee/metacognition/results/study_2026_04_16_metacot_v8_status_report.md). This older report remains useful as a phase log, but some interpretations below were superseded by the 2026-04-15/16 strict rerun and step-300 re-analysis.

**Author**: Seungpil Lee  
**Date**: 2026-04-13  
**Project**: metacognition  
**Model**: Qwen/Qwen3-8B  
**Compute**: msrresrchvc (4x A100 80GB, Premium)

---

## Executive Summary

Meta-CoT V8는 수학 추론에서 `<|meta|>` 블록을 통해 confidence-conditioned controller를 학습하고, 이를 RL로 강화하는 실험이다. 4개 Phase에 걸쳐 SFT, behavior analysis, 6-head RL, 2-head RL을 수행했다.

| Phase | Experiment | Key Metric | Meta | Base | Gap |
|---|---|---|---|---|---|
| 1 | Strict Paired SFT | Accuracy (1560) | 75.4% | 75.5% | -0.1pp |
| 2 | Behavior Analysis | Meta emission | 99.94% | N/A | controller acquired |
| 2 | Confidence Routing | Low-conf redirect | 89.68% | N/A | strong routing |
| 2 | Calibration (ECE) | AIME ECE | 0.074 | N/A | best on hardest |
| 2 | Entropy | After-meta delta | +0.300 nats | N/A | entropy INCREASES |
| 3 | E21 RL (6-head) | Val @step30 | 41.8% | 48.0% | **-6.2pp** |
| 4 | E21R-v2 (2-head) | Val @step50 | 44.9% | — | gap shrinking |

**핵심 발견**: 모델은 WHEN to redirect를 정확히 학습했으나 (low confidence에서 89.68% redirect), HOW to redirect를 효과적으로 실행하지 못한다 (redirect 성공률 47.3%, 긴 redirect는 9%만 성공). RL에서는 6-head GDPO의 reward dilution이 치명적이었으며 (-6.2pp), 2-head로 축소 후 gap이 줄어들고 있다.

---

## 1. Research Questions

### RQ1: Can SFT learn a confidence-conditioned controller?

**의도**: `<|meta|>` 블록이 단순한 스타일 문구가 아니라, confidence에 따라 verify/redirect를 결정하는 controller로 작동하는지 검증한다.

**기대 결과**: meta emission rate >90%, confidence-conditioned routing (low conf → redirect, high conf → verify), ECE 개선.

### RQ2: Can RL strengthen the controller via verifiable rewards?

**의도**: SFT로 형성된 controller behavior를 RL reward로 강화하여, base correctness-only GRPO 대비 competitive하거나 우월한 성능을 달성할 수 있는지 본다.

**기대 결과**: meta RL이 base GRPO 대비 동등하거나 우위.

### RQ3: Can the controller enable downstream adaptation?

**의도**: 학습된 meta state가 failure diagnosis를 만들어 retrieval/adaptation trigger로 사용 가능한지 확인한다.

**현황**: RQ1-RQ2 결과 확보 후 진행 예정. 본 보고서 범위 밖.

---

## 2. Phase 1: Strict Paired SFT

### 2.1 의도 (Intent)

Raw Qwen/Qwen3-8B에서 시작하는 clean paired SFT로, meta 데이터와 base 데이터가 정확히 같은 문제를 공유하면서 controller representation이 생기는지 확인한다. 기존 V5-V7의 문제점 (initializer 오염, 데이터 비대칭)을 근본적으로 해결하기 위해 strict paired contract를 도입했다.

### 2.2 가설 (Hypothesis)

1. 4264 paired samples (동일 문제, 동일 정답, meta/base만 다른 completion)으로 SFT하면, meta lane은 `<|meta|>` 블록을 안정적으로 emission하고, base lane은 emission하지 않는다.
2. 동일 initializer(raw Qwen3-8B)와 동일 hyperparameters에서 meta format 차이만으로 accuracy 차이가 발생한다면, 그것은 controller effect이다.
3. accuracy 차이가 없으면, SFT 단계에서 controller는 behavior만 학습하고 accuracy에는 중립적이다.

### 2.3 검증 방법 (Validation Method)

- **데이터**: 4264 paired samples, row-aligned (동일 문제, 동일 boxed answer)
- **학습**: raw Qwen/Qwen3-8B, 3 epochs, lr=2e-6, 4x A100
- **평가**: GSM8K 1030 + MATH500 500 + AIME2024 30 = 1560 problems, deterministic decoding (`do_sample=False`)
- **지표**: accuracy, meta emission rate, single-step collapse rate

### 2.4 결과 (Results)

#### 2.4.1 Accuracy

| Model | Overall | GSM8K | MATH500 | AIME2024 |
|---|---|---|---|---|
| Meta SFT | **1176/1560 = 75.4%** | 88.5% | 51.8% | 16.7% |
| Base SFT | **1178/1560 = 75.5%** | 88.5% | 51.6% | 26.7% |
| 차이 | -0.1pp | 0.0pp | +0.2pp | **-10.0pp** |

#### 2.4.2 Meta Emission

| Metric | Value |
|---|---|
| Meta block 포함 비율 | **99.94%** (1559/1560) |
| 정확히 1개 meta block | **99.94%** |
| 2개 이상 meta block | **0.06%** (1 sample) |

#### 2.4.3 Paired Comparison

| Category | Count | Ratio |
|---|---|---|
| Both correct | 1118 | 71.7% |
| Both wrong | 324 | 20.8% |
| Meta only correct | 58 | 3.7% |
| Base only correct | 60 | 3.8% |

### 2.5 해석 (Interpretation)

**RQ1 부분 지지**:

1. **Controller acquisition 성공**: 99.94% meta emission은 모델이 `<|meta|>` format을 완전히 학습했음을 의미한다. E19 (V6 데이터)의 15.9% emission과 비교하면 극적인 개선이다.
2. **Accuracy 중립**: meta format이 overall accuracy를 해치지 않았다 (75.4% vs 75.5%). 이는 meta block이 token budget을 소비하면서도 reasoning quality를 유지했다는 것이다.
3. **AIME 역전 (-10.0pp)**: AIME에서 meta가 base보다 10pp 낮다. 30 samples로 통계적 파워가 낮지만, hard subset에서 meta overhead가 해로울 수 있음을 시사한다. Base의 AIME 26.7%는 8/30 정답, meta의 16.7%는 5/30 정답이다.
4. **Single-step collapse**: 99.94%가 정확히 1개 meta block만 사용한다. 모델은 "meta를 1번 쓰는 것"만 학습했고, 필요할 때 여러 번 meta를 사용하는 multi-step metacognition은 학습하지 못했다. 이는 SFT 데이터의 구조적 한계이다.
5. **Paired comparison 대칭**: meta_only(58)와 base_only(60)가 거의 대칭이다. 이는 meta block이 "특정 문제에서 도움을 주고 특정 문제에서 방해하는" 양면성을 가짐을 시사한다.

---

## 3. Phase 2: Behavior & Entropy Analysis

### 3.1 Confidence-Conditioned Routing

#### 3.1.1 의도

Meta block 내부의 confidence 값이 실제로 verify/redirect 행동을 조건부로 결정하는지 확인한다. 만약 confidence와 action이 무관하다면 controller가 아니라 random noise이다.

#### 3.1.2 가설

Low confidence (<=0.5)는 redirect를, high confidence (>=0.7)는 verify를 유도해야 한다. 이 분리가 명확할수록 controller learning이 강하다.

#### 3.1.3 결과

**Confidence-conditioned routing**:

| Confidence Range | Primary Action | Rate |
|---|---|---|
| Low (<=0.5) | redirect | **89.68%** |
| High (>=0.7) | verify | **100%** |

**Benchmark별 routing**:

| Benchmark | Verify Rate | Redirect Rate | Avg Confidence |
|---|---|---|---|
| GSM8K | 99.0% | 0.8% | 0.764 |
| MATH500 | 15.6% | 86.0% | 0.360 |
| AIME2024 | 0% | 100% | 0.240 |

#### 3.1.4 해석

Controller routing은 매우 강하다. 세 가지 핵심 관찰:

1. **난이도 정합성**: GSM8K(쉬움) → high confidence + verify, MATH500(중간) → low confidence + redirect, AIME(어려움) → lowest confidence + 100% redirect. 이 gradient는 모델이 문제 난이도를 confidence로 정확히 반영함을 보여준다.
2. **Routing sharpness**: low conf에서 redirect 89.68%, high conf에서 verify 100%는 거의 deterministic routing이다. Fuzzy boundary가 없다.
3. **AIME 100% redirect**: AIME에서 verify가 0%인 것은 모델이 자신의 한계를 인식한다는 의미이지만, redirect를 100% 시도하면서도 AIME accuracy가 16.7%에 불과한 것은 redirect execution의 실패를 의미한다.

### 3.2 Calibration (ECE)

| Benchmark | ECE | 해석 |
|---|---|---|
| GSM8K | 0.121 | moderate miscalibration |
| MATH500 | 0.158 | moderate miscalibration |
| AIME2024 | **0.074** | best calibrated |

**해석**: 가장 어려운 벤치마크에서 ECE가 가장 낮다는 것은 반직관적이지만, AIME에서 모델이 일관되게 low confidence를 보고하고 실제로 정답률이 낮기 때문이다 (avg conf 0.240 vs accuracy 16.7%). 즉, "모른다"는 것을 정확히 안다. 반면 GSM8K에서는 high confidence를 보고하지만 88.5%만 맞추므로 11.5%의 과신이 존재한다.

### 3.3 Entropy Analysis

#### 3.3.1 의도

Token-level entropy를 `<|meta|>` 블록 전후로 측정하여, meta block이 모델 내부 상태를 실제로 변경하는지(resolution) 아니면 표면적 텍스트에 불과한지(decoration) 판별한다.

#### 3.3.2 가설

Meta block이 functional하다면, meta 이후 entropy가 감소해야 한다 (불확실성 해소). Meta block이 decorative하다면, entropy 패턴이 변하지 않거나 증가해야 한다.

#### 3.3.3 결과

| Region | Entropy (nats) |
|---|---|
| Before meta | 0.331 |
| During meta | 0.274 |
| After meta | 0.631 |
| **Delta (after - before)** | **+0.300** |

**Correctness별 delta**:

| Condition | Delta (nats) |
|---|---|
| Correct samples | +0.305 |
| Incorrect samples | +0.203 |

#### 3.3.4 해석

Entropy가 meta 이후 **증가**한다. 이는 resolution hypothesis와 정반대이다.

세 가지 해석이 가능하다:

1. **Meta block이 reasoning mode를 reset한다**: meta 이전은 집중된 derivation (low entropy), meta 이후는 새로운 탐색의 시작 (high entropy). 이 경우 entropy 증가는 "열린 탐색 → 집중 → 다시 열린 탐색"의 자연스러운 패턴일 수 있다.
2. **Meta block이 불확실성을 해소하지 못한다**: confidence를 보고하고 verify/redirect를 선언하지만, 실제 모델 내부의 token distribution은 더 불확실해진다. 이는 "말하기만 하고 실행하지 못하는" 패턴과 일치한다.
3. **Structural artifact**: meta block 이후 답을 작성하는 구간은 자연스럽게 diversity가 높은 텍스트 (수식, boxed answer 포맷 등)를 포함하므로 entropy가 높을 수 있다.

Correct samples에서 delta가 더 크다는 것(+0.305 vs +0.203)은 해석 1과 부분적으로 일치한다. 정답을 맞추는 경우에 meta 이후 더 다양한 탐색을 하고, 그중 올바른 경로를 찾는다는 해석이 가능하다. 그러나 이는 causal claim이 아니라 correlation이다.

### 3.4 Redirect Pattern Analysis

#### 3.4.1 의도

Meta block에서 redirect를 선언한 471 samples의 실제 성공률과 실패 패턴을 분석하여, redirect execution의 bottleneck을 정확히 파악한다.

#### 3.4.2 결과

| Metric | Value |
|---|---|
| Total redirect samples | 471 |
| Redirect success rate | **47.3%** |
| Template usage ("What is missing is...") | **99%** |

**Strategy별 성공률**:

| Strategy | Success Rate |
|---|---|
| substitution | 58% |
| combinatorial | 55% |
| recurrence | 55% |
| simplify | 52% |
| coordinate | 28% |
| geometric | **0%** |

**Post-redirect structure 영향**:

| Pattern | Effect on Success |
|---|---|
| Backtracking present | **-17.4pp** |
| Multiple attempts (>1) | **-23.5pp** |
| Short redirect (<=150w) | 55% success |
| Long redirect (>400w) | **9% success** |

#### 3.4.3 해석: Detection vs Execution Gap

이것이 V8의 가장 중요한 발견이다.

**"모델은 WHEN to redirect를 정확히 알지만, HOW to redirect를 효과적으로 실행하지 못한다."**

구체적 증거:

1. **Template 고착 (99%)**: 거의 모든 redirect가 "What is missing is..." 단일 template을 사용한다. 이는 SFT 데이터의 diversity 부족을 직접 반영한다. 모델은 redirect의 형식을 학습했지만, 문제별로 적절한 진단과 전략 전환을 생성하는 능력은 부족하다.
2. **길수록 실패**: redirect가 150 words 이하일 때 55% 성공하지만, 400 words 초과시 9%로 급락한다. 긴 redirect는 모델이 올바른 전략을 찾지 못하고 배회하는 신호이다.
3. **Backtracking은 해로움 (-17.4pp)**: redirect 후 다시 원래 접근으로 돌아가는 것은 redirect의 목적(전략 전환)과 모순되며, 실제로 성공률을 크게 떨어뜨린다.
4. **Multiple attempts도 해로움 (-23.5pp)**: 여러 번 시도하는 것은 모델이 전략 실행에 확신이 없음을 의미하며, 오히려 성공률을 더 낮춘다.
5. **Geometric 전략 0%**: 기하학적 접근법은 한 번도 성공하지 못했다. 이는 8B 모델의 기하 추론 능력 한계를 반영한다.

이 gap은 Phase 3-4 RL에서도 핵심 bottleneck으로 작용한다. RL이 redirect의 "빈도"를 높여도 "품질"이 개선되지 않으면 accuracy 향상으로 이어지지 않기 때문이다.

---

## 4. Phase 3: E21 RL Anchor (6-head GDPO)

### 4.1 의도 (Intent)

SFT로 형성된 meta controller를 RL로 강화한다. 6개의 reward head를 사용하는 GDPO (Group Distributional Policy Optimization)로 meta behavior의 다양한 측면을 동시에 최적화한다.

### 4.2 가설 (Hypothesis)

1. Correctness + switch_v2 + verify_v2 + conf_traj + meta_floor + meta_count_bonus의 6 heads가 meta behavior를 분해 가능하게 유도한다.
2. meta_count_bonus가 SFT의 single-step collapse를 깨고 multi-meta usage를 유도한다.
3. Meta RL이 base correctness-only GRPO와 competitive하다.

### 4.3 검증 방법 (Validation Method)

- **알고리즘**: GDPO with 6 reward heads
- **Reward heads**: correctness, switch_v2, verify_v2, conf_traj, meta_floor, meta_count_bonus
- **Baseline**: Base GRPO (correctness-only, same hyperparameters)
- **비교 방식**: step-matched validation comparison (같은 step, 같은 validation set)
- **학습**: paired RL contract 준수 (Section 3.2 frozen keys)

### 4.4 결과 (Results)

E21은 43 steps 학습 후 validation 하락으로 중단되었다 (`side_evidence` 분류).

**Step 30 공정 비교** (같은 step, 같은 validation set):

| Metric | Meta E21 @30 | Base GRPO @30 | Gap |
|---|---|---|---|
| algebra | 76.5% | 78.4% | -1.9pp |
| prealgebra | 67.5% | 82.5% | -15.0pp |
| number_theory | 67.9% | 64.3% | +3.6pp |
| **Overall** | **41.8%** | **48.0%** | **-6.2pp** |

### 4.5 해석 (Interpretation)

**가설 기각**: 6-head GDPO는 base GRPO보다 -6.2pp 열등하다.

**Root cause 분석**:

1. **Correctness weight dilution**: 6 heads 중 correctness가 1개이므로, GDPO의 per-head normalization 후 correctness의 실질 비중이 약 **32%** (= 1/6 + noise)로 떨어진다. 나머지 68%의 gradient가 meta behavior 관련 보상에 할당되어, "정답을 맞추는 것"보다 "meta처럼 보이는 것"에 과도한 학습 압력이 가해진다.
2. **GDPO 논문 권장 위반**: GDPO 원 논문은 2-3 heads를 권장한다. 6 heads는 advantage collapse 위험이 있으며, 실제로 step 30 이후 validation이 하락했다.
3. **Prealgebra 붕괴 (-15.0pp)**: 쉬운 문제에서 가장 큰 하락이 발생했다. 이는 meta overhead가 쉬운 문제의 straightforward solving을 방해함을 의미한다. Meta block이 불필요한 곳에서도 강제되기 때문이다.
4. **Number theory 예외 (+3.6pp)**: 유일하게 meta가 우위인 영역이다. Number theory는 try-and-check 전략이 유효하여, verify behavior가 실제로 도움이 된 것으로 보인다.
5. **Constraint forgetting**: 실제 응답 검수에서, meta interrupt 후 문제의 constraint를 누락하는 패턴이 확인되었다 (예: digit <= 9 조건 빠뜨림). Meta block이 working memory를 reset하는 부작용이 있다.

**결론**: reward heads를 많이 넣을수록 좋은 것이 아니다. Correctness 지배력 유지가 핵심이다.

---

## 5. Phase 4: E21R-v2 (2-head GDPO)

### 5.1 의도 (Intent)

E21의 실패 교훈을 반영하여, GDPO 논문 권장대로 2 heads만 사용한다. Correctness를 지배적으로 유지하면서, outcome_calibration만 보조 신호로 사용한다. Meta behavior를 직접 강제하지 않고, "정답을 위해 meta가 필요하면 자연스럽게 사용"하도록 유도한다.

### 5.2 가설 (Hypothesis)

1. 2-head GDPO (correctness 77% + outcome_calibration 23%)는 E21의 6-head보다 correctness 학습 속도가 빠르다.
2. outcome_calibration의 wrong-high-confidence penalty (-0.3 x last_conf)가 과신 오답을 줄인다.
3. Meta format의 structural overhead에도 불구하고, base GRPO 대비 gap이 줄어든다.

### 5.3 검증 방법 (Validation Method)

- **알고리즘**: GDPO with 2 reward heads
- **GDPO heads**: correctness (x1.0, range [-1, +1]) + outcome_calibration (x1.0, range [-0.4, +0.4])
- **Combined score 추가**: meta_floor (x0.3, range [-0.5, 0]) — GDPO normalization 밖
- **Effective weights**: correctness ~77%, outcome_calibration ~23%
- **비교 대상**: E21 (6-head), Base GRPO

### 5.4 결과 (Results)

**Validation trajectory (step 0 → 50)**:

| Step | E21R-v2 Val |
|---|---|
| 0 | 39.9% |
| 10 | 39.4% |
| 20 | 40.5% |
| 30 | 43.8% |
| 40 | 42.2% |
| 50 | 44.9% |

**E21 대비 비교**:

| Metric | E21R-v2 | E21 | 차이 |
|---|---|---|---|
| Avg validation (step 0-50) | ~41.8% | ~39.5% | **+2.3pp avg** |
| Peak step 30 | 43.8% | 41.8% | +2.0pp |
| Trend after step 30 | 계속 상승 | 하락 시작 | E21R-v2 안정적 |

**Base GRPO 대비 gap 변화**:

| Step | Gap to Base |
|---|---|
| 10 | -7.6pp |
| 30 | -4.2pp |
| Trend | **Gap 축소 중** |

### 5.5 해석 (Interpretation)

**가설 부분 지지**:

1. **E21 대비 개선 확인**: E21R-v2는 모든 step에서 E21보다 약 +2pp 높다. 2-head 구조의 correctness 지배력(77%)이 6-head의 diluted correctness(32%)보다 효과적임이 확인되었다.
2. **학습 안정성 개선**: E21은 step 30 이후 하락했지만, E21R-v2는 step 50에서도 44.9%로 상승 추세를 유지한다. Reward head 수 감소가 gradient stability를 개선했다.
3. **Base 대비 gap 축소**: step 10의 -7.6pp → step 30의 -4.2pp로 gap이 줄고 있다. 추세가 유지된다면 step 100+ 에서 base와 competitive할 가능성이 있다.
4. **그러나 여전히 base보다 열등**: step 30 기준으로 여전히 -4.2pp이다. Meta block의 structural overhead (token budget 소비, working memory reset)가 calibration 이득을 상쇄하고 있다.
5. **추가 학습 필요**: current trajectory를 extrapolate하면, base와 동등해지는 시점은 step 80-120 근처로 추정된다. 하지만 이는 linear extrapolation의 한계가 있다.

---

## 6. Cross-Phase Analysis

### 6.1 Why Base GRPO Outperforms Meta RL

Base GRPO가 meta RL보다 우위인 근본 원인은 3가지이다:

**1. Token budget overhead**

Meta block은 평균 33.5 tokens을 소비한다 (entropy analysis의 `meta_length_tokens_mean`). 4096 token budget에서 이는 약 0.8%이지만, redirect를 실행하는 경우 post-redirect reasoning이 추가되어 실질적으로 response budget의 상당 부분이 meta 관련 텍스트에 할당된다. 특히 long redirect (>400 words)에서 이 overhead가 치명적이다 (성공률 9%).

**2. Working memory disruption**

Meta block이 reasoning flow를 중단한다. E21의 constraint forgetting 사례에서 확인되었듯이, meta block 이후 모델이 문제의 조건을 일부 잊어버리는 현상이 있다. Entropy가 meta 이후 +0.300 nats 증가하는 것도 이 disruption과 일치한다.

**3. Redirect execution failure**

Redirect의 47.3% 성공률은 동전 던지기보다 약간 나은 수준이다. 모델이 redirect를 선언해도 실제로 새로운 전략을 효과적으로 실행하지 못하면, redirect는 순수한 overhead가 된다. 특히 99% template 고착은 SFT 데이터의 diversity 부족이 직접적 원인이다.

### 6.2 The Structural Overhead of Meta Blocks

Meta block의 구조적 비용을 정량화하면:

| Cost | Evidence |
|---|---|
| Token budget | 평균 33.5 tokens/meta block |
| Working memory | Entropy +0.300 nats after meta |
| AIME accuracy | -10.0pp vs base SFT |
| E21 RL overall | -6.2pp vs base GRPO |
| Redirect failure | 52.7% of redirects fail |
| Long redirect | 91% failure rate (>400w) |

대비되는 benefit:

| Benefit | Evidence |
|---|---|
| Confidence routing | 89.68% redirect at low conf |
| AIME calibration | ECE 0.074 (best) |
| Detection accuracy | Controller correctly identifies difficulty |
| Meta-only wins | 58 samples (3.7%) |

**결론**: current implementation에서 meta block의 cost가 benefit을 초과한다. Benefit이 주로 "detection" 단계에 집중되어 있고, "execution" 단계에서는 cost가 지배적이다.

### 6.3 GDPO Head Count vs Learning Efficiency

| Config | Heads | Correctness Weight | Step 30 Val | Trend |
|---|---|---|---|---|
| E21 | 6 | ~32% | 41.8% | declining after 30 |
| E21R-v2 | 2 | ~77% | 43.8% | still rising at 50 |
| Base GRPO | 1 | 100% | 48.0% | N/A (different algo) |

Head count와 learning efficiency 사이에 명확한 역상관이 관찰된다. 이는 GDPO 논문의 2-3 heads 권장과 일치하며, 특히 수학 reasoning처럼 correctness가 명확한 도메인에서는 correctness 비중을 최대한 유지하는 것이 중요하다.

---

## 7. Proposed Improvements

### 7.1 SCoRe-style Multi-turn Correction

**의도**: 현재 redirect는 single-turn 내에서 전략 전환을 시도하지만, 47.3% 성공률에 불과하다. SCoRe (Self-Correction via Reinforcement Learning, Kumar et al., 2024)처럼 Turn 1에서 시도 → Turn 2에서 교정하는 multi-turn 구조를 도입한다.

**기대 효과**:
- Turn 1의 실패 diagnosis가 Turn 2의 명시적 입력이 되므로, redirect execution quality가 향상된다
- 긴 single-turn redirect (>400w, 9% 성공)를 짧은 multi-turn (<=150w per turn, 55% 성공 기대)으로 분해한다

**검증**: AIME/MATH500 hard subset에서 single-turn vs multi-turn redirect 성공률 비교.

### 7.2 Shorter Meta Blocks

**의도**: 현재 meta block은 template 기반으로 구조화되어 있지만 ("confidence: X.XX / assessment: ... / action: ..."), 핵심 정보만 남기고 축소한다.

**근거**: 
- Short redirects (<=150w) succeed 55% vs long (>400w) 9%
- Meta block 평균 33.5 tokens은 줄일 수 있다
- Entropy +0.300 nats는 meta block이 reasoning flow를 방해하므로, block 길이를 줄이면 disruption이 감소할 수 있다

**구체적 방안**: meta block을 `<|meta|>conf:0.7 action:verify<|/meta|>` 형태로 축소 (5-10 tokens).

### 7.3 Subgoal-Based Redirect

**의도**: 현재 redirect의 99% template 고착을 깨기 위해, redirect 시 subgoal을 명시하도록 SFT 데이터를 재구성한다.

**근거**:
- Strategy별 성공률 편차가 크다 (substitution 58% vs geometric 0%)
- 모델이 문제 유형에 맞는 전략을 선택하지 못하는 것이 핵심 병목이다
- Subgoal 명시 ("Step 1: express in polar coordinates, Step 2: apply De Moivre's theorem")가 execution quality를 높일 수 있다

**검증**: subgoal 포함 데이터로 re-SFT → redirect 성공률 변화 측정.

---

## 8. Limitations

1. **AIME sample size (n=30)**: AIME의 -10.0pp 차이는 30 samples 기반이므로 통계적 파워가 낮다. 95% CI는 넓다 (~+-15pp).

2. **Entropy analysis sample size (n=120)**: token-level entropy 분석은 120 samples에서만 수행되었다. 특히 incorrect samples는 6개뿐이므로 correctness-conditioned 해석에 한계가 있다.

3. **E21R-v2 진행 중**: Phase 4는 step 50까지만 결과가 있으며, step 100+ 에서의 convergence behavior는 확인되지 않았다.

4. **Base GRPO step mismatch**: Base GRPO는 step 105+까지 진행되었으나, 직접 비교에서는 step 30까지만 사용했다. 이후 step에서의 base trajectory는 본 보고서에 포함되지 않았다.

5. **Single template 원인 미검증**: redirect의 99% template 고착이 SFT 데이터의 diversity 부족인지, 모델의 mode collapse인지 분리 검증되지 않았다.

6. **Causal claims 제한**: behavior analysis는 모두 observational이다. Meta block이 accuracy에 causal 영향을 미치는지는 ablation (meta block 제거 후 inference)으로만 확인 가능하며, 아직 수행되지 않았다.

7. **Entropy 해석 모호성**: after-meta entropy 증가가 "mode reset" (해석 1), "resolution 실패" (해석 2), "structural artifact" (해석 3) 중 어느 것인지 구분되지 않았다.

---

## 9. Conclusion

Meta-CoT V8는 RQ1에 대해 **부분적 지지**를 제공한다:

- **성공**: confidence-conditioned controller를 SFT만으로 학습할 수 있다 (99.94% emission, 89.68% redirect at low conf, ECE 0.074 on AIME).
- **실패**: controller가 "detection"에서는 작동하지만 "execution"에서는 실패한다 (redirect 47.3% 성공, 99% template 고착, entropy +0.300 after meta).

RQ2에 대해서는 **초기 결과**만 있다:

- E21 (6-head GDPO)는 기각 (-6.2pp vs base, reward dilution)
- E21R-v2 (2-head GDPO)는 개선을 보이며 진행 중 (gap -7.6pp → -4.2pp, still trailing)

핵심 발견은 **Detection-Execution Gap**이다. 모델은 언제 redirect해야 하는지를 정확히 알지만, 어떻게 redirect할지를 효과적으로 실행하지 못한다. 이 gap을 해소하지 않는 한, RL이 redirect "빈도"를 높여도 accuracy 향상으로 이어지지 않는다.

---

## 10. Next Experiments

### 10.1 단기 (1-2주)

| Priority | Experiment | 의도 | Expected Outcome |
|---|---|---|---|
| P0 | E21R-v2 step 100+ 대기 | base GRPO 대비 gap convergence 확인 | gap < 2pp 여부 |
| P0 | Base GRPO step 150+ 결과 수집 | meta와의 fair comparison | base ceiling 확인 |
| P1 | Meta block 축소 ablation | 33.5 tokens → 5-10 tokens | entropy delta 감소, accuracy neutral 여부 |

### 10.2 중기 (2-4주)

| Priority | Experiment | 의도 | Expected Outcome |
|---|---|---|---|
| P1 | SCoRe-style multi-turn correction | redirect execution gap 해소 | redirect 성공률 47% → 60%+ |
| P1 | Subgoal-based redirect SFT data | template 고착 해소 | template diversity, strategy-specific success |
| P2 | Entropy-based RL reward 탐색 | entropy delta를 reward signal로 | meta 이후 entropy 감소 유도 |

### 10.3 장기 (1-2개월)

| Priority | Experiment | 의도 | Expected Outcome |
|---|---|---|---|
| P2 | E21S stepwise confidence delta | blockwise dense reward | 더 촘촘한 credit assignment |
| P3 | Phase 5 curriculum/RAG | meta diagnosis → retrieval trigger | RQ3 initial evidence |
| P3 | MCTS-lite confidence-bucket search | meta를 branching trigger로 | inference-time search gain |

### 10.4 의사결정 분기점

```
E21R-v2 step 100 결과
├── gap < 2pp → meta overhead가 거의 해소됨 → Phase 5 준비
├── gap 2-5pp → meta block 축소 + multi-turn 필요 → 7.1 + 7.2 실행
└── gap > 5pp → meta format 근본 재설계 필요 → redirect 방식 전환 또는 verify-only mode
```

---

## Appendix A: Data Provenance

| Artifact | Path | Description |
|---|---|---|
| Meta SFT data | `data/v8_meta_inside_strict.parquet` | 4264 paired samples |
| Base SFT data | `data/v8_base_matched_strict.parquet` | 4264 paired samples |
| Meta SFT checkpoint | `checkpoints/v8_meta_inside_strict_sft` | Qwen3-8B, 3ep, lr=2e-6 |
| Base SFT checkpoint | `checkpoints/v8_base_matched_strict_sft` | Qwen3-8B, 3ep, lr=2e-6 |
| Paired eval | `results/eval_v8_meta_inside_strict_sft/` | 1560 problems |
| Behavior analysis | `results/strict_pair_analysis_repro_2026_04_12/` | Paired behavior JSON |
| Entropy analysis | `results/entropy_strict_meta/` | Token-level entropy stats |
| Active plan | `results/plan_metacot_v8_active_2026_04_09.md` | Frozen mainline contract |
| HuggingFace | `datasets/iamseungpil/metacot` | All data synced |

## Appendix B: Hyperparameter Summary

### SFT

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-8B |
| Epochs | 3 |
| Learning rate | 2e-6 |
| Training samples | 4264 (paired) |
| GPUs | 4x A100 80GB |
| Max length | 8192 |

### RL (Frozen Shared Keys)

| Parameter | Value |
|---|---|
| prompt_length | 2048 |
| response_length | 4096 |
| train_batch_size | 64 |
| rollout.n | 4 |
| learning_rate | 1e-6 |
| kl_coef | 0.001 |
| temperature | 0.7 |
| total_training_steps | 300 |

### E21 (6-head GDPO)

| Head | Weight |
|---|---|
| correctness | ~32% (1/6 effective) |
| switch_v2 | ~13% |
| verify_v2 | ~13% |
| conf_traj | ~13% |
| meta_floor | ~13% |
| meta_count_bonus | ~13% |

### E21R-v2 (2-head GDPO)

| Head | Weight | Range |
|---|---|---|
| correctness | ~77% | [-1, +1] |
| outcome_calibration | ~23% | [-0.4, +0.4] |
| meta_floor (outside GDPO) | x0.3 | [-0.5, 0] |
