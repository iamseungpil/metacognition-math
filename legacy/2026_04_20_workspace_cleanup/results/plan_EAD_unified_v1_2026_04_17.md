# Epistemic Alignment Distillation (EAD) — Unified Paper Plan v1

**Date**: 2026-04-17
**Status**: Draft v1, pending critic iteration
**Scope**: Single paper integrating (i) working note EV alignment theory, (ii) Meta-CoT V8 empirical observations, (iii) Meta-RLSD / N3 / planned B, D, F methods

---

## 0. Intent (의도)

Self-distillation이 meta-reasoning (epistemic verbalization, 이하 EV) controller를 **answer imitation으로 붕괴시키는** 현상을 단일 원리로 설명하고, 그 원리로부터 도출된 **control-preserving distillation family (EAD)** 가 collapse를 방지하는 것을 이론·관찰·방법 세 층위에서 보인다.

**한 줄 요약**:  
> Naive self-distillation은 EV token alignment를 보존하지 못해 필연적으로 collapse를 일으킨다. EAD는 alignment 조건을 loss로 직접 translate하여 collapse-free distillation을 만든다.

## 1. Why — 3-단계 motivation

### 1.1 Theory layer (working note: `네 가지 추론 습관의 PPO 단계 정보이론 메커니즘`)
- Llama-3.2-3B Countdown-3to4: 5 PPO 조건 × 정확도 27배 분산
- 통합 정리 (Prop 1 부호 정리): EV token이 $\Delta H_{t_e} > 0$ AND $\gamma_{t_e} > 0$ (정렬) 성립 시 $\Delta U_T > 0$ (성공), 정렬 실패 ($\gamma \le 0$) 시 실패
- 4개 경로: Opener / Compression / Scaffold / Alignment-failure

### 1.2 Observation layer (Meta-CoT V8)
- Qwen3-8B + MATH/AIME/GSM8K
- Meta SFT controller 성립: +3.88 pp (75.92 → 79.81 %), Δentropy = +0.300 nats (paper Opener signature)
- RL step 300 붕괴: wrap 100 → 88.2 %, confidence 0.96 collapse 98.9 %, AIME truncation 13/30, Δentropy = **−0.052** nats (alignment 반전)
- Naive self-distill (D2 rebuilt): controller 복원 (Δ +0.231) but AIME 47 % truncated, 정확도 60 %

→ **Observation**: RL + naive distill 모두 EV alignment 보존 실패. Meta token이 많음/적음이 아니라 **정렬된 drift 방향**이 핵심.

### 1.3 Method gap
기존 self-distill은 answer trace imitation만 한다. EV alignment를 loss에 explicit 하지 않음 → 학습이 answer-wrong-but-well-formed teacher를 그대로 복제하거나 (naive D2 실패), structural collapse에 대해 indifferent함 (RL 실패).

## 2. Core claim + 3 sub-claims

**Core**: Self-distillation은 control-preserving하려면 reasoning 본문 이 아니라 **EV alignment geometry**를 distill 대상으로 삼아야 한다.

- **C1** (Theory ↔ Observation bridge): Meta-CoT의 failure mode가 working note의 alignment-failure 경로와 일치한다 — Qwen3-8B에서도 $\gamma > 0$ 조건 붕괴 시 collapse.
- **C2** (EAD family 설계 원리): 4 제약 축 (distill 범위, teacher filter, token weighting, forbidden pattern)의 교집합으로 collapse-free 설계.
- **C3** (Empirical 검증): M1/N3/B/D/F 중 하나 이상이 naive baseline을 초과하며 (AIME ≥ baseline + 3 pp) controller 보존 (wrap ≥ 95 %, Δentropy 부호 양).

## 3. EAD framework (4 × 6 matrix)

**4 제약 축**:
1. **What to distill** (범위)
2. **Which teacher** (filter)
3. **Which token** (weighting)
4. **What to forbid** (penalty)

**6 instantiations** (working note + EAD 통합 설계):

| ID | Name | Axis 1: scope | Axis 2: teacher | Axis 3: token | Axis 4: forbid |
|---|---|---|---|---|---|
| **A** | Meta-only KL | meta span만 | raw correct | binary 0/1 | — |
| **B** | Alignment-filtered | meta + post-meta | meta_commit_quality > τ ∧ no_boxed=0 ∧ decoherence=0 | same as A | — |
| **C** | Contrastive T+/T- (N3) | meta only | T+ = correct, T- = decoy (deterministic) | binary | — |
| **D** | Entropy-shape regularizer | meta + post-meta | correct | 0/1 | Δentropy target +0.3 nats |
| **E** | Counterfactual | mixed (epistemic vs overconfident) | contrastive pair | binary | — |
| **F** | Commit-aware | meta + post-meta | correct | **control-critical 가중치** (confidence 1.5, diagnosis 1.25, verify 1.10) | meta-loop 반복, no-boxed, boxed-after-drift |

**Main run (EAD-Main)**: **A ∧ B ∧ F** 조합  
**Ablation**: +C (N3), +D, +E 각각  
**Baseline**: Naive D2 (no EAD), RL E21R-v2 step 300

## 4. Hypotheses (falsifiable)

### H1 (Theory-Observation bridge): Meta-CoT에서 EV signature 재현
- RL step 300의 Δentropy 부호 = 음 (alignment 실패 — working note 4th path)
- Meta SFT의 Δentropy 부호 = 양 (Opener — 1st path)
- D2 rebuilt의 AIME truncation = Compression path의 over-attention 실패 signature
- **Prediction**: 5 조건 × 2 모델 (Llama + Qwen) cross-validation 모두 Prop 1 부호 예측과 일치
- **Falsification**: Qwen3-8B에서 Prop 1 예측 부호와 어긋남이 50 % 이상

### H2 (EAD-Main > naive)
- 1030-problem 16k eval
- EAD-Main vs naive D2: Overall ≥ +3 pp, AIME ≥ +5 pp, meta wrap ≥ 95 %, AIME truncation ≤ 20 %
- **Falsification**: EAD-Main 정확도 ≤ naive D2 + 1 pp

### H3 (B filter 효과 isolation)
- EAD-Main vs (EAD-Main \ B): B 제거 시 collapse rate 증가
- **Prediction**: B filter 제거 → Δentropy 감소, wrap rate 감소
- **Falsification**: |ΔΔentropy| < 0.05 nats

### H4 (C contrastive 신호 추가성)
- EAD-Main vs EAD-Main+C (N3 통합): AIME +2 pp 이상
- **Falsification**: AIME 차이 < 1 pp

### H5 (D entropy-shape 정렬 강화)
- D 미포함 vs 포함: meta-token 이후 5-token Δentropy 분포가 +0.3 nats target에 가까워짐
- **Falsification**: 수렴 분포 평균이 +0.2 nats 미만

### H6 (Cross-model generalization)
- Working note (Llama-3.2-3B Countdown) 와 Meta-CoT (Qwen3-8B MATH) 의 alignment signature 양성률 > 0.7 공통
- **Falsification**: 한쪽 모델에서만 signature 관측

## 5. Verification methodology

### 5.1 Metric suite (BU analysis 재사용 + 확장)

(working note §2의 4 signatures 모두 재구현 예정)

1. **ΔH_{t_e±5}**: EV marker 전후 5-token 윈도우의 평균 entropy 차이
2. **d_M (Mahalanobis distribution rearrangement)**: $(H_t, \text{top1}, \text{top1-top2})$ 3-axis 공간에서 EV marker 위치 pair와 중립 pair 거리 비교
3. **$I(M_c; Y \mid D)$**: trace의 meta count capped × correctness, difficulty tercile 조건부
4. **$C_t = \sum_s (1 - H_s / \log_2 V)$ drift**: post-marker 5-token 누적 gain, SFT vs PPO Cohen's d

+ **Meta-CoT 확장 metrics**:
- AIME truncation rate (no_boxed in 16k budget)
- Boilerplate share (top-1 assessment 비중)
- Confidence distribution mode + entropy
- Wrap rate (`<|meta|>`/`<|/meta|>` balanced pair 비율)

### 5.2 Experimental matrix

**Primary runs** (student init = v8 meta SFT Qwen3-8B):

| Run | Method | Teacher | 예상 wall-time |
|---|---|---|---|
| Naive-D2 (baseline) | SFT only | D2 rebuilt teacher data | 3h |
| RL-step300 (baseline) | verl-GDPO | - | 완료 (분석 기존) |
| M1 (A instance) | Meta-RLSD | single priv | 3h |
| N3 (A ∧ C) | Contrastive | T+ / T- | 3h |
| EAD-B (A ∧ B) | Filter + meta-only | filtered priv | 3h |
| EAD-F (A ∧ F) | Commit-aware + meta-only | filtered | 3h |
| **EAD-Main (A ∧ B ∧ F)** | combined | filtered | 3h |
| EAD-Full (A ∧ B ∧ C ∧ D ∧ F) | all axes | filtered | 4h |

→ 8 runs × 2 seeds = 16 jobs, 평균 3.5h → **56 GPU-hours**. 4 노드 병렬 → **14 wall-hours**.

**Cross-validation**: working note Llama-3.2-3B Countdown-3to4 metric 재실행 (현재 BU codebase) + Meta-CoT 동일 metric 적용.

### 5.3 Success / failure criteria

| 레벨 | 성공 | 실패 |
|---|---|---|
| Theory | H1 cross-validation 통과 | 2 모델 중 하나에서만 signature |
| Method | EAD-Main > Naive-D2 (+3 pp overall, +5 pp AIME) | EAD-Main ≤ Naive + 1 pp |
| Ablation | B, D, F 제거 시 각각 유의 감소 | 모든 제거 실험에서 < 1 pp 차이 |
| Cross-model | Llama + Qwen 모두 alignment signature 공통 | 한쪽 모델 전용 |

## 6. Paper structure (integration)

```
Title: Epistemic Alignment Distillation — 
       Theory, Observation, and Control-Preserving Self-Distillation

Abstract
  EV alignment theorem → empirical collapse 관찰 → EAD family → Qwen/Llama 검증

§1 Introduction
  - Self-distillation의 paradox: controller 복원 시도가 accuracy 손상
  - 원인 가설: answer imitation이 alignment 신호 masking
  - Contributions: (i) theory, (ii) observation, (iii) 방법

§2 Epistemic Verbalization Alignment (Theory — working note § 2)
  - EV token과 hidden state shift
  - Prop 1 (4 pathway 부호 정리)
  - Alignment 가정 A_EA

§3 Empirical Collapse in Self-Distillation (Observation — Meta-CoT)
  - Meta SFT 성립 근거
  - RL E21R-v2 step 300 붕괴 증거
  - Naive D2 rebuilt trade-off

§4 Epistemic Alignment Distillation (Method)
  - 4 axis framework
  - 6 instantiation 표
  - Main: A ∧ B ∧ F

§5 Experiments
  - Cross-model alignment signature (Llama Countdown + Qwen MATH/AIME)
  - EAD-Main vs baselines
  - Ablation (B, D, F 각각 제거)

§6 Discussion
  - Alignment-first perspective로 distillation literature 재해석
  - Limitation: EV alignment assumption (A_EA)가 학습된 성질이라는 heuristic

§7 Related Work
  - RLCD, REDI, DistiLLM-2, RLSD arXiv:2604.03128, OPSD, GATES, HDPO
  - 차별화: alignment-as-loss, 4-axis framework

§8 Conclusion
```

## 7. Risks + mitigation

| Risk | Mitigation |
|---|---|
| Cross-model (Llama vs Qwen) alignment signature 불일치 | Qwen3-8B base model로 working note 실험 재실행 (plan §5.2의 cross-validation) |
| EAD-Main compute 초과 | 4-node parallel; HF checkpoint 매 20 step push로 preempt 복구 |
| Naive D2 baseline이 이미 약함 | SFT baseline (v8 meta SFT) 별도 비교 대조 |
| H5 (entropy-shape D) target `+0.3 nats` heuristic | Working note Theorem 7.2의 Opener path information ceiling으로 정당화 |
| Decoy quality (C 의 N3) | H3 random vs rule-based ablation |

## 8. Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Plan iteration (this doc) | 0.5 day | Plan approval |
| B, D, F code addition | 2 days | Extended MetaRLSDTrainer |
| Smoke (8 runs × 10 prompts) | 1 day | Bug-free |
| Full run (8 × 2 seeds) | 3.5 days (14 wall-hours × 6 cycles with preempt) | Training done |
| Eval + analysis (EV metrics) | 1 day | Metric tables |
| Cross-validation (Llama Countdown) | 1 day | Paper table 3 |
| Paper draft | 2 days | Complete draft |

**Total**: 11-12 days for full paper readiness.

## 9. Acceptance for coding phase

Plan v1 → critic iteration:
- ✅ Intent explicit (§0)
- ✅ Theory-Observation bridge (§1, §2)
- ✅ Falsifiable hypotheses (H1-H6)
- ✅ Operational metric suite (§5.1)
- ✅ 8-run experiment matrix (§5.2)
- ✅ Success/failure criteria (§5.3)
- ✅ Paper structure (§6)
- ✅ Risk + mitigation (§7)

→ Critic agent로 소독 통과 시 B, D, F 코드 구현 phase 진입.
