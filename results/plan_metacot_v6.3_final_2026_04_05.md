# Meta-CoT V6.3 Final Plan: SFT Seed + RL with Structural Switch & Calibration

> Historical document only. This plan was superseded after the `E11` gate result.
> Do not use this file to decide current runtime behavior.
> Active plan: `results/plan_metacot_v6.4_active_2026_04_06.md`

**Date**: 2026-04-05
**Codex Reviews**: 6차 (최종 통과)
**Core Intent**: Meta cognition → 실제 전환 → test time 문제 풀이 능력 향상

---

## 핵심 의도

1. **과신 검산**: confidence 높을 때 독립 검산 → 오류 발견 시 수정
2. **Low confidence 전환**: confidence 낮을 때 다른 방법으로 구조적 전환
3. **맞추기**: 위 행동이 실제 정답률 향상으로 이어짐

---

## V6.1 Reward 설계 (Codex 통과)

```
R1: correctness           weight=1.0   정답 1.0, 오답 0.0
R2: structural_switch     weight=0.3   keyword-based binary
                                        + 불필요 전환 시 -0.1
R3: calibration (Brier)   weight=0.2   1 - (conf - correct)^2
R4: verify_outcome        weight=0.2   검산+정답 → +0.2
                                        검산+오답 → -0.1
R_len: efficiency         weight=0.1   정답 시 (1 - len/max_len) * 0.1
```

**Weight 합산**: 1.8 (R1이 55.6%로 dominant)

---

## 3노드 병렬 실험

### E11 (EVAL 노드): SFT Seed

**의도**: 전환+검증 행동을 E9의 action space에 도입
**가설**: SFT 164개 seed → approach_change 1.8% → 5%+
**Base**: E9 checkpoint (62.1%)
**데이터**: 전환87 + 검증77 = 164 seed + E9 기존 1476 = 1640 mix (seed 10%)
**Config**: LR 1e-6, epochs 2, max_completion_length 2048
**검증**: approach_change >= 5%, accuracy >= 60%
**Early stop**: MATH accuracy < E9(44.0%) - 2pp = 42.0% → 즉시 중단

### E12 (TRAIN_B 노드): RL R1+R2 (전환 강화)

**의도**: 전환 행동을 정답률 기반으로 강화 — 유익한 전환만 남기기
**가설**: SFT에서 심은 전환을 RL이 최적화 → approach_change 10%+, accuracy >= 62%
**Base**: E11 checkpoint
**Reward**: R1(1.0) + R2(0.3) — 2개만 (간결, E10 교훈)
**Config**: num_gen 8, temp 0.9, steps 800, data mixed_train 4996
**검증**: approach_change >= 10%, accuracy >= 62%
**Go/No-go**: 
  - change >= 7% AND acc >= 60% → 성공, E13 결과와 비교
  - change < 7% OR acc < 60% → E11 SFT data 증량 후 재시도

### E13 (E8 노드): RL Full (전환+과신+검산+효율)

**의도**: 전환 + 과신 체크 + 검산 + 효율성을 종합 최적화
**가설**: 전체 reward 조합이 accuracy >= 63%, ECE < 0.15 달성
**Base**: E11 checkpoint
**Reward**: R1(1.0) + R2(0.3) + R3(0.2) + R4(0.2) + R_len(0.1) — 5개
**Config**: num_gen 8, temp 0.9, steps 800, data mixed_train 4996
**검증**: accuracy >= 63%, ECE < 0.15, approach_change >= 10%
**목표**: accuracy >= 67.1% (base_sft 이상)

---

## 타임라인

```
Day 0 (즉시):
  EVAL: E11 SFT 시작 (2-4시간)
  TRAIN_B + E8: 환경 셋업 + dry run 준비

Day 0.5:
  E11 완료 → checkpoint 배포
  EVAL: E11 eval (3시간)
  TRAIN_B: E12 RL 시작
  E8: E13 RL 시작

Day 1-3:
  TRAIN_B: E12 RL (800 steps)
  E8: E13 RL (800 steps)
  EVAL: E12/E13 중간 checkpoint eval

Day 3:
  E12/E13 eval → 비교 분석
  → E14 설계 (entropy-aware, logprobs 기반)
```

---

## 비교 평가 (Step 4)

| 모델 | 설명 | 검증 메트릭 |
|---|---|---|
| base_sft | No meta (67.1%) | accuracy baseline |
| E9 | Best RL no switch (62.1%) | accuracy, meta% |
| E11 | SFT seed only | approach_change, accuracy |
| E12 | RL R1+R2 (switch) | switch_rate, accuracy |
| E13 | RL full V6.1 | accuracy, ECE, switch, verify |

---

## Go/No-Go Gates

| Gate | 조건 | 결과 |
|---|---|---|
| E11 → E12/E13 | change >= 5% AND acc >= 60% | 진행 |
| E11 fail | change < 5% | seed 추가 200개 생성 |
| E12 success | change >= 7% AND acc >= 60% | 결과 비교 |
| E12 fail | change < 7% OR acc < 60% | SFT data 증량 |
| E13 success | acc >= 63% AND ECE < 0.15 | **목표 근접** |
| **최종** | acc >= 67.1% | **base_sft 초과 — RQ1 해소** |
