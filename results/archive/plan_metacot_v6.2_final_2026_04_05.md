# Meta-CoT V6.2 Final Plan: Minimal SFT Seed + RL Structural Diversity

**Date**: 2026-04-05
**Codex Reviews**: 4차 (최종 통과)
**Step 0 Result**: E9 구조적 전환 1.8% (17/960) → SFT seed 필수

---

## Step 0: 전환 빈도 측정 ✅ 완료

**결과**: E9에서 구조적 전환 1.8% (17/960), 부분 전환 19.8% (190/960)
**판정**: 1.8% < 5% → SFT seed 필수, RL만으로 불가

---

## Step 1: 전환 SFT Seed 데이터 생성 (~1일)

**의도**: 200개 고품질 전환 trajectory를 생성하여 E9의 action space에 "구조적 전환"을 도입.
**가설**: 독립 생성(진짜 실패 + 진짜 성공) + stitching → V5 "연극적 전환" 문제 해소.

**방법**:
1. TRAPI GPT-5.4-mini로 MATH train 500문제 × 2 독립 풀이 (=1,000 API 호출)
2. (오답, 정답) 쌍 + `_approaches_differ` 필터 → ~100-150 유효 쌍
3. Stitching (GPT-5.4-mini) → ~100개 전환 trajectory (yield 71%)
4. 독립 검산 100개 추가 (풀이 context 제거, 다른 방법으로 검증)
5. 합계: ~200개 seed

**검증**:
- 수동 검수 50개: 전환 지점이 실제 오류 위치와 일치하는가?
- `_approaches_differ`가 pre/post에서 True인가?
- confidence < 0.4이 전환 지점에 있는가?
- 최종 답이 gold와 일치하는가?

---

## Step 2: Minimal SFT (~0.5일)

**의도**: E9 위에 전환 seed를 catastrophic forgetting 없이 심기.

**가설**: 전환 200개를 E9에 가볍게 심으면, approach_change가 1.8% → 5%+ 상승.

**방법**:
- Base: E9 checkpoint (62.1%, meta 93%)
- 데이터: **전환 200개 + E9 기존 데이터 1,800개 = 2,000개 mix (전환 10%)**
  - Codex 수정 반영: 20% → 10%로 비율 하향 (forgetting 방지)
- LR: 1e-6 (매우 낮게)
- Epochs: 2
- max_completion_length: 2048
- **Early stopping**: MATH accuracy가 E9(44.0%) 대비 -2pp (42.0%) 이하면 즉시 중단

**검증 (go/no-go gate)**:
- approach_change >= 5% → Step 3 진행
- approach_change 1-5% → seed 200개 추가 생성 후 재시도 (4K mix, 전환 10%)
- approach_change < 1% → 방법 재검토 (seed 품질 문제)
- accuracy >= 60% (E9 62.1% 대비 -2pp 이내)

---

## Step 3: RL with Structural Diversity Reward (~3-5일)

**의도**: seed로 도입된 전환 행동을 RL이 정답률 기반으로 최적화.

**가설**: SFT에서 approach_change 5%+ 달성 후, RL의 diversity reward가 유익한 전환은 강화하고 무익한 전환은 약화.

**Reward 설계** (Codex 수정 반영):
```python
# Rule-based proxy (비용 0, RL loop 안에서 즉시 계산)
def structural_diversity_reward(completion):
    """Binary: 전환 전후 method family가 구조적으로 다른가?"""
    meta_pos = completion.find('<|/meta|>')
    if meta_pos < 0:
        return 0.0
    pre = completion[:completion.find('<|meta|>')]
    post = completion[meta_pos + len('<|/meta|>'):]
    if _approaches_differ_lightweight(pre, post):  # keyword-based, no LLM
        return 1.0
    return 0.0

# 최종 reward
reward = correctness * (1.0 + 0.3 * structural_diversity)
# correctness=0이면 diversity도 0 → 연극적 전환 보상 불가
```

**Training config**:
- num_generations: 8 (탐색 확대)
- temperature: 0.9
- max_steps: 800
- 데이터: mixed_train 4,996개
- max_completion_length: 2048
- Reward: correctness(1.0) + structural_diversity(0.3) — **2개만** (E10 교훈)

**검증**:
- approach_change >= 10%
- accuracy >= 62% (E9 유지)
- switch_success_rate > 30% (전환 후 정답 비율)
- **최종 목표**: accuracy >= 67.1% (base_sft 이상)

---

## Step 4: 비교 평가

5개 모델 비교 (n=1030):

| 모델 | 설명 |
|---|---|
| base_sft | No meta baseline (67.1%) |
| E9 | RL only, no switch reward (62.1%) |
| E_seed (Step 2) | E9 + SFT 200개 seed |
| E_seed+RL (Step 3) | E_seed + RL with diversity |
| V5_redirect | 기존 V5 redirect SFT (대조군) |

**Metrics**: accuracy, approach_change, switch_success_rate, ECE, pass@1 vs pass@8

---

## Go/No-Go Gates 요약

| Gate | 조건 | 결과 |
|---|---|---|
| Step 0 → Step 1 | 전환 빈도 < 5% | ✅ 1.8% → SFT 필수 |
| Step 2 → Step 3 | change >= 5% AND acc >= 60% | 측정 후 |
| Step 2 fail | change < 5% | seed 200개 추가, 4K mix 재시도 |
| Step 3 → 성공 | acc >= 67.1% (base_sft) | 최종 목표 |
| Step 3 부분 성공 | acc >= 62% + change >= 10% | RL 파라미터 조정 |

---

## 노드 할당

| 노드 | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---|---|---|---|
| TRAIN_B | 데이터 생성 (API) | SFT 학습 | - | eval |
| EVAL | - | eval | eval | eval |
| E8 | - | - | RL 학습 | eval |
