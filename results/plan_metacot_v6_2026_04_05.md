# Meta-CoT V6 Plan: Diagnosis-Execution Gap 해소

**Date**: 2026-04-05
**Based on**: study_metacot_v5_2026_04_05.md
**Core Problem**: Meta가 100% 정확한 진단을 하지만 0% 전환 실행 — SFT 데이터에 전환 trajectory 부재

---

## Phase 0: Pass@k 분석 (전환 가능성 upper bound 확인)

**의도**: 모델이 대안적 풀이 경로를 가지고 있는지 확인. 전환을 학습시키기 전에 전환할 경로가 존재하는지 먼저 검증.

**가설 H0**: Qwen3-8B는 MATH500에서 pass@1 < pass@8이며, gap >= 10pp. 즉, 첫 시도에 실패해도 다른 시도에서 성공할 수 있는 문제가 충분히 존재한다.

**검증 방법**:
- base_sft로 MATH500 500문제에 대해 k=1,4,8,16 sampling (temperature=0.7)
- per-problem pass rate 계산
- pass@k curve 그리기
- **판정 기준**:
  - pass@8 - pass@1 >= 10pp → Phase 1로 진행 (전환 가치 있음)
  - pass@8 - pass@1 < 5pp → 전환 학습 무의미 → inference-time 방법으로 전환
  - 5pp <= gap < 10pp → 부분적 가치 → meta-lite로 overhead 줄이기 우선

**실험 코드**: `scripts/compute_pass_at_k.py`
**필요 자원**: 1 GPU, ~4-8시간 (500문제 × 16 samples)

---

## Phase 1: 전환 Trajectory SFT 데이터 생성

**의도**: "진단 → 구체적 전환 → 성공" 패턴을 학습 데이터에 추가하여 Diagnosis-Execution Gap 해소.

**가설 H1**: SFT 데이터에 "wrong route → meta diagnosis → explicit switch → correct route → answer" trajectory를 50-100개 추가하면, AIME/MATH500에서 approach_change > 0이 되고, partial_switch 비율이 현재 3/28(11%)에서 최소 30%로 증가한다.

**검증 방법**:
- Phase 0의 pass@k 데이터에서 "첫 시도 실패 + 다른 시도 성공" 문제 추출
- 두 trajectory의 풀이 방법이 구조적으로 다른 것만 필터링
- Meta token 삽입: `<|meta|>confidence: 0.3\nroute is weak: [구체적 이유]\nswitch to: [구체적 대안 방법]<|/meta|>`
- 이 데이터로 verify_sft 위에 추가 SFT (기존 4,996 + 전환 50-100)
- Eval: AIME 30 + MATH500에서 approach_change 비율 측정
- **판정 기준**:
  - approach_change >= 30% AND accuracy >= verify_sft(60.3%) → Phase 2로
  - approach_change >= 30% AND accuracy < verify_sft → 전환은 되지만 품질 부족 → 데이터 정제
  - approach_change < 10% → 데이터 양 부족 → 100→500개로 확대

**실험 코드**: `scripts/build_switch_trajectory_data.py`
**필요 자원**: GPT-5.4-mini API (trajectory 생성), 1 GPU (SFT 학습)

---

## Phase 2: Conditional RL (전환 + 정답 보상)

**의도**: RL로 전환 행동을 강화하되, "전환 + 정답"에만 보상하여 불필요한 전환 방지.

**가설 H2**: Phase 1 SFT를 warm-start로 사용하고, route_switch_evidence_reward에 conditional gate (정답 시에만 보상)를 추가하면, accuracy가 verify_sft(60.3%) 이상이면서 과신률 < 20%를 달성한다.

**검증 방법**:
- Phase 1 SFT checkpoint에서 GDPO RL 시작
- 데이터: mixed_train 4,996개 (SFT와 같은 규모)
- max_completion_length: 2560 (meta overhead 허용)
- num_generations: 4, max_steps: 800-1200
- Reward: correctness(3.0) + conditional_switch_reward(1.0) + calibration(0.5) + omission_floor(0.5)
- **conditional_switch_reward**: 전환 증거가 있고 정답일 때만 보상, 전환했는데 오답이면 0 (패널티 아님)
- **판정 기준**:
  - accuracy >= 63% (base_sft 67.1%의 94%) + ECE < 0.15 → 성공
  - accuracy >= 60% + approach_change > 20% → 부분 성공 → reward 가중치 조정
  - accuracy < 58% → RL이 능력 훼손 → correctness weight 더 높이기

**실험 코드**: `src/training/grpo_v2.py` (E11 mode 추가)

---

## Phase 3: Inference-time Re-routing (학습 불필요)

**의도**: meta의 "감지" 능력을 inference-time에서 활용. meta가 "route is weak" 출력 시 현재 답을 버리고 re-sample.

**가설 H3**: meta가 "route is weak"을 출력한 문제에서 re-sample하면, pass@1 대비 +3-5pp 향상. 특히 MATH500에서 효과 기대.

**검증 방법**:
- verify_sft 또는 all_sft로 1030문제 생성
- meta가 "route is weak" + confidence < 0.4인 문제 식별 (~200개 예상)
- 이 문제들에 대해 meta 없이 재생성 (temperature 0.9로 다양성 확보)
- 1차 생성 vs 2차 재생성 중 더 나은 답 선택 (oracle: gold와 비교)
- **판정 기준**:
  - oracle re-routing accuracy > 1차 accuracy + 5pp → meta 감지가 유용한 routing signal
  - oracle re-routing ≈ 1차 accuracy → meta 감지가 routing에 무용 → 다른 signal 필요

**실험 코드**: `scripts/meta_rerouting_experiment.py`
**필요 자원**: 1 GPU, ~6시간

---

## Phase 4: Meta-lite Format (토큰 overhead 최소화)

**의도**: meta 블록을 ~10 tokens로 축소하여 풀이 토큰 보존. Phase 1-2가 효과 있으면 적용.

**가설 H4**: meta-lite(`<|meta|>c=L;a=verify;r=calc<|/meta|>`, ~10 tokens)로 SFT하면, 현재 full meta SFT 대비 accuracy +3-5pp이면서 같은 수준의 meta 행동 보존.

**검증 방법**:
- 기존 4,996 SFT 데이터의 meta 블록을 meta-lite로 변환
- SFT 학습 후 eval
- **판정 기준**:
  - accuracy > all_sft(57.1%) + 3pp = 60% 이상 → lite format 채택
  - accuracy ≈ all_sft → lite가 정보 손실 → verbose meta 유지

---

## 실행 순서 및 의존성

```
Phase 0 (pass@k) ──── 결과에 따라 ────┐
                                        │
                   gap >= 10pp ──→ Phase 1 (전환 SFT) ──→ Phase 2 (conditional RL)
                                        │
                   gap < 5pp ───→ Phase 3 (re-routing) + Phase 4 (meta-lite)
                                        │
                   5-10pp ──────→ Phase 3 + Phase 1 (smaller scale)
```

**Phase 3은 Phase 0과 병렬 실행 가능** (기존 모델로 즉시 가능)
**Phase 4는 Phase 1-2 결과를 보고 결정**

---

## 목표 메트릭

| 메트릭 | 현재 Best | 단기 목표 | 최종 목표 |
|---|---|---|---|
| Overall accuracy | 60.7% (E9c) | >= 63% | >= 67.1% (base_sft) |
| AIME accuracy | 10.0% (E9c) | >= 13.3% (+1문제) | >= 20% (+3문제) |
| approach_change rate | 0% | >= 20% | >= 40% |
| ECE | 0.129 (E8) | < 0.15 | < 0.10 |
| Overconfidence rate | 10% (E8) | < 15% | < 10% |
