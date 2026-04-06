# Meta-CoT V6.1 Plan: Diagnosis-Execution Gap 해소 (Codex 2차 검토 통과)

**Date**: 2026-04-05
**Based on**: study_metacot_v5 + Codex 비판 반영
**Core Problem**: Meta 진단 100% 정확, 전환 실행 0%
**Codex Review**: 2차 통과 (2건 보완 반영)

---

## Phase 0: 진단 (~4시간, 즉시 시작)

### P0-1: Full Eval 완료 대기
**의도**: partial GSM8K score로는 모델 선택 불가. full 1030 결과만 사용.
**상태**: E5/E9 (EVAL), E6/E7 (E8) 진행 중. 자동 완료 대기.

### P0-2: Pass@k Quick Trend
**의도**: 모델이 대안적 풀이 경로를 가지고 있는지 빠르게 확인.
**가설**: pass@8 - pass@1 >= 10pp이면 전환 학습의 가치가 있다.
**방법**: base_sft + all_sft 각 100문제 × 16 samples, 2+2 GPU 샤딩.
**검증**: 
- gap >= 10pp → Phase 1 full (전환 trajectory 500-800개)
- 5pp <= gap < 10pp → Phase 1 축소 (전환 trajectory 300개)
- gap < 5pp → Phase 1 skip → meta-lite (Phase 4)만

### P0-3: 토큰 Budget Ablation
**의도**: max_completion_length가 성능에 미치는 영향 정량화.
**가설**: meta 포함 모델은 2048보다 긴 budget이 필요할 수 있다.
**방법**: base_sft를 max_tokens 512/1024/2048/4096으로 100문제 eval.
**검증**: 2048→4096 accuracy gap > 2pp이면 training length 확대 필요.

### P0-4: Full Pass@k (승자만)
**의도**: quick trend의 대표성 보강.
**방법**: P0-2 승자만 500문제 full pass@k.
**전제**: P0-2 완료 후.

---

## Phase 1: 전환 학습 데이터 (Phase 0 후, ~3일)

### P1-1: 전환 Trajectory 데이터 생성
**의도**: SFT 데이터에 없는 "진단→전환→성공" 패턴 추가.
**가설**: 500-800개 전환 trajectory로 approach_change > 0% 달성.
**방법**: 2단계 생성
  1. 문제별 GPT-5.4-mini로 2+ 서로 다른 풀이 생성
  2. 첫 번째 오답 + 두 번째 정답 쌍 필터링
  3. Stitching: 첫 시도 → meta("route is weak, switch to X") → 두 번째 시도
**목표**: pass@k gap >= 10pp → 500-800개, 5-10pp → 300개
**검증**: 데이터 수, trajectory 품질 (수동 50개 검수)

### P1-2: Few-shot 전환 진단
**의도**: 전환 실패가 지식 부족인지 실행 능력 부족인지 진단.
**가설**: few-shot으로 전환 예시를 보여주면 accuracy +3pp 이상.
**방법**: 0/2/4-shot × base_sft/all_sft, MATH500 100문제.
**검증**: 
- few-shot accuracy - 0-shot accuracy >= 3pp → 실행 문제 (SFT로 해결)
- gap < 1pp → 지식 문제 (RAG/curriculum 필요)

### P1-3: 전환 SFT 학습
**의도**: 전환 행동을 모델에 심기.
**방법**: 기존 4,996 + 전환 500-800개로 SFT.
**Base model**: Phase 0 full eval 최고 성적 meta_sft 모델.

### P1-4: 전환 SFT Eval
**의도**: 전환이 실제로 일어나는지 확인.
**검증**: 
- **성공**: approach_change >= 20% AND accuracy >= 60% → Phase 2 full
- **부분 성공**: approach_change >= 10% AND accuracy >= 58% → Phase 2 pilot (축소)
- **실패**: approach_change < 10% → 데이터 확대 (800→2000) 또는 방법론 재검토

---

## Phase 1.5: Curriculum/RAG 진단 (Phase 1과 병렬, ~1일)

### P1.5-1: Curriculum Ordering
**의도**: 학습 순서가 전환 학습에 영향 미치는지 확인.
**가설**: easy→medium→hard 순서가 random보다 전환 학습에 유리.
**방법**: 같은 step budget, curriculum vs random.
**검증**: curriculum accuracy - random accuracy >= 2pp이면 Phase 2 학습 순서를 curriculum으로 고정.

### P1.5-2: RAG Diagnostic
**의도**: 비슷한 풀린 문제를 context로 주면 전환이 개선되는지.
**방법**: exemplar bank에서 top-2 retrieval, inference-only.
**검증**: RAG accuracy - no-RAG accuracy >= 3pp이면 Phase 2에 retrieval 통합.
**Phase 2 연결**: RAG가 유효하면 RL inference에 retrieval-augmented generation 추가.

---

## Phase 2: RL (Phase 1 성공 후, ~5일)

### P2-1: V3 Additive Reward
**의도**: V2의 chicken-and-egg 문제를 회피한 단순한 reward 설계.
**V2 처분**: 폐기. same_route_repetition_penalty, route_switch_evidence_reward, confidence_omission_floor 모두 drop.
**V3 설계**: 
```
R = 1.0 * task_correct 
  + 0.1 * format_valid 
  + 0.2 * transition_valid (전환 증거 AND 정답)
  - 0.05 * overlength_penalty
```
**가설**: additive reward로 전환 + 정확도 동시 개선.

### P2-2: Reward Ablation
**방법**: 3개 조건 비교
  - correct only (1.0 * task_correct)
  - correct + format (1.0 + 0.1)
  - V3 full (1.0 + 0.1 + 0.2 - 0.05)
**검증**: V3 full이 correct only 대비 accuracy 유지 + approach_change 증가.

### P2-3: 전제조건
- Phase 1 P1-4 "성공" 또는 "부분 성공" 통과
- Phase 0 base model 선택 완료
- Phase 1.5 결과 반영 (curriculum/RAG 유효하면 통합)

### P2-4: 판정 기준
- **성공**: accuracy >= 63% + ECE < 0.15
- **부분 성공**: accuracy >= 60% + approach_change > 20%
- **실패**: accuracy < 58% → correctness weight 조정 + KL penalty 증가

---

## 판정 기준 요약

| Gate | 조건 | 결과 |
|---|---|---|
| Phase 0 pass@k | gap >= 10pp | Phase 1 full (500-800) |
| Phase 0 pass@k | 5-10pp | Phase 1 축소 (300) |
| Phase 0 pass@k | < 5pp | Phase 4 (meta-lite) only |
| Phase 1 P1-4 | change >= 20% + acc >= 60% | Phase 2 full |
| Phase 1 P1-4 | change >= 10% + acc >= 58% | Phase 2 pilot |
| Phase 1 P1-4 | change < 10% | 데이터 확대 또는 재검토 |
| Phase 2 P2-4 | acc >= 63% + ECE < 0.15 | 성공! |

---

## 목표 메트릭

| 메트릭 | 현재 Best | 단기 | 최종 |
|---|---|---|---|
| Overall accuracy | 60.7% (E9c) | >= 63% | >= 67.1% |
| AIME accuracy | 10.0% | >= 13.3% | >= 20% |
| approach_change | 0% | >= 20% | >= 40% |
| ECE | 0.129 (E8) | < 0.15 | < 0.10 |
