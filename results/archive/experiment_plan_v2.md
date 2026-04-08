# Meta-CoT Research Plan v2

## Research Goal
"자기가 뭘 모르는지 아는 모델" — metacognitive reasoning을 외현화하여
confidence를 verifiable 수치로 표현하고, 이것이 실제 정확도와 일치하도록 학습.

## 4 Stages

### Stage 1: Meta-CoT SFT ✅
V2 data (4,996 chains, gpt-5.4-mini TRAPI):
- Diverse confidence (mean 0.745, >0.95 = 0%)
- Error→fix patterns (24%)
- Final verification (96%)
- \boxed{} 100%, short meta (18 words/block)

### Stage 2: Calibrated GRPO 🔄
V2+E3 training (step 20/500, reward 0.71):
- correctness + format + meta_quality + group_doubt
- GDPO per-reward normalization
Results so far:
- AIME overconfidence: 97%(V1) → 36%(V2 data) → 14%(V2+E7 GRPO)
- AIME ECE: 0.870 → 0.712 → 0.610

### Stage 3: Selective Abstention ✅ (initial)
V2 E7 conf≥0.7: 60.5% accuracy > Base SFT 58.9%
→ "확신 있을 때만 답하면 더 정확" 증명
→ 1,030 문제로 재검증 필요

### Stage 4: Self-Curation Learning 📋
Meta가 약점 진단 → RAG로 유사 문제 검색 → targeted training
→ AIME rollout에서 "뭘 모르는지" 추출
→ 해당 유형 문제로 curriculum 구성

## Immediate Plan

### Phase A: V2+E3 학습 완료 (~5시간)
- 500 step, 4×A100, ~48s/step

### Phase B: 4 GPU 병렬 eval (max_tokens=4096, 1,030 문제)
- GPU 0: Base SFT
- GPU 1: V2 SFT
- GPU 2: V2+E3
- GPU 3: V2+E7
- 벤치마크: GSM8K 500 + MATH-500 500 + AIME 30
- 전체 completion 저장 (정성분석용)

### Phase C: 종합 분석
- Accuracy table (4 모델 × 3 벤치마크)
- ECE per benchmark
- Selective abstention curve (accuracy vs coverage)
- Bootstrap 95% CI
- Confidence 분포 히스토그램
- 정성 분석: 틀린 문제 패턴, meta 내용

### Phase D: 커리큘럼 러닝 (Phase C 결과가 좋으면)
- AIME rollout → meta 약점 진단
- RAG 검색 → targeted curriculum
- 재학습 → 재평가
