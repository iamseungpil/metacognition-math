# 12 GPU 병렬 실험 계획 (E11 분석 결과 기반)

> Historical branch table only. It was written before scenario-C became the active branch.
> Do not use this file as the current execution contract.
> Active plan: `results/plan_metacot_v6.4_active_2026_04_06.md`

**의도**: E11 pilot 분석 후, 3노드 12GPU를 최대 활용하여 다양한 가설을 병렬 검증

---

## E11 분석 후 분기별 실험 배치

### 시나리오 A: approach_change >= 5% + acc >= 60%
"전환 seed 효과 확인됨 → RL 변형 비교"

| 노드 | GPU | 실험 | 의도 | 가설 |
|---|---|---|---|---|
| EVAL 0-1 | 2 | E12: RL R1+R2 | 전환만 강화 | switch reward로 change 10%+ |
| EVAL 2-3 | 2 | E11 추가 eval (AIME 상세) | 전환 품질 분석 | AIME에서 전환 후 정답 비율 |
| TRAIN_B 0-3 | 4 | E13: RL R1+R2+R3+R4+R_len | 전환+과신+검산+효율 | acc >= 63%, ECE < 0.15 |
| E8 0-3 | 4 | E14: RL R1+R2+checkpoint_reward | 전환+Information Checkpointing | rise 패턴 + acc >= 62% |

### 시나리오 B: approach_change 1-5% + acc >= 58%
"방향 맞지만 전환 부족 → seed 확대 + RL 탐험"

| 노드 | GPU | 실험 | 의도 | 가설 |
|---|---|---|---|---|
| EVAL 0 | 1 | 추가 seed 생성 (TRAPI) | seed 164→500 확대 | pair rate 개선 |
| EVAL 1-3 | 3 | E12b: RL num_gen=16 high exploration | 탐험 강화 | num_gen 증가로 전환 발견 |
| TRAIN_B 0-3 | 4 | E15: SFT 500seed + RL R1+R2 | 확대 seed + 전환 RL | change >= 5% 달성 |
| E8 0-3 | 4 | E16: Information Checkpointing RL | rise 패턴 보상 | 복수 meta → 전환 기회 증가 |

### 시나리오 C: approach_change ≈ 0%
"E9 base 너무 rigid → 근본 재설계"

| 노드 | GPU | 실험 | 의도 | 가설 |
|---|---|---|---|---|
| EVAL 0-3 | 4 | V6 10K 데이터 생성 (TRAPI) | clean redirect + verify + straight | V5 theatrical 우회 |
| TRAIN_B 0-3 | 4 | E17: base_sft + 5K clean SFT | base에서 처음부터 | V5 습관 없이 학습 |
| E8 0-3 | 4 | E18: base_sft + RL directly | SFT 없이 RL만 | RL exploration으로 충분? |

---

## 공통 실험 (시나리오 무관)

E11 eval 완료 동시에 시작 가능:

| 실험 | 의도 | GPU |
|---|---|---|
| **V6 10K clean data 생성** | 어떤 시나리오든 본격 10K 필요 | CPU만 (TRAPI API) |
| **E9 baseline 상세 분석** | E11과 동일 5-dimension 분석 비교 | CPU만 |

---

## 가설별 검증 메트릭

| 가설 | 실험 | 검증 메트릭 | 성공 기준 |
|---|---|---|---|
| H1: 전환 seed가 approach_change를 올린다 | E11 pilot | approach_change | >= 5% |
| H2: RL이 유익한 전환만 강화한다 | E12 | switch_success_rate | >= 30% |
| H3: Brier calibration이 과신을 줄인다 | E13 | ECE, FP rate | ECE<0.15, FP<25% |
| H4: Information Checkpointing이 rise 패턴을 유도한다 | E14/E16 | rise_pattern_rate | >= 10% |
| H5: 독립 검산이 실질적 오류를 잡는다 | E13 | verify_effectiveness | > E9(0%) |
| H6: clean 10K가 theatrical 10K보다 낫다 | E17 vs V5 | accuracy | E17 > V5 all_sft(57.1%) |
| H7: num_gen=16 탐험이 전환을 유발한다 | E12b | switch in rollouts | > 0% |
