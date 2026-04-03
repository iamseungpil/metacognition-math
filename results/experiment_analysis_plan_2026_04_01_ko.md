# Meta-CoT 실험 계획 및 분석 계약서 (2026-04-01)

## 1. 큰 질문

이 프로젝트의 큰 질문은 하나다.

`OOD test-time setting에서 학습된 metacognitive control이 스타일이 아니라 실제 행동 변화를 통해 문제 해결을 개선할 수 있는가`

이 질문을 세 개의 연구 질문으로 나눈다.

1. `RQ1: Meta-CoT`
   - 모델이 ordinary CoT와 분리된 metacognitive control state를 학습하고, 그것이 test-time adaptation의 기반이 될 수 있는가
2. `RQ2: Meta-RL`
   - 그 metacognitive state를 verifiable reward로 바꿔 control policy로 학습시킬 수 있는가
3. `RQ3: Curriculum`
   - 같은 metacognitive state를 diagnosis, retrieval, adaptation의 trigger로 확장할 수 있는가

## 2. 프로젝트 의도

### 2.1 의도 A: Meta-CoT는 self-talk가 아니라 controller여야 한다

우리가 원하는 것은 답을 길게 쓰게 하는 것이 아니다.

1. `confidence`는 말버릇이 아니라 제어 변수여야 한다
2. `something feels off`나 의미 있는 confidence drop은 redirect로 닫혀야 한다
3. `confidence는 높은데 근거가 얇다`는 상태는 verify로 닫혀야 한다
4. `diagnosis`는 왜 현재 경로가 약한지 설명해야 한다
5. `study_need`는 무엇이 부족한지 parseable하게 드러내야 한다
6. meta block은 ordinary derivation과 엄격히 분리되어야 한다

즉 meta block에는 다음만 들어가야 한다.

1. local confidence
2. anomaly / conflict notice
3. failure diagnosis
4. next control action
5. optional `study_need`

ordinary algebra 전개나 전체 CoT는 meta 안에 들어오면 안 된다.

### 2.2 의도 B: Meta-RL은 이 행동들을 verifiable하게 만들어야 한다

RL의 목적은 reward를 한 번에 다 섞는 것이 아니다.

1. 먼저 calibration을 맞춘다
2. 그 다음 intervention 주변의 confidence revision을 맞춘다
3. 그 다음 verify / redirect / diagnosis를 분해해서 본다
4. 마지막에만 full controller를 합친다

이 분해는 최종 점수만이 아니라 어떤 reward가 어떤 행동을 만드는지 보기 위한 것이다.

### 2.3 의도 C: Curriculum은 diagnosis에서 시작해야 한다

Curriculum과 RAG는 uncertainty만으로 켜지면 안 된다.

1. 왜 못 푸는지 진단해야 한다
2. 무엇을 더 배워야 하는지 `study_need`로 드러나야 한다
3. 그때만 retrieval이나 one-example adaptation이 정당화된다

장기 목표는 다음 loop다.

`diagnose -> expose study_need -> retrieve/adapt -> retry`

## 3. 의도 / 가설 / 검증 방법

### 3.1 RQ1: Meta-CoT

`의도`

1. 순수 metacognitive control state를 학습시킨다
2. meta와 derivation을 분리한다
3. 그 state가 hard / OOD setting에서 실제 test-time adaptation을 가능하게 하는지 본다

`가설`

1. `H1a`
   - meta-supervised SFT는 ordinary CoT와 분리된 parseable한 control state를 형성할 수 있다
2. `H1b`
   - 학습된 meta trace는 decorative CoT가 아니라 이후 verify / redirect / diagnosis 같은 test-time adaptation 행동을 실제로 유도하는 controller state를 나타낸다
3. `H1c`
   - diagnosis와 `study_need`는 이후 curriculum이나 retrieval trigger로 사용할 수 있을 만큼 안정적으로 parseable해질 수 있다
4. `H1d`
   - 같은 base reasoning 능력을 크게 깨지 않으면서도, hard slice에서는 retry / adaptation gain을 만들어 낼 수 있다

`검증 방법`

1. meta parse rate
2. confidence extraction rate
3. meta purity
   - meta 안에 diagnosis / control state / next action이 들어가는가
   - meta 안에 ordinary derivation이 섞이지 않는가
4. adaptation precursor coverage
   - hard problem에서 verify / redirect / diagnosis를 정당화하는 meta signal이 실제로 나타나는가
5. adaptation lift
   - hard slice에서 `first_completion -> intervention/retry completion`의 정확도 delta
   - redirect / verify가 있는 sample과 없는 sample의 retry gain 비교
6. difficulty-sliced qualitative analysis
   - AIME / hard math failure에서 meta가 어떤 방식으로 행동을 바꾸는가
7. accuracy retention

### 3.2 RQ2: Meta-RL

`의도`

1. confidence calibration을 개선한다
2. intervention-local confidence revision을 학습시킨다
3. verify / redirect / diagnosis 효과를 분리해서 본다
4. 마지막에 이 효과들이 합쳐져 controller가 되는지 본다

`가설`

1. `H2a`
   - `E3`는 explicit behavior reward 없이도 calibration을 개선한다
2. `H2b`
   - `E5`는 conflict나 anomaly 주변에서 confidence revision을 개선한다
3. `H2c`
   - `E8`은 `E5`보다 hard wrong case의 과신을 더 잘 낮춘다
4. `H2d`
   - `E9 / E9b / E9c`는 verify / redirect / diagnosis 효과를 분해해 보여준다
5. `H2e`
   - `E10`은 개별 reward만 있을 때보다 더 강한 full controller를 만든다
6. `H2f`
   - `E6 / E7`은 probe가 single trajectory correctness가 아니라 `p(correct | prefix)`를 맞출 때만 의미가 있다

`검증 방법`

1. benchmark accuracy
2. ECE / Brier / wrong-answer mean confidence / wrong high-confidence rate
3. conflict-conditioned confidence drop
4. high-confidence 상황에서의 verify precision
5. redirect-conditioned strategy switch와 recovery
6. diagnosis consistency와 usefulness
7. hard problem에서 repeated intervention quality
8. reward 평균뿐 아니라 qualitative response 분석

### 3.3 RQ3: Curriculum

`의도`

1. diagnosis를 action으로 연결한다
2. `study_need`를 parseable retrieval trigger로 만든다
3. failure analysis를 no-training RAG 혹은 one-example adaptation과 연결한다

`가설`

1. `H3a`
   - decorative low confidence만으로 retrieval이 켜지면 안 된다
2. `H3b`
   - diagnosis와 `study_need`가 있을 때 retrieval precision이 높아진다
3. `H3c`
   - diagnosis가 의미 있을 때만 retrieved example이나 one-example adaptation이 retry accuracy를 높인다

`검증 방법`

1. retrieval trigger precision
2. diagnosis / `study_need` coverage
3. retry prompt artifact logging
4. retry accuracy delta
5. 재현 가능한 저장 산출물

## 4. 실험 매트릭스

| 실험 | 의도 | 가설 | 검증 방법 |
|---|---|---|---|
| `V2 / V3 / V5 SFT` | parseable meta representation 확보 | H1a, H1b, H1c | parseability, purity, confidence extraction, accuracy retention |
| `control_v5_verify_sft` | verify controller 단독 학습 | verify를 redirect 없이도 학습 가능 | high-confidence verify precision |
| `control_v5_redirect_sft` | redirect controller 단독 학습 | redirect를 verify 없이도 학습 가능 | redirect recovery, strategy switch |
| `control_v5_all_sft` | unified controller SFT | verify / redirect / diagnosis 공존 | unified behavior with limited accuracy loss |
| `E3` | pure calibration | H2a | ECE, Brier, wrong high-confidence rate |
| `E5` | calibration + confidence revision | H2b | conflict-conditioned confidence drop, no-drop wrong-commit |
| `E6` | probe calibration | H2f | `|confidence - p_hat_probe|`, probe-aligned ECE |
| `E7` | probe + blockwise stepwise | H2f | block-level probe gap, intervention-local calibration |
| `E8` | anti-overconfidence shaping 강화 | H2c | hard-slice calibration, wrong-high-confidence suppression |
| `E9` | verify-only decomposition | H2d | verify precision, verify-conditioned error |
| `E9b` | redirect-only decomposition | H2d | redirect recovery, real switch fraction |
| `E9c` | diagnosis-only decomposition | H2d | diagnosis consistency, `study_need` usefulness |
| `E10` | full combined controller | H2e | verify + redirect + diagnosis closure |
| `Curriculum / RAG` | weakness-conditioned retrieval/adaptation | H3a, H3b, H3c | trigger precision, retry gain, `study_need` quality |

## 4.1 현재 probe-free RL 완료 범위

현재 probe 없이 해석 가능한 RL 축은 아래 실험들이다.

1. `E3`
   - calibration only
2. `E5`
   - calibration + confidence revision
3. `E8`
   - `E5` + anti-overconfidence shaping
4. `E9`
   - `E8` + verify only
5. `E9b`
   - `E8` + redirect only
6. `E9c`
   - `E8` + diagnosis / decomposition only
7. `E10`
   - full controller

즉 현재 probe gate가 없어도 비교 가능한 RL 분해 축은 이미 준비되어 있고,
`E6/E7`만 prefix-probe gate 이후에 추가된다.

## 5. Reward 분해 계약

reward family는 아래처럼 고정한다.

1. `E3`
   - pure calibration baseline
2. `E5`
   - `E3 + confidence_revision`
3. `E6`
   - `E3 + probe_calibration`
4. `E7`
   - `E6 + stepwise_probe`
5. `E8`
   - `E5 + overconfidence shaping`
6. `E9`
   - `E8 + verify only`
7. `E9b`
   - `E8 + redirect only`
8. `E9c`
   - `E8 + diagnosis / decomposition only`
9. `E10`
   - full combined controller

이 분해는 다음 질문들에 답하기 위해 필요하다.

1. calibration만으로 무엇이 바뀌는가
2. revision이 추가되면 무엇이 바뀌는가
3. 어떤 behavior reward가 어떤 행동을 바꾸는가
4. full controller가 decomposition 대비 추가 이득을 주는가

## 6. Probe 계약

probe는 style classifier가 되면 안 된다.

1. 의도된 target은 `p(correct | prefix)`다
   - 여기서 prefix는 completion 조각 alone이 아니라 reward-time과 동일한 `prompt + completion-prefix` 객체여야 한다
2. single rollout의 final correctness는 prefix-local uncertainty supervision으로 충분하지 않다
3. `E6/E7`은 prefix마다 multiple continuation에서 얻은 target이 준비됐을 때만 진행한다
4. 그 전까지는 probe-free RL만 진행하고 probe-dependent RL은 멈춘다

probe의 최소 검증은 다음과 같다.

1. held-out Brier
2. held-out ECE
3. stated confidence와 `p_hat_probe`의 상관
4. held-out target이 binary할 때만 AUROC를 참고한다
5. temperature scaling 이후 probe 자체 calibration
6. `problem_id` 기준 group split

## 7. Curriculum 계약

curriculum retrieval은 아래 조건을 모두 만족할 때만 유효하다.

1. low confidence alone으로 retrieval이 켜지지 않는다
2. diagnosis 또는 `study_need`가 존재한다
3. retrieved example이 실제 retry prompt에 들어간다
4. retry artifact가 전부 저장된다

즉 curriculum의 목표는 `uncertain하면 일단 retrieve`가 아니다.
`왜 현재 경로가 부족한지 알 때만 retrieve`다.

## 7.1 Curriculum 벤치 선택 계약

curriculum / RAG의 1차 주 무대는 `AIME`나 일반 `open math`의 hard slice가 맞다.

이유는 다음과 같다.

1. 이 구간에서만 redirect / diagnosis / study_need가 실제로 필요해진다
2. easy slice에서는 retrieval이 token cost만 늘리고 행동 차이를 왜곡할 수 있다
3. 장기적으로는 `AIME -> broader open math` 순으로 확장하는 것이 맞다

따라서 1차 검증은 다음 순서로 둔다.

1. `AIME / hard benchmark`에서 trigger precision과 retry gain 검증
2. 이후 `broader open math`에서 generalization 확인

## 8. 행동 분석 및 저장 계약

행동 분석은 정량만으로 끝나면 안 된다.

반드시 아래 두 층위를 함께 본다.

### 8.1 정량 분석

1. accuracy
2. confidence coverage
3. bucketed ECE / Brier
4. wrong-answer mean confidence
5. wrong high-confidence rate
6. verify / redirect / diagnosis / `study_need` rate
7. benchmark별 집계

### 8.2 정성 분석

1. `AIME` hard wrong case에서의 overconfidence suppression
2. anomaly 이후 실제 redirect가 일어나는지
3. diagnosis가 단순 장식이 아니라 실패 원인 설명인지
4. difficulty에 따라 meta behavior가 어떻게 달라지는지
5. retrieved example 이후 retry answer가 어떻게 변하는지

### 8.3 저장 산출물

eval은 나중에 전수 분석이 가능하도록 아래를 저장해야 한다.

1. `full_question`
2. `completion`
3. `first_completion`
4. `avg_confidence`
5. `meta_confidences`
6. `rag_used`
7. `retrieved_questions`
8. `retrieval_scores`
9. `rag_diagnosis`
10. metadata json + parquet bundle

즉 summary만 저장하는 것이 아니라, 응답 단위 raw artifact를 반드시 남긴다.

## 9. 실행 게이트

새로운 대규모 실험 launch는 아래 게이트를 만족할 때만 진행한다.

### Gate 1. 코드 안정성

필수 조건:

1. tokenizer compatibility가 local과 remote `transformers`에서 모두 안전하다
2. reward config가 문서의 decomposition과 일치한다
3. curriculum smoke가 통과한다
4. launch script와 core module이 compile된다

### Gate 2. Probe-free RL

Gate 1이 통과하면 허용된다.

실행 대상:

1. `E3`
2. `E5`
3. `E8`
4. `E9`
5. `E9b`
6. `E9c`
7. `E10`

### Gate 3. Probe-dependent RL

아래 조건을 모두 만족할 때만 허용된다.

1. prefix-conditioned target이 존재한다
2. probe smoke가 통과한다
3. held-out probe metric이 기준 이상이다
4. reward pipeline에서 probe output을 실제로 load할 수 있다

실행 대상:

1. `E6`
2. `E7`

### Gate 4. Curriculum

held-out 분석에서 diagnosis와 `study_need` 품질이 충분할 때만 허용된다.

실행 대상:

1. redirect-triggered in-context retrieval
2. one-example adaptation
3. 이후 필요하면 self-distill 혹은 RLVR-style follow-up

## 9. 3노드 계획

메인 프로젝트는 정확히 3개 training node를 쓰고, 나머지 1개 node는 다른 작업을 위해 비워 둔다.

1. `metacognition_eval`
   - 역할: probe-free calibration lane
   - 순서:
     `E3 -> E5 -> E9`
2. `metacognition_train_b`
   - 역할: probe-free behavior lane
   - 순서:
     `E8 -> E9b -> E9c`
3. `metacognition_e8`
   - 역할: gated lane
   - 순서:
     `probe target generation -> probe smoke -> E6 -> E7`
   - probe gate가 충족되지 않으면 fallback:
     `E10`

핵심 원칙은 다음이다.

`모든 RL run은 같은 unified SFT 초기화에서 시작한다`

운영 흐름을 더 구체적으로 적으면 다음과 같다.

1. 먼저 `probe-free RL`을 두 노드에서 병렬로 진행한다
2. 수정된 probe는 한 노드에서 학습하고 검증한다
3. probe gate가 통과할 때만 `E6/E7`을 진행한다
4. 완료된 체크포인트는 즉시 Hugging Face에 올린다
5. `base_sft`, control-v5 SFT, 완료된 RL 체크포인트를 모두 평가한다
6. 정량 + 정성 분석이 끝난 뒤에만 behavior analysis나 curriculum으로 넘어간다
7. eval 전에 local checkpoint가 없으면 Hugging Face에서 자동 materialize한 뒤 평가한다

## 10. Smoke / Critic / 개선 루프

모든 코드 경로는 같은 루프를 통과해야 한다.

1. smoke 또는 unit check 하나 실행
2. mismatch 하나를 찾는다
3. mismatch 하나만 고친다
4. 관련 smoke를 다시 돌린다
5. smoke나 guard가 좋아진 경우에만 변경을 유지한다

최소 필수 체크:

1. `tests/test_rewards.py`
2. `tests/test_gdpo.py`
3. `tests/test_tokenizer_utils.py`
4. `tests/test_probe_pipeline.py`
5. `tests/test_control_rag.py`
6. `scripts/smoke_control_rag.py --skip-model`
7. core launch / probe script에 대한 `python -m py_compile`

## 11. 현재 판단

방향 자체는 여전히 align돼 있다.

1. `Meta-CoT`
   - parseable한 controller state를 학습한다
2. `Meta-RL`
   - 그 state를 verifiable behavior reward로 바꾼다
3. `Curriculum`
   - diagnosis와 `study_need`가 충분히 안정화된 뒤에만 확장한다

따라서 올바른 순서는 여전히 다음과 같다.

`representation -> calibration/revision -> decomposed behavior control -> full controller -> curriculum`

지금 중요한 것은 또 다른 개념 전환이 아니다.
데이터, 코드, reward, launch 조건을 이 계약서와 정확히 맞추는 것이다.
