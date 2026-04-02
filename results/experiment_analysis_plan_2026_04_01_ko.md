# Meta-CoT 실험 계획 및 분석 계획 (2026-04-01)

## 1. 이 문서의 목적

이 문서는 두 가지를 분리해 정리한다.

1. `실험 계획`: 앞으로 무엇을 학습시키고 어떤 순서로 비교할 것인가
2. `분석 계획`: 그 결과가 실제로 의도한 메타인지 제어를 학습했는지 어떻게 판단할 것인가

이 문서에서 가장 중요한 것은 "무엇을 잘하고 싶은가"를 정확하게 적는 일이다. 현재 프로젝트의 목적은 단순히 meta text를 늘리는 것도 아니고, calibration 점수 하나를 예쁘게 만드는 것도 아니다. 목적은 `confidence를 내부 제어 변수로 쓰는 OOD test-time control policy`를 학습하는 것이다.

## 2. 중심 의도

현재 계획의 중심 의도는 다음 두 가지다.

### 2.1 막힐 때의 메타 제어

모델이 문제를 풀다가 다음과 같은 신호를 만나면 현재 경로를 밀어붙이지 않아야 한다.

- 계산 또는 대입 결과가 앞선 가정과 충돌함
- 논리적으로 설명되지 않는 점프가 생김
- 자신감이 눈에 띄게 하락함
- "뭔가 이상하다"는 anomaly를 감지함

이때 기대하는 행동은 단순한 자기반성 문장이 아니다. 기대하는 행동은

`trigger / anomaly -> confidence 하향 -> 왜 안 풀리는지 짧게 진단 -> 다른 풀이 방법으로 redirect`

이다.

### 2.2 자신 있을 때의 메타 제어

중요한 점은 `쉬운 문제라서 verify`가 아니라는 것이다. 모델이 현재 답이 충분히 맞을 것 같다고 느끼더라도, 그 confidence가 실제 근거보다 앞서 나가고 있거나, 너무 빨리 commit하고 있거나, 하나의 약한 경로만 보고 과감히 확신하고 있다면 그때 independent verification이 필요하다. 반대로 confidence가 충분히 안정적이고 calibration도 잘 맞는 상태라면 메타를 줄이는 것이 맞다. 따라서 두 번째 중심 의도는

`sufficient-confidence + overcommit / calibration-gap signal -> final answer 전 독립적 verify`

이다.

이 두 축이 현재 계획의 핵심이며, 앞으로의 모든 SFT, RL, eval, curriculum은 이 두 축을 얼마나 잘 구현하느냐를 기준으로 정렬되어야 한다.

## 3. 무엇을 목표로 하지 않는가

현재 단계에서 일부러 피해야 할 오해도 분명히 적어둔다.

1. `meta block` 개수가 많아지는 것 자체는 목표가 아니다.
2. wrong-answer confidence가 내려가는 것만으로 성공이라 말할 수 없다.
3. "다른 방법을 써보자"라는 문구가 들어가는 것만으로 redirect가 학습됐다고 볼 수 없다.
4. retrieval이나 curriculum을 붙여서 점수를 올렸다고 해도, 그 전에 self-diagnosis가 없으면 본래 의도와는 다르다.

즉, 현재 계획은 benchmark hacking보다 `조건부 행동 제어`를 더 중요하게 둔다.

## 4. 용어 정의

이 문서에서 `trigger`, `verify`, `redirect`는 같은 수준의 태그가 아니다.

| 용어 | 역할 | 설명 |
|---|---|---|
| `trigger` | 상황 신호 | contradiction, failed substitution, unsupported assumption, unit mismatch, confidence drop 같은 이상 신호 |
| `verify` | 안정화 행동 | 현재 답이 맞을 것 같더라도 최종 답 전에 독립 근거로 다시 확인하는 행동 |
| `redirect` | 전환 행동 | trigger 이후 confidence를 낮추고 실제로 다른 풀이 전략으로 바꾸는 행동 |
| `diagnosis` | 원인 해석 | 왜 현재 경로가 막혔는지 짧게 설명하는 행위 |
| `decomposition` | 실패 원인 분해 | 문제 풀이 계획이 아니라 왜 현재 접근이 안 풀리는지, 무엇이 빠졌는지, 어떤 하위 능력이 필요한지를 분해하는 행위 |

중요한 점은 다음과 같다.

1. `trigger`는 목표 행동이 아니라 `redirect`를 촉발하는 신호다.
2. `verify`와 `redirect`가 진짜 목표 행동이다.
3. `diagnosis`와 `decomposition`은 redirect를 질적으로 좋게 만드는 내부 구성 요소다.
4. `meta`는 CoT와 엄격히 분리되어야 하며, 계산식이나 실제 풀이 step은 들어가면 안 된다.

## 5. 현재 결과가 말해주는 것

현재까지의 핵심 해석은 다음과 같다.

1. meta 형식 자체는 이미 학습 가능하다.
2. calibration reward는 hard wrong answer에서 overconfidence를 낮추는 데 일부 효과가 있다.
3. 그러나 confidence 하향만으로는 AIME 같은 어려운 OOD 문제의 정답률이 안정적으로 오르지 않는다.
4. verification은 비교적 쉽게 학습되지만, redirect는 여전히 약하다.
5. 현재의 강한 메타 응답 중 상당수는 diagnosis-driven control이 아니라 local self-talk 또는 local patch에 가깝다.

이 해석 때문에 현재 계획은 `format 중심 -> calibration 중심 -> behavior 중심`으로 이동했다. 이 방향 전환은 임시방편이 아니라, 지금까지 얻은 증거와 잘 맞는 수정이다.

## 6. 왜 현재 실험 구성 계획이 의도와 align 되는가

현재 실험군은 역할이 분리되어 있다는 점에서 의도와 잘 맞는다.

### 6.1 V2 / V3

이 단계는 `표현 가능성`을 확인하는 단계다.

- explicit meta block
- confidence 표현
- 메타 텍스트와 수학 성능의 공존 가능성

여기서는 "meta를 말할 수 있는가"를 본다.

### 6.2 E3 / E8 축

이 단계는 `confidence calibration과 self-monitoring`을 실험하는 단계다.

- wrong high-confidence를 줄일 수 있는가
- meta intervention이 어려운 문제에서 더 자주 발생하는가
- confidence 값이 실제 오류 가능성과 어느 정도 맞아지는가

여기서는 "confidence를 보고할 수 있는가"가 아니라 "confidence가 오류 위험과 더 정렬되는가"를 본다.

### 6.3 Behavior SFT 축

이 단계는 `행동 자체`를 가르치는 단계다.

- `behavior_verify_sft`: high-confidence verify를 분리해서 학습
- `behavior_redirect_sft`: stuck/anomaly 이후 redirect를 분리해서 학습
- `behavior_all_sft`: straight / verify / redirect를 함께 넣어 conditional policy를 형성

이 단계에서 처음으로 "confidence가 바뀌면 행동도 바뀌는가"를 직접 실험한다.

### 6.4 차기 E10 축

차기 RL은 기존 calibration reward를 버리는 것이 아니라, calibration 축 위에 behavior reward를 추가하는 방식이어야 한다.

즉, 연구적으로는 다음 비교가 필요하다.

1. `E3`: calibration baseline
2. `E8`: stronger calibration / confidence shaping
3. `E10`: `E8 + behavior rewards`

이렇게 해야 "confidence reward가 듣는지"와 "behavior reward가 추가로 필요한지"를 분리해서 볼 수 있다.

## 7. 다음 데이터 설계 원칙

다음 데이터는 `control-v5`로 재설계한다. 핵심은 `meta purity + confidence-conditioned control + RAG/curriculum 연결 가능성`을 동시에 만족시키는 것이다.

### 7.1 유지할 것

1. `<meta>` 류의 명시적 meta 구간은 유지한다.
2. 각 meta block 안에는 추출 가능한 confidence 형식이 들어가야 한다.
3. 어려운 문제에서는 meta intervention이 여러 번 나올 수 있어야 한다.

### 7.2 바꿀 것

1. rigid field인 `trigger:`, `confidence_before:`, `confidence_after:` 같은 스키마는 제거한다.
2. meta 하나를 여러 개의 기계적 stage로 분해하지 않는다.
3. 대신 meta block 하나는 자연어 개입 하나를 나타내야 한다.
4. meta 안에는 실제 계산, 치환, case split, 검산 CoT, 풀이 계획을 넣지 않는다.

### 7.3 반드시 들어가야 할 의미

1. confidence self-monitoring
2. anomaly 또는 calibration-gap 감지
3. 왜 안 풀리는지에 대한 짧은 diagnosis
4. 필요한 경우 `study_need:`를 통한 missing skill / perspective 명시
5. 필요한 경우 failure decomposition
6. control-level next action만 짧게 선언

추가로 `verify`와 `redirect`는 분리해 설계한다.

1. `verify`: 현재 답이 맞을 것 같지만 과신/조기 commit 신호가 있을 때만 발동한다.
2. `redirect`: confidence drop, anomaly, stuck 신호가 있을 때만 발동한다.
3. `straight`: confidence가 충분히 잘 맞고 과신 신호가 없을 때는 meta를 거의 쓰지 않는다.

핵심은 `자연스러운 meta intervention`, `reward에서 추출 가능한 formatting`, 그리고 `후속 retrieval에 바로 쓸 수 있는 failure/study signal`을 동시에 만족시키는 것이다.

## 8. 다음 reward 계획

다음 RL reward는 하나로 뭉개면 안 된다. 효과를 분해해 봐야 한다.

### 8.1 유지할 축

1. `calibration_reward`
2. `confidence_revision_reward`
3. `overconfidence_penalty_reward`
4. `effective_verification_reward`
5. `effective_redirection_reward`

여기서 `calibration_reward`는 삭제 대상이 아니라 핵심 축이다. 특히 E8에서 효과가 있었던 confidence shaping을 유지한 상태에서 behavior reward를 추가해야 `confidence 교정만으로 되는지`, `행동 reward가 추가로 필요한지`를 분해해 볼 수 있다.

### 8.2 자연어 기반으로 바꿀 축

다음 reward는 rigid field 존재 여부가 아니라 자연어 의미를 보고 계산해야 한다.

1. `diagnosis_reward`
2. `decomposition_reward`

### 8.3 추가 비교 축

1. `anomaly_notice_reward`
2. `repeated_intervention_reward`
3. `overconfidence_verify_reward`

권장 RL 비교축은 다음과 같다.

1. `E3`: calibration baseline
2. `E5`: calibration + confidence revision 계열
3. `E8`: stronger calibration / overconfidence shaping
4. `E10`: `E8 + behavior rewards`

연구적으로 중요한 것은 reward가 많아지는 것이 아니라, 어떤 축이 실제 행동 변화를 만들었는지를 분리해 확인하는 것이다.

## 9. 실험 계획

### Phase A. 현재 비교 세트 정리

먼저 현재까지 만들어진 모델들의 eval과 artifact를 정리한다.

1. Base SFT
2. V2 SFT
3. V3 SFT
4. E3, E5, E7 prev, E7 current, E8
5. behavior_all_sft, behavior_redirect_sft, behavior_verify_sft

이 단계의 목적은 다음 RL 설계를 위해 출발점을 확정하는 것이다.

### Phase B. control-v5 데이터 재생성

다음 데이터는 pilot -> critic -> 개선 -> main run 순서로 만든다.

1. `straight`, `verify`, `redirect`가 모두 남는지 smoke QC
2. difficulty별 샘플 정성 점검
3. meta가 CoT와 엄격히 분리되어 있는지 점검
4. `verify`가 난이도 기반이 아니라 과신 신호 기반으로만 발동하는지 점검
5. `redirect`에서 diagnosis와 failure decomposition이 실제로 "왜 못 푸는지"를 말하는지 점검
6. `study_need:`가 짧고 parseable하며 RAG에 쓸 수 있는지 점검
7. hard trajectory에서는 필요할 때만 2회 이상 개입하는지 점검

pilot이 통과되기 전에는 10k main run으로 가지 않는다.

### Phase C. SFT 비교

기본 출발점은 raw base가 아니라 `qwen3_base_sft` 같은 강한 기존 SFT checkpoint다.

권장 비교:

1. `base_sft -> control_v5_all_sft`
2. `base_sft -> control_v5_verify_specialist`
3. `base_sft -> control_v5_redirect_specialist`

여기서 보는 것은

1. 정확도 유지
2. verify effectiveness
3. redirect effectiveness
4. confidence-conditioned behavior change

이다.

### Phase D. RL 비교

SFT 비교 후 가장 좋은 기반 하나 또는 둘을 골라 RL로 간다.

권장 ablation:

1. `E3`
2. `E5`
3. `E8`
4. `E10 = E8 + behavior rewards`

필요하면 `verify-specialist`와 `redirect-specialist`를 따로 유지하되, 메인 결론은 하나의 unified controller가 가능한지로 가져간다.

### Phase E. Curriculum / RAG 진입

Curriculum / RAG는 아래가 충족될 때만 진행한다.

1. anomaly 뒤 confidence drop이 실제로 나타남
2. confidence drop 뒤 redirect가 실제로 나타남
3. high confidence일 때 verify가 실제로 나타남
4. 위 행동들이 1,030 문제 기준과 hard slice 기준에서 반복적으로 관찰됨

그 다음에야 "못 푼 문제를 보고, 약점을 진단하고, `study_need`를 추출하고, 관련 예시나 retrieval을 붙여서 test-time에 회복하는 loop"가 연구적으로 의미를 가진다.

## 10. 분석 계획

### A. calibration 분석

필수 지표:

1. benchmark별 ECE
2. wrong high-confidence 비율
3. correct low-confidence 비율
4. wrong-answer average confidence

### B. 중간 confidence revision 분석

필수 지표:

1. contradiction-conditioned confidence drop
2. confidence drop 이후 redirect rate
3. confidence drop 이후 correctness recovery rate
4. no-drop wrong-commit rate

이 축은 "confidence가 실제로 행동을 제어하는가"를 가장 직접적으로 본다.

### C. verify 분석

필수 지표:

1. verify fraction
2. independent-check fraction
3. high-confidence with verify error rate
4. high-confidence without verify error rate
5. answer-change-after-verify rate

### D. redirect 분석

필수 지표:

1. redirect fraction
2. strategy-switch fraction
3. redirect-conditioned recovery accuracy
4. redirect-without-real-switch fraction

여기서는 문구가 아니라 실제 풀이 방법이 바뀌었는지를 본다.

### E. diagnosis / decomposition 분석

필수 정성 분석:

1. 왜 현재 경로가 안 되는지 명시했는가
2. 필요한 subgoal을 제대로 재설정했는가
3. 이후 행동이 그 진단과 일관되는가

이 축은 이후 curriculum과 RAG 연결 가능성을 판단하는 기준이 된다.

### F. difficulty-conditioned compute allocation

필수 지표:

1. 난이도별 completion length
2. 난이도별 meta block 수
3. easy 문제에서 불필요한 과개입 비율
4. hard 문제에서 추가 verify / redirect 비율

## 11. 현재 결론

현재 계획에 들어있는 중심 의도는 올바르다.

1. confidence를 자기 보고 숫자가 아니라 제어 신호로 다루려는 점
2. low confidence에서 redirect를, high confidence에서 verify를 요구하는 점
3. curriculum을 그 다음 단계로 미루고, 먼저 self-diagnosis와 control을 안정화하려는 점

이 세 가지는 서로 잘 맞는다.

또한 현재 실험 구성 계획도 대체로 align 되어 있다.

1. V2/V3는 표현 가능성을 확인한다.
2. E3/E8은 calibration 축을 확인한다.
3. behavior SFT는 행동 정책을 직접 가르친다.
4. 차기 E10은 calibration과 behavior를 합친다.
5. curriculum / RAG는 그 이후 단계로 게이트한다.

따라서 지금 필요한 것은 방향 수정이 아니라, 이 방향을 더 정확히 구현하는 일이다. 데이터는 더 자연스럽고 균형 있게 만들어야 하고, reward는 더 분해해서 비교해야 하며, 분석은 "meta를 말했다"가 아니라 "행동이 바뀌었다"를 기준으로 해야 한다.
