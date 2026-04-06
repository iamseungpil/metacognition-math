# Control-V5 Eval Readout (2026-04-04)

## 1. 한 줄 결론

현재 pilot에서는 `meta behavior change`의 흔적은 보이지만, 그 변화가 아직 `OOD test-time adaptation gain`으로 안정적으로 이어지지는 않았다.

가장 큰 원인은 세 가지다.

1. verify가 자주 `같은 풀이를 다시 말하는 장식적 self-check`로 끝난다.
2. redirect / diagnosis가 나와도 실제 대안 풀이로 전환되지 못하는 경우가 많다.
3. 일부 RL 설정은 confidence coverage를 낮추는 방식으로 calibration pressure를 회피하고 있다.

## 2. 주요 정량 요약

Pilot eval은 각 모델당 `n=90`이다.

| model | acc | AIME | GSM8K | MATH500 | conf_cov | ECE | wrong high-conf |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3_base_sft` | 42.2 | 13.3 | 80.0 | 33.3 | 0.0 | n/a | n/a |
| `qwen3_metacot_control_v5_all_sft` | 33.3 | 6.7 | 63.3 | 30.0 | 88.9 | 0.398 | 51.7 |
| `qwen3_metacot_control_v5_verify_sft` | 36.7 | 3.3 | 66.7 | 40.0 | 82.2 | 0.492 | 80.7 |
| `qwen3_metacot_control_v5_redirect_sft` | 35.6 | 10.0 | 63.3 | 33.3 | 48.9 | 0.160 | 0.0 |
| `control_v5_E3` | 30.0 | 3.3 | 63.3 | 23.3 | 100.0 | 0.515 | 71.4 |
| `control_v5_E5` | 40.0 | 3.3 | 80.0 | 36.7 | 1.1 | 0.790 | 0.0 |
| `control_v5_E8` | 38.9 | 10.0 | 80.0 | 26.7 | 23.3 | 0.322 | 3.6 |
| `control_v5_E9` | 41.1 | 6.7 | 90.0 | 26.7 | 91.1 | 0.301 | 50.9 |
| `control_v5_E9b` | 40.0 | 6.7 | 83.3 | 30.0 | 0.0 | n/a | n/a |
| `control_v5_E9c` | 36.7 | 6.7 | 73.3 | 30.0 | 7.8 | 0.264 | 1.8 |
| `control_v5_E10` | 35.6 | 6.7 | 66.7 | 33.3 | 5.6 | 0.340 | 0.0 |

읽는 법은 다음과 같다.

1. `E8`은 과신 억제는 가장 잘 보인다.
2. `E9`는 정확도는 상대적으로 높지만, verify-only가 과신을 해결하지는 못했다.
3. `E5`, `E9b`, `E10`은 confidence coverage가 너무 낮아서 calibration 해석 자체가 불안정하다.

## 3. 의도 / 가설 / 검증 기준 관점 해석

### RQ1. Meta-CoT가 test-time adaptation의 기반이 되는가

현재 증거는 약하다.

1. `all_sft`가 base SFT보다 정확도와 AIME에서 개선을 보이지 못했다.
2. 실제 오답 completion을 보면 meta가 들어가더라도 `verify -> same route restatement`가 많다.
3. AIME hard failure에서는 repeated intervention이 거의 없다.

즉 `parseable meta`는 일부 형성됐지만, 아직 `controller state`로 충분히 작동한다고 보기 어렵다.

### RQ2. Meta-RL이 behavior를 verifiable하게 학습시키는가

부분적으로 그렇다.

1. `E8`은 hard wrong case의 과신 억제 방향으로 가장 유망하다.
2. `E9`는 verify behavior를 늘리지만, high-confidence wrong case를 많이 남긴다.
3. `E10`은 full controller 가설을 아직 지지하지 않는다.
4. 일부 조건은 confidence를 덜 내거나 meta를 덜 내는 방식으로 reward pressure를 회피한 정황이 있다.

### RQ3. Curriculum trigger로 확장 가능한가

아직 보류다.

1. `study_need` 자체는 여러 오답에서 추출된다.
2. 하지만 diagnosis가 실제 recovery로 이어지지 않아 retrieval trigger 품질을 논하기 이르다.
3. 현재 상태로 RAG를 바로 붙이면 `uncertain -> retrieve`에 가까워질 위험이 있다.

## 4. 실제 completion에서 보인 반복 실패 패턴

아래는 실제 응답을 읽고 정리한 핵심 failure mode다.

### 4.1 Overconfident verify failed

대표 모델: `E3`, `E9`, `verify_sft`

반복 패턴:

1. 모델이 초반 풀이를 끝낸다.
2. `<meta>`에서 “single route라서 verify하겠다”고 말한다.
3. verification에서 같은 계산을 다시 서술한다.
4. 잘못된 가정은 유지된 채 confidence만 유지된다.

예:

1. `AIME 540` (`E9`)
   - meta는 “동시 최대화가 맞는지 verify하겠다”고 말한다.
   - 하지만 verify는 동일한 extremal intuition을 다시 쓰고 `324`를 유지한다.
2. `GSM8K restart` (`E9`)
   - restart 시점과 reset semantics를 확인하겠다고 말한다.
   - verification에서도 시간을 다시 잘못 세어 `180`을 유지한다.

핵심 문제:

`verify`가 `independent falsification`이 아니라 `same-route paraphrase`에 가깝다.

### 4.2 Diagnosis without recovery

대표 모델: `redirect_sft`, `E8`, `E10`

반복 패턴:

1. meta가 현재 route가 약하다고 정확히 진단한다.
2. `study_need`까지 잘 적는다.
3. 그런데 실제 본문은 기존 계산을 계속 밀어붙인다.

예:

1. `AIME octagon coloring` (`E8`)
   - meta: orbit/cyclic counting이 필요하다고 진단
   - 본문: 여전히 부정확한 케이스 카운팅을 이어가며 `535`
2. `AIME Aimeville sets` (`E10`)
   - meta: inclusion-exclusion이 필요하다고 진단
   - 본문: 결국 `exactly 2 + exactly 3 + exactly 4 = 900` 같은 잘못된 식으로 `229`

핵심 문제:

`diagnosis`는 생성되지만 `redirect execution`이 없다.

### 4.3 Single intervention only

대표 모델: `all_sft`, `E8`, `E10`

반복 패턴:

1. meta가 한 번 나온다.
2. 그 뒤 같은 trajectory 안에서 다시 self-monitoring이 이어지지 않는다.
3. hard problem에서도 repeated intervention이 거의 없다.

해석:

현재 Meta-CoT는 `one-shot comment`에 가깝고, `ongoing controller`로는 약하다.

### 4.4 Missing confidence / missing meta as reward escape

대표 모델: `E5`, `E10`, 일부 `E9c`

반복 패턴:

1. 오답인데 confidence가 빠져 있거나 meta가 거의 없다.
2. 그러면 calibration metric이 해석 불가능해진다.

해석:

confidence reward가 `잘 맞춘 confidence`를 학습시키기보다 `confidence emission 자체를 줄이는` 방향으로 새는 설정이 일부 있다.

## 5. 왜 Meta-CoT가 아직 효과를 못 봤는가

핵심 원인은 아래 세 문장으로 요약된다.

1. 현재 meta는 `내가 왜 틀릴 수 있는지`는 자주 말하지만, `그래서 어떤 alternative solver를 실제로 호출할지`는 약하다.
2. verify는 falsification operator가 아니라 repetition operator로 학습된 경우가 많다.
3. redirect는 diagnosis는 만들지만, 본문 계산을 진짜로 바꾸는 execution scaffold가 없다.

즉 지금의 병목은 `meta detection`보다 `meta execution` 쪽이다.

## 6. 바로 이어서 고쳐야 할 것

1. verify reward를 `check wording`이 아니라 `same-route repetition penalty + assumption-cross-check reward`로 바꾼다.
2. redirect reward를 `study_need 기재`가 아니라 `route switch evidence`에 더 강하게 건다.
3. confidence reward에는 `coverage floor`를 넣어 omission으로 도망가지 못하게 한다.
4. hard problem에서는 repeated intervention을 허용하고, 최소 2회 개입이 필요한 샘플을 별도로 추적한다.
5. curriculum/RAG는 diagnosis quality가 더 안정화되기 전까지 gate를 유지한다.

## 7. 생성 산출물

이번 분석에서 생성한 파일:

1. `results/local_mirror/summary_evalnode_v4.md`
2. `results/local_mirror/summary_evalnode_v4.json`
3. `results/local_mirror/failure_analysis_2026_04_04.md`
4. `results/local_mirror/failure_analysis_2026_04_04.json`
