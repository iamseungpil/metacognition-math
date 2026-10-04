# S3 «2-시도 GRPO + 반성문 크레딧» 설계 (0915)

근거: G8(`RESULTS_cd9` 0915 11:00) **사실만 고지 .6167 > 눈감음 .5825 > 본문 제시 .5125**,
오염 분석(`docs/analysis/CONTAM_activation_0915.md`) «손실은 원 오답의 재방출»,
LaMer(arXiv:2512.16848) 크로스-에피소드 리턴 `G(n) = g(n) + Σ_{m>n} γ_traj^{m−n} g_0(m)`,
γ_traj=.6, 어블레이션 «궤적-only 34.8 vs 반성문-only 56.4».

## 1. 무엇을 만드는가

문제 x 마다 **문맥을 버리는** 두 시도를 굴린다.

1. **시도 1**: 평소의 `math_opt` 프롬프트 → 응답 a1. 채점 R1 = grade_math(a1, gold).
2. R1=0 인 행만: **반성문 생성**. a1 을 보여 주고 «무엇이 잘못됐는지 한 문장, 숫자·답 금지»
   를 묻는다(≤ NOTE_MAX_TOKENS=64). 누출 가드가 숫자/`\boxed`/a1 의 최종 답을 담은 노트를
   버리고 일반 문구로 되돌린다. a1 은 **꼬리 NOTE_CTX_MAX_TOKENS=2048 토큰만** 이 프롬프트에
   들어간다(틀린 최종 답과 마무리 논증이 꼬리에 있다) — 이 호출은 `cf_prefix_agent` 라
   verl 의 `rollout.prompt_length` cap(chat-템플릿 경로 전용)을 우회하므로 상한이 없으면
   그 호출의 판 폭이 시도-1 과 어긋나고(0915 스모크 5120 vs 8788) 학습 배치 폭도 그만큼
   불어난다. 폭 자체는 `verl_sdc._s3_harmonize` 가 세 단계 공통 판으로 맞춘다.
3. **시도 2**: 문맥을 **통째로 버리고** `blind_external_prompt(tok, variant, x)`
   (= `math_activation_gate.note_prompt(..., extra=r)`, extra="" 이면 바이트 동일) + 반성문
   r 한 문장. 응답 a2, R2 = grade_math(a2, gold).

반성문 자리에는 **오답 본문이 한 글자도 들어가지 않는다** — G8 이 확정한 −10.4pp 끌개를
피하는 것이 이 팔의 존재 이유다. NOTE_MODE=none 이면 r 이 없고 프롬프트는 `.617` 참조와
바이트 동일해진다(레퍼런스 팔).

## 2. verl 0.9 에 어떻게 얹는가 — 슬롯 규약(멀티턴 기계 없음)

verl 의 `fit()`(ray_trainer.py:1515-1516)은 `batch.repeat(n)` 뒤 `union(gen_batch_output)`
이므로 **행 수를 늘릴 수 없다**. 그래서 행을 늘리지 않고 **rollout.n 을 3K 로 잡아 슬롯을
빌린다**. `interleave=True` 라 문제 p 의 3K 행은 연속이고 uid·problem·gold 가 동일하다.

| 슬롯 | 내용 |
|---|---|
| 0 … K−1 | 시도 1 (K 샘플) |
| K … 2K−1 | 슬롯 K+j = 샘플 j 의 반성문 행 |
| 2K … 3K−1 | 슬롯 2K+j = 샘플 j 의 시도 2 |

건드리는 자리는 딱 둘이다.

* `SDCRayPPOTrainer.init_workers` — `_s3_trial2` 게이트가 켜지면
  `async_rollout_manager.generate_sequences` 를 `_s3_generate_sequences` 로 감싼다
  (`_dcpo_cf_generate_sequences`(verl_sdc.py:5519)·`_bci_generate_sequences`(5484)와 **같은**
  자리·같은 규약: 엔진 replica 가 깨어 있는 동안 2·3차 생성을 더 부른다).
* `_s3_generate_sequences` — ① 시도-1 슬롯만 골라 원본 generate 호출 ② 채점 ③ 오답 행의
  반성문 생성(`max_tokens=64`) ④ 시도-2 프롬프트 생성 ⑤ 세 출력을 `DataProto.concat` 한 뒤
  `select_idxs(perm)` 로 슬롯 순서로 되돌린다. 텐서를 손으로 짜지 않는다 — 세 호출 모두
  엔진이 같은 폭(`max_prompt_length`/`max_response_length`)으로 패딩한 DataProto 를 준다.
  `validate=True` 배치는 **그대로 통과**(평가는 시도 1 뿐).

`_dcpo_cf_call_engine`(5834)의 «청크 정합을 위해 패딩하고 뒤를 버린다» 규약을 그대로 쓴다.

## 3. 보상·어드밴티지 — 새 advantage 코드 0줄

`patched_compute_advantage`(6438) → `_math_populate_token_rewards`(4552) → 원본 GRPO
중심화 순서를 그대로 탄다. 훅은 `_math_populate_token_rewards` 안의 분기 하나뿐:

1. 행별 스칼라 보상을 `trial2.row_rewards(arm, r1, r2, stage, active, gamma)` 로 얻어
   마지막 유효 토큰에 얹는다.
2. **uid 를 단계별로 갈라 쓴다**: `uid#a1` / `uid#note` / `uid#a2`, 비활성 슬롯은
   `uid#dead{i}`(싱글턴 → 중심화 뒤 정확히 0).

그러면 원본 GRPO(Dr.GRPO, `norm_adv_by_std_in_grpo=false`)의 그룹 평균 빼기가
**그대로 단계별·문제별 중심화**가 된다(`dcpo_region.group_mean_subtract` 와 같은 정의).

| 스팬 | CREDIT (`M_TRIAL2_CREDIT`) | OUTCOME (`M_TRIAL2_OUTCOME`) |
|---|---|---|
| 시도 1 | `center_{a1}(R1 + γ·R2)` | `center_{a1}(R1)` |
| 반성문 | `center_{note}(γ·R2) = γ·(R2 − mean R2)` | `0` (보상 0 → 상수 → 중심화 0) |
| 시도 2 | `center_{a2}(R2)` | `center_{a2}(R2)` |
| 비활성 | 0 | 0 |

R1=1 행은 R2≡0(시도 2 가 없다). **사행(sandbagging) 불가 증명**: 시도-1 행의 보상은
`R1 + γ·R2 ≤ 1 + 0` (R1=1) vs `0 + γ·R2 ≤ γ = 0.6` (R1=0). γ<1 이므로 «틀리는 편이 낫다»가
되는 조합이 없고, R1=0 에 붙는 보너스는 **오직 γ·R2** 뿐이다(별도 실패 보너스 없음).
보강으로 `ATTEMPT1_KL_COEF`(기본 = 트레이너의 기존 `actor.kl_loss_coef`)를 둔다.

## 4. 방어하는 실패 모드

* **노트 누출** — `trial2.clean_note`(= `math_activation_gate.clean_label` 로직을 공유
  모듈로 뽑은 것): `\boxed`·숫자·a1 의 최종 답 문자열 → 버리고 일반 문구, 첫 줄·12단어 컷.
  통계(`digit/boxed/answer/empty`)를 스텝마다 찍는다.
* **빈 노트** — 일반 문구로 대체하고 `note_generic` 비율을 기록; 100% 일반이면 팔이
  NOTE_MODE=none 과 바이트 동일해지므로 중단 규칙 대상.
* **사행** — §3 부등식 + 텔레메트리 `r1_mean`(M_G1 대비 −1pp 이상 하락 3스텝 연속 시 중단,
  기존 `MATH_ACC_FLOOR` 규약 재사용).
* **8k 절단** — 시도 1/2 각각 RESP_LEN 4096(총 예산은 행마다 따로다 — 같은 문맥에 이어
  쓰지 않으므로 M_RETRY 의 6144 가 필요 없다). `trunc_rate` 규칙 그대로.
* **오답 본문 오염** — 본문은 반성문 *생성* 프롬프트에만 들어가고 시도-2 프롬프트에는
  절대 들어가지 않는다. 테스트가 시도-2 프롬프트에 a1 부분문자열이 없음을 고정한다.

## 5. 계산 비용 (M_G0 = 1.0)

M_G0: K=8 × 4096 토큰. S3: 시도 1 (K=8, 4096) + 반성문 (오답률 ≈ .26 × 8 ≈ 2행, 64토큰,
≈ 0.004) + 시도 2 (≈ 2행, 4096, 0.26) ≈ **1.26× 생성**. 학습 쪽은 행이 3K 로 늘지만 비활성
슬롯은 1토큰 더미라 실제 학습 토큰은 ≈ 1.26×. 슬롯 차용 때문에 **엔진 청크 패딩**이 붙어
실측은 1.3~1.5× 로 본다. 메모리는 M_G0 과 같은 need_mb 70000.

## 6. 평가 규약

`scripts/local/math_trial2_eval.py`(미구현, 보고에 명시) 가 `math_activation_gate` 의
`blind_external_prompt`·`clean_label`·`bootstrap_ci` 를 그대로 불러 홀드아웃 MATH-500 에서:

1. **시도-1 pass@1 ≥ .744**(M_G1 스텝 30; M_G0 은 .740) — 이게 깨지면 팔은 기각.
2. **2-시도 정확도**(R1 ∨ R2) vs **사실만 재시도 .617**(G8, 난이도 5 오답 150 × K=8).
   같은 선별·같은 8,192 예산·같은 부트스트랩으로 재야 비교 가능하다.
3. 대조: NOTE_MODE=none 팔이 곧 `.617` 재현 팔이다 — 반성문의 순수 기여 = (self − none).

### 6b. NOTE_MODE 확장 (0916, F1 결과 반영)

F1 재요약(`/hdd_data/seungpil/scratch/eval/reset_content_s1_resum`)에서 리셋 자리의 내용
세 팔이 모두 사실만 고지(fact)보다 구제율이 높았다 — effort +.041 · switch +.040 ·
notx +.069(Holm p<.03), switch−donor +.037(CI 0 제외). 그래서 NOTE_MODE 를
`{self, none, switch, notx, switch_notx}` 로 넓혔다(기본값은 여전히 `self`).

* `switch` — 노트 단계의 질문을 `LABEL_ASK` 로 바꿔 **자기 방법 라벨**을 얻고, 시도-2 는
  `SWITCH_TMPL` 한 문장을 fact 줄 뒤에 붙인다 → F1 `fact_switch` 프롬프트와 **바이트 동일**
  (`tests/test_trial2.py` 가 고정). 크레딧 스팬은 그대로 노트 토큰(= 라벨 토큰)이다.
* `notx` — 사전 패스가 **없다**(노트 자리는 죽은 자리). X 는 a1 의 `\boxed` 값이며, 시도-2 는
  채점기가 이미 R1=0 이라 판정한 행에서만 돌므로 **gold 노출이 아니다**. `\boxed` 가 없으면
  그 절을 빼고 fact 전용으로 되돌린다. 노트 스팬이 없으므로 CREDIT 과 OUTCOME 은 시도-1 행의
  γ·R2 항 하나만으로 갈린다.
* `switch_notx` — 위 두 문장을 그 순서로 공백 하나로 이은 것(정확한 문자열은 테스트가 고정).
* ⚠️평가 스크립트에서 **정답 행에는 notx 계열을 돌리지 않는다** — 그 행의 `\boxed` 는 gold 라
  «답은 X 가 아니다»가 곧 누출이다(오답 행에서는 아니다).

또한 F1 의 flip 절은 절대 상한 .10 에서 **상대 규칙**(arm ≤ blind + .02)으로 고쳤다 —
신호 없는 blind 재표본도 정답 난이도-5 행을 .341 뒤집으므로 .10 은 통과 불가능한 오측정이었다
(`math_activation_gate.FLIP_MARGIN` 주석).

⚠️ LaMer 어블레이션에 «반성문-only 문맥»이 이미 있다 — 새롭다고 주장하지 말 것. 우리 것은
**단일턴 수학**에서 그 구조가 p@1 을 깎지 않는지를 묻는 것이다(LaMer 는 p@1 −7.9/−5.6).

## 7. M_RETRY 는 이걸 이미 하는가 — **아니다**

M_RETRY/M_AGREE 는 **한 응답 안에서** 첫 답 → decision → "Second attempt:" 를 잇는다
(문맥 유지). CONTAM 분석이 그 구조의 구제율을 0.000 으로 실측했고 G8 이 원인을 «문맥 오염»
으로 확정했다. S3 는 정확히 그 반대(문맥 폐기)라 재사용할 코드 경로가 없다. 재사용하는
것은 채점기(`grade_math`/`last_boxed`)·중심화(`group_mean_subtract`)·엔진 2차 호출 규약
(`_dcpo_cf_call_engine`)·누출 가드(`clean_label`) 뿐이다.
