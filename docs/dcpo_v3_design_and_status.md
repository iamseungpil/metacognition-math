# TRIOBJ_DCPO_V3 — 설계·상태 보고서 (2026-06-10)

> 이 문서는 "왜 이렇게 설계했는지"를 잊지 않기 위한 단일 참조점이다.
> 세부 스펙: `docs/superpowers/specs/2026-06-09-dcpo-v3-counterfactual-design.md`(인과 R_meta),
> `docs/superpowers/specs/2026-06-10-dcpo-v3-format-tier-design.md`(형식 3-tier).

## 1. 전체 의도 (north-star)

**메타인지는 목적이 아니라 정확도를 끌어올리는 수단이다.** RL은 "유용한 메타인지"만 선택적으로
강화해야 한다. 이를 위해 메타의 가치를 상관이 아닌 **인과**로 측정한다:

```
R_meta = c_with − c_without
  c_with    = 본 rollout(메타 포함)의 정답 여부
  c_without = 같은 prefix(메타 직전까지 토큰 단위로 동일)에서 메타를 억제하고
              다시 생성한 counterfactual의 정답 여부
```

CF는 **짝지어진 동일-prefix 절제 실험**이다 — "메타 있는 집합 vs 없는 집합" 비교가 아니라,
rollout 하나하나에 대해 "그 시점에 메타를 한 결정"의 효과만 잰다. CF rollout은 inference 전용
(GRPO 그룹에 불포함, advantage 없음)이고 c_without 스칼라 하나만 기여한다.
calibration(R_cal)은 부분 목표일 뿐이다.

## 2. 마스킹·라우팅 세부 (보상이 토큰 어디로 흐르는가)

head별로 **독립적으로** 그룹 평균을 빼고(Dr.GRPO, /std 없음, 전역 재정규화 없음)
**자기 영역에만** 라우팅한다. 영역을 잘못 잡으면 보상이 잘못된 토큰을 강화하므로
(v3j에서 실측: rollout의 17%에서 답 토큰이 META_CONTENT로 분류돼 R_corr을 못 받음)
영역의 정확성이 이 방법의 생명선이다.

| 토큰 | 마스크 | 받는 advantage | w | 의미 |
|---|---|---|---|---|
| 본문 추론·최종 답 | ANSWER_REGION | R_corr | 1.0 | 정답률 |
| `<|meta|>` (opener) | META_CONTENT | R_meta | 0.5 | **"언제 멈출지" 결정 자체를 인과 효용으로 학습** |
| 메타 내용 | META_CONTENT | R_meta | 0.5 | 유용한 메타 강화/해로운 메타 억제 |
| confidence 숫자 | CONF | R_cal (Brier) | 0.3 | 보정 |
| `<|/meta|>` (closer, 정상) | FORMAT_OK | R_format **+측** (그룹 센터링) | 0.1 | 올바른 닫기 강화 |
| 잘못된 구분자 (드리프트의 `</think>`, 쓰레기 태그) | FORMAT_VIOLATION | R_format **−1** | 0.1 | 실수 토큰 그 자리에 벌점 |
| 구분자 외 태그/패드 | (없음) | 0 (중립) | — | 구분자 |

핵심 교훈 (v3k에서 발견): 태그를 전부 중립으로 두면 **신호 비대칭**이 생긴다 —
드리프트 rollout이 정답일 때 실수 토큰(`</think>`)이 ANSWER_REGION에서 R_corr(+1.0)을 받아
**틀린 닫기가 오히려 강화**되고, 올바른 `<|/meta|>`는 영원히 0이다. 그래서 양방향 format head가 필요했다.

## 3. 형식 교정 3-tier 전략 (대체 → 버리기 → 보상)

SFT 모델의 샘플링 습관상 정상 형식은 17%뿐이다 (rollout 6,400개 실측):

| 클래스 | 실측 비율 | 전략 | 처리 |
|---|---|---|---|
| 정상 `<|meta|>…<|/meta|>` | 17% | — | §2 표 그대로 |
| 스왑 `</think>…<|/meta|>` | 8–25%* | **① 대체** | 잘못된 구분자를 올바른 태그로 **1:1 토큰 교체** → 정상 라우팅 |
| dup-open `<|meta|>…<|meta|>` | ~11% | **① 대체** | 두 번째 open을 close로 교체 |
| 역순 `<|/meta|>…<|meta|>` | ~7% | **① 대체** | 두 태그 맞바꿈 |
| 드리프트 `<|meta|>…</think>` | ~3–19%* | **③ 보상** | 삽입이 필요해 교체 불가 → 내용-앵커로 영역 복구(R_meta·conf 흐름) + 이중역할 `</think>` 토큰에 −1 |
| 교차/중첩 (CCK/CK/KOK 등) | ~30% | **② 버리기** | 영역 신뢰 불가 → 3-head 전부 0 + 쓰레기 구분자에 −1 + CF 스킵 |
| truncation | ~3% | 게이트 | R_meta 0, CF 스킵, **무벌점** (길이 문제지 습관 아님) |

\* 측정 방법(약식 텍스트 count vs 토큰 파서)에 따라 달라진 범위. 파서 기준이 정본.

**대체(①)가 on-policy를 깨지 않는 이유**: verl은 old_log_prob을 생성 *후* 별도 actor 패스에서
계산한다. 교체는 CF wrap(생성 직후, log-prob 패스 전)에서 responses+input_ids에 동일 길이로
일어나므로 비율이 교체된 시퀀스 기준으로 일관되고, 교정된 태그 위치에 흐르는 +advantage가
올바른 태그를 직접 가르친다(STaR식 교정의 토큰-국소 버전). 교체된 행은 R_format=0
(같은 위치에 벌점을 주면 +학습과 충돌). 라운드트립 검증: 교체 계획 1,175건 전부
재파싱 시 wellformed (100%).

**버리기(②)가 30%로 큰 것**: 안전(오배달 0)하지만 데이터 손실. 라이브 `dcpo/discard_rate`가
25%+ 유지되면 tier-1을 CCK/CK/KOK 형태로 확장(v3m 후보). 파서 단일 진실 원천 +
하니스(`scripts/format_parser_harness.py`)가 있어 확장 검증이 싸다.

**문헌 근거**: DeepSeek-R1(별도 format reward) + Qwen-Math(관대한 추출/자격 게이트) +
STaR/RFT(교정 후 학습). 우리는 셋을 영역 라우팅 위에 결합했고, 페널티를 위반 토큰
위치에 정밀 라우팅하는 점이 R1보다 한 발 더 나간 부분.

**CF 누출 가드**: CF 생성 시 두 태그 id(151669, 151670) 모두 logit_bias −100으로 억제
(스왑형이 증명했듯 opener 없이도 메타 내용이 나올 수 있음). 그래도 무형식 메타 시그니처
(`confidence:/assessment:/action:`)가 CF에 새면 그 행은 보수적으로 미채점(R_meta 0).

## 4. 5중 동기화 규칙 (크래시 3번의 교훈)

reward key를 추가/변경할 때 다음 다섯 곳이 **반드시 함께** 움직여야 한다. 하나라도 빠지면
부트 검증 또는 step-1 GDPO 단언에서 죽는다 (v3f: yaml 길이 불일치, v3g: populator 누락):

1. `REWARD_CONFIGS['TRIOBJ_DCPO_V3']` funcs/keys/weights (verl_sdc.py)
2. `configs/triobj_dcpo_v3_h100_4x4k.yaml` gdpo_reward_keys/weights (감사용 미러)
3. `_populate_dcpo_region_keys`의 non_tensor_batch 쓰기 (async 경로) + sync `__call__` 미러
4. `compose_dcpo_region_advantage` 파라미터/마스크
5. `build_dcpo_region_masks`/`classify_dcpo_format`의 마스크 키

소스-레벨 회귀 테스트가 1↔2↔3을 잠근다 (tests/test_dcpo_v3.py).

## 5. 실험 이력과 운영 교훈 (2026-06-09~10)

| run | 결과 | 교훈 |
|---|---|---|
| v3b | R_meta가 상관 신호였음 (gt="" → c_without≡0 + np.float32 NaN 누출) | 채점은 GT가 있는 consumer에서; NaN은 `x==x`로 |
| v3e | H200 Basic 용량 고갈로 큐 고착 | mlc 잡과 비교해 H100 Standard로 전환 |
| v3f | 부트 크래시: yaml 키 3 ≠ funcs 4 | 동기화 규칙 #1↔#2 |
| v3g | step-0 val 완주 + CF 파이프라인 실증 후 step-1 크래시 (populator에 meta_emission 없음) | 동기화 규칙 #3 |
| v3h | 노드 하드웨어 장애 3회 (선점 아님; H100 Standard 154 GPU 여유 확인) | H100엔 Premium 없음, 리전 선택 불가 → save_freq로 대응 |
| v3j | step 0–3 커밋, rmeta_pos 상승 확인 — **형식 오배달 발견으로 중단** | 768 rollout 감사가 17% 오배달 적발 |
| v3k | 게이트 작동 실증 (acc_with 0.50 > acc_without 0.35–0.42 첫 역전), step 12 도달 | 게이트만으론 신호 희소화; 태그 중립의 비대칭 발견 |
| v3l | 3-tier 형식 전략 (이 문서) — 진행 중 | |

**amlt 운영 함정**: ① `amlt cancel`은 **repo 디렉토리 안에서만** 동작 — 밖에서 실행하면
조용히 실패하므로 반드시 `status`로 killed 확인. ② 같은 HF ckpt repo를 쓰는 새 run 제출 전
`global_step_*` 잔재 확인·삭제 (v3j·v3k 둘 다 gs10을 남겨 다음 run을 오염시킬 뻔함).
③ `amlt logs tail`은 재연결마다 전체 로그를 재전송 — 스트리머에 라인-카운트 dedup 필요.

## 6. 관측 지표 사전 (wandb)

| 지표 | 의미 | 성공 판정 |
|---|---|---|
| `dcpo/acc_with` vs `acc_without` | 배치 평균 인과 효과 | gap > 0 유지·확대 |
| `dcpo/rmeta_pos_rate` / `neg_rate` | 메타가 답을 살림/해침 비율 | pos↑ neg↓ |
| `dcpo/rmeta_mean_meta_rows` | 메타 행의 평균 인과 델타 | + 유지·상승 |
| `dcpo/meta_emit_rate` | 메타 발행률 | 붕괴 금지 (v3b: 42→23% 붕괴 전례) |
| `dcpo/wellformed_rate` | 정상 형식 비율 | **상승** = 형식 학습 증거 |
| `dcpo/replaced_rate` | tier-1 교체 비율 | 하락 (교정 학습되면 교체할 게 줄어듦) |
| `dcpo/discard_rate` | tier-2 버림 비율 | 하락; 25%+ 고착 시 v3m 트리거 |
| `dcpo/drift_rate` | tier-3 드리프트 비율 | 하락 |
| `dcpo/rollouts` Table | 행마다 main_tail + cf_tail 짝 + fmt_class | 텍스트 수준 눈검수 |
| `val-aux/<벤치>/...` | step-0 SFT 베이스라인 대비 정확도/발행/보정 | RL > SFT |

최종 판정은 gs300의 standalone eval (1030문제, 16k): AIME 포함 정확도+ECE를
base 0.786 / inject 0.609 / v1 / e8 0.742와 비교.
