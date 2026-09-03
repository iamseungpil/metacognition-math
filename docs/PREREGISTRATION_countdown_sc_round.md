# 사전등록 — Countdown SC 라운드 (cd7, 로컬 H100)

작성 2026-09-04 02:10 · **결과 0건 시점**(학습 전 기준선만 관측) · 머신 로컬 H100 80GB × 4 (GPU 0~3)

## 1. 묻는 것

«막히면 새 계열로 탐색하고, 과신하면 실제로 재계산해 검증하는» 메타 습관을 **모델 자신의
신호**(프리픽스 시도 수 · 자기 보고 confidence/decision · 메타 뒤 행동 · 정답 여부)만으로 보상하면
정답률이 오르는가. 근거-진리(완전 열거)는 SC 에서 쓰지 않는다.

## 2. 설계

- 모델 Qwen3-4B(하이브리드, thinking off) · SFT 없음 · Countdown 4수 · 프롬프트 **new** · 응답 상한 2560
- GRPO(verl 0.7.1, `countdown_6arm` config) · 64 프롬프트 × 8 롤아웃 · lr 1e-6 · **100 스텝**, 체크포인트 5스텝, 판정 30·50·100
- 팔 하나 = GPU 한 장. 데이터 train 8000(seed 1) / val 500(seed 2), `scripts/local/make_data.sh`.

| 팔 | 데이터 | 항 (서명은 `arm_signature`) |
|---|---|---|
| N0 | plain | corr@1 + format@0.35 — 메타 요구 없음(순수 GRPO 기준선) |
| A | new | corr + format + meta_floor@0.02 — 메타 요구, 내용 미채점 |
| **SC** | new | A + explore@1·w + verify@0.5·w + early_cost@0.25 |
| SC_GH | new | SC − corr, **20스텝** — 굿하트 압력시험(관문 G-F), 학습 주장에 쓰지 않음 |

SC 항 정의(`countdown_selfcontrol.sc_row`, `countdown_rewards.r_explore/r_verify/r_early`):
- stuck = 1[메타 앞 등식 시도 ≥ 4] · early = 1[메타 앞 시도 0] · hi = 1[confidence ≥ 0.8]
- novel = 메타 뒤 첫 시도 쌍이 프리픽스에 없던 쌍 (new 프롬프트엔 next: 필드가 없음) · followed = 1
- checked = 메타 뒤 ~ boxed 앞에 산술적으로 참인 등식 ≥ 1
- explore = stuck × redirect × novel × followed · verify = hi × (verify×checked − (1−checked)×1[오답]) · early_cost = −early
- w 워밍업 0→20스텝 선형. 항별 [−1,1] 포화 후 무게.

## 3. 학습 전 기준선 (500문제 × 8, 씨앗 11, `/hdd_data/seungpil/scratch/eval/gs0_Qwen3-4B_*`)

| 프롬프트 | 정답률 | 발화율 | 발화 행 중 stuck | early | hi | novel | checked | 비고 |
|---|---|---|---|---|---|---|---|---|
| plain | 0.426 | 0 | — | — | — | — | — | N0 기준 |
| new | 0.390 | 0.398 | 0.869 | 0.031 | 0.297 | 0.047 | 0.414 | SC 기반 |
| p3 | 0.099 | 0.882 | 0.033 | **0.960** | 0.029 | 0.831 | 0.228 | ⛔ 부적합 — 메타가 시도 전, 확신 0.6~0.7 고정 |

new 에서 explore 발동률 0.0145(발화 행 기준), verify 항 비영 0.02. **낮다.** RL 이 올려야 할 양이며,
20스텝 안에 발동률이 0 에 머물면 §5 의 «침묵 항» 판정을 받는다.
확인된 교란: checked 행의 정답률 0.079 vs 미checked 0.457 — «등식이 더 있다 = 계속 헤맨다» 성분이 섞여 있다.
verify 항은 hi(30%)에서만 켜지므로 영향은 제한적이나, 이것이 verify 항의 알려진 약점이다.

## 4. 주 지표와 해상도

- **1차**: held-out 500×8 정답률, 판정 지점 30/50/100, 씨앗 11 (`countdown_gs0_eval.py`, 병합본).
- **2차**: 구조율(rescue), 발화율, explore/verify 발동률, 메타 위치 분포, 오답 길이.
- 해상도: 이전 라운드 σ_run 2.89pp(시드 1개끼리). **1파도는 스크리닝**이다. 2파도에서 씨앗 2·3을 더해
  팔당 3씨앗이 되기 전에는 팔 간 우열을 «확증»으로 쓰지 않는다.

## 5. 판정 밴드 (결과 전 고정)

| 결과 | 판정 |
|---|---|
| SC_GH 20스텝에서 메타 판박이율 > 0.5 또는 숫자 나열/빈 메타 급증 | SC 항 **게이밍 가능** → 항 재설계 전 SC 결과를 주장하지 않음 |
| SC 의 explore·verify 발동률이 50스텝까지 각각 < 0.01 | 항 **침묵** → 널을 «효과 없음»으로 읽지 않음 |
| SC − A ≥ +3pp 두 지점 이상 & 발화율 ≥ 0.3 유지 | 내부 신호 보상 양성(스크리닝) → 2파도 씨앗 확대 |
| SC − N0 ≥ 0 | «메타 요구가 손해가 아니다» 확보 (gs0 에서는 −3.6pp) |
| A ≈ N0 ≈ 원본 (±2pp) | GRPO 가 안 움직인 것 — 팔 비교 무의미, 난이도·스텝 재검토 |

## 6. 중단 규칙 (코드, 매 스텝, 3연속) — `countdown_rewards.check_abort` 기본값 유지
emit_rate < 0.2 · boilerplate > 0.5 · answer_leak > 0.1 · arith_in_meta > 0.02(new 기저 0.0019) 등.

## 7. 2파도 (1파도 결과와 무관하게 큐에 넣는다)
N0 s2, A s2, SC s2, G(길이 위약) s1 → 이후 씨앗 3.
