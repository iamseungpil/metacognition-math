# 논문 실험 계획: Calibration ≠ Metacognition (Matched-Base 후속)

**Date**: 2026-07-03  **Status**: DRAFT (self-critiqued, 승인 대기 → 승인 후 superpowers 브레인스토밍 → 본 문서 갱신)
**선행 문서**: `docs/superpowers/specs/2026-06-29-matched-base-clean-meta-comparison-design.md` (matched-base 설계)
**관련 메모리**: `meta-fails-root-correlated-self-verification-0625`, `valcore-reward-meta-shaped-use-valaux-correctness-0703`, `math500-grader-broken-aime-premature-assertion-0625`, `directional-self-distill-heldout-additive-beats-mult-0625`

---

## 0. 지금까지의 결과 (논문 헤드라인 후보)

pmi-shift 메타(반사실 R_meta를 `<|meta|>` 영역에 연속 로그확률 시프트로 적용)가 **교란 제거된 matched-base**(VANILLA_GRPO, 메타 메커니즘만 제거, 나머지 byte-identical)를 **동일-스텝 수학 정확도**에서 앞선다.

- val-aux/correctness(순수 correctness) 동일-스텝: base vs pmishift = 25:+0.021, 75:+0.104, 100:+0.074, 125:+0.096 (겹치는 5스텝 전부 메타 우위)
- pmishift는 메타 arm 중 정확도 최고이며, gs300까지 안정적으로 높은 유일 arm (stage3b는 gs125부터 붕괴)
- 채점 교란 없음: 양 arm 동일 문제·동일 math_verify·finding#2(추출 아티팩트) 0% flip으로 기각

**아직 미확정 (make-or-break)**: 위는 전부 val-aux(held-in). held-out 1030(특히 AIME)은 base gs300 도달 후 확정. 과거 메타 모델들은 *교란된* base GRPO에 held-out 7–9pp 뒤졌음(`meta-fails-root...`) — matched-base가 그 교란을 제거하지만 held-out 판정은 미지수.

---

## 1. 논문의 지적 정체성 (사용자 질문에 대한 답: "calibration·메타의 성질·원리를 정확히 분석할 필요가 있는가?")

**답: 필수다. 선택 사항이 아니라 논문의 정체성이다.** 이유:

이 분야는 **calibration**과 **metacognition**을 상습적으로 혼동한다. 둘은 다르다:

- **Calibration** = 자기 확신도(conf)를 실제 정답률(c_with)에 맞추는 것. Brier `R_cal = −(conf − c_with)²`가 강제. **그 자체로는 정확도를 올리지 않는다** — 완벽히 캘리브된 모델은 "나는 40% 확신"이라고 정확히 말할 뿐, 여전히 40%다.
- **Functional metacognition** = 자기감시를 통해 답을 인과적으로 개선(redirect/verify/backtrack)하는 것. 반사실 PMI `R_meta = c_with − c_without`가 강제. **이것이 정확도를 올릴 수 있다.**

우리 보상은 `R_corr + R_meta(pmi-shift) + R_cal`로 둘을 모두 담고 있어, **정확도 이득이 어느 쪽에서 오는지 분리하지 않으면 기여 주장 자체가 성립하지 않는다.**

**논문 thesis (제안)**:
> *"Calibration ≠ metacognition. 보상 분해로 둘을 분리하면, 반사실 보상을 받은 functional metacognition은 matched-base 대비 수학 정확도를 올리며(그 상한은 검증-오류 독립성에 묶인다), calibration 압력 단독으로는 정확도를 올리지 못하고 hard-task 능력을 오히려 해칠 수 있다."*

이 thesis는 우리 메모리와 정합적이다:
- `meta-fails-root-correlated-self-verification`: 메타가 정확도 못 올리는 뿌리 = 자기검증⊥자기오류 실패(같은 분포). 천장은 slip(독립)오류량에 묶임.
- `math500-grader-broken...`: confidence-rv가 easy 메타데이터로 학습→hard-math 능력 퇴화 = **calibration 압력이 능력을 해친 직접 증거.**

→ calibration/metacognition의 성질·원리 분석은 "부록"이 아니라 **본론(Result의 spine)**이다.

---

## 2. 태스크를 더 넓혀야 하나? (사용자 질문에 대한 답)

**답: 코어 논문은 넓히지 말 것 (depth > breadth). 단 RLMF식 factual-QA 1-arm은 generality 부록으로 선택적.**

- 현재 태스크(gsm8k → hendrycks_math → aime)는 이미 **난이도 스펙트럼**을 제공 → "메타가 slip(쉬움·독립오류)에서 더 돕고 capability(어려움·상관오류)에서 덜 돕는가"라는 **일반화 축이 태스크 내부에 이미 있다.**
- 새 도메인(factual QA/commonsense/code)으로 넓히면 초점이 흐려지고 비용 폭증. 메커니즘 논문은 **넓게보다 깊게** 가야 함.
- ARC/sudoku/crystal(메모리)은 **다른 프로젝트**(TTSO/energy) — 이 논문에 끌어오면 scope creep.
- 예외: RLMF(Z_g self-awareness + 데이터 선택)를 같은 세팅에 붙인 1-arm은 (a) 노벨티 대비(연속 PMI-shift 반사실 vs 그들의 self-awareness 스케일링), (b) generality 근거로 가치 → **여유 시 부록.**

---

## 3. 실험 목록 (자기비판 embed, 비용·의존성 표기)

범례: 💻=새 GPU 런 필요, 🔬=기존 데이터/eval 분석(경량 GPU or GPU-free), ⭐=must-have, ○=nice-to-have

### Part A — 분석 (경량, 기존/예정 eval 산출물 기반)

**A1 ⭐🔬 Calibration vs Metacognition 서술적 분해 (mechanistic)**
- 목적: pmishift 롤아웃에서 정확도 이득의 상관 구조를 서술. 메타 블록이 존재/부재한 문제, conf-정확도 정합, redirect 발생 여부별 정답률.
- 데이터: pmishift 1030 eval parquet + rollout_dump(존재 확인됨).
- **자기비판**: A1은 *상관/관찰*일 뿐 인과 분리가 아니다. calibration vs meta 기여의 진짜 분리는 A/B의 **E1 보상 ablation**(R_cal-only vs R_meta-only arm)에서만 나온다. A1은 그 인과 결과의 *서술*로만 쓸 것. 단독으로 "calibration이 아니라 meta가 원인"이라 주장 금지.

**A2 ⭐🔬 Flip 분석 (선택성)**
- 목적: base-오답→메타-정답(구제) vs base-정답→메타-오답(파괴)을 문제 단위로 분해. 이득이 "선택적 redirect"인지 "무차별 재유도"인지 판정.
- 데이터: base_matched_1030 parquet + pmishift 1030 parquet, **동일 문제 join**.
- **실현성 확인됨**: eval_vllm_1030이 문제별 `is_correct`+`completion`+`question` 저장. 단 저장된 `is_correct`는 **깨진 check_correctness** → **math_verify로 전량 재채점 필수**(메모리 `math500-grader-broken`).
- **자기비판**: 문제별 페어링은 *동일 문제·동일 decode*를 요구. 두 eval이 같은 문제집합·같은 max_tokens/temp인지 검증 후 join. 시드 1개라 flip 일부는 decode 노이즈 → pass@k나 다중 시드로 강건화 권장(비용↑).

**A3 ⭐🔬 독립성 프로브 (난이도 층화: slip vs capability)**
- 목적: thesis의 핵심 예측(메타는 독립오류=slip에서 돕고 상관오류=capability에서 못 돕는다) 직접 검증.
- 방법: 오류를 slip(base가 pass@k 높은 온도/다샘플에서 정답) vs capability(어떤 샘플서도 오답)로 프록시 라벨 → 각 층에서 메타의 flip율 비교. 보조지표: 오답 길이(메모리: aime 오답이 *더 김* = capability 결손).
- **자기비판**: slip/capability 라벨은 **프록시**이고 pass@k 샘플링이 필요(💻 경량 GPU). 리뷰어가 라벨 정의를 공격할 수 있음 → 정의를 사전 등록(pre-register)하고 민감도 분석(k 스윕) 첨부. 이 계획에서 **가장 논쟁적인 분석** — 결과가 약하면 본론이 아니라 discussion으로 강등.

### Part B — 인과 ablation (새 GPU 런)

**E1 ⭐💻 보상항 분해 ablation (thesis의 인과 증명)**
- arm: (a) R_corr만 = **matched-base(이미 있음)**, (b) +R_meta만(R_cal off), (c) +R_cal만(R_meta off), (d) full = pmishift(이미 있음). → **신규는 (b),(c) 2 arm.**
- 판정: (b)가 base 위로 올리고 (c)는 안 올리면 → "정확도=functional meta, calibration 아님" 인과 확정. thesis 성립.
- **자기비판**: (b),(c)는 각 ~gs300 풀런이라 선점 하에 느리고 비쌈. **비용 절감**: val-aux 격차가 gs125에서 이미 뚜렷하므로 ablation arm은 **gs150까지만** 돌려 *상대* 기여만 보고, 승자만 연장. (c) R_cal-only가 hard-math 해치는지도 여기서 관찰(메모리 예측).

**E2 ⭐💻/🔬 토큰 예산 통제 ("길게 생각해서 이긴 것" 반박 차단)**
- 1차(🔬, 경량): 기존 eval에서 base·meta의 정확도-vs-실제응답토큰 곡선. 같은 토큰 예산 구간에서도 메타가 위면 compute 설명 기각.
- 2차(💻, 필요시): base에 길이 인센티브/패딩 준 arm이 못 오르면 확증.
- **자기비판**: 1차는 상관적. 메타 토큰=`<|meta|>`가 정보이지 순수 padding이 아님 → "메타는 유용한 토큰을 더 쓴다"는 반론엔 A2/A3(선택성·독립성)로만 반박 가능. E2 단독으론 부족, A2/A3와 세트.

**E3 ⭐💻 시드 (에러바)**
- pmishift·matched-base 각 ≥3시드. 격차 > 시드분산 입증.
- **자기비판**: 6런×~13h 선점 = 최고비용. **그러나** base val-aux가 노이지(gs75=0.589 급락) → "base가 낮다"는 주장이 1개 불운한 시드일 수 있음 = 시드는 생각보다 *더* 긴급. **결정 게이트 적용**(§4): held-out(E4) 이긴 뒤 시드 투입. 비용절감: 헤드라인 2 arm만, 필요시 gs150 지평.

### Part C — 확증·포지셔닝

**E4 ⭐💻 Held-out 1030 (gs300, incl AIME) — 이미 예정, 자동 발사**
- base gs≥290 도달 시 `h100std_base_matched_1030_eval.yaml` 자동. 4k/16k 양쪽.
- **이것이 전체 계획의 make-or-break 게이트**(§4).

**E5 ○🔬 16k degeneration** — eval yaml에 이미 포함. pmishift가 다른 메타처럼 16k서 무너지나, base-on-meta_mix도 무너지나(=degeneration이 데이터 탓인가). 사실상 공짜.

**E6 ○💻 RLMF 베이스라인** — 최소는 개념 비교표(🔬 무료), 여유 시 1-arm 실증(💻). future work 허용.

---

## 4. ★결정 게이트 구조 (선형 아님 — 자기비판의 핵심)

**가장 큰 리스크: held-out AIME가 메타≤base로 나오면 §0~1 프레이밍이 붕괴**(과거 7–9pp 적자 재현 가능). 그러므로 계획은 **분기**한다:

```
E4 (held-out 1030, gs300) ──┬─ 메타 WIN ──→ 풀 플랜: E3 시드 + E1 ablation + A1/A2/A3 메커니즘
                            │                 = "functional meta > base, 원리는 독립검증" 강한 논문
                            │
                            └─ 메타 NEUTRAL/LOSE ──→ 피벗:
                                  (i) "in-distribution 개선 + 왜 held-out 전이 안 되나"(독립성 상한)
                                  (ii) "when/why functional meta helps"(난이도-조건부, A3 중심)
                                  = 여전히 출판가능한 메커니즘 논문, 단 헤드라인 변경
```

**함의: E4(자동, 곧) 전에는 E3 시드/E1 ablation에 대규모 컴퓨트를 태우지 않는다.** 지금 당장 가치있는 건 **GPU-경량 분석**(A1/A2/E2-1차)과 **E4 대기**다.

부차 리스크:
- base 단일시드 노이즈(gs75 급락) → "base 열등"이 시드운일 수 있음 → E3로만 해소(단 게이트 후).
- val-aux=held-in → in-dist 격차가 전이 안 될 수 있음 → E4가 유일한 진짜 시험.
- A3 slip/capability 라벨은 프록시 → pre-register + 민감도.

---

## 5. 즉시 착수 가능 (GPU-경량, 승인 시)

E4 게이트 전에 **지금** 할 수 있고 어느 분기에서도 버려지지 않는 것:
1. **A2 flip 분석 파이프라인** 구축 (base_matched_1030 ↔ pmishift_1030 parquet join + math_verify 재채점). base 1030 나오면 즉시 실행.
2. **A1 서술 분석** (pmishift rollout: 메타 존재/부재·conf정합·redirect별 정답률).
3. **E2-1차** 정확도-vs-토큰 곡선(기존 eval에서).
4. **E5** 16k degeneration 판정(eval 산출물에서).

이들은 held-out 결과와 무관하게 논문에 들어가는 서술/그림을 만든다.

---

## 6. 미해결·확인 필요 (브레인스토밍에서 다룰 것)
- A3 slip/capability 프록시의 정확한 조작정의 + pass@k 예산.
- E1 ablation의 지평(gs150 vs gs300)과 정확한 arm 수(R_meta-only, R_cal-only, +R_meta+R_cal?).
- E3 시드 수·지평·arm 범위(비용 vs 리뷰어 요구 절충).
- RLMF 실증 arm을 코어에 넣을지 future work로 뺄지.
- pmi-shift와 CF(gs100 +0.040)·asymcf_v2(gs125 0.720)의 관계를 논문에서 어떻게 서술할지(pmi-shift만 vs 반사실 계열 전체).

---

## 2026-07-05 업데이트

### (a) 프레이밍 확정 — 메타인지적 자기증류로 OOD-강건 학습 (사용자 승인)

논문의 북극성 프레이밍을 다음으로 확정한다: **"메타인지적 자기증류(metacognitive self-distillation)로 OOD에 강건한 수학 추론을 학습한다."** pmi-shift 보상은 모델 *자신의* gold/decoy 로그확률 대비 — 즉 외부 교사가 아니라 자기 신호 — 를 `<|meta|>` 구간에 증류하는 self-distillation이며, 핵심 가설은 이렇게 학습된 메타인지 행동이 in-distribution 정답 암기보다 OOD(더 어려운 도메인, 특히 AIME)에 더 강건하게 일반화한다는 것이다. §1의 calibration ≠ metacognition 분해는 그대로 유효하되, 이제 그 분해가 "무엇이 OOD 강건성을 만드는가"라는 상위 질문 아래에 놓인다.

**RLRT와의 관계 (직교 포지셔닝).** RLRT는 Qwen3-8B-Base에서 no-SFT GRPO 대비 반성적 추론 보상으로 정확도를 올린 동시대 작업이다(no-SFT GRPO 83.6 MATH500 / 19.8 AIME24 → RLRT 84.4 / 27.9, 20480 tokens, avg@16). 우리는 이를 정면 비교(head-to-head) 대상으로 삼지 않는다 — 세팅(SFT 유무, 토큰 예산, 베이스 체크포인트)이 달라 숫자 비교가 불공정하고, 무엇보다 메커니즘이 다르다. RLRT는 반성 *행동 형태*를 보상하는 반면, 우리는 **gold-anchored meta-region self-distillation** — gold/decoy 반사실 로그확률이라는 접지된 자기 신호를 메타 구간에만 국소적으로 증류 — 이다. 따라서 포지셔닝은 "우리가 RLRT를 이긴다"가 아니라 "반성 RL 계열과 직교하는, 접지-신호 기반 메타인지 학습 축을 연다"로 간다. RLRT 수치는 관련연구 표에서 참고 수치(their setting)로만 인용한다.

### (b) RQ4/T4 — Calibration은 보조 지표로 상시 측정 (구 RQ6, (g)의 4-RQ 압축 반영)

연구 질문 RQ4: **Calibration은 어떻게 되는가?** 대응 표 T4는 arm별 ECE(15-bin), Brier score, 과신율(overconfidence rate)을 담는다 — **순수 calibration만이며, OOD-vs-ID 강건성 부분은 RQ3/T3(층화)으로 이동했다**((g) 참조). 명확히 못박아 둘 것: **주-지표는 정확도이고 calibration은 보조 지표다.** §1의 thesis가 말하듯 calibration 그 자체는 정확도를 올리지 않으므로, calibration 개선을 헤드라인으로 팔지 않는다. 다만 (i) "메타인지가 좋아지면 calibration도 따라온다"는 부수 예측의 검증, (ii) R_cal-only arm(E1c)이 calibration만 얻고 정확도를 잃는지의 대조를 위해 모든 eval에서 함께 산출한다. 측정은 held-out 1030 채점 파이프라인(math_verify)에 confidence 추출을 붙여 arm 공통으로 수행한다 (`experiments/analysis/calibration.py`).

### (c) eval 프로토콜 상향: 4k n=1 → 16k avg@8

기존 4k max_tokens, n=1 프로토콜을 논문 표준으로 쓰기에는 (i) 16k degeneration 이슈(E5)를 가리고, (ii) 단일 샘플 노이즈가 AIME 30문항에서 치명적이다. 논문 최종 수치는 다음으로 상향한다: **max_tokens 16k, avg@8 (AIME는 avg@16), temperature 0.7, 양 arm을 반드시 같은 eval 잡에서 같은 시드로** 돌린다(METRIC RULES와 일치). 4k n=1 결과는 학습 중 빠른 모니터링용으로만 유지하고 논문 표에는 넣지 않는다. 기존 `h100std_*_1030_eval.yaml` 계열은 라이브 런 보존 원칙(additive-only)에 따라 수정하지 않고, 상향 프로토콜용 eval 설정을 새로 추가한다.

### (d) 인프라 사후분석 — gs245 제논 루프와 수정 사항

base arm이 gs245에서 세 번 연속 같은 자리로 되돌아오는 제논(Zeno) 루프에 빠졌던 원인을 사후분석했다. 구조는 이렇다: MSR 클러스터의 6h max_run 선점 윈도우 안에서, 체크포인트 업로드가 스텝당 ~16GB를 all-or-nothing 커밋으로 올리다가 윈도우 끝에 잘리면 그 스텝 전체가 유실된다 → 다음 윈도우는 더 오래된 완결 스텝에서 재개 → 다시 같은 지점에서 잘림 → 진행이 0인 채 세 윈도우를 소모했다.

수정은 두 갈래다. 첫째, **per-file 내구 푸셔**(`scripts/push_ckpts_to_hf.py`): 파일 단위 업로드 + `/scratch/.pushed_<config>.json` 장부로 선점-재개 시 재업로드를 건너뛰어, 윈도우가 잘려도 이미 올라간 파일은 살아남는다. 둘째, per-file 업로드는 필연적으로 HF 리포에 PARTIAL 스텝 디렉토리를 남기므로, **완결성 규칙(actor 모델 샤드 ≥ 4개 = COMPLETE)** 을 세 곳에서 통일 적용했다: (i) 푸셔의 retention(`push_ckpts_to_hf.py` — partial을 keep 카운트에 세지 않아 유일한 complete 체크포인트 축출 방지), (ii) 재개 풀러(`scripts/pull_resume_ckpt.py` — COMPLETE 스텝만 재개 후보), (iii) 파이프라인 yaml의 RGS 프로브(`h100std_base_matched_pipeline.yaml`의 resume-gs 계산 — HF에 gs>0이 있는데 로컬 풀이 실패하면 gs0 콜드스타트로 계보를 오염시키는 대신 ABORT). 세 곳 중 한 곳이라도 partial을 complete로 오인하면 크래시-재개 루프나 계보 오염이 재발하므로, 규칙의 단일화 자체가 수정의 핵심이다.

### (e) Stage 0–3 로드맵 + 결정 게이트 + 협업자 트랙 분배

§4의 E4 게이트 구조를 스테이지로 재편한다.

- **Stage 0 (현재)**: base arm gs300 확보(제논 루프 수정 후 재개 중) + 상향 eval 프로토콜((c)) 준비 + GPU-경량 분석(A1/A2/E2-1차/E5) 선행. 어느 분기에서도 버려지지 않는 작업만 태운다.
- **Stage 1 (T1 판정 게이트)**: base gs300 vs pmishift gs300을 held-out 1030 + 9-domain에서 상향 프로토콜로 판정. **이것이 유일한 make-or-break 게이트**이며 §4의 분기 구조가 그대로 적용된다.
- **Stage 2**: 게이트 **WIN** 시 — T1 참조행 확보(REF-0 no-SFT GRPO·raw Qwen3-8B·SFT-only 평가), Gandhi-arm(meta-SFT 후 VANILLA_GRPO, RQ2), 그리고 T1 프로토콜에 내장된 시드 ×3·token-budget control을 병렬 발사. 게이트 **LOSE/NEUTRAL** 시 — T2 메커니즘 분석(flip save/derail + placebo shuffled-meta) 중심으로 피벗하고 헤드라인을 "when/why"로 변경.
- **Stage 3**: SFT v2 파일럿(R1류 long-CoT에 메타 주석). 게이트 결과와 **독립적으로** 시작 가능 — 어느 분기에서도 다음 세대 SFT의 근거가 되므로 Stage 1을 기다리지 않는다.

**협업자 트랙 분배 (A–D)** — `experiments/README.md` 6절과 동일한 구분을 쓴다: 트랙 A = 클러스터 학습(base gs300 완주, Stage 2 arm 발사·재개·HF 릴레이 감시); 트랙 B = 분석, GPU 불필요 — **T2/T3/T4 분석이 이 트랙 담당이다**(T2 flip·placebo, T3 난이도 층화, T4 calibration, 파케이 join + math_verify 재채점, 표 통합 `aggregate_tables.py` T1–T4); 트랙 C = SFT v2 데이터(Stage 3 파일럿); 트랙 D = 집필·사이트(논문 숫자는 지표 규약 통과분만, PRELIMINARY 표기 유지). 트랙 간 의존성은 A→B(체크포인트가 있어야 eval·분석)뿐이며 C와 D는 게이트와 독립적으로 진행 가능하다.

### (f) `experiments/` 폴더 도입 — 과학/인프라 분리

일회성 과학 스크립트가 `scripts/`(학습·인프라 코드)에 섞이면서 어떤 파일이 라이브 런의 의존성인지 식별이 어려워지는 문제를 막기 위해, 리포 최상위 `experiments/` 폴더를 분리 도입했다. 규칙: **`scripts/` = 인프라**(학습 부트스트랩, HF 푸셔/풀러, 데이터 빌드 등 라이브 런이 tarball로 물고 가는 코드 — additive-only 원칙의 보호 대상), **`experiments/` = 과학**(가설 검증용 프로브·분석 — 자유롭게 추가·폐기 가능, 라이브 런이 참조하지 않음). 현재 구조는 `experiments/common/`(공유 유틸: `env.py` 시크릿/경로, `grading.py` math_verify 채점, `vllm_gen.py` 생성 헬퍼, `probe_utils.py`)과 `experiments/probes/`(개별 프로브 스크립트)이며, 이후 Stage별 분석(A1/A2/A3, T2 flip·placebo)도 `experiments/` 아래에 서브폴더로 추가한다. 새 분석을 짤 때 채점은 반드시 `experiments/common/grading.py`(math_verify)를 재사용하고 깨진 check_correctness를 다시 들여오지 않는다.

### (g) RQ 구조 압축 — 6개 → 4개 (사용자 결정)

연구 질문을 6개에서 4개로 압축한다. 질문 수를 줄이는 것이 목적이 아니라, 별도 RQ로 세워둘 만큼 독립적이지 않던 항목(시드·토큰통제는 프로토콜, SFT 비용은 참조행)을 제자리에 넣고 층화 분석을 명시적 질문으로 승격하는 재배치다.

- **RQ1 (T1, 메인 표)**: PMI-shift가 실제로 정확도를 올리는가 — matched-base gs300 vs pmishift gs300, held-out 1030 + 9도메인, 16k tokens, avg@8(AIME avg@16). 구 RQ5(시드 ×3, token-budget control)는 별도 질문이 아니라 **T1의 프로토콜에 내장**한다(시드별 mean±std, 응답 토큰 수 통제열 보고). 구 RQ2(SFT 능력비용)와 REF-0은 별도 실험이 아니라 **T1의 참조행**(raw Qwen3-8B, SFT-only)으로 흡수해 SFT 비용이 같은 표 안에서 읽히게 한다.
- **RQ2 (T2, 분해·메커니즘 표)**: 효과는 무엇이며 어디서 오는가 — Gandhi-arm(meta-SFT + VANILLA_GRPO)으로 SFT-프라이밍 vs RL-보상 기여 분해 + flip 분석(SAVE/DERAIL) + placebo(셔플 메타). 구 RQ3+RQ4의 통합.
- **RQ3 (T3, 층화 표)**: 난이도·문제 유형·OOD에 따라 메타 효과가 어떻게 달라지는가 — 난이도 사분위 층화 정확도+방출률(Simpson 발견: Q1 easy 방출 10%, Q3 mid-hard 메타 0.83 vs base 0.67 — PRELIMINARY), 도메인별, AIME 등 어려운 도메인의 OOD 강건성. 지표 규약에 묻혀 있던 층화 분석을 명시적 RQ로 승격 — OOD-강건 self-distillation 프레이밍((a))과 직결된다.
- **RQ4 (T4)**: Calibration은 어떻게 되는가 — arm별 ECE(15-bin)/Brier/과신율. 순수 calibration만이며 구 RQ6의 OOD 부분은 RQ3으로 이동. 주-지표는 정확도, calibration은 보조라는 위계((b))는 그대로다.

분석 스크립트 매핑: `experiments/analysis/flip.py`+`placebo.py`→T2, `stratify.py`→T3, `calibration.py`→T4, `aggregate_tables.py`가 T1–T4 전부를 하나의 markdown으로 생성. 협업자 트랙 분배는 (e)에 반영 — 트랙 B(분석)가 T2/T3/T4를 담당한다. 구 T5/T6 번호는 리포 전체에서 폐기한다.
