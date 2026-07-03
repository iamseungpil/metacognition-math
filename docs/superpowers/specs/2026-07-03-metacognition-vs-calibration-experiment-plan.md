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
