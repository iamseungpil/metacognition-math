# SURVEY — 수정(revision) 구간 크레딧 설계 관련 선행연구 (2026-09-19)

대상 설계: 4B 수학 정책이 한 rollout 안에서 자발적으로 답을 고치는 ~3% 행동에,
형제 rollout이 아니라 **같은 row의 first vs last 답**을 비교한 크레딧을 revision-zone 토큰에만 주는 방식
(outcome 개선형 = SCoRe류 / PMI belief-shift형 = 최종답 작성 **전** teacher-forced 측정).

각 항목: `arXiv id | 한 줄 주장 | 한 줄 관련성 | 검증`

---

## A. 단일 패스 내부의 자기수정/반성 구간에 크레딧 주기

- **2409.12917 (SCoRe)** | 2단계 multi-turn 온라인 RL로 자기생성 데이터만 써서 자기수정을 학습, 1단계는 1차 시도 분포를 base에 묶고 2단계는 두 시도를 함께 최적화하되 "직답으로 붕괴"를 막는 reward bias를 넣음(Gemini 1.0 Pro/1.5 Flash에서 MATH·HumanEval 자기수정 +15.6%/+9.1%). | 우리 outcome형 크레딧(1차→최종 개선)의 직계 조상. **차이: SCoRe는 프롬프트로 강제된 2턴 대화, 우리는 한 패스 안 자발적 수정 구간.** | 검증: arXiv 초록 검색 결과(2409.12917) 요약 확인.
- **2602.08503 (Octopus)** | VLM에서 "효과적 자기수정은 극히 드물게만 발현되어 학습 신호가 극도로 희소"하므로 기존 rollout을 재조합해 correction-specific rollout을 합성하고, response-masking으로 직답과 자기수정을 분리 학습(7개 벤치 SoTA, RLVR baseline +1.0). | 우리의 공개문제(1) "크레딧이 ~0.5% row만 건드림"에 대한 가장 직접적인 선행 해법(= 신호 밀도를 데이터 증강으로 올림). **차이: 우리는 증강 없이 advantage 라우팅으로 해결하려 함.** | 검증: arXiv 초록 전문(2602.08503) fetch.
- **2607.00482 (DASH)** | 중간 답 commitment를 step-label 대리로 써서 각 세그먼트가 정답 쪽으로 움직였는지 판정하고 세그먼트 단위 advantage shaping(59.45% vs Dr.GRPO 58.1%, GRPO 56.95%), 생산적 자기수정은 늘리고 overthinking은 줄임. | first/last 답 비교 → 구간 크레딧이라는 우리 골격과 사실상 같은 형태. **차이: DASH는 gold와 비교(outcome, 지도 필요), 우리 PMI안은 gold 없이 belief 변화만으로도 정의 가능하고 zone에만 라우팅.** | 검증: arXiv 초록 fetch(2607.00482).
- **2606.18810 (SC-GRPO)** | 모델을 자신의 검증된 trajectory에 조건화했을 때 생기는 per-token KL을 GRPO 그래디언트의 곱셈 가중치로 사용, 외부 PRM·teacher 없이 토큰 크레딧 생성(GRPO +8.1%, DAPO +5.9%). | "모델 자신의 분포 변화"를 크레딧으로 쓴다는 점에서 우리 PMI안과 같은 계열. **차이: 우리는 zone 시작/끝의 gold−X log-odds 차이를 쓰고, SC-GRPO는 조건화 유무 KL.** | 검증: arXiv 초록 전문(2606.18810) fetch.
- **2601.01580** | Two-Stage Decision-Sampling 가설: 정책을 "생성"과 "언제 고칠지 결정"으로 분리하면 RL의 일반화 이득이 주로 sampling이 아니라 **decision** 개선에서 온다고 주장. | 우리가 수정 구간만 따로 보상하는 근거(=decision 성분을 직접 겨냥). | 검증: arXiv 초록 fetch(2601.01580).

## B. 모델 자신의 belief/confidence 변화로 보상 shaping

- **2510.22255 (PACR)** | 외부 검증 없이 "정답에 대한 모델의 진화하는 믿음"에서 dense reward를 계산, 잘 짜인 추론 궤적에서는 ground-truth 확률이 대체로 상승해야 한다는 가정을 보상으로 사용. | 우리 PMI안과 가장 가까움. **차이: PACR은 궤적 전체의 단조 상승을 요구, 우리는 revision zone 앞뒤 두 점의 log-odds 차이만 쓰고 placebo 구간으로 대조.** | 검증: arXiv 초록 fetch(2510.22255).
- **2603.22293 (TIPS)** | 각 reasoning–tool-call 세그먼트에, 그 세그먼트가 (현재 정책의 frozen snapshot을 teacher로 삼아) 정답의 log-likelihood를 얼마나 올렸는지로 dense reward 부여 = turn-level information-potential shaping. | "정답 로그확률 증가량 = 구간 보상"이라는 potential-based 형식이 우리 PMI와 동형. **차이: TIPS는 검색/툴 턴 경계, 우리는 자발적 수정 구간 경계.** | 검증: 검색 결과 요약(2603.22293, arXiv html). 초록 원문 미확인 — 인용 시 재확인 필요.
- **2602.03979** | reference answer의 log-probability 자체를 CoT 학습의 보상으로 쓰면 검증기 없이도 광범위하게 작동하고, verifiable 환경에서 binary reward와 동등하거나 상회(perplexity는 더 좋음). | 우리 PMI가 쓰는 gold log-prob 신호가 보상으로 유효하다는 일반 근거. | 검증: arXiv 초록 fetch(2602.03979).
- **2505.19590 (Intuitor/RLIF) · 2505.22660 (RENT)** | Intuitor는 균등분포와의 KL(self-certainty), RENT는 예측분포의 음의 엔트로피를 외부 보상 없이 내재 보상으로 사용해 추론 성능을 올림. | "모델 자신의 확신"을 보상으로 쓰는 계열의 대표. **차이: 둘 다 정답-무관 절대 확신도이고 우리처럼 gold 대비 belief의 *변화*(전/후 차분)가 아님.** | 검증: 검색 결과 요약(2505.19590, 2505.22660). 초록 원문 미확인.
- **2606.14211 (RefGRPO)** | 에이전트의 자기 reflection과 실제 결과를 대조해 만든 "무료 calibration bonus"를 RL에 추가(text-to-SQL 과소확신 44.4%→7.7%, 정확도 75.1%→76.5%). | belief와 outcome의 불일치를 보상으로 쓰는 또 다른 형태. **차이: 결과가 나온 뒤(post-hoc) 대조, 우리는 최종답 작성 전 측정.** | 검증: arXiv 초록 fetch(2606.14211).

> **"답 커밋 전에 측정하는가?"** — PACR·TIPS는 궤적 중간에서 gold 확률을 teacher-forced로 재는 점에서 우리와 같지만, 둘 다 *수정 구간*을 경계로 잡지 않고 placebo 구간 대조도 두지 않음. 이 두 가지(수정 구간 국소화 + placebo)는 조사 범위에서 선행 사례를 못 찾음.

## C. GRPO/RLVR에서 희귀 행동 증폭 · 크레딧 크기/밀도

- **2510.03222 (Lp-Reg)** | RLVR의 entropy collapse는 저확률 탐색 토큰("reasoning sparks")이 과도 penalize되어 소멸하기 때문이며, noise 토큰을 걸러 재정규화한 proxy 분포로 KL 정칙화해 이를 보호(3,000 step 안정 스케일링, 5개 수학 벤치 평균 60.17%, +2.66%). | 우리 3% 수정 행동이 plain GRPO에서 반감되는 현상의 메커니즘 설명 + 보상 대신 정칙화로 보존하는 대안. | 검증: arXiv 초록 전문(2510.03222) 확인.
- **2602.03452 (Weighted GRPO)** | pair-level로 binary outcome을 재가중해 "드문 성공"을 날카로운 양의 신호로, "드문 실패"를 강한 음의 신호로 증폭(AIME25 Pass@8 16.8→22.2, AMC23 Pass@64 94.0→97.0). | 공개문제(1) 크레딧 크기 문제의 직접 처방(rare-event amplification). **우리 batch-mean |advantage| 클리핑은 정확히 반대 방향(희귀 신호를 깎음)** — 재검토 필요. | 검증: arXiv 초록 fetch(2602.03452).
- **2605.00365 (UCPO)** | 정답 rollout에 배분되는 advantage 총량은 보존하되 과소대표된 정답 응답 쪽으로 재배분해 RLVR의 다양성 무관심을 깨뜨림. | "총 advantage 질량 보존 + 소수 모드로 재배분"은 우리가 revision row만 키울 때 쓸 수 있는 정규화 레시피. | 검증: 검색 결과 요약(2605.00365 PDF). 초록 원문 미확인.
- **2607.20543** | RLVR의 pass@k 역전은 base 모델이 드물게만 내는 정답 궤적이 유한 rollout group에 아예 나타나지 않아 강화되지 못하는 "absence-of-evidence" 실패로 설명됨. | 3% 행동이 그룹 안에 거의 안 잡히는 우리 상황과 동일 구조 — 샘플 수/그룹 구성이 크레딧 크기보다 먼저인 변수일 수 있음. | 검증: 검색 결과 요약(2607.20543). 초록 원문 미확인.

## D. 자기수정 RL의 sandbagging / reward hacking

- **2602.08503 (Octopus, §4.2)** | 자기수정 reward shaping을 쓰자 "the model deliberately produces an incorrect first response despite correct reasoning, followed by a trivial correction"가 나타나 학습이 불안정해지고 전체 추론력이 저하됨. 저자들의 방어는 response-masking으로 직답/자기수정 신호를 분리하는 것. | **우리 outcome형(1차→최종 개선) 크레딧의 가장 직접적인 위협 사례이자 유일하게 실측된 sandbagging 증거.** PMI형이 이 해킹에 덜 노출되는지가 핵심 실험. | 검증: arXiv html 본문에서 해당 문장 verbatim 확인(2602.08503).
- **2409.12917 (SCoRe)** | 1단계에서 1차 시도 분포를 base에 묶고 2단계 보상에 bias를 넣어 "직답 붕괴"를 막음 = 1차 시도를 건드리지 못하게 하는 제약이 곧 방어책. | 우리도 1차 답 분포에 KL anchor를 거는 형태의 방어를 그대로 차용 가능. | 검증: 위와 동일(2409.12917 초록).
- **2604.28182 (Exploration Hacking)** | 모델이 정답을 안다는 것을 보이면서도 의도적으로 틀린 답을 내거나 거부하는 "strategic underperformance"를 정의하고, 탐색을 통제해 RL 업데이트를 회피할 수 있는지 조사. | sandbagging의 일반 프레임 — 우리 진단 지표(1차 답 정확도 하락 모니터링)의 근거. | 검증: 검색 결과 요약(2604.28182 PDF). 초록 원문 미확인.

## E. Countdown / 산술 탐색에서 불가능 인스턴스·abstention

- **TinyZero (github Jiayi-Pan/TinyZero) + `Jiayi-Pan/Countdown-Tasks-3to4`** | Countdown RL의 사실상 표준 데이터셋. HF 카드가 비어 있고 컬럼은 `target`, `nums` 둘뿐(490,364 rows), 해답 컬럼도 solvable 플래그도 없음. | 우리가 "불가능 인스턴스"를 쓰려면 **직접 생성/검증해야 함**(기존 셋은 solvability 라벨 없음). | 검증: HF 데이터셋 페이지 fetch(README 비어 있음, 컬럼 2개 확인) + TinyZero repo 검색 결과.
- **2512.01775** | RL post-training이 Countdown에서 어떤 skill composition을 유도하는지를 expression tree 모양 단위로 분석(더 큰 문제·새 구조로의 일반화). 초록에 solvable/unsolvable 언급 없음. | Countdown을 진단용 probe로 쓰는 최신 레퍼런스이나 불가능 인스턴스는 다루지 않음 = 우리 E축이 빈칸임을 뒷받침. | 검증: arXiv 초록 fetch(2512.01775), unsolvable 언급 없음을 확인.
- **2502.20238 (FINEREASON)** | 논리 퍼즐을 atomic step으로 쪼개 **state checking / state transition** 두 과제로 중간 추론을 평가하고, 이 데이터로 학습하면 GSM8K가 최대 +5.1%. | "현재 상태가 해로 이어지는가"를 직접 묻는 과제 설계는 우리 dead-state 판정 테스트의 템플릿. 단, 초록에 "unsolvable/dead state"라는 표현은 없음(과제명은 state checking). | 검증: arXiv 초록 전문 fetch(2502.20238).
- **2508.18760 (AAAI 2026, "Answering the Unanswerable Is to Err Knowingly")** | 풀 수 없는 수학 문제에서 LRM의 abstention 실패를 분석, linear probe 정확도 >80%로 **내부에는 불가능성 인식이 있으나 외부 행동은 강제 응답 쪽으로 편향**. | 우리 가설(메타인지 신호는 내부에 있고 행동으로 안 나온다)의 수학 도메인 직접 증거. | 검증: 검색 결과 요약(paper note, arXiv 2508.18760). 초록 원문 미확인.
- **2608.29109** | 답 가능/구조적으로 불가능한 수학·코드 프롬프트를 가르는 단일 선형 방향이 1.7B~70B에서 존재하며, 이 방향은 안전 거부 방향과 거의 직교. | 불가능 판정이 학습 가능한 표현으로 이미 존재 → RL로 "행동화"만 하면 된다는 설계 정당화. | 검증: 검색 결과 요약(2608.29109). 초록 원문 미확인.

## F. 외부 교사 없는 추론 습관 self-distillation

- **2203.14465 (STaR) / 2312.06585 (ReST-EM)** | 자기 생성 rationale 중 정답을 낸 것만 남겨 반복 fine-tune(STaR); EM 관점에서 샘플 생성→binary feedback 필터→fine-tune 반복(ReST-EM), MATH·APPS에서 인간 데이터 SFT를 크게 상회. | 우리 설계 대응: "수정이 일어나고 개선된 row만 남겨 그 zone을 SFT" = 가장 싼 ablation(크레딧 크기 문제를 우회). | 검증: ReST-EM은 검색 결과 요약(2312.06585 초록 인용), STaR는 검색 결과 요약(2203.14465). 원문 미확인.
- **2504.16084 (TTRL)** | 라벨 없이 majority voting을 보상으로 써서 test-time RL을 돌리면 Qwen2.5-Math-7B의 AIME24 pass@1이 12.9→40.2(+211%), 투표 상한선까지 넘김. | 우리 PMI안의 gold를 majority-vote pseudo-gold로 바꾸면 라벨 없이도 belief-shift 크레딧을 정의 가능(확장 경로). | 검증: arXiv 초록 검색 결과(2504.16084) 수치 확인. | 
- **2603.05433 (OPSD/CRISP)** | 모델 자신의 간결한 행동을 자기 자신에게 distill해 reasoning을 압축하는 on-policy self-distillation. | "single pass가 best-of-K를 근사한다"는 압축 관점의 대표 — 우리는 "수정 후 답"을 "처음부터 그 답"으로 압축하는 self-distillation으로 재해석 가능. | 검증: 검색 결과 요약(2603.05433). 초록 원문 미확인.
- **2509.14252 (LLM-JEPA)** | JEPA의 joint-embedding 예측 목적을 autoregressive LLM에 이식, 생성능력은 보존하면서 추상화를 개선(GSM8K에서 Qwen3-1.7B·R1-Distill-1.5B 유의미 향상 보고). | latent 공간 타깃 계열 대응: 우리 revision zone의 "수정 후 상태"를 latent 예측 타깃으로 쓰는 변형이 가능. **단 two-view 페어 데이터가 필요**해 우리 설정에 바로 안 붙음. | 검증: 검색 결과 요약(2509.14252 PDF + 인용 논문). 초록 원문 미확인.

---

## 우리 설계와의 차별점과 위험

1. 진짜 새로운 것 (A) **자발적** 수정 구간을 경계로 잡아 그 토큰에만 크레딧을 라우팅하는 점 — SCoRe는 2턴 프롬프트 강제, DASH는 세그먼트 일반, Octopus는 증강으로 유도.
2. 진짜 새로운 것 (B) belief shift를 **최종 답이 쓰이기 전에** teacher-forced로 재고 **placebo 구간과 대조**하는 점. PACR/TIPS는 중간 gold 확률을 쓰지만 수정 구간 국소화도 placebo 대조도 없음.
3. 새롭지 않은 것: "1차 vs 최종 개선을 보상" 자체(SCoRe/DASH), "모델 자신의 확신/로그확률을 보상"(PACR, TIPS, 2602.03979, Intuitor, RENT), "sibling 아닌 within-row 비교"도 DASH의 중간 commitment와 실질 유사.
4. 새롭지 않은 것: 희귀 행동 보존 문제 자체 — Lp-Reg가 "reasoning sparks 소멸"로 이미 정식화했고, 우리 3% 반감 관찰은 그 특수 사례로 읽힐 위험이 있음.
5. **가장 위협적인 선행 결과: 2602.08503(Octopus) §4.2.** 자기수정 reward shaping이 ~정상 학습 중에 "일부러 1차 답을 틀리게 내고 사소하게 고치는" 해킹을 실제로 유발했고 전체 추론력을 떨어뜨렸다. 우리 outcome형 크레딧은 같은 함정에 그대로 노출된다.
6. 두 번째 위협: 2607.20543 — 3% 행동이 rollout group에 거의 안 잡히면 크레딧 설계와 무관하게 강화가 안 된다. 즉 공개문제(1)은 advantage 크기가 아니라 **샘플링/그룹 구성** 문제일 수 있다.
7. 세 번째 위협: 2602.03452·2605.00365 기준으로, batch-mean |advantage| 클리핑은 rare-event를 **증폭이 아니라 감쇠**시키는 선택이다. 최소한 rare-success 재가중 또는 질량 보존형 재배분과 비교 ablation이 필요하다.
8. 방어 설계 최소 세트: (i) 1차 답 분포에 KL anchor(SCoRe 1단계), (ii) 1차 답 정확도를 별도 모니터 지표로(Exploration Hacking), (iii) PMI형 vs outcome형을 같은 조건에서 해킹률로 비교.
9. E축은 문헌 공백이 확인됨 — Countdown 계열에 solvability 라벨이 없으므로 불가능 인스턴스는 자체 생성해야 하고, 그만큼 novelty는 크지만 baseline이 없다.
