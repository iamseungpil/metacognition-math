# 서베이 — 셀프 디스틸·특권 정보·온폴리시 증류 RL 논문은 비교군과 절제를 어떻게 고르나 (2026-10-12)

질문: 최근 셀프 디스틸 / 특권 정보 / 온폴리시 증류(OPD) RL 논문들은 비교군·절제·평가·행동 분석을 어떻게 짜는가?
그리고 우리 논문(«풀이 중간 멀티 교사 셀프 디스틸», Qwen3-4B-Instruct-2507, DAPO-Math-17k, RLRT 프로토콜)에 무엇이 빠졌는가?

범위: 연구만 했다(코드·GPU·큐·git·내려받기 없음). 모든 논문은 arXiv abs/HTML 페이지에서 제목·저자·ID를 확인했다.
표의 내용은 HTML 본문에서 뽑았다. «미보고»는 본문(앞 100k자)에서 찾지 못했다는 뜻이다. 부록 끝이 잘린 논문은 표시했다.
우리 쪽 현황은 사전등록 수정 70–72b와 `paper/draft_v2/sections/experiments.tex`를 기준으로 했다.

---

## 0. 한 줄 결론

- 이 장르의 **사실상 표준 비교군**은 «base · GRPO(DAPO 레시피) · 직계 셀프 디스틸 2–3개(SDPO·OPSD·RLSD 중)»다. 여기에 **오프폴리시 자기 증류 SFT(RFT/STaR/«선생님 성공 SFT»)**를 넣는 논문이 많다(SDPO·OPSD·SD-Zero·SCoRe·Kim 2603).
- 우리 비교군(GRPO·RLSD·RLRT·SDPO식·끈만·리셋만·ours·TT-reset·maj@k)은 RLRT 논문보다 넓다. 빠진 것 중 **반드시 넣을 것은 셋**이다.
  (i) **오프라인 리셋 증류 SFT**: 같은 «눈 가린 나» 출력을 RFT/STaR식으로 SFT한다. «온폴리시가 왜 필요한가»에 답한다.
  (ii) **계산량·토큰 회계표와 GRPO 연장 짝**: 이미 계획한 GRPO-100이 여기에 해당한다.
  (iii) **본 시험(avg@16 표본)에서 토큰을 맞춘 maj@k**: 추가 생성이 없어 공짜다.
  넣으면 좋은 것은 SRPO식 «틀린 행만 풀이 선생님» 팔, 정보를 뒤섞은 쪽지 대조(Rethinking-PI 식), 두 번째 모델이다.
  OPSD 정식판과 더 강한 선생님 증류는 생략한다. DAPO에 참조 풀이가 없고, 우리 주장도 «외부 선생님 없음»이기 때문이다.
- 행동 분석은 **엔트로피 곡선 + 길이 + 반성 단어 수(어휘 지표) + LLM 분류(Four Habits)** 조합이 일반적이다.
  Four Habits 원 프로토콜은 이렇다. GPT-4o-mini, 온도 0, 출력 512토큰, **궤적 하나에 행동별 질문 4개**를 따로 던져 «서로 다른 출현 횟수»를 세고, 궤적 평균을 낸다. 신뢰도는 100궤적 ICC로 확인했다.

---

## 1. 논문 × (비교군 · 절제 · 평가 · 행동 분석)

### 1a. 직계: RLVR + 셀프 디스틸 (보상 있음)

| 논문 (arXiv / 발표처) | 모델 · 데이터 | 비교군 | 계산량 맞춤 | 절제 | 평가 | 행동 분석 |
|---|---|---|---|---|---|---|
| **RLRT** Kim·Jeon·Li·Yang (2605.10781, 프리프린트) | Qwen3-4B/8B-Base, 4B-Instruct, Qwen3-8B(생각) · DAPO-Math-17k | base · GRPO(DAPO) · SDPO · SRPO · RLSD · (탐색 비교: 엔트로피 보너스·DIVER, pass@k만) · **4B-Instruct 표에는 SRPO 행 없음** | 같은 레시피(배치 256·롤아웃 8·20,480·LR 1e-6, SDPO 1e-5·SRPO 5e-6). **총 스텝 수·씨앗·CI 미보고** | 보상 게이트(RLRT-all) · ε_w 탐색 {.2,.5,1} · λ 감쇠(instruct·thinking 30스텝; 탐색 결과는 미보고) · 선생님 문맥은 절제 안 함 | AIME24/25/26 · HMMT26 · AMC23 · MATH500, **avg@16 / pass@16**, pass@k 곡선 k≤256, 온도 .7·top-p .8·top-k 20·38,912 · OOD 없음 | 엔트로피·길이 학습 곡선 · 탐색/수확 표지어 log-odds(«wait/another/consider» 대 «conclude/correct/final») · **«Wait, let me reconsider.» 끼워 넣기 인과 개입**(max-D̄ / 무작위 / min-D̄ 자리, 어려운 묶음 flip→R · 쉬운 묶음 flip→W) · JS·top-k 겹침 분포 이동 |
| **SDPO** Hübotter 외 (2601.20802, 프리프린트) | Qwen3-8B(+0.6–8B 규모), Olmo3-7B-Instr · SciKnowEval, ToolAlpaca, LCBv6 | 개선 GRPO(하이퍼 탐색) · 온폴리시 GRPO · **선생님 성공 샘플 SFT(오프폴리시 자기 증류)** · base · 시험 때 best-of-k·다중 턴 | **벽시계 1h/5h**, LCB는 생성 수, 5씨앗(시험 때) | 선생님 문맥(피드백만·자기 풀이만·둘 다) · 선생님 정규화(비정규·얼림·EMA·신뢰영역) · 크레딧 단위(logit/토큰/시퀀스) · SDPO+GRPO 혼합 λ .9 · 규모 | avg@16 · discovery@k · **망각: IFEval·ArenaHard-v2·MMLU-Pro** | 길이(3× 짧아짐) · «Hmm/Wait» 일화 · 선생님 문맥별 엔트로피 · 선생님 정확도 곡선 |
| **RLSD** Yang 외 (2604.03128, 프리프린트) | Qwen3-VL-8B-Instr · MMFineReason-123K | base · GRPO · OPSD · SDPO · GRPO+OPSD | 같은 배치·롤아웃(LR만 다름), 명시적 맞춤 없음 | 예비: OPSD / 선생님 top-1 / 학생 top-1(누출) · λ 50스텝 감쇠 고정 | MMMU 등 5개, 정확도 · avg@k·OOD 없음 | 엔트로피(GRPO 붕괴 대 RLSD 유지) · clip 비율 · 토큰 크레딧 열지도 |
| **SRPO** G. Li 외 (2604.02288, 프리프린트) | Qwen3-4B/8B(생각 끔) · SDPO 과제 5종 | 강화 GRPO · SDPO (OPSD·DAPO 미비교) | **벽시계 1/5/10h**, 배치·롤아웃 같음 | 이점 섞기 대 표본 라우팅 · 동적 가중 · **SDPO를 틀린 표본만 / 맞은 표본만**(맞은 표본에 쓰면 나빠짐) | avg@16(온도 .6) · OOD·pass@k 없음 | 길이 · 스텝 시간 · 라우팅 비율 · 선생님 엔트로피 |
| **AntiSD** Shen 외 (2605.11609, 프리프린트) | Qwen3-8B, **4B-Instruct-2507**, Olmo3-7B-Instr/Think, 30B-A3B · DAPO-Math-17k | GRPO · 기본 SD(+u_t) · 역KL 상승 | **같은 200스텝**, 도달 스텝 비(2–10×) | 부호 · 발산(JSD/역KL) · 게이트 임계 · 가산 대 곱 · 선생님 없음 · 게이트 신호 · GRPO 체크포인트에서 이어서 | AIME24/25/26 · HMMT25 avg@32 · Minerva · HMMT pass@k · 코드(HumanEval+·MBPP+)=전이 | PMI로 «Wait/Maybe» 숙고 토큰 분석 · 선생님·학생 엔트로피 · 길이 |
| **CEPO** Heakl 외 (2605.19436, 프리프린트) | Qwen3-VL-2B/4B, LoRA · Geo3k | GRPO · OPSD · SDPO · RLSD | **같은 50스텝·배치·G**, 벽시계 보고 | 선생님 출처(현재 정책/동기화/ref) · 피드백 출처 5종(앞부분만·뒷부분만 대조 포함) · λ 일정(상수 대 감쇠) · ε_w | 5개 벤치 정확도 ± 표준오차 | 대조 델타 비율 · 토큰 가중 열지도 |
| **HSD** Y. Li 외 (2606.15576, 프리프린트) | Qwen3-8B/32B · 수학 94k + 코드 19k | GRPO · Dr.GRPO · GSPO · **DAPO** · OPSD · SDPO · RLSD · GRPO+OPSD · (부록) SDFT | **같은 1,500스텝·G 8·배치 128**, 스텝당 벽시계 | 끝점(OPSD) 대 경로 문맥 · 갈림 자리 τ 주변 크레딧 · 혼합 계수 · 일정 · G·β·온도·동료 선택 규칙 | MATH-500·AIME24/25·LCB·HE+, pass@1(16표본 평균) | 없음 |
| **E²-OPSD** Liu 외 (2610.05048, 프리프린트) | Qwen3-0.6B/1.7B(생각), **4B-Instruct-2507**, Olmo3-7B-Instr · DeepMath | base · OPSD · SDPO · GRPO · RLSD · OPSD+EOPD | **같은 360스텝·롤아웃** | 구성 요소 누적 · 고정 KL 방향 대 라우팅 · 선생님 문맥 탐침 | AMC23·AIME24/25·HMMT25 **mean@16 × 3씨앗**, pass@2/4 · **OOD: ZebraLogic·HumanEval** | 학생·선생님 엔트로피(«엔트로피 과잉») · KL 방향 비율 · 길이 |

### 1b. 특권 정보 · 순수 증류 (보상 없이, 또는 진단 논문)

| 논문 | 모델 · 데이터 | 비교군 | 계산량 | 절제 | 평가 | 행동 분석 |
|---|---|---|---|---|---|---|
| **OPSD / Self-Distilled Reasoner** Zhao 외 (2601.18734, 프리프린트) | Qwen3-1.7B/4B/8B, LoRA · OpenThoughts 수학 30K | base · **SFT(같은 데이터·같은 표본 수)** · GRPO | 토큰 효율 주장(1롤아웃·1,024 대 8×16k), 체크포인트는 각자 최고 | 발산(정KL/역KL/JSD) · 생성 방식 짝 · 토큰별 clip · 생성 길이 · 전체 어휘 대 표본 토큰 | AIME24/25 · HMMT25 **avg@12**(온도 1·38,912) | 스타일 토큰 KL 분해 |
| **PAST** Feng 외 (2608.08726, 프리프린트) | Qwen3-1.7B(생각), 4B(씨앗 1) · 참조 풀이 있는 과제 | base · SFT · GRPO · OPSD · SD-Zero · H2SD | **같은 업데이트 100**, 표 10에 선생님 표본·검증 호출·토큰·GPU-h·메모리 | 궤적 제거 · **문제 간 뒤섞기** · 2×2 요인 · 맞은/틀린/합동 · KL β · 목적 5종 · 적응 대 고정 표본 | AIME24/25 · HMMT25 avg@12, **문제·씨앗 층화 짝 부트스트랩 CI** | 없음 |
| **Rethinking PI** Shrestha·Tessier (2608.18271, 프리프린트) | Qwen3-1.7B/4B/8B, LoRA · NuminaMath, MegaScience | 자기 조건들만(base · 맞는 참조 / **다른 문제 참조** / 참조 없음 · 생각 켬/끔 · 참조 5종) | 미보고, 조건당 1회 | 선생님 모드 · 참조 유무·종류 · 규모 · 데이터 | MATH-500 avg@4 · AIME·HMMT·**GPQA-D** avg@12 · **MMLU-Pro**·UGPhysics avg@1 · 짝 부트스트랩 | 엔트로피(생각 끈 선생님이면 급락) · 생각 방향 정렬 · **결론: 맞는 참조의 이득이 일관되지 않고, 다른 문제 참조도 이득을 상당 부분 재현** |
| **Kaur 외** (2607.05184, 프리프린트, Arora 그룹) | Qwen3-1.7B/4B/8B, 4B-Think/Instr-2507, OLMo-7B · OpenThoughts 15k, Countdown | base · 바닐라 OPD(8B 선생님) · OPD+정답 시범 · OPSD · OPSD(생각 끔) · 희소/조밀 문맥 | 미보고(같은 4,096 학습 상한·1에폭) | 문맥 밀도 · 생각 켬/끔 · 토큰 마스크 | AIME24/25 · HMMT25 **avg@16 · pass@16**, 예산 4k–38,912 | **어휘 표지어(wait·hmm·maybe) /1k 토큰** · fork/lock 비율 · 길이 · 결론: instruct는 OPSD로 오르고 thinking은 최대 −17% |
| **Kim 외** (2603.24472, 프리프린트; RLRT와 같은 그룹) | R1-Distill-Qwen-7B, Qwen3-8B, 1.7B, Olmo3-7B · DAPO-Math-17k | GRPO · SDPO · **SFT(유도 없음/풀이 유도)** · (부록) OPSD | — | 문맥(풀이 / think 뺀 풀이) · 고정 대 이동 선생님 · **GRPO에서 인식 토큰 벌점(인과)** · 과제 범위 | AIME24/25 · AMC23 · MATH500(OOD) · pass@16(부록) | **인식 토큰 10종 수(wait·hmm·perhaps·maybe…)** · **GPT-5.4 불확실성 판정** · 길이 · 결론: 풍부한 문맥이 인식 표현을 눌러 AIME24 최대 −40% |
| **Baumann 외** (2609.21561, 프리프린트, Krause 그룹) | Qwen3-4B(혼합), **4B-Instruct-2507** · DeepMath 10k, Group Anagrams | GRPO(Dr.GRPO+DAPO clip) · 끌어당김 SD · 밀어냄 SD | 미보고 | 밀어냄 선생님 문맥(맞은 풀이 대 실패 롤아웃) · **토큰 크레딧 창 1/10/100/1,000/전체 + 순열 대조** · 양의 출처 | AIME24/25/26 · AMC23 · HMMT25 · MATH-500(128) mean@4 | 길이 · **잘림 비율** · think 닫힘 |
| **ROSD** Zhao 외 (2605.28014, 프리프린트) | Qwen3-4B/8B, Llama-3.2-3B · SDPO 과제 | 강화 GRPO · SDPO(공식) · (OPSD·SDFT는 정답 풀이가 필요해 제외) | 수렴 기준 에폭, 스텝 시간(부록) | 반성 없이(원 참조) · **국소 증류 없이(응답 전체)** | mean@16 · Self-BLEU·Distinct-2 · **OOD: AIME24·교차 도메인** | 길이 · 다양성 · Claude 판정 «첫 오류 자리» |
| **Feedback Alignment** Kara·Ersoy (2606.11173, **ICML 2026 RLxF 워크숍**) | Qwen3-1.7B (+QwQ-32B 비평가) · OpenMathReasoning 282문제 | GRPO · RefSol(참조 풀이 선생님) · StepAlignFB | GRPO G=8 대 SD G=1, 같은 에폭·LR | 정량 절제 없음, 피드백 구성 정성 비교(전체 반복·정답 단계 생략·첫 오류까지 반복) | 30문제 Avg/Maj/Pass@12 | 토큰 이점 그림 · **«전체를 그대로 반복하면 교정 신호가 사라진다(유도 헤드 복사)»** |
| **SD-Zero** He 외 (2604.12002, 프리프린트) | **Qwen3-4B-Instruct**, Olmo3-7B-Instr · OpenR1-Math 15K, Codeforces | **SFT(R1 시범)** · **RFT** · GRPO(=DAPO) · SDFT | 같은 문제 집합, 생성 수 ~60K(부록) | 손실 항 · 1단계 생략 · 데이터 분할 | AIME24/25 · HMMT25 · MATH · AMOBench avg@8, pass@8 | **자기 수정 키워드 빈도**·길이의 단계별 변화 · 생성 후 수정 평가 |
| **SDFT** Shenfeld 외 (2601.19897, 프리프린트) | 연속 학습 과제 | SFT | — | — | 새 과제 + 이전 능력(망각) | — |

### 1c. 고전 기준점 (OPD · 자기 개선 · 자기 교정 · 행동 프로토콜)

| 논문 | 핵심 설정 | 비교군 | 우리에게 주는 것 |
|---|---|---|---|
| **GKD** Agarwal 외 (2306.13649, **ICLR 2024**) | T5 학생 / T5-XL 선생님, XSum·WMT·GSM8K·FLAN | SeqKD · 지도 KD · ImitKD · f-distill | 발산(정·역KL·JSD β) × 학생 데이터 비율 λ 절제 · RL 보상과 섞기(α) — «증류 + RL 혼합» 원형 |
| **On-Policy Distillation** K. Lu / Thinking Machines (블로그, 2025-10-27) | Qwen3-8B-Base ← Qwen3-8B/32B, OpenThoughts-3 | 오프폴리시 SFT(외삽) · RL(Qwen3 보고) · 오프폴리시 증류 | **FLOPs·GPU-h 회계표**(9–30×) · 토큰별 역KL · 개인화 망각 회복(IF-eval 85→45→83) |
| **STaR** Zelikman 외 (2203.14465; NeurIPS 2022로 알려짐, arXiv 페이지에는 표기 없음) | 정답 낸 근거로 반복 SFT + 정답을 주고 다시 쓰게 하는 rationalization | 답 직접 예측 SFT · few-shot · 30× 큰 모델 | «자기 성공만 SFT»의 원형. 우리의 «리셋 출력 SFT» 기준선 근거 |
| **RFT** Z. Yuan 외 (2308.01825, 프리프린트) | 기각 표집 SFT, GSM8K | SFT | 서로 다른 정답 경로가 많을수록 효과가 크다 — RFT 기준선의 표준 인용 |
| **SCoRe** Kumar 외 (2409.12917; ICLR 2025로 알려짐, arXiv 페이지에는 표기 없음) | Gemini 1.0 Pro / 1.5 Flash, MATH·HumanEval, 2턴 자기 교정 | base · Self-Refine(프롬프트) · **STaR** · **Pair-SFT** · 2턴째만 최적화하는 RL | **지표: Acc@t1 · Acc@t2 · Δ(t1,t2) · Δi→c · Δc→i**(= 우리 사분면) · **계산량 맞춘 자기 일관성**(K개 병렬 + 교정 1회 > 2K개 병렬) · 절제(1단계 없이·보상 성형 없이·STaR로 대체) · 결론: 오프라인 교정 SFT는 분포 불일치·행동 붕괴로 실패 |
| **Four Habits** Gandhi 외 (2503.01307, 프리프린트) | Qwen-2.5-3B / Llama-3.2-3B, Countdown, PPO 250스텝 | 프라이밍 조건 5개 + 빈 CoT·자리표시 CoT 대조 + **틀린 풀이 프라이밍** | 행동 분류 프로토콜(§3c) |

---

## 2. 이 장르의 «표준 비교군»과 우리에게 빠진 것

### 2a. 빈도 (위 18편 중 해당 비교군을 실제로 돌린 논문 수, 대략)

| 비교군 | 빈도 | 대표 |
|---|---|---|
| base | 거의 전부 | — |
| GRPO / DAPO 레시피 | 16/18 | RLRT·SDPO·SRPO·HSD(DAPO 따로)·E²-OPSD |
| SDPO | 8 | RLRT·RLSD·SRPO·CEPO·HSD·E²·ROSD·Kim |
| OPSD | 7 | RLSD·CEPO·HSD·E²·PAST·Kaur·Kim(부록) |
| RLSD | 4 | RLRT·CEPO·HSD·E² |
| **SFT / RFT / 선생님 성공 SFT (오프폴리시)** | 6 | OPSD·PAST·SDPO·SD-Zero·Kim·SCoRe(STaR·Pair-SFT) |
| 더 강한 선생님 증류(OPD) | 2 | Kaur(8B 선생님), TM 블로그 |
| 시험 때 방법(best-of-k·SC·재프롬프트) | 2 | SDPO(best-of-k·다중 턴), SCoRe(SC 맞춤) |
| 같은 스텝/벽시계 맞춤 명시 | 7 | SDPO·SRPO(벽시계) · HSD·CEPO·E²·AntiSD·PAST(스텝; PAST는 GPU-h 표) |
| 씨앗 ≥ 2 또는 CI | 4 | PAST(CI)·E²(3씨앗)·Rethinking-PI(CI)·CEPO(표준오차) — **RLRT는 없음** |
| OOD / 망각 | 6 | SDPO(IFEval·ArenaHard·MMLU-Pro)·Rethinking-PI(GPQA·MMLU-Pro)·E²(ZebraLogic·HE)·AntiSD(코드)·ROSD·Kim(MATH500) |
| 모델 ≥ 2 | 15/18 | 단일 모델은 RLSD·Feedback Alignment 정도 |

### 2b. 우리 현황 대비 빠진 것 — 판단

계산 단위 «1 팔»은 본 프로토콜 학습 1회(60스텝 × 64문제 × 8롤아웃 × 20,480)와 avg@16 평가(수학 4종 + MMLU-Pro 650)를 합친 것이다.
아직 SMOKE 시간을 재지 않았다. hc2(32문제·스텝 ~30분/GPU)에서 늘려 잡으면 **1 팔 ≈ GPU 1장 2–3일**로 추정한다(추정치, 실측 아님).

| 빠진 비교군 | 우리 주장에 중요한가 | 2×H100 비용 | 권고 |
|---|---|---|---|
| **오프라인 리셋 증류 SFT** (= STaR/RFT/Pair-SFT식: DAPO 앞부분에서 «눈 가린 나»가 다시 풀어 맞힌 출력을 첫 답 뒤에 이어 붙여 SFT) | **높음.** «셀프 디스틸»이라 부르면 심사자가 먼저 «그냥 자기 성공을 SFT하면 안 되나?»를 묻는다. SDPO·OPSD·SD-Zero·SCoRe가 모두 이 대조를 둔다. SCoRe는 이 방식이 행동 붕괴로 실패한다고 보였다. 같은 결과가 나오면 «온폴리시 + 행 안 크레딧»의 필요성이 선다 | 생성 ~9,600 앞부분 × k4 ≈ 0.3–0.5 팔, SFT 수 시간, 평가 1회. **≈ 0.6 팔** | **must** |
| (같은 축) 일반 RFT(문제에서 처음부터 k개 표집, 맞은 것만 SFT) | 중간. «데이터 효과» 대조. 위 리셋 SFT가 더 직접적이다 | ≈ 0.6 팔 | nice (리셋 SFT를 먼저) |
| **계산량 회계 + GRPO 연장** (PAST 표 10·TM 블로그식 GPU-h·생성 토큰·선생님 forward, GRPO-100 대 ours-60) | **높음.** ours는 선생님 forward(+5–15분/스텝)와 고정 앞부분 생성 비용을 더 쓴다. «계산을 더 써서 이겼다»는 반론을 막아야 한다 | 회계는 공짜. GRPO 60→100 이어 학습은 이미 계획(수정 72) | **must**(표 1개 + GRPO-100 행) |
| **토큰을 맞춘 자기 일관성 maj@k (본 시험)** | 높음. hdev에서는 이미 했다(maj@3·4가 더 정확하지만 2–3× 토큰). 본 시험 표에도 «토큰 대 정확도» 점이 있어야 한다 | **공짜**: base·GRPO의 avg@16 16표본에서 maj@k와 평균 토큰을 계산 | **must** |
| 시험 때 리셋(TT-reset) 본 시험판 | 중간. hdev에서는 구간에 0이 포함돼 «내면화»로 주장을 낮췄다. 본 시험에서도 보이면 주장이 일관된다 | 수학 4종 base 재풀이 1회 ≈ 0.2 팔 | nice |
| **앞부분 데이터 맞춘 대조** (끈만 팔이 같은 고정 앞부분으로 학습하는지 = 초안 TODO) | **높음.** ours는 고정 앞부분, GRPO는 처음부터 푼다. «데이터(앞부분) 효과 대 선생님 효과»가 섞여 있다. 예비 프로토콜의 «균등»이 이 대조였다 | 0(끈만 팔 정의만 확정). 본 프로토콜에 균등을 따로 두면 +1 팔 | **must**: 끈만 팔을 앞부분 위에서 학습시켜 데이터 맞춘 짝으로 쓴다 |
| SRPO / «틀린 행만 풀이 선생님» | 중간–높음. SRPO의 핵심(틀린 표본만 SDPO)이 ours의 «틀린 첫 답 행만»과 겹친다. SDPO식 팔을 틀린 행으로 제한하면 ours와 «선생님 문맥(눈 가림 대 정답 풀이)·자리(첫 답 뒤 대 전체)»만 다른 짝이 된다. RLRT 4B-Instruct 표에도 SRPO가 없어 비교 의무는 약하다 | sdpo 경로에 행 마스크만 추가 ≈ 1 팔. 예비 프로토콜이면 ≈ 0.3 팔 | **nice**: 예비 프로토콜 절제로 |
| 정보를 뒤섞은 쪽지 대조 («전에 X»의 X를 다른 문제 답·무작위 수로) | 높음(절제로서). Rethinking-PI와 PAST가 «다른 문제 참조도 이득을 재현한다»를 보였다. 위약(자리 섞기)만으로는 «쪽지 내용이 중요하다»를 못 보인다 | 예비 프로토콜 1 팔 ≈ 0.3 팔 | **nice → 4.3에 강력 권고** |
| OPSD 정식판 (참조 풀이 선생님, 보상 없음, 전체 어휘 KL) | 낮음. DAPO-Math-17k에는 참조 풀이가 없다. RLSD(정답 조건)·SDPO식(형제 풀이)·RLRT가 «특권 풀이 선생님» 축을 이미 덮는다. Kaur가 «instruct 모델에선 OPSD가 오른다»고 보였으니 언급은 필요하다 | 풀이 확보 + 전체 어휘 KL 구현 = 큼 | **skip**(관련 연구에서 설명) |
| 더 강한 선생님 증류(Qwen3-8B / 4B-Thinking OPD) 상한 참조 | 낮음. 우리 주장은 «외부 선생님 없이»다. 상한선이 있으면 해석에 도움은 된다 | 20,480 길이 선생님 채점·메모리 ≈ 1.5 팔 이상 | **skip**(한계 절) |
| 두 번째 모델 | 중간. 18편 중 15편이 모델 ≥ 2. 두 번째 모델 데이터 관문은 이미 미달(수정 69) | 팔 2개(GRPO·ours) ≈ 2 팔 + 데이터 파이프 | nice(본 실험 뒤 여력이 있으면, 아니면 한계로 명시) |
| 씨앗 ≥ 3 + 짝 부트스트랩 | 우리 강점. RLRT에는 없다 | 본 실험 seed 3은 이미 계획 | 유지 |

**정정 하나.** 초안 TODO에 «RLRT reports both [SDPO/SRPO] below GRPO on Qwen3-4B-Instruct»라고 적혀 있다. 실제 RLRT 표 3의 4B-Instruct에는 **SDPO와 RLSD만 있고 SRPO 행은 없다.**
그 표의 avg@16은 AIME24 기준 GRPO 69.4, SDPO 53.1, RLSD 61.5, RLRT 70.4다.
또 RLRT는 **총 스텝 수를 보고하지 않는다**(스텝 100 체크포인트 언급만). 수정 72의 «RLRT ≈ 100 × 256» 비율은 추정이라고 밝혀 두자.

---

## 3. 실험 절 구조 · 행동 그림 · 우리 4.1–4.3 권고

### 3a. 구조 패턴

- **성능 → (학습 동역학/행동) → 절제**가 다수다. RLSD(주 결과 → 학습 동역학 → 사례), ROSD(주 결과 → 동역학 → 다양성 → 절제), E²-OPSD(주 결과 + 엔트로피 → 절제), AntiSD(주 결과 → 동역학 → 절제 → GRPO 포화 너머)가 이 순서다.
- RLRT는 **성능 → 인과 개입(Wait 끼워 넣기) → 분포 이동 → 다른 탐색법 비교 → 절제** 순서다. 우리 «성능 → 메타인지 → 절제»와 거의 같아 그대로 따라도 된다.
- 진단 논문(Kaur·Kim·Rethinking-PI)은 «주장 하나 = 소절 하나» 제목을 쓴다(예: «3.2 Degradation is specific to teacher-side privileged context»). 초안의 질문형 첫 문장과 잘 맞는다.

### 3b. 행동 그림에서 쓰는 것

| 그림 종류 | 쓰는 논문 | 우리 대응 |
|---|---|---|
| 학습 중 actor 엔트로피 곡선 | RLSD·RLRT·AntiSD·E²·Rethinking-PI | 이미 있음(균등 좁아짐 대 리셋+끈 유지·증가). 본 실험 팔 전체로 확장 |
| 응답 길이 · 잘림 비율 곡선 | 거의 전부, Baumann은 잘림까지 | 잘림 −14pp가 핵심 결과 중 하나 → 그림으로 |
| 어휘 표지어 수(wait·hmm·maybe /1k 토큰) | Kim(10종)·Kaur·SD-Zero·RLRT(log-odds) | **첫 박스 뒤 구간만** 따로 세면 우리만의 그림이 된다. 비교 대상: SDPO식은 이 표지어를 누를 것이라는 Kim·RLRT의 예측 |
| LLM 분류 행동 | Four Habits(GPT-4o-mini)·Kim(GPT-5.4 불확실성)·ROSD(Claude 첫 오류) | §3c |
| 인과 개입(Wait 끼워 넣기, 자리별) | RLRT §6.2 | 수정 70의 «Wait 끼워 넣기 개입»을 RLRT와 같은 형식으로: 첫 박스 직후 대 무작위 자리 대 base/GRPO/ours, 어려운 묶음 flip→R · 쉬운 묶음 flip→W |
| 사분면 Δi→c / Δc→i | SCoRe | 우리 R→R / R→W / W→R / W→W(첫 박스 대 최종). SCoRe 지표 이름을 병기하면 독자가 바로 알아본다 |

### 3c. Four Habits 프로토콜 (재현용 세부)

- **네 행동**(정의는 논문 문구):
  - backtracking: 오류를 알아채고 접근을 명시적으로 바꿈, 예: «This approach won't work because…»
  - verification: 중간 결과를 체계적으로 확인, 예: «This sequence results in 1, which is not equal to 22»
  - subgoal setting: 문제를 하위 단계로 나눔
  - backward chaining: 목표에서 거꾸로 추론
- **분류기:** 행동 분석은 **GPT-4o-mini**(비용·능력 균형), 사전학습 데이터 거르기는 **Qwen-2.5-32B**(vLLM)를 썼다. **온도 0, 출력 512토큰.**
- **프롬프트 구조:** 궤적 하나에 **행동별 질문 4개**를 따로 던진다. 각 질문에는 해당 행동의 예시가 들어 있고, 서로 다른 출현 횟수를 세어 보고하게 한다. 지표는 **궤적당 평균 횟수**다.
  원문 프롬프트는 논문 부록에 없고 저자 GitHub에 있다고만 적혀 있다(본 조사에서 원문 미확인). 원 프롬프트는 Countdown 전용이므로 수학용으로 고쳐 써야 한다.
- **단위:** 궤적 전체. 생성은 온도 1.0, 최대 1,024토큰(Countdown)이었다. 그림 1은 학습 스텝별 정확도·길이·네 행동 빈도이고, 스텝당 표본 수는 미보고다.
- **신뢰도(부록 F):** 100궤적 ICC3. Claude 대비 검증 .70 · 되돌림 .74 · 하위목표 .88 · 역추론 .78. 사람 두 명 대비 .65–.70 / .67 / .70–.77 / .40–.55(역추론이 가장 낮음).
- **우리용 권고:**
  1. **세는 구간을 둘로 나눈다.** (a) 첫 박스 전, (b) 첫 박스 뒤. 우리 주장은 (b)에서 verification·backtracking이 늘고 (a)는 그대로라는 것이다. 응답이 수만 토큰이라 길이로 나눈 값(/1k 토큰)도 함께 보고한다(Kaur식).
  2. 행동 넷 가운데 우리에게 의미 있는 것은 verification·backtracking 둘이다. subgoal·backward는 대조 행동(변하지 않아야 함)으로 보고한다.
  3. 분류기: 로컬 open 모델(Qwen2.5-32B-Instruct 또는 Qwen3-32B, GPU 1장 bf16)을 주로 쓰고, 100궤적에 대해 사람 또는 다른 LLM과 ICC를 보고한다. 원 논문의 GPT-4o-mini를 그대로 쓰려면 TRAPI 사용 승인이 필요하다.
  4. 32k 응답은 분류기 문맥을 넘을 수 있으므로 (b) 구간 위주로 자르고, 자른 사실을 적는다.
  5. 표본: 팔마다 hdev 288 × 2표본 + OOD(MMLU-Pro) 200개 정도면 충분하다(CI는 문제 부트스트랩).

### 3d. 4.1 성능 권고

- **표 1(본 프로토콜):** AIME24 · AIME25 · HMMT25 · MATH500(+ 확보되면 AIME26·HMMT26·AMC23 — RLRT 벤치 구성과 맞춤)을 **avg@16 / pass@16**으로 낸다.
  행은 base · GRPO · 끈만 · 리셋만 · RLSD · RLRT · SDPO식 · ours · **리셋 SFT(새로 추가)**다. 아래에 계산량 열(GPU-h, 생성 토큰/스텝)을 단다.
- **표 2:** hdev · MATH L5 · MMLU-Pro(OOD). SDPO·Rethinking-PI가 MMLU-Pro를 «망각/OOD»로 썼으니 «손해 없음» 판정선과 잘 맞는다.
- **사분면 표(SCoRe식):** 벤치마다 첫 박스 acc · 최종 acc · Δ · W→R · R→W를 둔다. 이득이 첫 박스 뒤에서 나온다는 주장의 직접 증거다.
- **pass@k 곡선**(AIME24/25, k ≤ 16): RLRT와 Kaur가 pass@k를 본다. 셀프 디스틸은 pass@k를 평탄하게 만든다는 보고가 있으니(Nicolicioiu 2606.26091, Kaur), ours가 pass@16을 잃지 않는지 보여야 한다.
- **토큰 대 정확도 그림:** ours · GRPO · base maj@k · TT-reset을 한 평면에 놓는다. 본 시험 판도 공짜로 만들 수 있다.
- **예산 곡선**은 유지한다. 8k에서 진다는 사실을 숨기지 않은 점은 Kaur의 «긴 예산에서만 나타나는 차이» 논의와 맞물린다.

### 3e. 4.2 행동 권고

- **(1) 학습 동역학:** 엔트로피·길이·잘림 곡선을 팔 전체로 그린다(RLSD·RLRT·E² 형식).
- **(2) 첫 박스 뒤 행동 사슬:** 안 엶 · 재확인 · 새로 풀어 고침 · 맞은 답 유지 · 반복을 hdev와 MMLU-Pro에서 센다. 우리 고유 지표다.
- **(3) 어휘 표지어:** Kim의 인식 토큰 10종과 RLRT 탐색 표지어를 첫 박스 전과 뒤로 나눠 /1k 토큰으로 센다. 예측은 «SDPO식·RLSD는 줄고 ours는 첫 박스 뒤에서만 는다». 문헌 예측과 정면으로 비교할 수 있는 그림이다.
- **(4) Four Habits LLM 분류:** §3c 방식으로 verification·backtracking을 첫 박스 뒤 구간에서 센다.
- **(5) 인과 개입:** RLRT §6.2처럼 «Wait, let me reconsider.»를 끼워 넣는다. 자리 조건은 첫 박스 직후 / 무작위 / 그 밖이고, 대상 모델은 base·GRPO·ours다. 지표는 어려운 묶음 flip→R, 쉬운 묶음 flip→W다. 오라클 쪽지(«틀렸다»: 고침 .026 → .216) 결과를 같은 그림에 놓으면 «병목은 알아채기»를 보일 수 있다.
- **(6) 내부 신호:** 첫 박스 답 엔트로피 AUC(base .586 → ours .649)는 보조 그림으로 둔다.

### 3f. 4.3 절제 권고 (예비 프로토콜로 돌려도 됨 — PAST·CEPO도 작은 설정에서 절제)

| 질문 | 팔 | 문헌 근거 |
|---|---|---|
| 왜 두 교사인가 | 끈만 · 리셋만 · ours (+ 위약: 자리 섞기) | SRPO 혼합 대 라우팅, GKD α |
| 리셋 선생님 문맥 형태 | **fact(«전에 X»)** · mask(«답 가림, 처음부터») · other(«다른 학생이 X») · **뒤섞은 X(다른 문제 답)** · 정답 풀이(= SDPO식) · 문맥 없음 | SDPO 표 6, Rethinking-PI(다른 문제 참조), PAST(문제 간 뒤섞기), CEPO(피드백 출처 5종), Feedback Alignment(«그대로 반복하면 교정 신호가 사라진다» — 우리 «풀이 가림»의 근거) |
| 어느 행·어느 말에 | 틀린 첫 답 행만(ours) 대 모든 행(dskla: 파괴 가드 중단) · 첫 박스 뒤만 대 응답 전체 | SRPO(맞은 표본에 SDPO를 쓰면 나빠짐) — 우리 dskla 결과와 같은 방향, ROSD(국소 증류), Baumann(크레딧 창 + 순열) |
| 세기·일정 | β ∈ {.25, .5, 1} 중 한 점 추가 · 끈 λ 한 점 | RLRT ε_w, PAST β_KL, RLSD/CEPO λ 감쇠 |
| 잘림 처리 | 잘림 mask 켬/끔 · 잘림률 보고 | Baumann(잘림 비율), Kaur(긴 예산) |
| 고정 앞부분 대 온폴리시 | 온폴리시 리셋(탈락, 고침 −.090) | Kim·CEPO(고정 대 이동 선생님) |
| 스텝 | 20 대 40(예비), 60 대 100(본) | AntiSD(도달 스텝), OPSD(100 대 500) |
| 실패한 선생님 9종 | 부록 표 | 닫힌 축 기록 |

**숨은 끈 명시(수정 72b).** 리셋·SDPO식·RLRT 추정량은 모두 [log ref − log π_old] 항을 가진다. 따라서 «끈만 대 리셋만» 해석에서 «리셋만»에도 약한 역KL 끈이 들어 있다는 점을 4.3 본문에 한 줄 적어야 한다.

---

## 4. 확인한 문헌 (제목 — 제1저자, arXiv, 발표처)

- Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR (RLRT) — J. Kim, 2605.10781, 프리프린트
- Reinforcement Learning via Self-Distillation (SDPO) — J. Hübotter, 2601.20802, 프리프린트
- Self-Distilled RLVR (RLSD) — C. Yang, 2604.03128, 프리프린트
- Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing (SRPO) — G. Li, 2604.02288, 프리프린트
- Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs (OPSD) — S. Zhao, 2601.18734, 프리프린트
- On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) — R. Agarwal, 2306.13649, ICLR 2024
- On-Policy Distillation — K. Lu, Thinking Machines Connectionism 블로그, 2025-10-27
- PAST: Privileged Adaptation from Complete Student Trajectories for OPSD — Y. Feng, 2608.08726, 프리프린트
- Rethinking Privileged Information in On-Policy Self-Distillation — S. Shrestha, 2608.18271, 프리프린트
- Localizing Credit at the Divergence: Path-Conditioned Self-Distillation (HSD) — Y. Li, 2606.15576, 프리프린트
- CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization — A. Heakl, 2605.19436, 프리프린트
- Beyond Imitation: Reflective On-Policy Self-Distillation (ROSD) — Z. Zhao, 2605.28014, 프리프린트
- The Role of Feedback Alignment in Self-Distillation — S. Kara, 2606.11173, ICML 2026 RLxF 워크숍
- STaR: Bootstrapping Reasoning With Reasoning — E. Zelikman, 2203.14465
- Scaling Relationship on Learning Mathematical Reasoning with LLMs (RFT) — Z. Yuan, 2308.01825, 프리프린트
- Training Language Models to Self-Correct via RL (SCoRe) — A. Kumar, 2409.12917
- Cognitive Behaviors that Enable Self-Improving Reasoners (Four Habits) — K. Gandhi, 2503.01307, 프리프린트
- Anti-Self-Distillation for Reasoning RL via PMI (AntiSD) — G. Shen, 2605.11609, 프리프린트
- E²-OPSD: Taming Entropy Overshoot in OPSD — Y. Liu, 2610.05048, 프리프린트
- Rethinking On-Policy Self-Distillation for Thinking Models — S. Kaur, 2607.05184, 프리프린트
- Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? — J. Kim, 2603.24472, 프리프린트
- On Repulsive and Attractive Teachers: Separating Correctness from Behavior in Self-Distillation — A. Baumann, 2609.21561, 프리프린트
- Self-Distillation Zero (SD-Zero) — Y. He, 2604.12002, 프리프린트
- Self-Distillation Enables Continual Learning (SDFT) — I. Shenfeld, 2601.19897, 프리프린트

미확인·주의: Four Habits 분류기 원문 프롬프트(저자 GitHub에 있다고만 함) · STaR/SCoRe 학회 표기(arXiv 페이지에 없음) · HSD 본문의 «모든 열 최고» 주장은 자기 표의 HE+ 열과 어긋남 · SDPO 부록 표 9(다른 RLVR 비교군)는 잘려서 읽지 못함.
