# 서베이: 셀프 디스틸레이션에서 «교사가 무엇을 보느냐» (2026-10-08)

질문: 교사가 얼린(또는 갱신되는) 자기 자신이고 **문맥만 다를 때**, 문맥에 무엇을 넣으면 학생이 어떻게 바뀌는가.
범위: 수학·추론 중심. 모든 논문은 arXiv 페이지에서 제목·제1저자·ID를 확인했다(목록은 맨 아래).
주의: 수치는 초록·HTML 본문을 자동 추출해 옮긴 것이다. 설계에 직접 쓰는 숫자(★ 표시)는 PDF 원문 재확인을 권한다.

## 0. 한 줄 결론
- 교사 문맥이 **풍부할수록**(전체 풀이 > 정답만 > 지시문) 학생은 **짧고 확신에 찬** 말투로 바뀐다.
  이득은 분포 안 짧은 예산에서 나고, «Wait/Hmm» 같은 불확실 표지가 줄어 긴 예산·분포 밖에서 손해를 본다(Kim 2026, Kaur 2026).
- 이 손해는 **thinking 모델에서 크고 instruct 모델에서는 거의 없다**. Qwen3-4B-Instruct는 오히려 이득이었다(Kaur 2026 ★).
- 학생 자신의 이전 시도를 교사 문맥에 넣으면 교사가 그 시도에 **닻을 내린다**(엔트로피 감소, SDPO ★). 우리 «눈 가린 나»가 이 방향과 맞는다.
- «멈춰라» 지시는 같은 모델에 약하게만 듣는다. 지시를 **문제 앞**에 두고 **교사를 주기적으로 갱신해 누적**시켜야 압축이 커졌다(CRISP).
- 내부 신호(일치·확신)만으로는 **일관된 오답(self-consistent error)을 못 가린다**(Tan 2025). 체계적 오답에는 형제·검증기 같은 바깥 정보가 필요하다.

## 1. 분류표 — 교사가 보는 것 → 학생에게 생기는 일

| # | 교사 문맥에 추가되는 것 | 대표 논문 | 주효과 | 대표 부작용 |
|---|---|---|---|---|
| 1 | 지시문·시스템 프롬프트·예시 | Askell 2021, Snell 2022, CRISP 2026, SDFT 2026 | 지시 없이도 그 행동을 내면화 | 지시가 약하면 신호도 약함 |
| 2a | 정답 풀이 전체(gold CoT) | OPSD 2026, Kim 2026, Kaur 2026 | 토큰 효율↑, 짧은 예산 정확도↑ | 불확실 표지 억제, 길이 압축, thinking 모델 성능↓ |
| 2b | 정답(최종 답)만 | Kaur 2026, STaR 2022 | 2a보다 약한 이득 | 2a보다 덜 망가짐 |
| 2c | 성공한 형제 롤아웃 | SDPO 2026, Nicolicioiu 2026 | 표본 효율↑, 3배 짧은 응답 | 다양성↓, pass@k 평탄화 |
| 2d | 환경 피드백·오류·보상 | SDPO 2026, SD-Zero 2026, Critique-GRPO 2025 | 실패 행에 촘촘한 신호 | 피드백이 빈약하면 무효 |
| 3 | 고치기 시범·비평 | SCoRe 2024, RISE 2024, S²R 2025 | 2턴 고치기 | 오프라인 SFT는 «안 고침»으로 붕괴 |
| 4 | 멈춤 문구·자르기·길이 벌점 | SBT 2025, DEER 2025, s1, L1, LC-R1, ThinkPrune, ShorterBetter, JET, ARLCP, CaRT | 토큰 30~80%↓ | 정확도 소폭↓, 드문 교정까지 잘림 |
| 5 | 이전 답만(풀이 가림)·새 출발 | SDPO 절제, LEMMA 2025, Chen 2026 | 닻 해제, 재풀이 | 이전 답을 그대로 재도출 |
| 6 | 내부 신호로 어디를 증류할지 | DASD, EGRSD, SEAD, SRPO, E²-OPSD, 80/20 | 갈림길 보존, 안정화 | 일관된 오답은 못 거름 |
| 7 | 성공·실패 대조, 교사 라우팅 | Baumann 2026, RLCD 2023, SRPO, ATESD, VISTA | 행동 이동 상쇄, 길이 안정 | 반발 교사 단독은 길이 폭주 |

## 2. 부류별 상세

### 2.1 지시문·문맥 증류
- **Askell et al. 2021**(2112.00861): HHH 프롬프트를 붙인 모델과 안 붙인 모델 사이 KL을 줄이는 context distillation. 큰 모델일수록 이득, 능력 저하 없음.
- **Snell et al. 2022**(2209.15189): «지시+입력 → 스크래치패드+답»을 «입력 → 답»으로 내면화. 지시·단계 추론·예시 모두 내면화된다. SPIDER에서 직접 경사하강보다 +9%.
- **CRISP / OPSDC, Sang 2026**(2603.05433): 교사 = 같은 모델 + «간결하게 풀어라» 지시(**문제 앞에 붙임**), on-policy 역KL.
  지시만으로 추론 길이가 14~37% 줄었다(비율 .63~.86). 학습 후 MATH-500 길이 −56%(정확도 +3.3), AIME24 −38%(±1점).
  어려운 문제는 교사도 길게 써야 하므로 압축이 자동으로 약해진다(난이도 적응). 엔트로피는 유지됐다(«Wait» 억제 없음).
  교사를 매 스텝 갱신(M=1)하면 붕괴했다(정확도 ~2%). M=40~60 주기로 갱신해야 안정적으로 누적됐다.
- **SDFT, Shenfeld 2026**(2601.19897): 교사 = 시범 예시를 문맥에 넣은 자기 자신. SFT보다 새 과제 정확도가 높고 망각이 적다.
- 시사점: 우리 괄호 쪽지(«같은 답이 또 나왔으니 멈춰라», 비율 .875)는 CRISP 지시 효과의 약한 끝과 같은 크기다.
  차이는 두 가지다. 위치(이어쓰기 중간 vs 문제 앞)와 대상(특정 말 한 마디 vs 전반적 간결함).

### 2.2 특권 정보 교사 (정답·형제·피드백)
- **OPSD, Zhao 2026**(2601.18734): 교사 = 문제 + 검증된 정답 풀이(«합리화하라» 지시), 학생 롤아웃 위 토큰별 JSD.
  Qwen3-8B AIME25 69.8 vs GRPO 68.7. GRPO 대비 토큰 효율 8~12배. 1.7B에서는 실패했다(능력 문턱).
- **SDPO, Hübotter 2026**(2601.20802): 교사 = 현재 모델 + 피드백(런타임 오류 등) 또는 **성공한 형제 풀이**.
  응답이 GRPO보다 평균 3배 짧고, GRPO의 «Hmm/Wait» 반복 고리를 피한다. 아주 어려운 문제의 시험 시 발견을 3배 가속했다.
  ★ 절제: **학생 원래 시도를 교사 문맥에 넣으면** 엔트로피가 .41→.23으로 떨어지고 성능이 나빠졌다(교사가 시도에 끌림).
  교사 정규화(신뢰영역·EMA) 없이는 발산했다. 얼린 교사도 동작하지만 약간 낮다(48.8 vs 50.6). 1.5B에서는 GRPO보다 못했다.
- **Kim 2026**(2603.24472, COLM): 수학에서 자기 증류가 응답 길이를 줄이며 성능을 최대 40%까지 떨어뜨린 사례.
  원인은 «인식적 언표(epistemic verbalization)» 억제다. 교사 문맥이 풍부할수록 억제가 커지고 분포 밖에서 손해가 크다.
- **Kaur 2026**(2607.05184, Arora 그룹): 특권 문맥 증류가 thinking 모델 다섯 개에서 avg@16을 상대 최대 17% 떨어뜨렸다.
  «wait/but/maybe»의 이득 부호가 일반 OPD에서는 +인데 특권 문맥에서는 −로 뒤집힌다. 갈림 비율 8.3%→3.5%.
  전체 풀이(dense)가 정답만(sparse)보다 더 망가진다. ★ **instruct 모델은 이득**: Qwen3-4B-Instruct .527→.559, 갈림 비율 변화 없음.
  해결책은 제시하지 않았다.
- **Nicolicioiu 2026**(2606.26091): 표본 시범을 본 교사는 모델의 기존 선호를 증폭한다. 다양성이 줄고 pass@k가 평탄해진다.
- **π-Distill / Penaloza 2026**(2602.04942): 행동만 담긴 특권 정보로도 증류가 된다(에이전트 과제). 전체 CoT 감독 SFT+RL보다 낫다.
- **STaR, Zelikman 2022**(2203.14465): 정답을 힌트로 주고 근거를 다시 쓰게 함(rationalization). 사후 합리화 근거가 섞일 위험이 있다.

### 2.3 고치기 교사
- **SCoRe, Kumar 2024**(2409.12917) ★ Gemini 1.5 Flash, MATH:
  기본 모델 t1 52.6 → t2 41.4(맞던 것 망침 15.8%). 고치기 흔적 SFT(STaR·Pair-SFT)는 Δ +0.4/+1.8에 그쳤다.
  붕괴 양상은 «거의 안 고치기»다. 첫 답 정확도는 오르고 2턴 수정은 사라진다.
  원인은 (a) 남의 오답과 자기 오답 사이의 분포 차이, (b) 한 가지 고치기 양식으로 수렴하는 것이다.
  해법: 1단계에서 **첫 시도에 KL 끈**을 걸어 결합을 끊는다. 2단계에서 보너스 α·(r₂−r₁)을 준다.
  결과 60.0→64.4(Δi→c 5.8, Δc→i 1.4).
- **RISE, Qu 2024**(2407.18219): 다회 MDP로 재귀 수정. 큰 모델일수록 이득이 크고 1턴 능력은 유지된다.
- **Huang 2023**(2310.01798): 바깥 피드백 없는 내재적 자기 수정은 성능을 떨어뜨리기도 한다.
- **Self-Refine, Madaan 2023**(2303.17651): 같은 모델이 피드백을 쓰고 고친다. 추론 수학에서는 효과가 제한적이다(Huang 2023의 반론).
- **S²R, Ma 2025**(2502.12853): 3.1k 검증·수정 시범으로 SFT 후 결과·과정 RL. Qwen2.5-Math-7B 51.0→81.6.
- **Critique-GRPO, Zhang 2025**(2506.03106): 자연어 비평을 보고 고친 답을 함께 학습. 숫자 보상만 쓸 때 생기는 정체·«계속 실패»를 뚫는다.
- **SD-Zero, He 2026**(2604.12002): 교사(reviser) = 자기 응답 + **이진 보상**을 본 자기 자신. 토큰 단위로 고칠 자리를 스스로 찾는다. 기본 대비 +10% 이상.
  → 우리 «오라클 쪽지(X는 틀렸다)»와 같은 축이다(우리는 닫음).
- **First Try Matters, Kang 2025**(2510.08308) ★: 8개 추론 모델에서 첫 답 이후 반성의 90% 이상이 **확인용**(같은 답)이다. 교정은 대부분 2% 미만이다.
  반성 많은 데이터로 학습하면 늘어나는 것은 «첫 답 정확도»이지 고치기가 아니다.

### 2.4 멈춤 교사 — «말하고 멈추기» vs «자르기·벌점»
| 방법 | 멈춤 말을 배우나 | 기제 | 효과 | 고치기 영향 |
|---|---|---|---|---|
| **SBT, Zhao 2025**(2505.14604) | **예** | 첫 정답 이후 군더더기를 자르고 «Wait, I've gotten the same answer multiple times, time to end the thinking»류 문구를 삽입해 SFT. 잘린 군더더기는 손실에서 마스크 | 토큰 최대 −60%. Qwen2.5-Math-7B MATH500 92.7→90.8, AIME 40.8→38.8 | 별도 분석 없음. 1~3점 손해 |
| **CaRT, Liu 2025**(2510.08517) | **예**(종료 이유를 말로) | «멈추는 게 맞는 궤적 / 조금 바꿔 계속해야 하는 궤적» 반사실 짝 + 종료 근거 SFT | AIME25에서 더 적은 토큰으로 더 높은 정확도 | RL을 덧대도 이득 없고 길어짐 |
| **Thinking Intervention, Wu 2025**(2503.24370) | 추론 시 삽입만 | 생각 토큰 중간에 지시 삽입(학습 없음) | 지시 따르기 +6.7% 등 | 해당 없음 |
| **DEER, Yang 2025**(2504.15895) | 아니오 | «Wait» 자리에서 시험 답을 내게 하고 확신도가 높으면 종료 | 길이 −19~80%, 정확도 +0.3~5 | 학습 없음 |
| **Answer Convergence, Liu & Wang 2025**(2506.02536) | 아니오 | 답 일치로 조기 종료, 종료 신호 확률 올리기, 활성값 탐침 | 수학은 60% 지점에서 답이 수렴 | — |
| **Probe, Zhang 2025**(2504.05419) | 아니오 | 은닉 상태 탐침이 중간 답의 정오를 보정된 확률로 예측 | 토큰 −24%, 성능 유지 | — |
| **s1, Muennighoff 2025**(2501.19393) | 아니오 | 강제 종료 / «Wait» 덧붙여 연장 | — | «Wait» 연장이 정확도를 올림(= 되짚기의 가치) |
| **L1/LCPO, Aggarwal 2025**(2503.04697) | 아니오 | 프롬프트의 목표 길이를 지키게 RL | 길이 제어 | — |
| **ThinkPrune, Hou 2025**(2504.01296) | 아니오 | 예산을 넘으면 보상 0, 예산을 점점 조임 | — | — |
| **ShorterBetter, Yi 2025**(2504.21370) | 아니오 | 가장 짧은 정답 길이를 목표로 보상 | 길이 −50~80% | 과도한 자기 검증이 줄어듦(교정 포함 여부 불명) |
| **LC-R1, Cheng 2025**(2506.14755) | 아니오 | 길이 보상 + 첫 정답 이후 «무효 구간» 압축 보상 | 길이 ~−50%, 정확도 ~−2% | 정답 이후 재확인을 겨냥 |
| **JET, Han 2025**(2509.23392) | 아니오 | 롤아웃 중간 자르기 + 품질 통제 길이 보상 | 1.5B Olympiad +4.6%, 길이 −46% | — |
| **ALP, Xiang 2025**(2506.05256) / **ARLCP, Yu 2026**(2602.12113) | 아니오 | 풀이율에 반비례하는 벌점 / 반성 벌점 + 복잡도별 길이 벌점 | 길이 −35~53%, 정확도 +2.7~5.8 | 어려운 문제는 벌점을 면제해 보호 |
| **Hosseini 2026**(2609.31619) | 아니오(확신도만 학습) | 중간 지점 확신도를 자기 지도로 학습 | 토큰 −25% | 추론 구성 보존 |
- 정리: 멈춤 **말**을 직접 가르친 것은 SBT(SFT, 삽입)와 CaRT(SFT, 반사실 짝)뿐이다. 둘 다 **오프라인 편집 데이터**를 쓴다.
  얼린 교사에게 «멈춰라»를 말해 logit으로 가르친 사례는 찾지 못했다. 가장 가까운 것이 CRISP(전반적 간결함)다.
- 멈춤 압력이 고치기를 해쳤는지 직접 잰 논문은 드물다. 간접 증거만 있다.
  (a) First Try Matters: 자르면 정확도가 2.9%까지 떨어진다. (b) s1: «Wait»를 덧붙이면 오른다.
  (c) Kim/Kaur: 불확실 표지를 억제하면 성능이 떨어진다. (d) ALP/ARLCP: 난이도에 맞춰 벌점을 조절해야 손해가 없다.
  우리 관측(길이·멈춤 벌점이 «다시 보지 마라»로 새어 고침이 줄었다)과 같은 방향이다.

### 2.5 이전 답만 보는 교사·새 출발·닻
- **SDPO 절제**(위 2.2): 원래 시도를 문맥에 넣으면 엔트로피가 떨어지고 성능이 나빠진다. 교사에게서 학생 풀이를 가리는 쪽이 낫다.
- **LEMMA, Pan 2025**(2503.17439): 오답 풀이 → 반성 → (a) 그 자리에서 고쳐 이어가기 / (b) **새로 시작**, 두 갈래 데이터로 SFT.
- **Self-Correction Bench, Tsui 2025**(2507.02778): 같은 오류라도 **자기 출력이면 못 고치고 남의 것이면 고친다**(14개 모델 평균 맹점 64.5%). «Wait» 한 단어로 맹점이 89.3% 줄었다.
- **Chen 2026**(2606.05976) ★: 오류 주장은 그대로 두고 **출처 역할만 바꿨다**(생각 → 사용자/도구/메모리). 명시적 교정률이 23~93%p 올랐다. 수학에서는 `<memory>`가 가장 좋았다.
- **Thought Anchors, Bogdan 2025**(2506.19143): 계획·되돌아가기 문장이 이후 추론을 과도하게 좌우한다(닻 문장).
- 시사점: 우리 «눈 가린 나»(문제 + «전에 X», 풀이 가림)는 SDPO 절제와 LEMMA «새 출발»의 사이에 있다.
  «전에 X»를 **다른 출처**(다른 학생·메모)로 붙이는 변형은 Chen 2026이 직접 지지한다.

### 2.6 내부 신호로 거른 증류
- **80/20, Wang 2025**(2506.01939): 엔트로피 상위 20% «갈림 토큰»에만 경사를 주어도 RLVR 성능이 유지되거나 오른다.
- **DASD, Zhang 2026**(2605.22263): 고엔트로피 토큰은 특권 교사에서 **밀어내고**, 저엔트로피 토큰은 끌어당긴다(방향을 적응).
- **EGRSD, Ke 2026**(2605.13255): 교사 엔트로피 게이트로 고엔트로피 자리의 가중을 낮춘다(하한 유지). Qwen3-4B/8B에서 정확도-길이 경계가 앞으로 나간다.
- **SEAD, Lee 2026**(2606.28562): 교사·학생 공동 엔트로피로 토큰을 세 구역으로 나눈다(~50%는 건너뜀, 역KL·정KL 분담). 별도 교사 OPD.
- **E²-OPSD, Liu 2026**(2610.05048): OPSD에서 학생 엔트로피가 교사를 넘어서는 «엔트로피 과잉»을 교정. OPSD 대비 +4.3.
- **SRPO, Li 2026**(2604.02288): **맞은 샘플 → GRPO, 틀린 샘플 → SDPO**로 라우팅 + 엔트로피 가중. Qwen3-8B에서 GRPO 대비 +3.4.
  (우리 `PFX_DISTILL_ROWS=wrong`과 같은 구조.)
- **INTUITOR, Zhao 2025**(2505.19590): 보상을 자기 확신(self-certainty)으로 대체. 분포 안은 GRPO 수준, 분포 밖 일반화는 우세.
- **Tan 2025**(2505.17656) ★: **일관된 오답**은 모델이 커져도 줄지 않는다. 네 종류 탐지법이 모두 실패했고, 다른 모델의 은닉 상태를 섞은 탐침이 필요했다.
  Zhang 2026(2608.18795): 오답끼리도 정답만큼 강하게 일치할 수 있다.
- 시사점: PMI·확신·형제 일치 같은 내부 신호는 «어디서 증류할지» 고르는 데는 유효하다(갈림 토큰). 그러나 **틀린 합의를 맞은 합의와 구별하지 못한다**.

### 2.7 대조 교사·라우팅
- **Baumann 2026**(2609.21561, Hübotter·Krause 그룹) ★: 같은 틀에 «정답 풀이»를 넣은 **끌어당기는 교사**와 «실패 롤아웃»을 넣은 **밀어내는 교사**를 비교했다.
  끌어당기는 교사는 탐색 억제 + 짧고 확신에 찬 응답을 낳고 평가 성능은 평탄했다.
  밀어내는 교사는 길이가 폭주해 잘림 붕괴에 이르렀다(thinking 모드가 다시 깨어남).
  대조(½[log p₊ − log p₋])에서는 공통 행동 이동이 상쇄돼 **길이가 안정되고 성능이 올랐다**. Qwen3-4B-Instruct-2507 포함.
  토큰 이득을 응답 안에서 섞으면 85.5→62.2로 떨어졌다(자리별 크레딧이 핵심).
- **RLCD, Yang 2023**(2307.12950): 긍정·부정 프롬프트로 만든 출력 짝으로 선호 학습.
- **ATESD, Han 2026**(2605.11458): 교사가 정답 풀이를 **얼마나** 볼지(노출)를 학습 진도 보상으로 조절. 전체 노출이 최선이 아니다. +0.95~2.33.
- **VISTA, Ding 2026**(2608.28306): 검증된 롤아웃으로 교사를 학생 쪽으로 맞춘다. +0.6~2.1.
- **Meta-Hints / AAPO, Wang 2025**(2510.09388): 답을 드러내는 힌트는 분포 불일치로 학습을 흔든다. 추상적인 «인지 비계» 힌트가 더 안정적이다.
- **QuestA, J. Li 2025**(2507.13266): 어려운 문제에 부분 풀이를 붙여 학습. pass@k까지 오른다.
- **LUFFY, Yan 2025**(2504.14945): 오프폴리시 시범을 GRPO 그룹에 섞고 정규화 중요도 표본추출을 쓴다.

## 3. 우리 방법에 대한 함의
1. **눈 가린 나**는 문헌과 맞는다. SDPO 절제(시도를 가리면 엔트로피 보존), LEMMA 새 출발, Self-Correction Bench·Chen 2026(자기 출력 맹점)이 지지한다.
   다만 «전에 X»를 자기 것으로 표기하는 한 닻은 남는다. 출처 재표기가 싼 개선 후보다.
2. 우리 정책은 instruct 모델이다. Kaur 2026에 따르면 특권 문맥 손해가 적은 쪽이다.
   그러나 첫 `\boxed` 뒤 «Wait—» 이어쓰기는 thinking 모델의 반성과 닮았다. 불확실 표지 억제(Kim·Kaur)를 감시 지표로 둬야 한다.
3. **멈춤 쪽지 실패(.875)**는 이상하지 않다. CRISP의 지시 효과도 .63~.86이었다. CRISP는 문제 앞 배치와 교사 주기 갱신 누적으로 키웠다.
   그리고 특정 «말»을 가르친 선행 연구(SBT·CaRT)는 모두 **편집된 롤아웃**을 썼지 지시를 받은 교사를 쓰지 않았다.
4. **멈춤 벌점 → 되짚기 억제 누수**는 First Try Matters(교정은 2% 미만이라 자르면 같이 잘림)와 Kim·Kaur(불확실 표지 억제)로 설명된다.
   방어책은 (a) 맞은 행에만 거는 라우팅(SRPO·ALP식), (b) 대조로 공통 길이 이동을 상쇄(Baumann)하는 것이다.
5. **체계적 오답**(형제 ≥3/7이 같은 X)은 내부 신호로는 원리상 어렵다(Tan 2025). 눈 가린 나가 같은 X를 재도출하는 것도 예측과 같다.
   바깥 정보(검증된 형제 답·피드백)나 출처·관점을 바꾸는 개입이 필요하다. 이 중 검증기 정보를 쓰는 것은 «오라클 쪽지»(닫힌 축)와 겹친다.

## 4. 다음에 시도할 교사 문맥 설계(5개)
공통 관문 자료: 기존 이어쓰기 롤아웃의 맞음→맞음 반복확인 행, 틀린 행(그중 합의 오답 ≥3/7 부분집합). 학습 전에 교사만 샘플링·채점한다.

**D1. 시범 예시 멈춤 교사** — 목표 (i)
- 교사가 보는 것: **문제 앞**에 자기 롤아웃에서 편집한 짧은 예시 1~2개를 둔다.
  예시는 «같은 답을 두 번 확인 → "I keep getting X, so I'll stop here." → 최종 답» 형태다. 그 뒤 문제와 학생 앞부분이 온다. 지시문은 쓰지 않는다.
- 닮은 선행: SDFT(시범 조건 자기 교사), CRISP(문제 앞 배치), SBT(멈춤 문구).
- 새로운 점: 전반적 간결함이 아니라 **특정 상태에서의 특정 메타인지 문장**을 logit 증류로 가르친다. 겹침은 중간 정도다(SBT의 on-policy·logit 판).
- 기대: 반복 확인 자리에서 문구 확률이 올라가고, 맞음→맞음 행의 꼬리가 줄어든다.
- 위험: 틀린 행에도 멈춤이 새어(합의 오답 X도 «계속 X»라서) 고침이 준다. 불확실 표지가 억제된다.
- 관문: 교사 단독 샘플에서 맞음→맞음 행의 토큰 비율 ≤ .6, 문구 출현 ≥ 50%(쪽지 .875 대비).
  같은 교사를 틀린 행에 걸었을 때 고침률이 눈 가린 나 대비 상대 10% 이상 떨어지면 기각한다.

**D2. 자기 롤아웃 이어 붙이기 형제(반사실 짝)** — 목표 (i)
- 교사가 보는 것: 교사 logit을 쓰지 않는다. 같은 답이 두 번째로 확인된 자리에서 학생 롤아웃을 자르고 멈춤 문장 + 최종 답을 붙여 GRPO 그룹에 형제로 넣는다.
  같은 조건이 틀린 행에서 성립하면 같은 편집을 해 음의 이득을 받게 한다.
- 닮은 선행: SBT(자르기 + 문구 SFT), CaRT(반사실 짝), JET(롤아웃 자르기), LUFFY(오프폴리시 혼합).
- 새로운 점: SFT가 아니라 결과 보상 아래 on-policy 편집이다. 자르는 자리를 내부 신호(답 반복 + PMI 이동 ≈ 0)로 정한다. 겹침은 크다(CaRT와 개념적으로 가깝다).
- 기대: 문구 자체와 «틀리면 멈추지 말라»는 구별을 함께 배운다.
- 위험: 이어 붙인 토큰의 중요도 비율이 극단으로 간다. 조건이 틀린 행에서도 비슷하게 발화하면 구별 신호가 없다.
- 관문: 기존 롤아웃에서 «X가 두 번 반복» 조건의 정밀도(발화 중 최종 정답 비율)가 ≥ .9여야 한다. 합의 오답 행에서의 발화율도 함께 보고한다.
  학생 기준 문구 NLL이 상위 1% 토큰보다 크면 D1부터 해 둔다.

**D3. 출처 재표기 눈 가린 나** — 목표 (ii), 저비용
- 교사가 보는 것: 문제 + «다른 학생이 X라고 답했다»(또는 `<memory>` 블록). 현행 «전에 나는 X»를 이것으로 바꾼다.
- 닮은 선행: Chen 2026(역할 재표기, 교정 +23~93%p), Self-Correction Bench.
- 새로운 점: 추론 시 개입을 **증류 교사 문맥**으로 옮긴다. 겹침은 작다.
- 기대: 합의 오답 X에 대한 의심이 늘고, 재도출이 같은 X로 돌아가는 비율이 준다.
- 위험: 맞은 X도 의심해 망침이 는다. 효과가 판단 표명에 그치고 재계산으로 이어지지 않을 수 있다.
- 관문: 교사 단독, 합의 오답 행 고침률 ≥ 눈 가린 나 × 2. 맞은 행 망침 ≤ 고침의 1/3. 표본은 모두 기존 행으로 한다.

**D4. 정답만 쪽지 대조 교사(끌림 − 밀림)** — 목표 (ii)
- 교사가 보는 것: 두 문맥은 «다른 시도는 Y를 얻었다»(Y = 검증된 형제 답)와 «다른 시도는 X′를 얻었다»(X′ = 틀린 형제 답)다. 이 둘만 다르다.
  크레딧은 c_t = ½[log p(·|Y) − log p(·|X′)]다. 형제 중 정답이 하나라도 있는 행에만 쓴다.
- 닮은 선행: Baumann 2026(대조 자기 증류), SDPO(형제 성공), Kaur sparse(정답만), RLCD.
- 새로운 점: 전체 풀이가 아니라 **답 문자열 하나만 다른 최소 짝**이다. 첫 `\boxed` 뒤 이어쓰기에만 건다. 겹침은 중간~큼(Baumann의 축소판).
- 기대: 행동 이동(길이·확신)은 상쇄되고 «X를 떠나 Y로 가는» 토큰만 크레딧을 받는다.
- 위험: 검증기 정보에 의존한다. 닫힌 «오라클 쪽지» 축과의 경계를 사용자가 판정해야 한다. 정답 Y로 근거 없이 건너뛰는 사후 합리화(STaR식)도 위험이다.
- 관문: 기존 롤아웃에서 c_t 합이 고친 이어쓰기와 안 고친 이어쓰기를 가르는 AUC ≥ .65.
  교사(Y 문맥) 단독 샘플이 Y에 도달할 때 중간 유도식이 실제로 Y를 내는 비율 ≥ .7. 길이 비율은 1±.1.

**D5. 가정 점검 메타 힌트 교사(정답 없음)** — 목표 (ii)
- 교사가 보는 것: 문제 + «전에 X» + 메타 힌트다. 힌트 내용: «여러 시도가 X에 동의하지만 같은 오독을 공유할 수 있다. 조건과 해석을 하나씩 다시 읽어라».
  **형제 합의(≥3/7)일 때만** 켜며, 맞은 합의와 틀린 합의 모두에 걸린다.
- 닮은 선행: Meta-Hints(Wang 2025), Critique-GRPO, 실패한 «다른 방법으로 풀어라» 교사.
- 새로운 점: 합의라는 내부 신호로 켜는 비오라클 교사다. 겹침은 크다. «다른 방법» 교사가 실패했으므로 사전 기대는 낮다.
- 기대: 오독형 체계 오답 일부가 고쳐진다.
- 위험: Tan 2025대로 효과가 미미할 수 있다. 맞은 합의 행을 망칠 수 있다.
- 관문: D3과 같은 기준이다. 추가로 오답 유형(오독 vs 계산)을 표시해 오독 유형에서만 효과가 나는지 본다. 효과가 없으면 D5를 닫는다.

권장 순서: D3(가장 싸고 겹침 작음) → D1 관문 → D2 관문(정밀도만 재면 됨) → D4(닫힌 축 경계 판정 후) → D5.

## 5. 확인한 문헌 (제목 — 제1저자, 연도, arXiv)
- A General Language Assistant as a Laboratory for Alignment — Askell, 2021, 2112.00861
- Learning by Distilling Context — Snell, 2022, 2209.15189
- RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment — K. Yang, 2023, 2307.12950
- STaR: Bootstrapping Reasoning With Reasoning — Zelikman, 2022, 2203.14465
- Self-Refine: Iterative Refinement with Self-Feedback — Madaan, 2023, 2303.17651
- Large Language Models Cannot Self-Correct Reasoning Yet — J. Huang, 2023, 2310.01798
- Recursive Introspection (RISE) — Qu, 2024, 2407.18219
- Training Language Models to Self-Correct via Reinforcement Learning (SCoRe) — Kumar, 2024, 2409.12917
- S²R: Teaching LLMs to Self-verify and Self-correct via RL — R. Ma, 2025, 2502.12853
- Critique-GRPO — X. Zhang, 2025, 2506.03106
- LEMMA: Learning from Errors for MatheMatical Advancement in LLMs — Pan, 2025, 2503.17439
- Self-Correction Bench — Tsui, 2025, 2507.02778
- The Self-Correction Illusion: Role Relabeling Gates Explicit Error Flagging — K.-Y. Chen, 2026, 2606.05976
- First Try Matters: Revisiting the Role of Reflection in Reasoning Models — L. Kang, 2025, 2510.08308
- Thought Anchors: Which LLM Reasoning Steps Matter? — Bogdan, 2025, 2506.19143
- Too Consistent to Detect: Self-Consistent Errors in LLMs — H. Tan, 2025, 2505.17656
- Decomposing Wrong-Consensus Agreement in LLM Self-Consistency — L. Zhang, 2026, 2608.18795
- Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs (OPSD) — S. Zhao, 2026, 2601.18734
- Reinforcement Learning via Self-Distillation (SDPO) — Hübotter, 2026, 2601.20802
- Self-Distillation Enables Continual Learning (SDFT) — Shenfeld, 2026, 2601.19897
- Privileged Information Distillation for Language Models (π-Distill) — Penaloza, 2026, 2602.04942
- Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? — J. Kim, 2026, 2603.24472
- Rethinking On-Policy Self-Distillation for Thinking Models — Kaur, 2026, 2607.05184
- On-Policy Self-Distillation with Sampled Demonstrations Reduces Output Diversity — Nicolicioiu, 2026, 2606.26091
- On Repulsive and Attractive Teachers: Separating Correctness from Behavior in Self-Distillation — Baumann, 2026, 2609.21561
- CRISP: Compressed Reasoning via Iterative Self-Policy Distillation (구판 제목 On-Policy Self-Distillation for Reasoning Compression) — Sang, 2026, 2603.05433
- Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision — Y. He, 2026, 2604.12002
- Tailoring Teaching to Aptitude: Direction-Adaptive Self-Distillation (DASD) — H. Zhang, 2026, 2605.22263
- Respecting Self-Uncertainty in On-Policy Self-Distillation (EGRSD) — Ke, 2026, 2605.13255
- SEAD: Competence-Aware On-Policy Distillation via Entropy-Guided Supervision — C.-H. Lee, 2026, 2606.28562
- E²-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation — Y. Liu, 2026, 2610.05048
- Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing (SRPO) — G. Li, 2026, 2604.02288
- Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning (ATESD) — Z. Han, 2026, 2605.11458
- VISTA: Verifier-Informed Student-to-Teacher Adaptation for OPSD — Z. Ding, 2026, 2608.28306
- Don't Tell the Answer, Truly Guide the Reasoning During RL Rollouts — X. Wang, 2025, 2510.09388
- QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation — J. Li, 2025, 2507.13266
- Learning to Reason under Off-Policy Guidance (LUFFY) — J. Yan, 2025, 2504.14945
- Beyond the 80/20 Rule: High-Entropy Minority Tokens — S. Wang, 2025, 2506.01939
- Learning to Reason without External Rewards (INTUITOR) — X. Zhao, 2025, 2505.19590
- Let LRMs Break Free from Overthinking via Self-Braking Tuning — H. Zhao, 2025, 2505.14604
- Effectively Controlling Reasoning Models through Thinking Intervention — T. Wu, 2025, 2503.24370
- Dynamic Early Exit in Reasoning Models (DEER) — C. Yang, 2025, 2504.15895
- Answer Convergence as a Signal for Early Stopping in Reasoning — X. Liu, 2025, 2506.02536
- Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification — A. Zhang, 2025, 2504.05419
- s1: Simple test-time scaling — Muennighoff, 2025, 2501.19393 · L1: Controlling How Long A Reasoning Model Thinks With RL — Aggarwal, 2025, 2503.04697
- ThinkPrune — Hou, 2025, 2504.01296 · ShorterBetter — Yi, 2025, 2504.21370 · LC-R1 — Cheng, 2025, 2506.14755
- Your Models Have Thought Enough (JET) — J. Han, 2025, 2509.23392 · Just Enough Thinking (ALP) — Xiang, 2025, 2506.05256
- Stop Unnecessary Reflection (ARLCP) — Z. Yu, 2026, 2602.12113
- Learning to Stop without Learning to Stop — Hosseini, 2026, 2609.31619
- CaRT: Teaching LLM Agents to Know When They Know Enough — G. Liu, 2025, 2510.08517
