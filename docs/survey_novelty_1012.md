# 서베이: 노벨티 감사 — «Two Teachers in the Middle» (2026-10-12)

대상: `paper/draft_v2/sections/{intro,method,related}.tex`의 주장.
범위: 2023–2026(2025–2026 우선). 새로 찾은 논문은 모두 arXiv API(export.arxiv.org) 또는 abs/HTML 페이지에서 제목·제1저자·날짜를 확인했다.
세부 기제는 초록과 HTML 본문에서 옮겼다(★ = HTML 본문까지 확인).
이미 다룬 문헌(`survey_selfdistill_teacher_context_1008.md`, `survey_noticing_1012.md`, `survey_baselines_selfdistill_1012.md`)은 «기존» 표시만 하고 다시 설명하지 않는다.

## 0. 한 줄 결론
- **«틀린 첫 답 행에서만, 첫 답 뒤 이어쓰기에, 풀이를 가린 얼린 자기 자신을 교사로 쓰고, 기준 모델 KL 끈을 함께 거는» 조합을 한 논문은 찾지 못했다.** 조합으로서의 노벨티는 유지된다.
- 그러나 **부품은 각각 선행이 있다.** «두 개의 자기 교사»는 AD-OPSD·H²SD·대조 교사 계열이 이미 한다. «첫 답 뒤·앞부분 이어쓰기 학습»은 Re²·Failure-Prefix가 한다. «정오로 교사 라우팅»은 SRPO·H²SD·CAST가 한다. «재풀이 쪽지»는 Think Twice가 시험 때 쓴다.
- 따라서 기여 문장은 **«첫 multi-teacher 자기 증류»나 «첫 자동 자기 교정»이 아니라**, «학생 문맥의 *진부분집합*만 보는 교사», «첫 답 직후라는 *상태*에 국한한 증류», «바꾸기/지키기를 두 교사로 *나눈 역할 분담*»으로 좁혀야 안전하다.
- 가장 위협적인 새 논문은 **Re² (ICLR'26)**, **Critique-Conditioned OPSD (10-09)**, **H²SD**, **AD-OPSD** 넷이다. 넷 다 현재 `related.tex`·`refs.bib`에 없다. 기존 서베이에 있던 Failure-Prefix·Flawed Thinking·DenoiseRL·HSD·Feedback Alignment·CEPO·RLCSD도 아직 인용하지 않았다.

## 1. 주장별 판정

| # | 우리 주장 | 판정 | 가장 가까운 논문 | 이유 |
|---|---|---|---|---|
| 1 | 풀이 중간(첫 답 뒤) multi-teacher 자기 증류 — 두 얼린 자기 교사가 같은 이어쓰기 토큰에 작용 | **부분 노벨** | AD-OPSD 2607.10805 ★ | 특권 자기 교사 + 얼린 base 사전분포(학생 문맥)를 토큰마다 섞는다 = «두 얼린 자기 신호». 단 응답 전체이고, 엔트로피 게이트이며, RL 보상과 첫 답 위치가 없다. «두 교사» 자체는 새롭지 않고, 자리와 역할 분담이 새롭다 |
| 2 | 학생보다 *덜 아는* 교사(풀이를 가리고 이전 답 X만 봄) | **대체로 노벨** | CAST 2606.00172 ★ (비특권, 학생과 같은 문맥) · SDPO 절제(기존) | 비특권 교사는 있다(CAST는 정보가 같다). 학생 시도를 빼는 편이 낫다는 관찰도 있다(SDPO). 그러나 교사 문맥이 학생 문맥의 진부분집합인 설계를 학습 교사로 쓴 예는 찾지 못했다 |
| 3 | 교사-문맥 원리: 학생 긴 글을 보는 교사는 신호 ≈ 0, 가린 교사는 큰 신호 | **부분 노벨(관찰은 선행 있음)** | Feedback Alignment 2606.11173(기존) · HSD 2606.15576(기존) | 전체를 그대로 반복하면 교정 신호가 사라진다(복사)는 관찰, 답만 보는 교사는 이동이 23배 작다는 관찰, 시도를 넣으면 엔트로피가 준다는 관찰(SDPO)이 이미 있다. **반론 위험**: 2609.25623은 초기 교사-학생 KL 크기가 하류 성능을 예측하지 못한다고 보고했다. «신호 크기 = 유용성»으로 쓰면 안 된다 |
| 4 | 첫 `\boxed` 뒤 고정 앞부분에서 이어쓰기를 학습 | **부분 노벨** | Re² 2603.07197 ★ · Failure-Prefix 2601.20829(기존) | Re²는 on-policy 앞부분(무작위 0–80% 자름)마다 8개 이어쓰기 그룹으로 GRPO를 하고 «다시 풀기»를 배운다. 앞부분 이어쓰기와 재풀이 행동이 모두 겹친다. «base 모델의 첫 답에서 자른다»와 «교사 크레딧»이 차이다 |
| 5 | 첫 답의 정오로 교사 라우팅(틀린 첫 답 → T1) | **부분 노벨** | SRPO 2604.02288(기존) · H²SD 2607.18955 | 정오 라우팅은 흔하다(SRPO: 실패 → SD, 성공 → GRPO, 맞은 표본에 SD를 쓰면 나빠짐). 다만 모두 *롤아웃 최종 결과*로 라우팅한다. *앞부분 상태(첫 답)*로 라우팅하고 이어쓰기 결과와 분리한 예는 없다 |
| 6 | 리셋 교사(«a previous attempt answered X; solve again from scratch») | **학습 교사로서는 노벨** | Think Twice 2503.19855 (시험 때) · Re² | Think Twice는 «이전 답만 주고 다시 답하라»를 *시험 때 프롬프트*로 쓴다(학습 없음). Re²는 처음부터 다시 풀 때의 성공률로 «다시 풀기»의 가치를 매긴다(교사 없음). 이 쪽지를 얼린 교사의 문맥으로 써서 토큰 크레딧을 만든 예는 없다 |
| 7 | 메타인지 언어 습관의 자동 강화(사람 메타인지 데이터·문구 보상·외부 교사 없음) | **부분 노벨** | Re² · Critique-Conditioned OPSD 2610.13744 ★ | Re²도 SFT·외부 교사 없이 «redo» 행동을 0.5% → 30% 이상으로 키운다. 다만 고정 redo 문구 템플릿을 쓴다. Critique-OPSD는 비평 교사가 «Wait/But/Actually» 같은 되짚기 표지 확률을 올린다고 보였다(비평가는 gpt-5.2 또는 자기 비평). RLERR 2601.12720은 «효과적 반성»에 보상을 준다(SCFT 초기화) |
| 8 | 행동 지표(W→R 증가, R→W 유지) | **노벨 아님(지표)** | SCoRe(기존) Δi→c/Δc→i · S²R | 지표 이름을 병기한다. 결과 자체는 증거로만 쓴다 |
| 9 | OOD(MMLU-Pro) | **노벨 아님** | SDPO · Rethinking-PI(기존) | 이미 쓰는 평가다. 주장이 아니라 검증 항목이다 |

## 2. 가장 가까운 5편 × 축 A–F
기호: ✓ = 같은 요소 · △ = 부분적으로 겹침 · ✗ = 없음. 맨 아래 행은 우리 방법이다.

| 논문 (ID, 발표처) | A 다른 문맥의 얼린 자기 교사 | B 역할이 다른 교사 둘 이상 | C 첫 답 뒤·앞부분 이어쓰기 | D 첫 답 정오 라우팅 | E 이전 답만 보는 재풀이 | F 메타인지 습관 자동 학습 | 위협 |
|---|---|---|---|---|---|---|---|
| **Re²** 2603.07197 (ICLR'26) ★ | ✗ 교사 없음 | ✗ | △ on-policy 무작위 자름 앞부분에서 8개 이어쓰기 | ✗ (그룹 밖 표본으로 재풀이 가치) | △ «처음부터 다시 풀기» 보상(문맥 내용은 명시 안 됨) | △ redo 0.5→30%+, 고정 redo 문구 | **높음**(C·E·F) |
| **Critique-Conditioned OPSD** 2610.13744 (10-09) ★ | ✓ 같은 모델 + 비평(특권, 더 앎) | △ GRPO/RLCSD 위에 비평 항 추가 | △ 인용된 오류 문장 + 이후 50토큰에 가중. 시험 때는 자른 자리에서 이어씀 | △ 틀린 롤아웃만(최종 결과 기준) | ✗ | △ 되짚기 표지 argmax 7.7% → 23.9% | **중–높음**(A·C·D·F) |
| **H²SD** 2607.18955 | ✓ 성공: 검증된 응답 + 다시 말하기 지시 / 실패: 참조 힌트 | ✓ 결과에 따라 문맥이 다른 두 교사 | ✗ | △ 롤아웃 결과로 라우팅 | ✗ | ✗ | **중간**(B·D) |
| **AD-OPSD** 2607.10805 ★ | ✓ 정답 풀이를 본 얼린 초기 체크포인트 | ✓ 얼린 base 사전분포(학생 문맥)를 «인식 닻»으로 섞음 | ✗ 응답 전체, 고엔트로피 20% | ✗ 엔트로피·확률 차 게이트 | ✗ | △ 인식 토큰 억제를 막음 | **중간**(B — T2와 가장 비슷한 «base 닻») |
| **SRPO** 2604.02288 (기존) | ✓ SDPO 교사(피드백·형제) | △ GRPO + SD 혼합 | ✗ | △ 실패 → SD, 성공 → GRPO (맞은 표본에 SD → 악화) | ✗ | ✗ | **중간**(D) |
| *우리* | ✓ 학생 문맥의 진부분집합 | ✓ T1 바꾸기 / T2 지키기(KL) | ✓ base 첫 `\boxed`에서 자른 고정 앞부분 | ✓ 첫 답 X의 정오(검증기, 학습 때만) | ✓ «answered X; solve again from scratch» | △ 문구 보상 없이 첫 답 뒤 되짚기 | — |

차순위(표에서 뺀 것): Failure-Prefix 2601.20829(C, 기존) · SDPO 2601.20802(A·E 절제, 기존) · Think Twice 2503.19855(E, 시험 때) · CAST 2606.00172(A·D) · SCoRe 2409.12917(C·끈, 기존).

## 3. «틀린 첫 답 행 + 첫 답 뒤 + 눈 가린 자기 교사 + 끈»을 이미 한 논문이 있나
**없다(찾지 못했다).** 검색 축 A–F를 교차한 질의 20여 개와 arXiv API 확인 50여 편 중에 네 요소를 모두 가진 논문은 없다. 요소 셋을 가진 것도 없다.
가장 가까운 것은 Critique-Conditioned OPSD(2610.13744)다. 틀린 롤아웃에만, 오류 문장 뒤 구간에 교사 신호를 준다.
그러나 그 교사는 비평이라는 정보를 *더* 보고, 학생의 롤아웃 전체를 읽는다. 첫 답 위치·고정 앞부분·KL 끈은 없다.
주의: 이 분야는 매주 논문이 나온다(10-08~10-09에만 관련 논문 4편: ReTeach, 2610.11128, UPTURN, Critique-OPSD). 제출 직전에 다시 검색해야 한다.

## 4. 안전한 노벨티 문장과 추가해야 할 관련 연구

### 4.1 주장할 수 있는 것 / 없는 것
| 주장 | 가능 여부 | 권고 문구 |
|---|---|---|
| «첫 multi-teacher 자기 증류» | ✗ | AD-OPSD·H²SD·SGSD·대조 교사(Baumann·SIPO·RLCSD·CEPO·OGLS-SD)가 둘 이상의 교사 문맥을 쓴다. → «prior multi-context self-teachers *combine* privileged contexts to sharpen one signal; ours *split one decision* (change vs keep) between two teachers» |
| «KL 끈이 새로운 교사» | ✗ | 기준 모델 KL은 표준이다. LwF(1606.09282)·SCoRe 1단계 KL·AD-OPSD의 base 닻과 같은 계열이다. → «we *read* the standard reference KL as a second teacher and give it a distinct role» |
| «학생보다 덜 아는 교사» | ✓(좁혀서) | «To our knowledge, prior self-distillation teachers see the student's context plus extra information (OPSD, SDPO, RLSD, H²SD) or the same context (CAST). Ours sees a strict subset: the problem and the previous answer, not the reasoning» |
| «첫 답 뒤가 가르칠 자리» | △ | Re²·Failure-Prefix·Flawed Thinking·DenoiseRL이 앞부분 이어쓰기로 학습한다. → «we restrict *teacher credit* to the state right after a committed first answer»라고 위치를 좁히고, 이 넷을 인용한다 |
| «정오로 라우팅» | △ | «unlike SRPO and H²SD, which route by the outcome of the rollout being scored, we route by the correctness of the fixed first answer, a property of the state, so a re-solve that fails still receives T1 credit» |
| «자동 강화(문구 보상·사람 데이터·외부 교사 없음)» | △ | Re²도 SFT·외부 교사가 없다. → 차이를 «no phrase template and no redo token (unlike Re²); no critic (unlike Critique-OPSD)»로 명시한다 |
| «교사-문맥 원리» | △ | «consistent with» Feedback Alignment(복사로 신호 소멸)·SDPO 절제·HSD(답만 보는 교사의 작은 이동). 2609.25623(KL 크기 ≠ 유용성)을 인정해서 «*necessary* condition on signal size, not a sufficient condition for usefulness»로 쓴다 |

### 4.2 권고 기여 문장(intro 1번 항목 대체안)
> We propose a self-distillation scheme that acts only at the state right after a committed first answer. Two frozen copies of the model play complementary roles on the same continuation. A blindfolded self, whose context is a strict subset of the student's (the problem and the previous answer, without the reasoning), gives token credit for re-solving on wrong first answers. Yesterday's self, the standard reference-KL term read as a second teacher, keeps right answers. To our knowledge, no prior method uses a teacher that knows less than the student, or confines self-distillation credit to the post-answer state.

P6(«Prior self-distillation uses one teacher that knows more than the student»)은 **틀린 문장이 된다**(AD-OPSD·H²SD·대조 교사는 둘이다). «uses teachers that know at least as much as the student»로 고친다.

### 4.3 `related.tex`에 넣을 문장(인용 필요)
1. 자기 증류 문단: «Several methods already use two self-teacher contexts: AD-OPSD blends a solution-conditioned teacher with the frozen base prior~\citep{peng2026adopsd}, H²SD routes successful and failed rollouts to different teacher contexts~\citep{cai2026h2sd}, and contrastive teachers subtract a wrong-hint or failed-rollout context~\citep{baumann2026repulsive,yue2026sipo,pan2026rlcsd,heakl2026cepo}. All of them add information to the teacher, and all score the whole response.»
2. 같은 문단: «CAST uses a non-privileged self-teacher that sees exactly the student's context~\citep{li2026cast}; ours sees strictly less.»
3. 같은 문단: «Critique-conditioned self-distillation concentrates credit after the first flagged error and raises backtracking markers, but its teacher reads the full rollout and a critique~\citep{maniparambil2026critique}.»
4. 교사-문맥 원리 절: «A teacher that repeats the student's context loses the corrective signal~\citep{kara2026feedbackalign}, and answer-only teachers shift tokens little away from the answer span~\citep{li2026hsd}. Signal size alone does not predict usefulness~\citep{tian2026whatsee}.»
5. 자기 교정 문단: «Re² trains a model to abandon a prefix and re-solve, with the value of re-solving estimated from fresh attempts~\citep{wang2026re2}. Failure-prefix conditioning, flawed-thinking prefills and noisy prefixes train recovery from fixed prefixes~\citep{kim2026failureprefix,amjith2025flawed,xu2026denoiserl}. Multi-round thinking re-prompts with only the previous answer at test time~\citep{tian2025thinktwice}; we use this prompt as a teacher context during training.»
6. 정오 라우팅: 기존 SRPO 문장 뒤에 «H²SD and CAST also condition the teacher's role on rollout correctness~\citep{cai2026h2sd,li2026cast}.»
7. (선택) 자리 반대편: Dynamic Rollout Editing 2606.17890(정답 뒤 과잉 사고를 RL 중에 편집)을 «repetition after a right first answer» 문단에, Two-Stage Decision-Sampling 2601.01580(RL이 «틀리면 바꾸고 맞으면 유지»하는 결정 요소를 키운다는 이론)을 P2 «두 반쪽» 주장 옆에 인용한다.

## 5. 새로 찾은 논문 전체(확인됨)

| ID | 제목(줄임) · 제1저자 | 축 | 우리와 공유 | 우리에게 없는/그들에게 없는 것 | 위협 |
|---|---|---|---|---|---|
| 2603.07197 | Re²: RL with Re-solving · P. Wang (ICLR'26) ★ | C·E·F | 앞부분 이어쓰기 그룹 GRPO(DAPO-Math), 다시 풀기 행동, 외부 교사 없음 | 교사·토큰 크레딧 없음, redo 문구 템플릿, on-policy 무작위 자름 | 높음 |
| 2610.13744 | Critique-Conditioned OPSD · Maniparambil ★ | A·C·D·F | 틀린 롤아웃만, 오류 뒤 구간 가중, 되짚기 표지 증가, 자른 자리 이어쓰기(시험) | 교사가 더 앎(비평 + 전체 롤아웃), 끈 없음, 첫 답 아님 | 중–높음 |
| 2607.18955 | H²SD: Hybrid Hindsight SD · Q. Cai | B·D | 결과로 갈린 두 교사 문맥 | 둘 다 정보를 더함, 응답 전체 | 중간 |
| 2607.10805 | Thinking Collapse in OPSD (AD-OPSD) · K. Peng ★ | A·B·F | 특권 교사 + 얼린 base 닻, 인식 토큰 보존 | RL 보상 없음, 엔트로피 게이트, 응답 전체 | 중간 |
| 2606.00172 | CAST: Non-Privileged Asymmetric Self-Teaching · Y. Li ★ | A·D | 비특권 자기 교사, 정오별 부호 처리, DAPO-Math, Qwen3-4B | 교사 문맥 = 학생 문맥(덜 알지 않음), KL 0 | 중간 |
| 2609.25623 | What Should a Self-Teacher See? · K. Tian ★ | A(원리) | 교사 문맥 설계 원리, 답만 보는 문맥 | 학생 시도를 가리는 조건 없음. KL 크기 ≠ 유용성(반론 재료) | 중간 |
| 2609.36742 | SIPO · Z. Yue | A·B | 두 문맥(참조 답 / 그룹 내 실수) 차이로 토큰 크레딧, 결과가 방향 | 대조 목적, 응답 전체 | 중간 |
| 2606.11709 | RLCSD · L. Pan | A·B | 맞는 힌트 대 틀린 힌트 교사 대조 | 대조, 응답 전체 | 낮–중 |
| 2605.12400 | OGLS-SD · Y. Yang | A·B | 성공·실패 궤적으로 유도한 두 교사 logit 대조 | 대조 | 낮–중 |
| 2605.28791 | SGSD: Skill-Conditioned Gated SD · J. Huang | B | 여러 자기 교사 풀이 같은 롤아웃을 채점, 검증기로 극성 확인 | 기술·실수 문맥, 역할 분담 아님 | 낮–중 |
| 2605.20643 | AVSD: Adaptive-View SD · D. Nguyen | B | 여러 특권 관점의 합의 대 잔차 | 정보 추가형 | 낮음 |
| 2605.20258 | It Takes Two: Complementary SD · S. Park | B | 두 교사의 곱(PoE), 보존 대 최소 공개 | 맥락 무결성(프라이버시) 과제 | 낮음 |
| 2610.11529 | ReTeach · Y. Tang | A·D·E | 실패한 시도에서 반성·재시도로 교사 구성, 초기 정답/반성 교정/미해결로 가중 | 교사가 학생 시도 + 반성을 봄(더 앎) | 낮–중 |
| 2609.30652 | Recursive Self-Improvement via OPD · S. Yin | A·F | 정답 교사가 오답 끝에서 «Wait»를 거의 주지 않는다는 진단 | 공진화 교사 | 낮음 |
| 2608.09263 | Privileged Likelihood Is Not Automatically Value · X.-P. Nguyen | A(원리) | 토큰 점수가 정오를 거의 못 가름 | 진단 논문 | 낮음(인용 권장) |
| 2608.04794 | Privileged, but Biased · S. Harne | A(원리) | 특권 교사 목적은 과제 성공과 분리됨 | 진단 | 낮음 |
| 2608.06296 | OPSD without Any Supervision · Yijiang Li | A·E | 다수결 의사 정답을 교사 문맥에, 불일치 완성만 증류 | 이전 답이 «끌어당기는 목표»(우리는 «다시 풀 출발점») | 낮–중 |
| 2607.13643 | Consensus as Privileged Context (CANON) · Gkountouras | A·E | 합의 풀이를 본 얼린 스냅숏, 답만이면 이득의 ⅓ | 같음 | 낮음 |
| 2503.19855 | Think Twice: Multi-round Test-time Thinking · X. Tian | E | «이전 답만 주고 다시 답하라» 템플릿 | 학습 없음(시험 때) | 중간(E 선행) |
| 2601.20829 | Failure-Prefix Conditioning · M. Kim (기존) | C | 고정 실패 앞부분 RL | 교사 없음 | 중간 |
| 2610.11128 | Reference Guidance vs Free Generation · H. Wang | C·D | 참조 앞부분 이어쓰기, 전부 실패 그룹에만 | 정답 앞부분(우리는 오답 첫 답) | 낮음 |
| 2610.13672 | Inject a Turn (UPTURN) · J. Jeon | C·D | 실패 롤아웃의 자를 자리 학습 + 앞부분 보존 이어쓰기 | 시험 때, LLM 미세조정 없음 | 낮음 |
| 2606.17890 | Dynamic Rollout Editing · Z. Wei | C | 정답 이후 과잉 검증을 RL 중 편집 | 맞은 행 쪽(우리 미해결 칸) | 낮음 |
| 2601.12720 | Effective Reflection (RLERR) · H. Wang | F | 효과적 반성 보상 | SCFT 초기화, 반성 문구 기반 | 낮–중 |
| 2601.01580 | Two-Stage Decision-Sampling Hypothesis · Z. Zhao | F | RL이 «틀리면 바꾸고 맞으면 유지» 결정을 키운다 | 이론·분석 | 낮음(인용 권장) |
| 2510.15979 | Cog-Rethinker · Z. Sun | E·F | 이전 오답을 참조해 다시 풀기(0정확도 문제), «메타인지» 표방 | 문맥에 이전 오답 풀이를 넣음, SFT | 낮–중 |
| 2609.19156 | Reflective Recovery · Q. Chen | C | 실패 앞부분 + 프롬프트로 회복 데이터 | 자기 지도 데이터, 교사 없음 | 낮음 |
| 1606.09282 | Learning without Forgetting · Z. Li | B(T2) | 옛 모델을 «지키기» 교사로 | 비전, 연속 학습 | 인용 권장 |

## 6. 확인하지 못한 항목
- **«RL on Student-Induced States for On-Policy Distillation»**, **TRD(Trajectory-Refined Distillation)**, **CausalOPD**, **SCOPE(Zheng 2026)**, **Self-Routing**: 검색 요약과 GitHub 목록에만 나오고 arXiv ID를 찾지 못했다.
- **«On-Policy Self-Distillation for RL in LLM Post-Training: A Unified Framework and Survey»**(ResearchGate): 기준 모델 KL을 교사-학생 구조로 읽는다는 요약이 있다. 우리 «T2 = 교사» 해석의 선행일 수 있으나 arXiv ID를 찾지 못했다.
- **«Dual teachers for self-knowledge distillation»**(Pattern Recognition 2024, 비전): 과거 epoch 교사 + 현재 교사 구조다. arXiv 판은 확인하지 못했다.
- **H²SD·SIPO·RLCSD의 KL 끈 유무, 사용 모델**: 초록에 없다. 본문 확인이 필요하다.
- **Re²에서 재풀이가 이전 풀이를 보는지**: HTML 앞 10만 자에는 명시되어 있지 않다.
- **«User wording» 항목**: «Learning to self-correct from fixed wrong prefixes»라는 제목의 논문은 찾지 못했다. 가장 가까운 것은 Self-rewarding correction 2502.19613(기존)과 Failure-Prefix 2601.20829(기존)다. «Kim 2603»은 2603.24472(기존, 자기 증류 성능 저하)로 보았다.
- 날짜가 2026-10-08 이후인 논문(ReTeach·UPTURN·Critique-OPSD·2610.11128)은 v1만 있다. 내용이 바뀔 수 있다.
