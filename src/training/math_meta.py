r"""MATH_META — 수학 무대의 «분리 가능한 메타 스팬 보상» 팔 명세 + 행 계산 + 텔레메트리.

★왜 별도 모듈인가 (2026-09-14, cd9). Countdown 은 오라클(nums/target/witness/decoy)이
있어 메타의 «좋음»을 규칙으로 잴 수 있었다. 수학엔 그 오라클이 없다 — 있는 것은
(1) math_verify 정답 여부, (2) `scripts/local/math_sites.py` 가 **같은 자리 개입**으로
오프라인에서 캐낸 판단 라벨(best_decision ∈ verify/redirect/tie), (3) 외부 프로브
점수(아직 없음, `set_meta_scorer` 로 꽂는다) 뿐이다. 그래서 `countdown_rewards.py`
의 여덟 팔·오라클 항을 건드리지 않고, 수학은 이 파일 한 곳에서 정의한다.

구조는 COUNTDOWN_6ARM 과 같다 — 배치당 한 번 도는 프리패스(`verl_sdc.
_compute_math_arm_stash`)가 여기 함수들로 행을 계산해 스태시를 채우고, 얇은 보상
헤드(`verl_sdc.math_arm_reward`)는 그것을 읽기만 한다. 메타 스팬 항은 시퀀스 보상에
**넣지 않고** 스태시에 따로 두었다가 `verl_sdc._math_add_meta_region_advantage` 가
GRPO 어드밴티지 계산 뒤 메타 토큰 구간에만 얹는다(Countdown §13-b CHK_REGION 과
같은 경로). 답 스팬은 정답 보상만 받는다 — «메타 부분과 정답을 다르게 채점».

팔 (MATH_ARM_SPECS):
  M_G0    math_plain 프롬프트, 정답만.                      (세금 0 기준선)
  M_G1    math_opt 프롬프트, 정답만. 메타 허용, 메타 보상 없음. (허가의 세금)
  M_JUDGE math_opt; 메타 스팬 = 판단 일치 항(아래).
  M_PROBE math_opt; 메타 스팬 = 외부 프로브 점수(`set_meta_scorer`; 기본 0, 한 번 경고).
  M_RAND  M_JUDGE 와 같되 라벨을 **배치 안에서 섞는다** — 대조군. 이 팔이 M_JUDGE 와
          같은 성적이면 판단 항은 내용이 아니라 «메타 스팬에 잡음을 얹은 효과»다.
          ★감사 4(0914): 섞는 단위는 행이 아니라 **프롬프트 그룹(uid)** 이다 — uid→라벨
          표를 순열해 같은 그룹의 8 롤아웃은 같은(섞인) 라벨을 받는다. tie/무라벨 그룹은
          풀 밖. 행 단위로 섞으면 그룹 안 라벨이 갈려 «그룹 중심화 뒤 잡음»의 분산이
          M_JUDGE 와 달라져 대조군이 아니게 된다.

판단 일치 항 (M_JUDGE):
  +1  메타를 냈고 decision == best_decision
  −1  메타를 냈고 decision 이 best_decision 의 반대(verify↔redirect)
   0  tie / decision 없음 / 미발화 / 문제가 라벨표에 없음 / n_blocks>1(형식 위반)
  가중치 MATH_JUDGE_W(기본 0.5). 라벨표는 MATH_JUDGE_LABELS(json, problem→best_decision).
  ★감사 3: 위 «0» 은 보상 0 이 아니라 **항이 정의되지 않음**이다. 행마다 `meta_defined`
  (0/1)를 같이 내고, 그룹 중심화(`verl_sdc._math_add_meta_region_advantage`)는 member=
  meta_defined 로 미정의 행을 평균에서도 배제하고 중심화 값도 0 으로 둔다.

재시도 판단 팔 (M_RETRY / M_RETRY_RAND, 사전등록 수정 3, 2026-09-14):
  ★왜: cut/own 두 자리 설계는 «자리 라벨»(math_sites 오프라인 개입) 밀도가 게이트에 못 미쳤다
  (RESULTS_cd9 s3b/s5b/s6b — 판단 tie 88~99%, 자기 메타는 답이 정해진 뒤의 사후 진술). 그래서
  라벨을 밖에서 캐지 않고 **행 안에서 판단의 결과가 드러나는 구조**로 바꾼다: 첫 답(\boxed) →
  한 블록(decision) → redirect 면 "Second attempt:" 로 다른 방법으로 다시 푼다. 판단 항은
  «첫 답의 정오»와 «결정»의 조합만으로 채점된다(`retry_judgment_term`):
      첫 답 오답 & redirect 실행 & 최종 답 ≠ 첫 답   → +0.5 + 0.5·최종정답  (수정 3b)
      첫 답 오답 & redirect 실행 & 최종 답 == 첫 답  →  0   (말만 하고 베낌 — 탐색이 아니다)
      첫 답 오답 & verify                             → −1
      첫 답 정답 & verify                             → +1
      첫 답 정답 & redirect                           → −1 − LEN_COST·(메타 뒤 토큰 / 1000)
      decision 없음 / 미발화 / 다중 블록              → 항 미정의(meta_defined=0, 그룹 중심화 밖)
  `redirect 실행` = decision==redirect ∧ has_second_attempt. decision==redirect 인데 두 번째
  시도가 없으면(선언만) «베낌»과 같이 본다(오답 0 / 정답 −1, 길이 비용 없음).
  ★«최종 답 == 첫 답»(베낌)은 **수학 동치**(`answers_equivalent`: 문자열 동일 ∨ math_verify)다 —
  문자열 비교면 7→7.0, 0.5→\frac{1}{2} 가 «변경»으로 읽혀 공짜 크레딧이 된다(0914 검증 ①).
  ★수정 3b(굿하트 봉쇄, 0914): 오답∧redirect∧변경 의 크레딧이 +1 이면 «아무 답으로나 바꾸기»가
  «진짜 재시도»와 같은 값이다. +0.5(판단 크레딧) + 0.5·최종정답 으로 갈라 둘을 구별한다 —
  다른 오답으로 바꾸면 +0.5, 정답으로 구제하면 +1.0. 나머지 진리표는 불변.
  ★첫/최종 답 선택에서 <meta>…</meta> **안의** \boxed 는 제외한다 — 메타 안 박스는 «접근 평가»가
  아니라 답 누출이며(boxed_in_meta 가드), 첫 답으로 읽히면 그 가드가 무력해진다(0914 검증 ③).
  가중치 MATH_RETRY_W(기본 0.5), 길이 비용 MATH_RETRY_LEN_COST(기본 0.2 / 1k 토큰). 답 스팬
  보상은 **최종 답(마지막 \boxed)의 정오만**이다 — 판단 항은 메타 스팬에만 얹힌다.
  M_RETRY_RAND: 판단 항의 **부호를 프롬프트 그룹(uid) 단위로** 무작위로 뒤집는 대조군
  (M_RAND 의 그룹 순열과 같은 규약 — 행 단위로 뒤집으면 그룹 중심화 뒤 분산이 달라진다).
  ★뒤집는 것은 ±1 판단 부분뿐이다: term = sign·base − len_cost. 길이 비용까지 뒤집으면 «정답인데
  redirect» 행이 +1 + cost 로 길이 **보너스**가 되어 대조군이 길이를 학습한다(0914 검증 ⑤).
  ★발사 규약: M_RETRY 는 두 번 풀 수 있어야 하므로 **RESP_LEN=6144** 로 올린다
  (run_math_arm.sh 기본 4096 은 그대로 — 두 번째 시도가 잘리면 trunc_rate 규칙이 죽인다).

자기지도 판단 팔 (M_RETRY_SL, 2026-09-14):
  M_RETRY 와 프롬프트·보상이 바이트 동일하되, **결정 토큰에만** 자기지도 CE 를 얹는다.
  라벨은 그 행 자신의 첫 답 정오(교사 없음): 틀렸으면 redirect, 맞았으면 verify. 구현은
  어드밴티지 주입의 REINFORCE 형태(annotate_sl_rows docstring 에 유도) — 그룹 중심화는
  하지 않는다. 가중치 MATH_SL_W(기본 0.5), MATH_SL_WARMUP_STEPS(기본 10) 동안 2배.

형제 동의 예측 팔 (M_AGREE / M_AGREE_RAND, 2026-09-14):
  ★왜: 혼합 문제(0<pass_rate<1) 안에서 "이 샘플의 답이 형제 7개의 다수와 다르다"는 사실이
  오답↔정답을 AUC .77 로 가른다 — 첫 답 자체의 정답 여부만 보는 프로브(.58)보다 훨씬 강한
  신호다. 하지만 모델은 형제를 못 본다(자기 롤아웃 하나만 안다) — 그래서 이 팔은 모델에게
  "형제들과 얼마나 일치할지"를 **예측**하게 만들고(모니터링), 그 예측이 아니라 실제 형제
  동의율(agree_true, 배치 uid 그룹에서 사후 계산)을 기준으로 **행동**(verify/redirect)이
  옳았는지도 채점한다(통제) — M_RETRY 와 같은 첫 답→판단→재시도 구조를 그대로 쓰되, 판단
  블록에 `agreement:` 줄 하나가 늘고, "옳음"의 기준이 첫 답 정오(M_RETRY)가 아니라 다수
  일치(M_AGREE)로 바뀐다.
  메타 스팬 항 = MATH_AGREE_MON_W·모니터링 + MATH_AGREE_CTL_W·통제 (둘 다 기본 0.5).
    모니터링 = 1 − 2·|agree_pred − agree_true|  (∈[−1,1]; 둘 중 하나라도 None 이면 미정의)
      ★자가 보정 항이다 — «내가 형제들과 얼마나 갈릴지 아는가»를 직접 보상한다.
    통제 = retry_judgment_parts 와 같은 진리표이되 «첫 답 정오» 대신 «다수 일치»
      (majority = agree_true≥0.5)를 옳음의 기준으로 쓴다:
        소수(agree_true<0.5)·redirect·변경  → 0.5 + 0.5·최종정답
        소수·verify                         → −1
        다수(agree_true≥0.5)·verify         → +1
        다수·redirect                       → −1 − LEN_COST·(메타 뒤 토큰/1000)
        베낌(변경 없음) / decision 없음      → 0 (정의는 됨, 단 decision 없음은 미정의)
      ★agree_true 는 다수결(형제 동의)이지 정답 채점기가 아니다 — 그래서 이 항은 "다수를
      따르는 것이 옳다"고 가르칠 위험이 있다(굿하트: 순응 학습). 답 스팬 보상은 여전히
      **최종 답 GOLD 정오만**이므로(다수결 답이 아니다), 다수를 따라 봤자 답이 틀리면 답
      스팬에서 크레딧이 없다 — «다수를 따르는 습관»과 «정답을 맞히는 습관»이 분리된다.
  M_AGREE_RAND: 통제 항의 ±1 부분만 uid 그룹 단위로 부호 반전(모니터링 항은 불변 — 자가
  보정 자체는 대조군이 아니라 항상 같은 방식으로 채점된다; M_RETRY_RAND 와 같은 «길이 비용은
  반전 밖» 규약도 그대로).
  강제 행(forced_redirect=1)은 통제·모니터링 둘 다 미정의(0) — 답 스팬만 정상.
  가중치 MATH_AGREE_MON_W(기본 0.5), MATH_AGREE_CTL_W(기본 0.5). 길이 비용은 MATH_RETRY_LEN_COST
  를 그대로 공유한다(재시도 후 토큰 비용이라는 의미가 같다).

비평 정보이득 팔 (M_CRIT, 사전등록 수정 6, 2026-09-14):
  ★왜: 0914 정박 진단에서 **같은 문맥 안의 두 번째 시도는 값이 0**으로 실측됐다(오답 문맥
  1,248행 전부 구제 0.000, 86% 가 첫 답을 글자 그대로 재기입 — SCoRe 의 «행동 붕괴»). 그래서
  M_RETRY 계열은 판단을 배워도 이득이 0 이다(중단·보류). 남는 길은 **판단·비평은 문맥 안에
  두되 행동은 빼는 것**: 첫 답 → 한 블록(확신도 + 2~3문장 비평 + 결정) → 끝. 두 번째 시도를
  학습 중에 생성하지 않는다(프롬프트 math_crit 가 금지).
  그러면 «비평이 좋았는가»를 무엇으로 채점하는가 — **그 비평이 새 풀이를 얼마나 쉽게 만드는가**
  (정보 이득, information gain)다. 같은 문제의 **정답 형제** S+(그룹에서 가장 짧은 정답·미잘림
  롤아웃의 응답, 첫 \boxed 끝까지 자른 것)에 대해 **얼어붙은 채점기**(초기 정책, 학습되지 않는다)로
      IG        = mean_tok log p(S+ | 문제 + NOTE_PREFIX + 비평) − mean_tok log p(S+ | 문제)
      IG_donor  = 같은 것, 단 **다른 문제**의 비평(내용 대조)
      항        = clip((IG − IG_donor) / MATH_CRIT_SCALE, −1, +1)      가중치 MATH_CRIT_W
  ★IG_donor 대조가 핵심이다. IG 만 보면 «무슨 말이든 덧붙이면 다음 토큰이 쉬워진다»(문맥 길이·
  주의 효과)를 «비평 내용의 값»으로 오독한다 — cd9 EVC 판정과 같은 함정.
  ★채점기가 **얼어붙어야** 하는 이유: 학습 중인 정책으로 재면 «비평을 잘 쓰기»가 아니라 «채점기가
  좋아하도록 정책을 옮기기»가 최적해가 된다(자기 참조 굿하트). 초기 정책 고정 = 고정된 자.
  항이 정의되는 행: 발화 ∧ 단일 블록 ∧ 비평 문장 존재 ∧ **누출 아님**(critique_leaks) ∧ 그룹에
  정답 형제(S+)가 있음. 나머지는 미정의(meta_defined=0, 항 0, 그룹 중심화 밖).
  답 스팬 보상 = 그 (유일한) \boxed 의 정오 — 비평은 메타 스팬에만 얹힌다.
  ★S+ 는 **그룹 단위**로 고른다(자기 자신 배제 없음). 같은 그룹의 형제들이 서로 다른 표적을 쓰면
  그룹 중심화가 «비평의 질» 대신 «표적이 쉬웠나»를 비교하게 된다. 덤으로 plain 채점이 그룹당
  한 번이면 된다(캐시).

불일치 진단 팔 (M_DIS / M_DIS_RAND / M_DIS0, 2026-09-14):
  ★왜: 이 4B 비사고 정책에서 **한 풀이 안의** 메타 블록(verify/critique/redirect/plan)은 네
  관문 전부에서 own ≤ donor 로 인과적 무력이 실측됐다(docs/HYPOTHESIS_LEDGER_cd9.md). 문제
  안에서 살아 있는 유일한 신호는 **자기 샘플들 사이의 불일치**다(형제 답 점유율, 문제별 AUC
  .754). M_DIS 는 언어적 메타 습관을 바로 그 신호 위에서 돌린다 — 프롬프트가 이 정책 자신의
  후보 풀이 마무리 K=4 개를 보여 주고, 응답은 `<meta>…</meta>` 한 블록에 DIAGNOSIS(어느 후보가
  어디서 갈리고 어떤 가정·계산이 틀렸는지, 후보 번호 인용)를 쓰고 `Commit: <n>` 으로 닫은 뒤
  최종 답을 \boxed 로 낸다.
  답 스팬 = **gold 정오**(모든 팔과 같다). 메타 스팬 = **자기 증류 판단 크레딧**(src/training/
  math_dis.dis_row_credit): 라벨이 gold 가 아니라 **그 GRPO 그룹 형제들의 LEAVE-ONE-OUT 다수답**
  (이 행 자신의 최종 답은 뺀다)에서 온다 —
      +1  커밋한 후보의 저장된 답 ≡ LOO 형제 다수답(plurality, 수학 동치로 군집, 자기 자신 제외)
      −1  다르다
      미정의  커밋 미파싱/범위 밖 · 메타 없음 · 블록 다수 · 메타 안 \boxed · LOO 형제 2 미만 ·
              다수답 미정(동률/무답)
  ★0914 리뷰 D1: 자기 자신을 다수결에 포함시키면 이 행의 커밋이 자기 자신의 최종 답을 정하므로
  자기가 자기를 심판하는 표를 한 표(8분의 1) 보태는 순응 유인(conformity attractor)이 생긴다.
  LOO 는 이 프로젝트의 다른 자기증류 라벨(ledger c1.py 의 형제 답 점유율, M_RETRY 의
  leave-one-out p_retry)과 같은 관행이다.
  가중치 MATH_DIS_W(기본 0.5). ★판단 스팬에 한정한 TTRL 식 «다수결을 라벨로»이며 **gold 는
  메타 크레딧에 한 번도 들어가지 않는다**(들어가면 판단이 아니라 정답 복사를 보상한다).
  parquet 의 `cand_correct`(후보들의 gold 정오)는 **지표 전용** — 보상 경로가 읽지 않는다.
  M_DIS_RAND: 크레딧의 **부호를 uid 그룹 단위로** 고정 시드로 무작위 반전한 대조군 — M_RAND/
  M_RETRY_RAND 와 **같은** 규약(`_group_sign_flips`)이다(0914 리뷰 D2로 정정 — 이전에는 행
  단위로 뒤집어 그룹 중심화 뒤 분산이 M_DIS 와 달라지는 대조군 무효 버그가 있었다). 시드만은
  여전히 **고정**(DIS_RAND_SEED) — 스텝에 따라 흔들리지 않는다는 것이 이 팔의 원래 의도다.
  M_DIS0: 프롬프트는 같고 **메타 크레딧 가중치만 0** — 후보 4개를 본다는 사실 자체의 값
  (= 정답 보상만으로도 오르는 몫)과 판단 크레딧의 몫을 가른다.

난이도 판단 팔 (M_DIFF / M_DIFF_RAND / M_DIFF0, 2026-09-14e):
  ★왜: 이 정책의 **내부 상태**는 문제 난이도를 읽는다(문제 홀드아웃 은닉 프로브 풀링 AUC .825,
  자기 pass rate 와 Spearman +.51) 그러나 **말로 낸** confidence 는 그 축을 거의 못 담고(+.13),
  GRPO 그룹 중심화는 문제 간 축을 통째로 지운다(docs/HYPOTHESIS_LEDGER_cd9.md §C1). M_DIFF 는
  ① 그 판단을 **풀기 전에 말하게** 하고(응답은 `<meta>` 한 블록으로 시작, `difficulty:
  easy|medium|hard` + `why:` 한 문장) ② **중심화하지 않고** 보상하며 ③ 라벨을 gold 없이 그
  정책 자신의 형제 롤아웃에서 뽑는다.
  답 스팬 = **gold 정오**(모든 팔과 같다). 메타 스팬 = **방향 크레딧**(src/training/math_diff.
  diff_row_credit): 라벨은 **LEAVE-ONE-OUT 형제 답 동의도**(자기 제외; 참 pass rate 와 ρ≈.55)를
  세 버킷으로 이산화한 값이고 —
      +1  말한 버킷 == 동의도 버킷
       0  이웃(easy↔medium, medium↔hard)
      −1  반대(easy↔hard)
      미정의  블록 없음 · 블록이 맨 앞이 아님 · 버킷 미파싱 · why 없음 · 메타 안 \boxed ·
              블록 다수 · LOO 라벨 미정의(답 있는 형제 2 미만 / 다수 동률)
  ★**그룹 중심화를 하지 않는다**(verl_sdc._math_add_diff_meta_advantage) — 한 그룹의 형제는
  **같은 라벨**을 공유하므로 중심화하면 «어려운 문제라고 다같이 말했다»가 정확히 0 이 된다.
  그것이 바로 이 팔이 되살리려는 축이다. 대신 보너스가 정답 신호를 압도하지 못하게 배치 평균
  |답 어드밴티지| 로 절댓값을 자른다(Cheng et al. 2506.14758 의 «보너스는 지배하면 안 된다»).
  가중치 MATH_DIFF_W(기본 0.5). ★gold 는 크레딧에 **한 번도** 들어가지 않는다 —
  stated_vs_truepass_spearman(형제 r_corr 로 잰 실제 난이도)는 **지표 전용**이다.
  M_DIFF_RAND: 그룹↔라벨 대응만 깨는 대조군 — uid 그룹마다 **다른 그룹의 LOO 동의도**로 라벨을
  만든다(M_RAND 의 «그룹 단위 순열» 규약; 행 단위로 섞으면 같은 그룹 안 라벨이 갈려 분포가
  M_DIFF 와 달라진다). 라벨 분포는 그대로고 «이 문제가 어렵다»는 대응만 사라진다.
  M_DIFF0: 프롬프트는 같고 **메타 크레딧 가중치만 0** — 난이도를 말하게 하는 것 자체의 값
  (세금 포함)과 판단 크레딧의 몫을 가른다.

2-시도 팔 (M_TRIAL2_CREDIT / M_TRIAL2_OUTCOME, S3, 2026-09-15):
  한 응답 안에서 두 번 푸는 M_RETRY 와 **정반대**다: 시도 1 이 틀리면 **문맥을 통째로 버리고**
  (G8: 오답 본문 제시 −10.4pp · 사실만 고지 +3.4pp) 사실 줄 + 자기 반성문 한 문장만 들고
  다시 푼다. 크레딧은 LaMer(2512.16848) 의 크로스-에피소드 리턴 G(1) = R1 + γ·R2
  (γ = GAMMA_TRAJ, 기본 .6). OUTCOME 팔은 파이프라인이 같고 크레딧만 없다(시도1=R1·반성문=0).
  프롬프트·보상·슬롯 기하는 `src/training/trial2.py` 가 단일 진실 원천이고, 설계 원문은
  `docs/DESIGN_S3_trial2_0915.md`.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from src.training import countdown_rewards as _cdr
from src.training import retry_metrics as _rmet

SPEC_VERSION = "math-meta-0914"

# ── 팔 명세 ───────────────────────────────────────────────────────────────────
# `meta_term`: None(메타 항 없음) / "judge" / "probe" / "judge_shuffled".
# `require_meta`: 발화율 중단 규칙을 적용하는가 — 메타 항이 있는 팔만 True 다
#   (M_G1 은 허용만 하므로 발화 0 이 정상, Countdown OPT 계열과 같은 규약).
MATH_ARM_SPECS: dict[str, dict] = {
    "M_G0": {"variant": "math_plain", "meta_term": None, "require_meta": False,
             "note": "메타 지시문 없음. 정답만. 세금 0 기준선."},
    "M_G1": {"variant": "math_opt", "meta_term": None, "require_meta": False,
             "note": "메타 허용·비요구. 정답만. 허가 문장의 세금을 잰다."},
    "M_JUDGE": {"variant": "math_opt", "meta_term": "judge", "require_meta": True,
                "note": "메타 스팬에 판단 일치 항(±1·MATH_JUDGE_W), 답 스팬은 정답만."},
    "M_PROBE": {"variant": "math_opt", "meta_term": "probe", "require_meta": True,
                "note": "메타 스팬에 외부 프로브 점수. set_meta_scorer 로 꽂는다."},
    "M_RAND": {"variant": "math_opt", "meta_term": "judge_shuffled", "require_meta": True,
               "note": "M_JUDGE 의 라벨을 배치 안에서 섞은 대조군."},
    # ★사전등록 수정 3: 재시도 판단 팔. 라벨표 불필요(행 안에서 채점). RESP_LEN=6144 로 발사.
    "M_RETRY": {"variant": "math_retry", "meta_term": "retry", "require_meta": True,
                "note": "첫 답→판단→(redirect 면) 재시도. 메타 스팬 = 재시도 판단 항, 답 스팬 = 최종 답 정오."},
    "M_RETRY_RAND": {"variant": "math_retry", "meta_term": "retry_shuffled", "require_meta": True,
                     "note": "M_RETRY 의 판단 항 부호를 uid 그룹 단위로 무작위 반전한 대조군."},
    # ★0914 SL: M_RETRY 와 «바이트 동일»한 팔에 결정 토큰 자기지도(self-distilled judgment) 항만 더한다.
    #   왜: M_RETRY 는 step 1 텔레메트리에서 redirect=0.000(전 행 verify)이라 RL 이 희소 행동을
    #   아예 표집하지 못한다 — 강제 행은 프롬프트가 달라 판단 항이 미정의라 정상 프롬프트의 결정
    #   토큰엔 기울기가 한 방울도 안 간다. 그래서 «행 자신의 첫 답 정오»를 라벨로 삼아(교사 없음)
    #   결정 토큰에만 CE 를 얹는다. 구현은 math_meta.annotate_sl_rows + verl_sdc.
    #   _math_add_decision_sl_advantage 를 볼 것.
    "M_RETRY_SL": {"variant": "math_retry", "meta_term": "retry_sl", "require_meta": True,
                   "note": "M_RETRY + 결정 토큰 자기지도 CE(MATH_SL_W). 라벨은 그 행의 first_correct."},
    # ★형제 동의 예측(모듈 docstring 참조). math_retry 와 같은 첫답→판단→재시도 구조지만
    #   블록에 agreement: 줄이 늘고, variant 는 별도(math_agree) — 프롬프트가 갈린다.
    "M_AGREE": {"variant": "math_agree", "meta_term": "agree", "require_meta": True,
               "note": "메타 스팬 = W_MON·모니터링(자가 동의 예측 보정) + W_CTL·통제(다수 일치 기준 재시도 판단), "
                       "답 스팬은 최종 답 GOLD 정오만."},
    "M_AGREE_RAND": {"variant": "math_agree", "meta_term": "agree_shuffled", "require_meta": True,
                     "note": "M_AGREE 의 통제 항 부호만 uid 그룹 단위로 무작위 반전한 대조군(모니터링 항은 불변)."},
    # ★사전등록 수정 6: 비평 정보이득 팔(모듈 docstring 참조). 문맥 안 재시도 없음 → RESP_LEN 4096.
    "M_CRIT": {"variant": "math_crit", "meta_term": "crit", "require_meta": True,
               "note": "첫 답→한 블록(비평)→끝. 메타 스팬 = 얼어붙은 채점기가 잰 비평의 정보 이득 "
                       "(IG − IG_donor), 답 스팬 = 그 \\boxed 의 정오."},
    # ★0914d 불일치 진단 팔(모듈 docstring 참조). 프롬프트는 math_opt 시스템 + **행마다 다른**
    #   사용자 턴(문제 + 후보 4개 + 진단 지시)이라 parquet 이 완성된 턴을 싣는다.
    "M_DIS": {"variant": "math_dis", "meta_term": "dis", "require_meta": True,
              "note": "후보 4개 제시→한 블록(진단+Commit: n)→답. 메타 스팬 = 그룹 다수답 기준 "
                      "자기증류 판단 크레딧(±1·MATH_DIS_W), 답 스팬 = 최종 \\boxed 의 gold 정오."},
    "M_DIS_RAND": {"variant": "math_dis", "meta_term": "dis_shuffled", "require_meta": True,
                   "note": "M_DIS 의 판단 크레딧 부호를 행마다 고정 시드로 반전한 대조군."},
    # ★require_meta=False 인 이유(M_G1 과 같은 규약): 메타 크레딧이 0 인 팔에서 발화가 줄어드는
    #   것은 **관측 대상이지 사고가 아니다** — 발화·커밋 중단 규칙을 걸면 이 대조군은 «크레딧이
    #   없으면 습관이 사라진다»는 결과를 내기 전에 죽는다.
    "M_DIS0": {"variant": "math_dis", "meta_term": "dis_zero", "require_meta": False,
               "note": "M_DIS 와 프롬프트 동일, 메타 크레딧 가중치 0(정답 보상만) — 후보를 보는 "
                       "것 자체의 값과 판단 크레딧의 값을 가른다."},
    # ★0914e 난이도 판단 팔(모듈 docstring 참조). 프롬프트는 math_opt 시스템 + 문제 + 난이도
    #   지시(사용자 턴) — 행마다 다른 재료가 없어 build_math_parquet.py --variant math_diff 로 만든다.
    "M_DIFF": {"variant": "math_diff", "meta_term": "diff", "require_meta": True,
               "note": "풀기 전 <meta> 한 블록(difficulty+why)→풀이. 메타 스팬 = LOO 형제 동의도 "
                       "버킷 기준 방향 크레딧(+1/0/−1 · MATH_DIFF_W, **중심화 없음**), 답 스팬 = "
                       "최종 \\boxed 의 gold 정오."},
    "M_DIFF_RAND": {"variant": "math_diff", "meta_term": "diff_shuffled", "require_meta": True,
                    "note": "M_DIFF 의 라벨을 uid 그룹 단위로 순열한 대조군(라벨 분포 불변, "
                            "그룹↔라벨 대응만 파괴)."},
    # ★require_meta=False 인 이유(M_G1/M_DIS0 과 같은 규약): 크레딧이 0 인 팔에서 발화가 줄어드는
    #   것은 **관측 대상이지 사고가 아니다**.
    "M_DIFF0": {"variant": "math_diff", "meta_term": "diff_zero", "require_meta": False,
                "note": "M_DIFF 와 프롬프트 동일, 메타 크레딧 가중치 0(정답 보상만) — 난이도를 "
                        "말하게 하는 것 자체의 값과 판단 크레딧의 값을 가른다."},
    # ★S3 2-시도 팔(docs/DESIGN_S3_trial2_0915.md). 한 응답 안에서 두 번 푸는 M_RETRY 와
    #   **정반대**다: 시도 1 이 틀리면 **문맥을 통째로 버리고**(G8: 오답 본문 제시 −10.4pp,
    #   사실만 고지 +3.4pp) 사실 줄 + 자기 반성문 한 문장만 들고 다시 푼다. 크레딧은
    #   LaMer(2512.16848) 의 크로스-에피소드 리턴 G(1)=R1+γ·R2 (γ=GAMMA_TRAJ, 기본 .6).
    #   프롬프트·보상·슬롯 기하는 src/training/trial2.py 가 단일 진실 원천.
    "M_TRIAL2_CREDIT": {"variant": "math_opt", "meta_term": "trial2_credit", "require_meta": False,
                        "note": "2-시도. 시도1·반성문 스팬에 γ·R2 크로스-에피소드 크레딧, 시도2 스팬은 R2."},
    # ★0918 사전등록 수정 6: 자발적 답 수정(revision) 증폭. 프롬프트 주입 없음(math_opt) —
    #   «첫 \boxed → 재검토 → 다른 \boxed» 라는 정책이 **이미 하는** 행동에만 크레딧을 준다.
    #   크레딧은 그 행 안에서(첫 답 vs 마지막 답) 계산되고 수정 구간 토큰에만 얹힌다(중심화 없음).
    #   ★결과만 보는 대조 팔은 **M_G1 을 그대로 쓴다**(math_opt·meta_term None·require_meta False
    #   — 이 세 팔과 프롬프트가 글자 그대로 같다). 새 팔을 만들지 않는다.
    "M_REV_CF": {"variant": "math_opt", "meta_term": "revision_cf", "require_meta": False,
                 "note": "행 내부 정오개선 크레딧(+MATH_REV_W·save / −derail)을 수정 구간에. "
                         "추가 forward 없음. 대조는 M_G1."},
    "M_REV_PMI_GOLD": {"variant": "math_opt", "meta_term": "revision_pmi", "require_meta": False,
                       "note": "교사강제 믿음 이동(PMI_open→PMI_close, 앵커 gold_x)을 수정 구간에. "
                               "대조는 M_G1."},
    "M_REV_PMI_COMBO": {"variant": "math_opt", "meta_term": "revision_pmi", "require_meta": False,
                        "note": "M_REV_PMI_GOLD 와 같되 앵커 combo — 그룹 다수답이 틀린 그룹에서 "
                                "구제 보너스 2배. 대조는 M_G1."},
    # ★0919 혼합 팔(docs/analysis/HABIT_vs_PMI_0919): PMI 항은 «검산해서 믿음이 움직였는가»를
    #   보느라 결과가 나빠진 수정(derail)을 앵커 규칙으로 통째로 건너뛴다. 결과 항을 같이 얹어
    #   그 편애가 희석되는지(정밀도가 지켜지는지) 본다. 두 항은 **한 경로**에서 합산된다.
    "M_REV_PMI_CF": {"variant": "math_opt", "meta_term": "revision_pmi_cf", "require_meta": False,
                     "note": "PMI의 검산 편애를 결과 항이 희석하는지 보는 혼합 팔 "
                             "— HABIT_vs_PMI_0919. 앵커 gold_x, 가중치 MATH_REV_W_PMI/W_CF."},
    # ★0920 확인(confirm) 팔(docs/analysis/WHY_NO_GAIN_0920.md §4): 누락 질량의 63.45% 가
    #   «박스를 두 번 쓰고도 같은 답» — 의심은 하는데 못 바꾼다. 그 행을 크레딧 후보로 올려
    #   «가짜 확인»(믿음이 안 움직였는데 재확인)에 값을 주지 않는지 본다. 앵커 gold_x.
    "M_REV_PMI_CONF": {"variant": "math_opt", "meta_term": "revision_pmi", "require_meta": False,
                       "note": "M_REV_PMI_GOLD + **첫 답이 틀린** 확인 행 크레딧(모드 1 을 팔이 강제). "
                               "가짜 확인은 shift≈0 이라 값을 못 받는다 — WHY_NO_GAIN_0920 §4."},
    # ★0921 H2(docs/PLAN_h2_twoturn_0921.md §1·§5): 밀도 1 리셋 2턴. 고르기를 배우게 하지
    #   않고(문제 내 AUC .51~.55) 파괴율 .078 → .04 를 표적으로 비대칭 Δ 보상을 건다.
    "M_TRIAL2_SCORE": {"variant": "math_opt", "meta_term": "trial2_score", "require_meta": False,
                       "note": "2-시도. 시도2 스팬 = R2 + α⁺·max(R2−R1,0) − α⁻·max(R1−R2,0) "
                               "(SCORE_ALPHA_POS/NEG, 기본 1.0/2.0) — «맞던 것을 깨뜨리지 않기»."},
    "M_TRIAL2_OUTCOME": {"variant": "math_opt", "meta_term": "trial2_outcome", "require_meta": False,
                         "note": "M_TRIAL2_CREDIT 과 파이프라인 동일, 크로스-에피소드 크레딧만 제거 "
                                 "(시도1=R1·반성문=0) — 결과만 보는 대조군."},
}

# ★"retry_sl" 도 재시도 계열이다 — 행 계산(_compute_retry_rows)·텔레메트리·중단 규칙·
#   RESP_LEN 6144 규약을 M_RETRY 와 글자 그대로 공유하고, SL 항만 따로 얹는다.
_RETRY_TERMS = frozenset({"retry", "retry_shuffled", "retry_sl"})
_RETRY_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"] in _RETRY_TERMS)

_AGREE_TERMS = frozenset({"agree", "agree_shuffled"})
_AGREE_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"] in _AGREE_TERMS)
# ★ABORT_RULES 의 redirect_rate/trunc_rate/emit_rate 워밍업은 «재시도 구조를 가진 팔 전부»에
#   적용된다 — M_AGREE 도 첫답→판단→재시도 구조를 그대로 쓰므로 M_RETRY 와 같은 취급이 맞다.
_RETRY_LIKE_ARMS = _RETRY_ARMS | _AGREE_ARMS
# ★verl_sdc._compute_math_arm_stash 가 truncated/tok_len_fn/forced_redirect/group_pass_rate 재료를
#   싣는 조건 — M_AGREE 도 M_RETRY 와 같은 재료가 필요하다(두 번째 시도 잘림, 메타 뒤 토큰 비용).
_RETRY_LIKE_TERMS = _RETRY_TERMS | _AGREE_TERMS

# ★비평 정보이득 팔(M_CRIT). 재시도 구조가 **없다** — 재시도 전용 규칙(redirect_rate/trunc_rate)
#   과 forced_redirect/group_pass_rate 재료를 공유하지 않는다.
_CRIT_TERMS = frozenset({"crit"})
_CRIT_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"] in _CRIT_TERMS)
# ★verl_sdc 가 truncated/tok_len_fn 을 실어야 하는 팔 — M_CRIT 은 «정답 형제 S+ 고르기»에서
#   잘린 롤아웃을 배제해야 하므로 truncated 가 필요하다(길이 비용은 없다).
_NEEDS_TRUNC_TERMS = _RETRY_TERMS | _AGREE_TERMS | _CRIT_TERMS

# ★불일치 진단 팔(M_DIS/M_DIS_RAND/M_DIS0). 재시도 구조도, 외부 채점기도 없다 — 필요한 재료는
#   parquet 의 후보 답(cand_answers)과 배치의 uid 그룹뿐이다.
_DIS_TERMS = frozenset({"dis", "dis_shuffled", "dis_zero"})
_DIS_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"] in _DIS_TERMS)

# ★난이도 판단 팔(M_DIFF/M_DIFF_RAND/M_DIFF0). 재시도 구조도, 외부 채점기도, parquet 의 추가
#   컬럼도 필요 없다 — 필요한 재료는 배치의 uid 그룹뿐이다(라벨이 형제 롤아웃에서 나온다).
#   ★이 팔의 메타 항은 **중심화 경로를 타지 않는다**(verl_sdc 가 별도 스태시 키로 받는다).
_DIFF_TERMS = frozenset({"diff", "diff_shuffled", "diff_zero"})
_DIFF_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"] in _DIFF_TERMS)

# ★수정(revision) 팔. 재시도 구조도, 외부 채점기도, parquet 추가 컬럼도 필요 없다 —
#   재료는 롤아웃 본문과 배치의 uid 그룹뿐이다(M_REV_PMI_* 만 frozen-ref forward 를 더 쓴다).
#   ★이 팔의 크레딧도 M_DIFF 처럼 **중심화 경로를 타지 않는다**(verl_sdc 가 별도 스태시 키로 받는다).
_REV_TERMS = frozenset({"revision_cf", "revision_pmi", "revision_pmi_cf"})
_REV_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"] in _REV_TERMS)
# ★M_REV_PMI_COMBO 만 combo 앵커가 기본이다 — 런처가 MATH_REV_ANCHOR 를 안 줘도 팔 이름이
#   처치를 정한다(무효 레버 방지: 앵커를 잊으면 두 PMI 팔이 바이트 동일해진다).
_REV_ARM_ANCHOR = {"M_REV_PMI_GOLD": "gold_x", "M_REV_PMI_COMBO": "combo",
                   "M_REV_PMI_CF": "gold_x", "M_REV_PMI_CONF": "gold_x"}
# ★확인(confirm) 크레딧을 **팔이 강제**하는 목록 — 런처가 MATH_REV_CONFIRM 을 잊어도
#   M_REV_PMI_CONF 는 M_REV_PMI_GOLD 와 바이트 동일해지지 않는다(무효 레버 방지).
_REV_ARM_CONFIRM = {"M_REV_PMI_CONF": 1.0}   # 모드 1 = 가짜 확인만


def rev_confirm(arm: str) -> float:
    """확인(confirm) 크레딧 **모드** — 0 = 끔(기본) / 1 = 첫 답이 **틀린** 확인 행만 /
    2 = 확인 행 전부(대조용). MATH_REV_CONFIRM 이 있으면 그것, 없으면 팔 명세가 정한 값.
    ★1 이 기본 처치다: 확인 행의 ~90% 는 첫 답이 맞는 행이라(참조 2,718 중 2,458) 전부
    받으면 «맞은 답 재박스»를 키우고 수정 신호를 질량으로 덮는다(WHY_NO_GAIN_0920 §4)."""
    v = (os.environ.get("MATH_REV_CONFIRM") or "").strip()
    if v:
        try:
            return float(v)
        except ValueError:
            return 0.0
    return float(_REV_ARM_CONFIRM.get(str(arm).upper(), 0.0))


def rev_weight() -> float:
    """MATH_REV_W (기본 1.0) — 수정 구간 크레딧의 가중치. 워커에서 읽는다(diff_weight 와 같은 규약)."""
    v = os.environ.get("MATH_REV_W")
    return 1.0 if v is None else float(v)


def rev_anchor(arm: str) -> str:
    """MATH_REV_ANCHOR 가 있으면 그것, 없으면 팔 명세가 정한 앵커(gold_x/combo)."""
    v = (os.environ.get("MATH_REV_ANCHOR") or "").strip()
    return v or _REV_ARM_ANCHOR.get(str(arm).upper(), "gold_x")


# ★SL 항(결정 토큰 자기지도)을 받는 팔. 지금은 M_RETRY_SL 하나.
_SL_TERMS = frozenset({"retry_sl"})
_SL_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"] in _SL_TERMS)

_META_TERM_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"])
_OPPOSITE = {"verify": "redirect", "redirect": "verify"}

# ★비평을 재풀이/채점 문맥에 심을 때의 접두. math_critique_resolve_gate 가 import 한다 —
#   학습(IG 채점)과 게이트(재풀이 생성)가 같은 문자열을 써야 두 산출물이 같은 조건이다.
NOTE_PREFIX = "\n\nNote from a previous attempt: "


def require_arm(arm: str) -> dict:
    """fail-closed. 조용한 기본값 금지 — 팔이 틀리면 다섯 잡이 같은 보상으로 돈다."""
    if arm not in MATH_ARM_SPECS:
        raise ValueError(
            f"[MATH] arm={arm!r} 가 MATH_ARM_SPECS {sorted(MATH_ARM_SPECS)} 에 없다 — "
            "런처가 ++algorithm.math_arm 을 넘겼는지 확인하라.")
    return MATH_ARM_SPECS[arm]


def judge_weight() -> float:
    """MATH_JUDGE_W (기본 0.5). ★워커에서 읽으므로 verl_sdc.main 의 Ray env 목록에 실려야 한다."""
    v = os.environ.get("MATH_JUDGE_W")
    return 0.5 if v is None else float(v)


def retry_weight() -> float:
    """MATH_RETRY_W (기본 0.5). 워커에서 읽는다 — verl_sdc.main 의 Ray env 목록에 실려야 한다."""
    v = os.environ.get("MATH_RETRY_W")
    return 0.5 if v is None else float(v)


def retry_len_cost() -> float:
    """MATH_RETRY_LEN_COST (기본 0.2) — «정답인데 redirect» 의 추가 토큰 1k 당 비용.
    ★왜 여기만 길이 비용인가: 틀린 답을 다시 푸는 토큰은 값을 하고, 맞은 답을 다시 푸는
    토큰은 순수 세금이다. 세금을 재시도 전체에 물리면 탐색 자체를 벌한다."""
    v = os.environ.get("MATH_RETRY_LEN_COST")
    return 0.2 if v is None else float(v)


def sl_weight() -> float:
    """MATH_SL_W (기본 0.5) — 결정 토큰 자기지도 항의 가중치. 워커에서 읽는다(Ray env 목록 필수)."""
    v = os.environ.get("MATH_SL_W")
    return 0.5 if v is None else float(v)


def sl_warmup_steps() -> int:
    """MATH_SL_WARMUP_STEPS (기본 10) — 이 스텝까지는 SL 가중치를 **2배**로 쓴다.
    ★왜 워밍업이 필요한가: vLLM 에이전트 루프엔 결정 토큰만 온도를 올릴 수단이 없다
    (MATH_DECISION_TEMP 는 구현 불가). 탐색을 못 키우니 «초기에 더 세게 민다»가 유일한
    대체 수단이다 — p(verify)≈1 인 첫 스텝들에서 verify 쪽 로짓을 빨리 끌어내린다."""
    v = os.environ.get("MATH_SL_WARMUP_STEPS")
    return 10 if v is None else int(v)


def sl_step_weight(step) -> float:
    """그 스텝에서 실제로 쓰는 SL 가중치(워밍업 구간은 2배)."""
    return sl_weight() * (2.0 if int(step or 0) <= sl_warmup_steps() else 1.0)


def agree_mon_weight() -> float:
    """MATH_AGREE_MON_W (기본 0.5) — M_AGREE 모니터링(자가 보정) 항 가중치. 워커에서 읽는다."""
    v = os.environ.get("MATH_AGREE_MON_W")
    return 0.5 if v is None else float(v)


def agree_ctl_weight() -> float:
    """MATH_AGREE_CTL_W (기본 0.5) — M_AGREE 통제(다수 일치 판단) 항 가중치. 워커에서 읽는다."""
    v = os.environ.get("MATH_AGREE_CTL_W")
    return 0.5 if v is None else float(v)


def crit_weight() -> float:
    """MATH_CRIT_W (기본 0.5) — 비평 정보이득 항의 가중치. 워커에서 읽는다(Ray env 목록 필수)."""
    v = os.environ.get("MATH_CRIT_W")
    return 0.5 if v is None else float(v)


def dis_weight() -> float:
    """MATH_DIS_W (기본 0.5) — 불일치 진단 팔의 자기증류 판단 크레딧 가중치.
    워커에서 읽는다 — verl_sdc.main 의 Ray env 목록에 실려야 한다."""
    v = os.environ.get("MATH_DIS_W")
    return 0.5 if v is None else float(v)


def diff_weight() -> float:
    """MATH_DIFF_W (기본 0.5) — 난이도 판단 팔의 방향 크레딧 가중치.
    워커에서 읽는다 — verl_sdc.main 의 Ray env 목록에 실려야 한다(드라이버 export 만으로는
    워커가 못 본다 — COUNTDOWN_INV 사고와 같은 모양)."""
    v = os.environ.get("MATH_DIFF_W")
    return 0.5 if v is None else float(v)


def crit_scale() -> float:
    """MATH_CRIT_SCALE (기본 0.05, 단위 nats/token) — (IG − IG_donor) 를 [−1,1] 로 사상하는 자.
    ★왜 0.05 인가: teacher-forcing 평균 로그확률의 조건 간 차이는 통상 1e-3~1e-1 nats/token 규모다
    (math_critique_ig_ruler 의 실측 범위). 자가 너무 크면 항이 늘 0 근처(무효 레버), 너무 작으면
    전 행이 ±1 로 포화해 «얼마나 좋은가»가 사라진다."""
    v = os.environ.get("MATH_CRIT_SCALE")
    return 0.05 if v is None else float(v)


# ── 채점: math_verify (scripts/local/math_rollout.py 와 **같은 함수**를 쓴다) ──────
_ANS_PREFIX_RE = re.compile(r"^(?:[a-zA-Z]|\\[a-zA-Z]+)\s*=\s*")
_TEXT_CMD_RE = re.compile(r"\\(?:text|mbox|textbf|textit|mathrm|mathbf)\s*\{([^{}]*)\}")
_FRAC_NOBRACE_RE = re.compile(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_BARE_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_DEG_RE = re.compile(r"(?:\^\s*\{?\s*\\circ\s*\}?|\\degree|\s*degrees?)\s*$")


def _norm_answer(s: str, *, gold_is_bare_num: bool = False, gold_has_prefix: bool = False) -> str:
    r"""채점용 문자열 정규화(의미 추론 없음 — 표기 차이만 지운다)."""
    t = str(s or "").strip()
    # 바깥 $ …$ / \(…\) / \[…\]
    for _ in range(3):
        t = t.strip()
        if len(t) >= 2 and t[0] == "$" and t[-1] == "$":
            t = t[1:-1]
        elif t.startswith("\\(") and t.endswith("\\)"):
            t = t[2:-2]
        elif t.startswith("\\[") and t.endswith("\\]"):
            t = t[2:-2]
        else:
            break
    t = t.strip()
    # 바깥 \boxed{...}
    for _ in range(3):
        m = _BOXED_RE.match(t)
        if not m:
            break
        got = _boxed_span_at(t, m)
        if not got or got[2] != len(t):
            break
        t = got[0].strip()
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for _ in range(3):
        nt = _TEXT_CMD_RE.sub(r"\1", t)
        if nt == t:
            break
        t = nt
    t = _FRAC_NOBRACE_RE.sub(r"\\frac{\1}{\2}", t)
    t = t.strip()
    while t.endswith("."):
        t = t[:-1].strip()
    t = t.strip()
    # "x=" 류 접두 — gold 에 없을 때만 벗긴다
    if not gold_has_prefix:
        t = _ANS_PREFIX_RE.sub("", t, count=1).strip()
    # 도 기호 / 퍼센트 — gold 가 맨숫자일 때만
    if gold_is_bare_num:
        t = _DEG_RE.sub("", t).strip()
        while t.endswith("%") or t.endswith("\\%"):
            t = t[:-2].strip() if t.endswith("\\%") else t[:-1].strip()
    # 통화/자릿점 — 순수 숫자에 한해
    t = t.replace("\\$", "").strip()
    if t.startswith("$"):
        t = t[1:].strip()
    if _THOUSANDS_RE.match(t):
        t = t.replace(",", "")
    t = re.sub(r"\s+", "", t)
    return t


# ── 객관식 선택지 표식(«(E)» 류) 벗기기 ──────────────────────────────────────
# `\boxed{\text{(E)}\ \dfrac{7}{2}}` 처럼 **문항 기호가 붙은 답**이 gold `\dfrac{7}{2}` 와
# math_verify·문자열 양쪽에서 갈려 정답이 0 으로 찍히던 거짓음성을 막는다(습관 계측의 가짜
# 정→오, 학습의 가짜 «파괴» 벌점). 표식을 벗긴 나머지(비면 글자 자체)는 **기존 경로가 전부
# 실패한 뒤에만** 추가로 시도하므로 기존 판정은 한 건도 뒤집히지 않는다(오답→정답만 가능).
_CHOICE_TEXT_RE = re.compile(r"^\\(?:text|textbf|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}")
_CHOICE_INNER_RE = re.compile(r"^\(\s*([A-Ea-e])\s*\)\s*[:.,]?$|^([A-Ea-e])\s*[).]\s*[:,]?$")
_CHOICE_BARE_RE = re.compile(r"^\(\s*([A-Ea-e])\s*\)|^([A-Ea-e])\s*\)")
_CHOICE_SEP_RE = re.compile(r"^(?:\\[,;:!\s]|\\quad|\\qquad|~|\s|[:,.])+")


def strip_choice_marker(s: str):
    r"""답 문자열 맨 앞의 객관식 표식을 (글자, 나머지)로 가른다. 표식이 없으면 None.
    ★맨 글자(`C`)는 표식으로 보지 않는다 — 괄호나 `)` 가 있어야 한다. 숫자 gold 에 대한
    `\boxed{C}` 는 지금도 오답이어야 하기 때문이다(자가검사 케이스)."""
    t = str(s or "").strip()
    m = _CHOICE_TEXT_RE.match(t)
    if m:
        mi = _CHOICE_INNER_RE.match(m.group(1).strip())
        if not mi:
            return None
        letter = mi.group(1) or mi.group(2)
    else:
        m = _CHOICE_BARE_RE.match(t)
        if not m:
            return None
        letter = m.group(1) or m.group(2)
    return letter.upper(), _CHOICE_SEP_RE.sub("", t[m.end():]).strip()


def grade_math(pred_text: str, gold: str) -> int:
    """math_verify parse+verify(관대한 gold 래핑 + 표기 정규화 문자열 동일). 예외는 0.

    0915 수리: `parse(gold)` 가 구분자 없는 맨 gold('t^7', '4x + 18', '\\csc 10')에서
    빈 리스트를 내 **정답을 0 으로** 만들던 거짓음성(측정 566/6400, 1207/12000)을 막는다.
    오답을 받아주지 않도록 의미 휴리스틱은 넣지 않는다(math_verify + 표기 정규화뿐)."""
    from math_verify import parse, verify  # noqa: PLC0415

    g = str(gold)
    pred = str(pred_text or "")

    def _p(x: str):
        try:
            return parse(x)
        except Exception:
            return []

    def _v(gp, pp) -> int:
        if not gp or not pp:
            return 0
        try:
            return int(bool(verify(gp, pp)))
        except Exception:
            return 0

    pp = _p(pred)
    # (a) 현행 동작
    if _v(_p(g), pp):
        return 1
    # (b)/(c) gold 래핑. ★괄호 없는 쉼표 목록 gold('1,2')는 math_verify 가 **집합**으로
    #   읽어 순서를 잃는다 — pred 가 괄호 쌍('(2,1)')이면 순서쌍 의도이므로 래핑 경로를 끈다.
    boxed = last_boxed(pred)
    bare_list_gold = ("," in g) and not any(ch in g for ch in "()[]{}")
    if bare_list_gold and boxed and boxed.strip()[:1] in "([":
        gold_wrapped = []
    else:
        gold_wrapped = [_p(f"${g}$"), _p("\\boxed{" + g + "}")]
    for gp in gold_wrapped:
        if _v(gp, pp):
            return 1
    # (d) pred parse 가 비었으면 박스 내용물을 같은 방식으로 감싸 재시도
    if not pp and boxed:
        for pw in (_p(f"${boxed}$"), _p("\\boxed{" + boxed + "}")):
            for gp in (_p(g), *gold_wrapped):
                if _v(gp, pw):
                    return 1
    # (e) 표기 정규화 문자열 동일
    if boxed:
        gold_is_num = bool(_BARE_NUM_RE.match(_norm_answer(g)))
        gold_has_prefix = bool(_ANS_PREFIX_RE.match(str(g).strip()))
        ng = _norm_answer(g, gold_is_bare_num=gold_is_num, gold_has_prefix=gold_has_prefix)
        np_ = _norm_answer(boxed, gold_is_bare_num=gold_is_num, gold_has_prefix=gold_has_prefix)
        if ng and ng == np_:
            return 1
    # (f) 객관식 표식 — 표식을 벗긴 나머지(또는 글자 자체)로 한 번 더
    got = strip_choice_marker(boxed) if boxed else None
    if got:
        letter, rest = got
        for cand in ([rest] if rest else []) + [letter]:
            if grade_math("\\boxed{" + cand + "}", g):
                return 1
    return 0


def selftest_math_verify() -> None:
    """math_verify timeout 래퍼가 워커 스레드에서 정답을 조용히 오답으로 만드는 함정
    (`scripts/patch_math_verify.py`)이 있다 — 깨져 있으면 즉사한다."""
    cases = [("\\boxed{42}", "42", 1), ("\\boxed{\\frac{1}{2}}", "0.5", 1),
             ("\\boxed{7}", "42", 0),
             ("\\boxed{t^7}", "t^7", 1),
             ("\\boxed{4x + 18}", "4x + 18", 1),
             ("\\boxed{\\csc 10}", "\\csc 10", 1),
             ("\\boxed{\\frac83}", "\\frac{8}{3}", 1),
             ("\\boxed{\\$347}", "347", 1),
             ("\\boxed{153}", "306", 0),
             ("\\boxed{24}", "243", 0),
             ("\\boxed{C}", "0.20", 0),
             ("\\boxed{(-5,7)}", "(-5,1)", 0),
             ("\\boxed{15}", "30^\\circ", 0)]
    got = [grade_math(p, g) for p, g, _ in cases]
    want = [e for *_, e in cases]
    if got != want:
        raise RuntimeError(
            f"math_verify 자가검사 실패: got={got} want={want} — "
            "scripts/patch_math_verify.py 를 먼저 적용하라(조용한 오채점 방지).")


# ★박스 스캐너는 **하나**다(0914 검증 ②): first/last/has_second_attempt/boxed_in_meta 가 전부 이
#   `_BOXED_RE`(\boxed 와 { 사이 공백 허용)를 쓴다. 예전엔 first/last 가 "\\boxed{" 문자열 검색,
#   나머지가 정규식이라 "\boxed {7}" 은 «두 번째 시도 있음»인데 «첫 답 없음»으로 갈렸다.
_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _boxed_span_at(t: str, m: re.Match) -> tuple[str, int, int] | None:
    r"""_BOXED_RE 매치 m 에서 (내용, 시작, 닫힘 뒤 오프셋). 안 닫히면 None."""
    j, depth = m.end(), 1
    while j < len(t) and depth:
        depth += (t[j] == "{") - (t[j] == "}")
        j += 1
    return (t[m.end(): j - 1].strip(), m.start(), j) if depth == 0 else None


def boxed_spans(text: str, *, exclude: Sequence[tuple[int, int]] = ()) -> list[tuple[str, int, int]]:
    r"""텍스트 안 모든 균형 \boxed{...} 의 (내용, 시작, 끝). `exclude` 구간(예: 메타 블록) 안에서
    시작하는 박스는 뺀다."""
    t = text or ""
    out = []
    for m in _BOXED_RE.finditer(t):
        if any(a <= m.start() < b for a, b in exclude):
            continue
        got = _boxed_span_at(t, m)
        if got:
            out.append(got)
    return out


def last_boxed(text: str) -> str:
    r"""마지막 \boxed{...} 안의 문자열(중괄호 균형; 없거나 안 닫히면 "")."""
    sp = boxed_spans(text)
    return sp[-1][0] if sp else ""


def first_boxed(text: str) -> str:
    r"""첫 \boxed{...} 안의 문자열(중괄호 균형; 없거나 안 닫히면 "")."""
    sp = boxed_spans(text)
    return sp[0][0] if sp else ""


def _meta_block_spans(text: str) -> list[tuple[int, int]]:
    """전체 텍스트의 모든 <meta>…</meta> 구간(첫 블록만이 아니다 — 누출 가드·답 제외에 쓴다)."""
    return [(m.start(), m.end()) for m in _cdr._META_BLOCK.finditer(text or "")]


# ★0914 수리: 콜론은 선택(스펙 지시 — "Second attempt" 만으로도 표식으로 인정).
_SECOND_ATTEMPT_RE = re.compile(r"second\s+attempt\s*:?", re.IGNORECASE)


def split_attempts(text: str) -> dict:
    r"""M_RETRY 행 분해 — 첫 답 / 그 **뒤**의 메타 블록 / 두 번째 시도 / 최종 답.

    ★메타는 첫 \boxed 뒤 구간에서만 파싱한다(form="math"). 첫 답 앞의 블록은 «답을 판단한
    것»이 아니므로 위치 위반 — 발화로 치지 않는다(n_blocks 는 전체 텍스트에서 세므로 다중
    블록 형식 위반은 기존 multi_block 경로가 잡는다).
    ★첫/최종 답은 **<meta>…</meta> 밖의** \boxed 에서만 고른다(0914 검증 ③). 메타 안 박스가 첫
    답으로 읽히면 «첫 답 → 메타» 순서가 뒤집혀 emitted 가 0 이 되고 boxed_in_meta 가드가 첫 답
    분리에 종속돼 무력해진다. 그래서 boxed_in_meta 는 첫 답 분리와 **무관하게** 전체 텍스트의
    블록(parse_meta(text,"math")["raw"] 및 모든 블록)에서 잰다.
    has_second_attempt = 블록 뒤에 "Second attempt" 표식(콜론 선택)이 있고, **그 표식 뒤에** \boxed
  가 있다(0914 수리 — 종전엔 표식 없이 블록 뒤 \boxed 만 있어도 두 번째 시도로 셌다: verify 행이
  답을 그냥 재진술한 \boxed 를 «재시도»로 오채점해 smoke 텔레메트리에서 second_attempt=.80 인데
  redirect=.002 인 모순이 났다. 표식만 있고 \boxed 가 없으면(잘림 등) 시도가 미완이므로 0 —
  최종 답은 여전히 첫 답으로 떨어진다).
    반환: first_answer / final_answer(메타 밖 첫/마지막 \boxed, 없으면 None), meta(parse_meta
    dict, 오프셋은 전체 텍스트 기준), n_blocks(전체), has_second_attempt, boxed_in_meta,
    n_chars_after_meta.
    """
    t = text or ""
    blocks = _meta_block_spans(t)
    whole = _cdr.parse_meta(t, "math")
    n_blocks_total = int(whole["n_blocks"])
    boxed_in_meta = int(bool(_BOXED_RE.search(whole.get("raw") or ""))
                        or any(_BOXED_RE.search(t[a:b]) for a, b in blocks))
    spans = boxed_spans(t, exclude=blocks)
    if not spans:
        first_answer, final_answer, fb_end = None, None, None
    else:
        first_answer, _, fb_end = spans[0]
        final_answer = spans[-1][0]
    if fb_end is None:
        m = _cdr._empty_meta()
    else:
        m = _cdr.parse_meta(t[fb_end:], "math")
        if m["start"] is not None:
            m["start"] += fb_end
            m["end"] += fb_end
    after = t[m["end"]:] if m["end"] is not None else ""
    # ★0914 수리: marker AND marker 뒤 \boxed 둘 다 있어야 두 번째 시도다(위 docstring 참조).
    marker_m = _SECOND_ATTEMPT_RE.search(after) if m["end"] is not None else None
    has_second = bool(marker_m and _BOXED_RE.search(after, marker_m.end()))
    return {
        "first_answer": first_answer,
        "final_answer": final_answer,
        "meta": m,
        "n_blocks": n_blocks_total,
        "has_second_attempt": int(has_second),
        "boxed_in_meta": boxed_in_meta,
        "n_chars_after_meta": len(after),
    }


# ── 판단 라벨표 ──────────────────────────────────────────────────────────────
def norm_problem(p: str) -> str:
    """라벨표 키. 공백 정규화만 — 문제 텍스트는 parquet 과 math_sites 가 같은 원문을 쓴다."""
    return re.sub(r"\s+", " ", str(p or "")).strip()


def load_judge_labels(path: str | None = None) -> dict[str, str]:
    """MATH_JUDGE_LABELS(json) → {norm_problem: best_decision}. 없으면 빈 표(항 전부 0).

    형식 두 가지를 받는다: {problem: decision} 사전, 또는 math_sites 의 sites.jsonl 을
    모은 [{problem, best_decision}, ...] 리스트. 값이 verify/redirect/tie 가 아니면 버린다.
    """
    p = path if path is not None else os.environ.get("MATH_JUDGE_LABELS", "")
    if not p:
        return {}
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw.items() if isinstance(raw, dict) else \
        ((r.get("problem"), r.get("best_decision")) for r in raw)
    out = {}
    for k, v in items:
        if v in ("verify", "redirect", "tie") and k:
            out[norm_problem(k)] = v
    return out


def check_labels_cover(parquet_path: str, labels_path: str) -> dict:
    """★감사 2: 라벨표 키가 학습 parquet 의 문제와 교집합이 0 이면 판단 항은 전 행 0 —
    M_JUDGE/M_RAND 가 M_G1 과 바이트 동일해진다(무효 레버). 발사 전에 즉사시킨다.
    반환 {n_train, n_labels, n_cover}. `problem` 컬럼이 없으면 extra_info.problem 을 쓴다."""
    import pandas as pd  # noqa: PLC0415
    df = pd.read_parquet(parquet_path)
    if "problem" in df.columns:
        probs = [norm_problem(x) for x in df["problem"].tolist()]
    elif "extra_info" in df.columns:
        probs = [norm_problem((e or {}).get("problem")) for e in df["extra_info"].tolist()]
    else:
        raise RuntimeError(f"[MATH][LABELS] {parquet_path}: problem/extra_info 컬럼이 없다: {list(df.columns)}")
    probs_set = {x for x in probs if x}
    labels = load_judge_labels(labels_path)
    n_cover = len(probs_set & set(labels))
    st = {"n_train": len(probs_set), "n_labels": len(labels), "n_cover": n_cover}
    if n_cover == 0:
        raise RuntimeError(
            f"[MATH][LABELS] 라벨 키가 학습 문제와 교집합이 없다: n_train={st['n_train']} "
            f"n_labels={st['n_labels']} n_cover=0 (parquet={parquet_path}, labels={labels_path}). "
            "math_sites 의 문제 텍스트와 build_math_parquet 의 문제 텍스트가 같은 원문인지 확인하라.")
    return st


def judgment_term(emitted, decision, best_decision) -> float:
    """진리표 — 모듈 docstring 참조."""
    if not _cdr._bool01(emitted) or decision not in _OPPOSITE or best_decision not in _OPPOSITE:
        return 0.0
    return 1.0 if decision == best_decision else -1.0


def answers_equivalent(first_answer, final_answer) -> bool:
    r"""«베낌» 판정 = 문자열 동일 ∨ math_verify 동치(7 ≡ 7.0, 0.5 ≡ \frac{1}{2}).
    ★왜 동치인가(0914 검증 ①): 문자열만 보면 표기만 바꾼 재시도가 «변경»으로 +크레딧을 받는다.
    동치 판정기가 예외를 내면 문자열 비교로 폴백한다(조용히 «변경»으로 읽히지 않게)."""
    a, b = str(first_answer or "").strip(), str(final_answer or "").strip()
    if a == b:
        return True
    if not a or not b:
        return False
    try:
        return bool(grade_math(f"\\boxed{{{b}}}", a))
    except Exception:
        return False


_LOOSE_WRAP_RE = re.compile(r"\\left|\\right|\\text\{[^{}]*\}|[\{\}]")
_LOOSE_SPLIT_RE = re.compile(r",|\\text\{\s*and\s*\}|\band\b")


def _loose_items(s: str) -> list[str]:
    """`answers_equivalent_loose` 보조 — 래퍼(`\\{ \\} \\left \\right \\text{...}`) 를 벗기고
    `,` / `and` / `\\text{ and }` 로 쪼갠 뒤 공백을 정리한 항 리스트."""
    stripped = _LOOSE_WRAP_RE.sub(" ", s)
    parts = [p.strip() for p in _LOOSE_SPLIT_RE.split(stripped)]
    return [p for p in parts if p]


def answers_equivalent_loose(a, b) -> bool:
    r"""«표기만 다른 재시도» 판정 — `answers_equivalent` 이거나, 둘을 각각 `\{ \} \left \right
    \text{...}` 래퍼를 벗기고 `,` / `and` / `\text{ and }` 로 쪼갠 항 다중집합이 서로 **쌍대응**
    (pairwise `answers_equivalent`) 하면 True. 예: `0 \text{ and } -3` ≡ `\{-3, 0\}` ≡ `0, -3`.
    항 개수가 다르거나 매칭이 안 되면 False — `3` vs `5`, `(2,1)` vs `(1,2)` 는 여전히 다르다
    (괄호 안 콤마는 항 구분자로 안 쪼갠다 — `_loose_items` 는 최상위 문자열만 나눈다는 뜻이 아니라
    실제로는 문자열 전체를 `,` 로 쪼개므로, 괄호 좌표 값은 애초에 항이 여러 개가 되어 개수가
    안 맞으면 자연히 False 가 된다)."""
    if answers_equivalent(a, b):
        return True
    sa, sb = str(a or "").strip(), str(b or "").strip()
    if not sa or not sb:
        return False
    ia, ib = _loose_items(sa), _loose_items(sb)
    if len(ia) < 2 and len(ib) < 2:
        return False  # 둘 다 단일 항이면 이미 위에서 answers_equivalent 로 끝났어야 한다
    if len(ia) != len(ib):
        return False
    remaining = list(ib)
    for x in ia:
        for i, y in enumerate(remaining):
            if answers_equivalent(x, y):
                del remaining[i]
                break
        else:
            return False
    return True


def retry_judgment_parts(first_correct, decision, has_second_attempt, final_answer, first_answer,
                         *, final_correct, extra_tokens: float = 0.0,
                         len_cost: float | None = None) -> tuple[float, float, int]:
    """재시도 판단 진리표(모듈 docstring) → (base, cost, 정의됨 0/1); 항 = base − cost.
    ★base/cost 를 갈라 돌려주는 이유: M_RETRY_RAND 는 base 의 부호만 뒤집고 cost 는 그대로 둔다."""
    if decision not in _OPPOSITE:
        return 0.0, 0.0, 0
    if len_cost is None:
        len_cost = retry_len_cost()
    right = bool(_cdr._bool01(first_correct))
    redirected = decision == "redirect" and bool(_cdr._bool01(has_second_attempt))
    if decision == "verify":
        return (1.0 if right else -1.0), 0.0, 1
    # decision == redirect
    if right:
        cost = len_cost * (float(extra_tokens) / 1000.0) if redirected else 0.0
        return -1.0, cost, 1
    if redirected and not answers_equivalent(first_answer, final_answer):
        # ★수정 3b: 판단 크레딧 +0.5 는 «틀렸다고 보고 바꿨다»에, 나머지 +0.5 는 구제했을 때만.
        return 0.5 + 0.5 * float(_cdr._bool01(final_correct)), 0.0, 1
    return 0.0, 0.0, 1                  # 말만 하고 베낌(또는 선언만) — 정의는 되지만 0


def retry_judgment_term(first_correct, decision, has_second_attempt, final_answer, first_answer,
                        *, final_correct, extra_tokens: float = 0.0,
                        len_cost: float | None = None) -> tuple[float, int]:
    """재시도 판단 진리표 → (항, 정의됨 0/1). decision 없음 → (0, 0). `final_correct` 는 필수
    (수정 3b — 기본값을 두면 구제 크레딧이 조용히 0 으로 빠진다)."""
    base, cost, defined = retry_judgment_parts(
        first_correct, decision, has_second_attempt, final_answer, first_answer,
        final_correct=final_correct, extra_tokens=extra_tokens, len_cost=len_cost)
    return base - cost, defined


# ── M_CRIT: 비평 누출 가드 · 길이 · donor 배정 (단일 진실 원천) ──────────────────
# ★여기 세 함수는 원래 scripts/local/math_critique_resolve_gate.py 에만 있었다. 학습(이 파일)과
#   게이트·IG 자(그 스크립트들)가 **같은 누출 판정**을 쓰지 않으면 «게이트가 통과시킨 비평»과
#   «RL 이 보상한 비평»이 갈린다 — 그래서 정의를 이리로 옮기고 그 스크립트가 import 한다.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"\w+")


def _numbers(s: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(s or ""):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def critique_leaks(critique: str, answer: str) -> bool:
    r"""비평이 답을 흘렸는가. (a) \boxed 가 있다 (b) 답 문자열이 통째로 들어 있다
    (c) 비평의 숫자 토큰 하나가 답 안의 숫자 하나와 **값이 같다**.
    ★(c) 가 느슨한 쪽으로(=거짓 양성 쪽으로) 기운 것은 의도다 — 누출된 비평 하나는 «비평의 값»이
      아니라 «답을 다시 보여 준 값»을 잰다. 누출 행은 항 미정의(보상 0)로 빠진다."""
    c = critique or ""
    if _BOXED_RE.search(c):
        return True
    a = str(answer or "").strip()
    if a and a in c:
        return True
    an = _numbers(a)
    if not an:
        return False
    cn = _numbers(c)
    return any(any(abs(x - y) < 1e-9 for y in an) for x in cn)


def critique_len(critique: str) -> int:
    """비평 길이(단어 수) — 짧으면 파싱 실패, 길면 모델이 비평 자리에서 다시 풀어 버린 것이다."""
    return len(_WORD_RE.findall(critique or ""))


def assign_donors(recs: Sequence[Mapping], rng: random.Random) -> list[int | None]:
    """각 행에 **다른 문제**(group_id 가 다른) 행을 하나씩 배정한다 → 인덱스 리스트.
    셔플한 순열을 회전시켜 짝을 짓고, 같은 문제끼리 붙은 자리만 다음 후보로 민다.
    후보가 없으면(모두 같은 문제) None. ★math_critique_resolve_gate 가 이 함수를 import 한다 —
    donor 대조의 «다른 문제» 정의가 학습과 게이트에서 갈리면 두 산출물을 나란히 읽을 수 없다."""
    n = len(recs)
    if n < 2:
        return [None] * n
    order = list(range(n))
    rng.shuffle(order)
    donors: list[int | None] = [None] * n
    for pos, i in enumerate(order):
        for step in range(1, n):
            j = order[(pos + step) % n]
            if recs[j]["group_id"] != recs[i]["group_id"]:
                donors[i] = j
                break
    return donors


# ── M_CRIT: 행 · 정보이득 항 ────────────────────────────────────────────────────
def truncate_to_first_boxed(text: str) -> str:
    r"""첫 \boxed{...} 의 **닫는 중괄호까지** 자른 텍스트(박스가 없으면 원문 그대로).
    ★S+ (정답 형제)의 표적 구간 정의다. 박스 뒤의 메타 블록·잡담까지 표적에 넣으면 IG 가
    «풀이가 얼마나 쉬워졌나»가 아니라 «메타 형식을 얼마나 잘 맞추나»를 잰다."""
    sp = boxed_spans(text or "")
    return (text or "")[:sp[0][2]] if sp else (text or "")


def parse_crit_row(text: str, gold: str, problem: str, *, truncated: int = 0,
                   tok_len_fn: Callable[[str], int] | None = None) -> dict:
    """M_CRIT 한 롤아웃의 원재료 — parse_retry_row 의 키 전부 + critique/leaked/crit_words.

    ★`first_correct` 가 이 팔의 답 스팬 보상이다(블록 앞의 **유일한** \\boxed). 프롬프트가 두 번째
    시도를 금지하므로 final==first 가 정상이고, 어기고 두 번째 박스를 쓴 행은 final_answer 가
    달라져 텔레메트리(second_attempt_rate)에 드러난다 — 그래도 보상은 첫 답으로 준다(계약이 그렇다).
    ★critique 는 parse_meta 의 `body` = 블록에서 confidence:/decision: 줄을 뺀 문장들이다.
    """
    r = parse_retry_row(text, gold, problem, truncated=truncated, tok_len_fn=tok_len_fn)
    crit = (r.get("body") or "").strip()
    r["critique"] = crit
    r["crit_words"] = critique_len(crit)
    r["leaked"] = int(bool(crit) and critique_leaks(crit, r.get("first_answer")))
    r["r_corr"] = int(r["first_correct"])
    return r


def select_s_plus(rows: Sequence[Mapping], keys: Sequence) -> list[int | None]:
    """그룹(uid)마다 **가장 짧은 정답·미잘림 롤아웃**의 인덱스 → 그 그룹의 모든 행에 같은 값.
    없으면 None. ★그룹 단위인 이유는 모듈 docstring(같은 표적이어야 그룹 중심화가 «비평의 질»을
    비교한다 + plain 채점 캐시). 동률이면 배치 순서가 빠른 쪽."""
    if len(keys) != len(rows):
        raise RuntimeError(f"[MATH][CRIT] keys 길이 {len(keys)} != 행 {len(rows)}")
    # ★0914 점검 수리: 정답 행이 «자기 자신»을 표적으로 삼으면 «내 비평이 내 풀이를 잘 설명하는가»가
    #   보상이 되어 자기 서술로 IG 를 올리는 구멍이 생긴다. 표적은 반드시 **자기 아닌** 형제.
    #   그룹마다 정답·미잘림 롤아웃을 길이순으로 최대 2개 보관해, 최단이 자기 자신이면 차순위를 쓴다.
    ranked: dict = {}
    for i, (r, k) in enumerate(zip(rows, keys)):
        k = str(k)
        if not int(_cdr._bool01(r.get("first_correct", 0))) or _cdr._bool01(r.get("truncated", 0)):
            continue
        lst = ranked.setdefault(k, [])
        lst.append((len(r.get("text") or ""), i))
        lst.sort()
        del lst[2:]
    out: list = []
    for i, k in enumerate(keys):
        cands = [j for _, j in ranked.get(str(k), []) if j != i]
        out.append(cands[0] if cands else None)
    return out


def crit_term(ig: float, ig_donor: float, scale: float | None = None) -> float:
    """항 = clip((IG − IG_donor)/scale, −1, +1). 유한하지 않으면 0."""
    if scale is None:
        scale = crit_scale()
    try:
        v = (float(ig) - float(ig_donor)) / float(scale)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    if not _cdr._finite(v):
        return 0.0
    return max(-1.0, min(1.0, v))


def crit_prompt_pair(scorer, variant: str, problem: str, critique: str | None) -> str:
    """채점 문맥. critique 가 None 이면 **생성 프롬프트와 바이트 동일**(plain), 있으면 그 끝에
    NOTE_PREFIX + 비평만 붙는다 — math_critique_resolve_gate.resolve_prompt 와 같은 규약이다
    (조건 사이의 유일한 차이가 비평 문자열이어야 «비평의 효과»가 «프롬프트가 달라진 효과»와
    안 섞인다). 조립은 스코어러가 들고 있는 토크나이저로 한다."""
    from src.metacot.math_meta_prompt import (  # noqa: PLC0415
        build_math_prompt, render_chat_messages, render_generation_prompt,
    )
    tok = scorer.tokenizer
    if critique is None:
        return render_generation_prompt(tok, variant, problem)
    msgs = build_math_prompt(problem, variant)
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + NOTE_PREFIX + str(critique).strip()}
    return render_chat_messages(tok, msgs)


def annotate_crit_ig(rows: Sequence[Mapping], keys: Sequence, *, scorer, variant: str,
                     rng: random.Random | None = None, weight: float | None = None,
                     scale: float | None = None) -> Sequence[Mapping]:
    """★M_CRIT 의 메타 스팬 항을 행에 채운다(`_compute_math_arm_stash` 가 배치당 한 번 부른다).

    행 조건: 발화 ∧ 단일 블록 ∧ 비평 존재 ∧ 누출 아님 ∧ 그룹에 정답 형제 S+ 있음 ∧ donor 배정됨.
    그 행마다 세 점수를 얼어붙은 채점기에서 얻는다 — plain(그룹당 한 번, 캐시) / note / donor.
    """
    w = crit_weight() if weight is None else float(weight)
    sc = crit_scale() if scale is None else float(scale)
    rng = rng or random.Random(0)
    sib = select_s_plus(rows, keys)
    for r in rows:
        r.setdefault("best_decision", None)
        r["multi_block"] = int(int(r.get("n_blocks", 0)) > 1)
        r["crit_ok"] = int(bool(_cdr._bool01(r.get("emitted", 0))) and not r["multi_block"]
                           and bool((r.get("critique") or "").strip())
                           and not _cdr._bool01(r.get("leaked", 0)))
        r["ig"] = r["ig_donor"] = r["ig_delta"] = float("nan")
        r["crit_term"] = 0.0
        r["judge"] = 0.0                   # SAMPLE 로그·judge_match 가 읽는 이름 유지
        r["meta_val"] = 0.0
        r["meta_defined"] = 0
        r["answer_total"] = float(_cdr._bool01(r.get("first_correct", 0)))
        r["s_plus_idx"] = None
    cand = [i for i, r in enumerate(rows) if int(r["crit_ok"]) and sib[i] is not None]
    if not cand:
        print("[MATH][CRIT] 정의된 행 0 — 비평·정답 형제가 있는 행이 없다(항 전부 미정의).",
              flush=True)
        return rows
    donors = assign_donors([{"group_id": str(keys[i])} for i in cand], rng)
    # ── 채점 요청 조립: plain 은 그룹당 하나(캐시), note/donor 는 행마다 하나 ──
    prompts: list[str] = []
    targets: list[str] = []
    # ★0914 점검 수리: 표적(S+)이 자기 자신을 제외하므로 그룹 안에서 행마다 다를 수 있다.
    #   plain 캐시와 표적 텍스트는 uid 가 아니라 «표적 행 인덱스» 로 키를 잡는다.
    plain_at: dict[int, int] = {}
    note_at: dict[int, int] = {}
    donor_at: dict[int, int] = {}
    s_plus_text: dict[int, str] = {}
    for pos, i in enumerate(cand):
        if donors[pos] is None:
            continue                      # 배치에 다른 문제가 없다 — 내용 대조가 불가능하므로 미정의
        t = int(sib[i])
        if t not in plain_at:
            s_plus_text[t] = truncate_to_first_boxed(rows[t].get("text") or "")
            plain_at[t] = len(prompts)
            prompts.append(crit_prompt_pair(scorer, variant, rows[i]["problem"], None))
            targets.append(s_plus_text[t])
        note_at[i] = len(prompts)
        prompts.append(crit_prompt_pair(scorer, variant, rows[i]["problem"], rows[i]["critique"]))
        targets.append(s_plus_text[t])
        dj = cand[donors[pos]]
        donor_at[i] = len(prompts)
        prompts.append(crit_prompt_pair(scorer, variant, rows[i]["problem"], rows[dj]["critique"]))
        targets.append(s_plus_text[t])
        rows[i]["s_plus_idx"] = t
        rows[i]["donor_idx"] = int(dj)
    if not prompts:
        print("[MATH][CRIT] donor 배정 가능한 행이 0 — 배치에 문제가 하나뿐인가(항 전부 미정의).",
              flush=True)
        return rows
    scores = list(scorer.score_meanlogp(prompts, targets))
    if len(scores) != len(prompts):
        raise RuntimeError(f"[MATH][CRIT] 채점기가 {len(scores)} 개를 돌려줬다(요청 {len(prompts)}).")
    n_def = 0
    for i in note_at:
        t = int(rows[i]["s_plus_idx"])
        plain, note, don = scores[plain_at[t]], scores[note_at[i]], scores[donor_at[i]]
        if not all(_cdr._finite(x) for x in (plain, note, don)):
            continue                      # 채점 실패 행은 미정의(0 으로 읽히면 «효과 없음»이 된다)
        r = rows[i]
        r["ig"] = float(note) - float(plain)
        r["ig_donor"] = float(don) - float(plain)
        r["ig_delta"] = r["ig"] - r["ig_donor"]
        r["crit_term"] = crit_term(r["ig"], r["ig_donor"], sc)
        r["judge"] = r["crit_term"]
        r["meta_val"] = w * r["crit_term"]
        r["meta_defined"] = 1
        n_def += 1
    print(f"[MATH][CRIT] scored={len(prompts)} groups={len(plain_at)} defined_rows={n_def}/{len(rows)} "
          f"w={w:.3f} scale={sc:.4f}", flush=True)
    return rows


def _compute_crit_rows(texts, golds, problems, spec, *, rng, uids, truncated, tok_len_fn) -> list[dict]:
    """M_CRIT 의 행. **정보이득 항은 여기서 채우지 않는다** — 얼어붙은 채점기(GPU)는 트레이너
    프로세스에만 있으므로 `verl_sdc._compute_math_arm_stash` 가 `annotate_crit_ig` 를 이어서
    부른다(annotate_sl_rows 와 같은 규약). 여기서는 파싱·기본값(미정의)만 낸다."""
    n = len(texts)
    trunc = list(truncated) if truncated is not None else [0] * n
    if len(trunc) != n:
        raise RuntimeError(f"[MATH][CRIT] truncated 길이 {len(trunc)} != 행 {n}")
    rows = [parse_crit_row(t, g, p, truncated=tr, tok_len_fn=tok_len_fn)
            for t, g, p, tr in zip(texts, golds, problems, trunc)]
    keys = [str(u) for u in uids] if uids is not None else [norm_problem(r["problem"]) for r in rows]
    if len(keys) != n:
        raise RuntimeError(f"[MATH][CRIT] uids 길이 {len(keys)} != 행 {n}")
    for r, k in zip(rows, keys):
        r["best_decision"] = None
        r["uid"] = k
        r["group_id"] = k
        r["forced_redirect"] = 0
        r["multi_block"] = int(int(r["n_blocks"]) > 1)
        r["redirected"] = 0               # ★이 팔엔 문맥 안 재시도가 없다(프롬프트가 금지).
        r["crit_ok"] = int(bool(_cdr._bool01(r["emitted"])) and not r["multi_block"]
                           and bool(r["critique"]) and not r["leaked"])
        r["meta_val"] = 0.0
        r["meta_defined"] = 0
        r["judge"] = 0.0
        r["answer_total"] = float(r["first_correct"])
    return rows


# ── 외부 프로브 스코어러(M_PROBE) ────────────────────────────────────────────
_SCORER: dict = {"fn": None, "warned": False}


def set_meta_scorer(fn: Callable[[Sequence[Mapping]], Sequence[float]] | None) -> None:
    """fn(rows) -> [float]*len(rows). None 이면 기본(0, 한 번 경고)으로 되돌린다."""
    _SCORER["fn"] = fn
    _SCORER["warned"] = False


def scorer_registered() -> bool:
    """★inert 레버 (i): M_PROBE 발사 전 fail-closed 체크(verl_sdc._compute_math_arm_stash)가
    쓴다. 등록 안 됐으면 score_meta_probe 는 조용히 0 을 돌리므로(경고 한 번뿐) 발사자가
    모르고 M_PROBE 를 M_G1 과 바이트 동일한 무효 레버로 돌릴 수 있다 — step 1 에 즉사시켜라."""
    return _SCORER["fn"] is not None


def score_meta_probe(rows: Sequence[Mapping]) -> list[float]:
    fn = _SCORER["fn"]
    if fn is None:
        if not _SCORER["warned"]:
            _SCORER["warned"] = True
            print("[MATH][PROBE][WARN] meta scorer 미등록 — M_PROBE 의 메타 항은 전부 0 이다. "
                  "set_meta_scorer(fn) 으로 꽂아라(이 경고는 한 번만 찍힌다).", flush=True)
        return [0.0] * len(rows)
    out = [float(x) for x in fn(rows)]
    if len(out) != len(rows):
        raise RuntimeError(f"[MATH][PROBE] scorer 가 {len(out)} 개를 돌려줬다(행 {len(rows)}).")
    return out


# ── 행 계산 ─────────────────────────────────────────────────────────────────
def parse_row(text: str, gold: str, problem: str) -> dict:
    """한 롤아웃의 원재료. 메타는 form="math"(decision 선택)."""
    m = _cdr.parse_meta(text or "", "math")
    raw = m.get("raw") or ""
    return {
        "text": text or "", "gold": str(gold), "problem": problem,
        "r_corr": grade_math(text or "", gold),
        "emitted": int(m["emitted"]), "n_blocks": int(m["n_blocks"]),
        "meta_start": m["start"], "meta_end": m["end"],
        "confidence": m["confidence"], "decision": m["decision"], "body": m["body"],
        # ★누출 가드: 메타 안에 \boxed 가 있으면 «접근 평가»가 아니라 답이다.
        "boxed_in_meta": int(bool(m["n_blocks"]) and bool(_BOXED_RE.search(raw))),
        "n_chars": len(text or ""),
    }


def parse_retry_row(text: str, gold: str, problem: str, *, truncated: int = 0,
                    tok_len_fn: Callable[[str], int] | None = None) -> dict:
    """M_RETRY 한 롤아웃의 원재료 — parse_row 와 같은 키 + first/final 답·정오·재시도 여부.
    `r_corr` 는 **최종 답**의 정오(답 스팬 보상). 첫 답은 `first_correct` 로 따로 낸다.
    `n_tok_after_meta`: tok_len_fn 이 있으면 실제 토큰 수, 없으면 chars/4 근사."""
    sp = split_attempts(text or "")
    m = sp["meta"]
    first_ans, final_ans = sp["first_answer"], sp["final_answer"]
    after = (text or "")[m["end"]:] if m["end"] is not None else ""
    n_after = (int(tok_len_fn(after)) if (tok_len_fn is not None and after)
               else int(round(sp["n_chars_after_meta"] / 4.0)))
    return {
        "text": text or "", "gold": str(gold), "problem": problem,
        "first_answer": first_ans, "final_answer": final_ans,
        "first_correct": grade_math(f"\\boxed{{{first_ans}}}", gold) if first_ans else 0,
        "final_correct": grade_math(f"\\boxed{{{final_ans}}}", gold) if final_ans else 0,
        "emitted": int(m["emitted"]), "n_blocks": int(sp["n_blocks"]),
        "meta_start": m["start"], "meta_end": m["end"],
        "confidence": m["confidence"], "decision": m["decision"], "body": m["body"],
        "has_second_attempt": int(sp["has_second_attempt"]),
        # 수학 동치 기준 «답이 바뀌었나»(둘 다 있을 때만 1) — 판단 항·held-out 곡선이 같은 정의를 쓴다
        "answer_changed": int(bool(first_ans) and bool(final_ans) and not answers_equivalent(first_ans, final_ans)),
        "boxed_in_meta": int(sp["boxed_in_meta"]),
        "n_chars": len(text or ""), "n_tok_after_meta": n_after,
        "truncated": int(_cdr._bool01(truncated)),
    }


# ── M_AGREE: agreement: 줄 파싱 ──────────────────────────────────────────────
# ★값이 아니라 «줄이 있는가»도 따로 낸다(agree_line) — 값이 범위를 벗어나 파싱 실패해도
#   «최소한 그 줄을 쓰려고 시도했다»는 형식 준수 신호는 다르다(중단 규칙 agree_line_rate).
_AGREE_LINE_RE = re.compile(r"agreement\s*:", re.IGNORECASE)
_AGREE_RE = re.compile(r"agreement\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def parse_agreement(raw: str) -> dict:
    """메타 블록 원문(raw, <meta>…</meta> 포함) → {agree_line: 0/1, agree_pred: float|None}.
    ★agree_pred 는 [0,1] 범위를 벗어나면 None(스펙 지시 — 파싱은 됐지만 값이 무효)."""
    t = raw or ""
    has_line = bool(_AGREE_LINE_RE.search(t))
    m = _AGREE_RE.search(t)
    val = None
    if m:
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                val = v
        except ValueError:
            val = None
    return {"agree_line": int(has_line), "agree_pred": val}


def parse_agree_row(text: str, gold: str, problem: str, *, truncated: int = 0,
                    tok_len_fn: Callable[[str], int] | None = None) -> dict:
    """M_AGREE 한 롤아웃의 원재료 — parse_retry_row 와 같은 키 전부 + agree_pred/agree_line.
    ★split_attempts 를 다시 부른다(parse_retry_row 안에서 이미 한 번 돎) — 형제 동의 예측은
    재시도 구조와 별개 관심사라 코드를 안 섞으려고 약간의 중복을 감수한다."""
    r = parse_retry_row(text, gold, problem, truncated=truncated, tok_len_fn=tok_len_fn)
    sp = split_attempts(text or "")
    ag = parse_agreement(sp["meta"].get("raw"))
    r["agree_pred"] = ag["agree_pred"]
    r["agree_line"] = ag["agree_line"]
    return r


def compute_agree_true(rows: Sequence[Mapping], keys: Sequence) -> list[float | None]:
    """행마다 «같은 uid 그룹의 다른 행들 중 first_answer 가 수학적으로 동치인 비율».
    자기 자신의 first_answer 가 없거나, 다른 행 중 first_answer 가 있는 것이 2개 미만이면 None
    (스펙 지시 — «형제가 2 미만»이면 다수/소수 자체가 안 정해진다)."""
    if len(keys) != len(rows):
        raise RuntimeError(f"[MATH][AGREE] keys 길이 {len(keys)} != 행 {len(rows)}")
    idx_by_key: dict = {}
    for i, k in enumerate(keys):
        idx_by_key.setdefault(k, []).append(i)
    out: list[float | None] = [None] * len(rows)
    for idxs in idx_by_key.values():
        for i in idxs:
            own = rows[i].get("first_answer")
            sibs = [rows[j]["first_answer"] for j in idxs if j != i and rows[j].get("first_answer")]
            if not own or len(sibs) < 2:
                out[i] = None
                continue
            n_eq = sum(1 for s in sibs if answers_equivalent(own, s))
            out[i] = n_eq / len(sibs)
    return out


def agree_monitoring_term(agree_pred, agree_true) -> tuple[float, int]:
    """모니터링 항 = 1 − 2·|agree_pred − agree_true| ∈ [−1,1]; 둘 중 하나라도 None 이면 (0,0)."""
    if agree_pred is None or agree_true is None:
        return 0.0, 0
    return 1.0 - 2.0 * abs(float(agree_pred) - float(agree_true)), 1


def agree_control_parts(agree_true, decision, has_second_attempt, final_answer, first_answer,
                        *, final_correct, extra_tokens: float = 0.0,
                        len_cost: float | None = None) -> tuple[float, float, int]:
    """M_AGREE 통제 진리표 — retry_judgment_parts 와 글자 그대로 같은 모양이되, «옳음»의
    기준이 첫 답 정오가 아니라 **다수 일치**(majority = agree_true≥0.5, 모듈 docstring 참조)다.
    agree_true 가 None 이면 미정의(0,0,0) — 형제가 모자라 다수/소수 자체가 안 정해진 행이다."""
    if decision not in _OPPOSITE or agree_true is None:
        return 0.0, 0.0, 0
    if len_cost is None:
        len_cost = retry_len_cost()
    majority = float(agree_true) >= 0.5
    redirected = decision == "redirect" and bool(_cdr._bool01(has_second_attempt))
    if decision == "verify":
        return (1.0 if majority else -1.0), 0.0, 1
    # decision == redirect
    if majority:
        cost = len_cost * (float(extra_tokens) / 1000.0) if redirected else 0.0
        return -1.0, cost, 1
    if redirected and not answers_equivalent(first_answer, final_answer):
        return 0.5 + 0.5 * float(_cdr._bool01(final_correct)), 0.0, 1
    return 0.0, 0.0, 1                  # 말만 하고 베낌(또는 선언만) — 정의는 되지만 0


def _group_sign_flips(keys: Sequence, rng: random.Random) -> dict:
    """★M_RETRY_RAND: uid 그룹마다 ±1 부호 하나(M_RAND 의 «그룹 단위 순열» 규약과 같은 이유 —
    행 단위로 뒤집으면 같은 그룹 안 항이 갈려 그룹 중심화 뒤 분산이 M_RETRY 와 달라진다)."""
    return {k: rng.choice((1.0, -1.0)) for k in sorted(set(map(str, keys)))}


def _permute_group_labels(best: list, keys: Sequence, rng: random.Random) -> list:
    """★감사 4: 그룹(uid)→라벨 표를 순열한다. 풀 = verify/redirect 라벨이 있는 그룹만
    (tie/None 은 그대로). 같은 그룹 안에 다른 문제·라벨이 섞여 있으면 즉사(규약 위반)."""
    by_key: dict = {}
    for k, b in zip(keys, best):
        by_key.setdefault(k, set()).add(b)
    mixed = {k for k, v in by_key.items() if len(v) > 1}
    if mixed:
        raise RuntimeError(f"[MATH][RAND] 같은 uid 그룹에 라벨(문제)이 둘 이상 섞였다: {sorted(map(str, mixed))[:5]}")
    pool = [k for k, v in by_key.items() if next(iter(v)) in _OPPOSITE]
    vals = [next(iter(by_key[k])) for k in pool]
    rng.shuffle(vals)
    table = dict(zip(pool, vals))
    return [table.get(k, b) for k, b in zip(keys, best)]


def compute_rows(texts: Sequence[str], golds: Sequence[str], problems: Sequence[str],
                 arm: str, *, labels: Mapping[str, str] | None = None,
                 rng: random.Random | None = None, uids: Sequence | None = None,
                 truncated: Sequence | None = None,
                 tok_len_fn: Callable[[str], int] | None = None,
                 forced_redirect: Sequence | None = None,
                 group_pass_rate: Sequence | None = None,
                 cand_answers: Sequence | None = None,
                 cand_correct: Sequence | None = None) -> list[dict]:
    """배치 전체의 행 + 메타 항. 반환 행마다 `answer_total`(시퀀스 보상)과
    `meta_val`(메타 스팬 전용 항), `meta_defined`(항이 정의된 행 = 그룹 중심화 member)이
    분리돼 있다.

    M_RAND: uid→best_decision **표**를 순열한다(`uids` 없으면 정규화 문제 텍스트가 그룹 키).
    라벨 분포는 그대로고 그룹↔라벨 대응만 깨진다 — 판단 «내용»만 제거한 대조군.
    ★감사 6: n_blocks>1 은 형식 위반 — 그 행의 메타 항은 0 이고 정의되지 않은 것으로 둔다.
    """
    spec = require_arm(arm)
    labels = labels or {}
    if spec["meta_term"] in _RETRY_TERMS:
        return _compute_retry_rows(texts, golds, problems, spec, rng=rng, uids=uids,
                                   truncated=truncated, tok_len_fn=tok_len_fn,
                                   forced_redirect=forced_redirect, group_pass_rate=group_pass_rate)
    if spec["meta_term"] in _AGREE_TERMS:
        return _compute_agree_rows(texts, golds, problems, spec, rng=rng, uids=uids,
                                   truncated=truncated, tok_len_fn=tok_len_fn,
                                   forced_redirect=forced_redirect, group_pass_rate=group_pass_rate)
    if spec["meta_term"] in _CRIT_TERMS:
        # ★정보이득 항은 채점기가 필요하다 — verl_sdc 가 annotate_crit_ig 로 이어서 채운다.
        return _compute_crit_rows(texts, golds, problems, spec, rng=rng, uids=uids,
                                  truncated=truncated, tok_len_fn=tok_len_fn)
    if spec["meta_term"] in _DIS_TERMS:
        # ★라벨은 배치 안(그룹 다수답)에서 나온다 — 표도 채점기도 필요 없다. 후보 답만 밖에서 온다.
        return _compute_dis_rows(texts, golds, problems, spec, uids=uids,
                                 cand_answers=cand_answers, cand_correct=cand_correct)
    if spec["meta_term"] in _DIFF_TERMS:
        # ★라벨은 배치 안(그룹의 LOO 형제 동의도)에서 나온다 — parquet 의 추가 컬럼도 필요 없다.
        return _compute_diff_rows(texts, golds, problems, spec, rng=rng, uids=uids)
    rows = [parse_row(t, g, p) for t, g, p in zip(texts, golds, problems)]
    best = [labels.get(norm_problem(r["problem"])) for r in rows]
    if spec["meta_term"] == "judge_shuffled":
        keys = [str(u) for u in uids] if uids is not None else [norm_problem(r["problem"]) for r in rows]
        if len(keys) != len(rows):
            raise RuntimeError(f"[MATH][RAND] uids 길이 {len(keys)} != 행 {len(rows)}")
        before = sum(1 for b in best if b in _OPPOSITE)
        best = _permute_group_labels(best, keys, rng or random.Random(0))
        n_lab = sum(1 for b in best if b in _OPPOSITE)
        print(f"[MATH][RAND] uid→label 표 순열: n_groups={len(set(keys))} "
              f"n_groups_in_pool={len({k for k, b in zip(keys, best) if b in _OPPOSITE})} "
              f"n_lab={n_lab} (before={before})", flush=True)
    for r, b in zip(rows, best):
        r["best_decision"] = b
        r["multi_block"] = int(int(r.get("n_blocks", 0)) > 1)
        r["judge"] = 0.0 if r["multi_block"] else judgment_term(r["emitted"], r["decision"], b)
        r["answer_total"] = float(r["r_corr"])
    if spec["meta_term"] in ("judge", "judge_shuffled"):
        w = judge_weight()
        for r in rows:
            r["meta_val"] = w * r["judge"]
            r["meta_defined"] = int(bool(_cdr._bool01(r["emitted"]) and not r["multi_block"]
                                        and r["decision"] in _OPPOSITE and r["best_decision"] in _OPPOSITE))
    elif spec["meta_term"] == "probe":
        for r, s in zip(rows, score_meta_probe(rows)):
            ok = bool(_cdr._bool01(r["emitted"]) and not r["multi_block"])
            r["meta_val"] = float(s) if ok else 0.0
            r["meta_defined"] = int(ok)
    else:
        for r in rows:
            r["meta_val"] = 0.0
            r["meta_defined"] = 0
    return rows


def _compute_retry_rows(texts, golds, problems, spec, *, rng, uids, truncated, tok_len_fn,
                        forced_redirect=None, group_pass_rate=None) -> list[dict]:
    """M_RETRY / M_RETRY_RAND 의 행. answer_total = 최종 답 정오, meta_val = W·판단 항(부호 반전
    은 RAND 만, uid 그룹 단위), meta_defined = 발화 ∧ decision ∧ 단일 블록.

    ★0914 forced-redirect 탐색: `forced_redirect[i]==1` 인 행은 build_math_parquet.py 의
    math_retry_forced 변형으로 생성됐다 — decision 이 정책 판단이 아니라 프롬프트 강제이므로
    judgment 항(meta_val/judge)은 **정의되지 않음**(meta_defined=0, base=cost=0) 처리한다.
    단 answer_total(최종 답 정오)은 정상 계산 — "redirect 가 실제로 답을 구할 수 있다"는
    신호는 강제 행에서도 학습해야 한다(스펙 지시). RAND(retry_shuffled)도 같은 처리를 공유
    한다(부호 반전 대상에서 자체적으로 빠진다 — base=0 이라 sign 을 곱해도 0)."""
    n = len(texts)
    trunc = list(truncated) if truncated is not None else [0] * n
    if len(trunc) != n:
        raise RuntimeError(f"[MATH][RETRY] truncated 길이 {len(trunc)} != 행 {n}")
    forced = ([int(_cdr._bool01(x)) for x in forced_redirect] if forced_redirect is not None
             else [0] * n)
    if len(forced) != n:
        raise RuntimeError(f"[MATH][RETRY] forced_redirect 길이 {len(forced)} != 행 {n}")
    # ★0914: build_math_parquet.py --forced_from_rollouts 가 extra_info.group_pass_rate 로 심은
    #   오프라인 롤아웃 pass rate. 있으면 retry_telemetry 의 within-problem 선택성 지표(mixed 판정)
    #   가 배치-로컬 first_correct 대신 이 값을 쓴다 — 배치에 문제당 샘플이 적어도 mixed 여부를 안다.
    gpr = (list(group_pass_rate) if group_pass_rate is not None else [None] * n)
    if len(gpr) != n:
        raise RuntimeError(f"[MATH][RETRY] group_pass_rate 길이 {len(gpr)} != 행 {n}")
    rows = [parse_retry_row(t, g, p, truncated=tr, tok_len_fn=tok_len_fn)
            for t, g, p, tr in zip(texts, golds, problems, trunc)]
    keys = [str(u) for u in uids] if uids is not None else [norm_problem(r["problem"]) for r in rows]
    if len(keys) != n:
        raise RuntimeError(f"[MATH][RETRY] uids 길이 {len(keys)} != 행 {n}")
    flips = (_group_sign_flips(keys, rng or random.Random(0))
             if spec["meta_term"] == "retry_shuffled" else None)
    if flips is not None:
        print(f"[MATH][RETRY_RAND] uid 그룹 부호 반전: n_groups={len(flips)} "
              f"n_neg={sum(1 for v in flips.values() if v < 0)}", flush=True)
    w, lc = retry_weight(), retry_len_cost()
    for r, k, fr, g in zip(rows, keys, forced, gpr):
        r["best_decision"] = None
        r["uid"] = k
        r["group_id"] = k                 # ★retry_metrics 는 group_key="group_id" 를 기본으로 쓴다
        r["group_pass_rate"] = float(g) if g is not None else None
        r["multi_block"] = int(int(r["n_blocks"]) > 1)
        r["redirected"] = int(r["decision"] == "redirect" and bool(r["has_second_attempt"]))
        r["forced_redirect"] = int(fr)
        emitted_ok = bool(_cdr._bool01(r["emitted"]) and not r["multi_block"])
        if emitted_ok and not fr:
            base, cost, defined = retry_judgment_parts(
                r["first_correct"], r["decision"], r["has_second_attempt"],
                r["final_answer"], r["first_answer"], final_correct=r["final_correct"],
                extra_tokens=r["n_tok_after_meta"], len_cost=lc)
        else:
            # ★강제 행: decision 이 정책 판단이 아니므로 judgment 항 undefined(0). 형식 위반 행도 동일.
            base, cost, defined = 0.0, 0.0, 0
        # ★RAND 는 ±1 판단 부분(base)만 뒤집는다 — 길이 비용은 뒤집지 않는다(뒤집으면 길이 보너스).
        sign = flips[k] if flips is not None else 1.0
        term = sign * base - cost
        r["judge"] = term                 # SAMPLE 로그·judge_match 가 읽는 이름 유지
        r["meta_val"] = w * term
        r["meta_defined"] = int(defined)
        r["r_corr"] = int(r["final_correct"])
        r["answer_total"] = float(r["final_correct"])
    return rows


def _compute_agree_rows(texts, golds, problems, spec, *, rng, uids, truncated, tok_len_fn,
                        forced_redirect=None, group_pass_rate=None) -> list[dict]:
    """M_AGREE / M_AGREE_RAND 의 행. answer_total = 최종 답 정오(M_RETRY 와 같다),
    meta_val = W_MON·모니터링 + W_CTL·통제(부호 반전은 RAND 의 통제 부분만, uid 그룹 단위).

    ★agree_true 는 **배치 전체**(uid 그룹, rollout.n=8)에서 사후 계산한다 — 모델은 형제를
    보지 않았지만 우리는 배치 안에서 다 갖고 있다(모듈 docstring). 강제 행(forced_redirect=1)
    은 M_RETRY 와 같은 이유로 모니터링·통제 둘 다 미정의(0) — decision 이 정책 판단이 아니다."""
    n = len(texts)
    trunc = list(truncated) if truncated is not None else [0] * n
    if len(trunc) != n:
        raise RuntimeError(f"[MATH][AGREE] truncated 길이 {len(trunc)} != 행 {n}")
    forced = ([int(_cdr._bool01(x)) for x in forced_redirect] if forced_redirect is not None
             else [0] * n)
    if len(forced) != n:
        raise RuntimeError(f"[MATH][AGREE] forced_redirect 길이 {len(forced)} != 행 {n}")
    gpr = (list(group_pass_rate) if group_pass_rate is not None else [None] * n)
    if len(gpr) != n:
        raise RuntimeError(f"[MATH][AGREE] group_pass_rate 길이 {len(gpr)} != 행 {n}")
    rows = [parse_agree_row(t, g, p, truncated=tr, tok_len_fn=tok_len_fn)
            for t, g, p, tr in zip(texts, golds, problems, trunc)]
    keys = [str(u) for u in uids] if uids is not None else [norm_problem(r["problem"]) for r in rows]
    if len(keys) != n:
        raise RuntimeError(f"[MATH][AGREE] uids 길이 {len(keys)} != 행 {n}")
    agree_true_list = compute_agree_true(rows, keys)
    flips = (_group_sign_flips(keys, rng or random.Random(0))
             if spec["meta_term"] == "agree_shuffled" else None)
    if flips is not None:
        print(f"[MATH][AGREE_RAND] uid 그룹 부호 반전: n_groups={len(flips)} "
              f"n_neg={sum(1 for v in flips.values() if v < 0)}", flush=True)
    w_mon, w_ctl, lc = agree_mon_weight(), agree_ctl_weight(), retry_len_cost()
    for r, k, fr, g, at in zip(rows, keys, forced, gpr, agree_true_list):
        r["best_decision"] = None
        r["uid"] = k
        r["group_id"] = k                 # ★retry_metrics 는 group_key="group_id" 를 기본으로 쓴다
        r["group_pass_rate"] = float(g) if g is not None else None
        r["agree_true"] = at
        r["multi_block"] = int(int(r["n_blocks"]) > 1)
        r["redirected"] = int(r["decision"] == "redirect" and bool(r["has_second_attempt"]))
        r["forced_redirect"] = int(fr)
        emitted_ok = bool(_cdr._bool01(r["emitted"]) and not r["multi_block"])
        if emitted_ok and not fr:
            mon, mon_def = agree_monitoring_term(r["agree_pred"], at)
            base, cost, ctl_def = agree_control_parts(
                at, r["decision"], r["has_second_attempt"], r["final_answer"], r["first_answer"],
                final_correct=r["final_correct"], extra_tokens=r["n_tok_after_meta"], len_cost=lc)
        else:
            mon, mon_def = 0.0, 0
            base, cost, ctl_def = 0.0, 0.0, 0
        # ★RAND 는 통제의 ±1 부분(base)만 뒤집는다 — 길이 비용은 그대로, 모니터링은 아예 안 건든다.
        sign = flips[k] if flips is not None else 1.0
        ctl_term = sign * base - cost
        r["mon_term"] = mon
        r["ctl_term"] = ctl_term
        r["judge"] = ctl_term              # SAMPLE 로그·M_RETRY 와 이름 호환
        r["meta_val"] = w_mon * mon + w_ctl * ctl_term
        r["meta_defined"] = int(bool(mon_def) or bool(ctl_def))
        r["r_corr"] = int(r["final_correct"])
        r["answer_total"] = float(r["final_correct"])
    return rows


# ★M_DIS_RAND 의 **고정** 시드(스펙 지시: "sign is randomized with a fixed seed"). 배치·스텝이
#   달라도 같은 uid 그룹은 같은 부호를 받는다 — 스텝마다 새로 뽑으면 대조군이 «부호가 매 스텝
#   흔들리는 팔»이 되어 M_DIS 와 분산이 달라진다. ★0914 리뷰 D2: **그룹 단위**로 뒤집는다
#   (M_RAND/M_RETRY_RAND 와 같은 규약 — 행 단위로 뒤집으면 같은 그룹 안 크레딧이 갈려 그룹
#   중심화 뒤 분산이 M_DIS 와 달라진다). 행 단위였던 이전 버전은 정정됨(모듈 docstring 참조).
DIS_RAND_SEED = 20260914


def _compute_dis_rows(texts, golds, problems, spec, *, uids, cand_answers, cand_correct) -> list[dict]:
    r"""M_DIS / M_DIS_RAND / M_DIS0 의 행. answer_total = 최종 \boxed 의 gold 정오,
    meta_val = W·자기증류 판단 크레딧, meta_defined = 크레딧이 정의된 행(그룹 중심화 member).

    ★라벨은 **배치의 uid 그룹**에서 나온다(math_dis.dis_row_credit): 그룹 롤아웃 중 **이 행
    자신을 뺀** 나머지(LOO)의 최종 답을 수학 동치로 묶은 다수답. 그래서 이 팔은 오프라인
    라벨표도, 얼어붙은 채점기도 필요 없다 — 필요한 것은 parquet 이 실어 준 후보 답
    (cand_answers)과 rollout.n 개의 형제뿐이다.
    ★`cand_correct` 는 **지표 전용**이다(commit_cand_correct_rate). 보상 계산은 이 값을 읽지
    않는다 — 읽으면 메타 스팬이 gold 를 받아 «판단»이 아니라 «정답 복사»를 보상한다.
    """
    from src.training import math_dis as _md   # noqa: PLC0415  (순환 import 방지 — math_dis 가 이 모듈을 읽는다)

    n = len(texts)
    ca = list(cand_answers) if cand_answers is not None else [None] * n
    cc = list(cand_correct) if cand_correct is not None else [None] * n
    if len(ca) != n or len(cc) != n:
        raise RuntimeError(f"[MATH][DIS] cand 길이 불일치: cand_answers={len(ca)} cand_correct={len(cc)} 행={n}")
    rows = [_md.parse_dis_row(t, g, p, a, c)
            for t, g, p, a, c in zip(texts, golds, problems, ca, cc)]
    keys = [str(u) for u in uids] if uids is not None else [norm_problem(r["problem"]) for r in rows]
    if len(keys) != n:
        raise RuntimeError(f"[MATH][DIS] uids 길이 {len(keys)} != 행 {n}")
    by_key: dict = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    # ★부호 반전은 **uid 그룹 단위**(0914 리뷰 D2) — M_RAND/M_RETRY_RAND 의 `_group_sign_flips`
    #   규약과 같다(행 단위로 뒤집으면 같은 그룹 안 크레딧이 갈려 그룹 중심화 뒤 분산이 M_DIS 와
    #   달라진다). 시드는 여전히 **고정**(DIS_RAND_SEED) — 이 팔만 스텝에 따라 흔들리지 않는다.
    flips = (_group_sign_flips(keys, random.Random(DIS_RAND_SEED))
             if spec["meta_term"] == "dis_shuffled" else None)
    if flips is not None:
        print(f"[MATH][DIS_RAND] uid 그룹 부호 반전(seed={DIS_RAND_SEED}): n_groups={len(flips)} "
              f"n_neg={sum(1 for v in flips.values() if v < 0)}", flush=True)
    # ★M_DIS0 은 가중치만 0 이다 — 프롬프트·행 계산·텔레메트리는 M_DIS 와 바이트 동일하고,
    #   메타 스팬에 얹히는 값만 사라진다(verl_sdc 의 영역 함수가 전 행 0 이면 no-op).
    w = 0.0 if spec["meta_term"] == "dis_zero" else dis_weight()
    for k, idxs in by_key.items():
        group = [rows[i] for i in idxs]
        plur = _md.plurality_answer([r.get("final_answer") for r in group])
        sign = flips[k] if flips is not None else 1.0
        # ★0914 리뷰 D1: 크레딧은 LEAVE-ONE-OUT 이다 — `self_idx`(그룹 안 이 행의 **위치**)를
        #   넘겨 이 행 자신의 최종 답이 자기 자신을 심판하는 다수결에 표를 보태지 못하게 한다
        #   (math_dis.dis_row_credit 의 ★설명 참조 — 순응 유인 차단).
        for local_i, i in enumerate(idxs):
            r = rows[i]
            credit, defined = _md.dis_row_credit(r, group, self_idx=local_i)
            f = _md.dis_row_flags(r)
            r.update(f)
            r["uid"] = k
            r["group_id"] = k
            r["best_decision"] = None
            r["plurality_answer"] = plur
            r["dis_credit"] = sign * credit
            r["dis_defined"] = int(defined)
            r["judge"] = r["dis_credit"]      # SAMPLE 로그·judge_match 가 읽는 이름 유지
            r["meta_val"] = w * r["dis_credit"]
            r["meta_defined"] = int(defined)
            r["answer_total"] = float(r["r_corr"])
    return rows


def _permute_group_donors(keys: Sequence, rng: random.Random) -> dict:
    """★M_DIFF_RAND: uid 그룹 → **다른 그룹**(도너) 표. 도너 그룹의 LOO 동의도로 라벨을 만든다.

    ★왜 «도너 표»인가(M_RAND 의 그룹 순열 규약): 라벨 분포는 그대로 두고 «이 문제가 어렵다»는
      그룹↔라벨 **대응만** 깬다. 행 단위로 섞으면 같은 그룹 안 라벨이 갈려 라벨 분포 자체가
      M_DIFF 와 달라진다(대조군 무효).
    ★기존 `_permute_group_labels` 를 못 쓰는 이유: 그 함수는 라벨이 verify/redirect(`_OPPOSITE`)
      일 때만 순열 풀에 넣는다 — 난이도 버킷은 풀이 비어 **조용히 아무것도 안 섞는다**(M_DIFF 와
      바이트 동일한 무효 대조군). 그래서 같은 규약의 별도 함수를 둔다.
    ★그룹이 하나뿐이면 도너가 자기 자신일 수밖에 없다 — 그때는 항등 표(대조군이 성립하지 않는
      배치이며, 학습 배치는 언제나 train_batch_size 개의 그룹을 갖는다).
    """
    ks = sorted(set(map(str, keys)))
    shuffled = list(ks)
    rng.shuffle(shuffled)
    if len(ks) > 1:
        # 고정점(자기 자신이 도너) 제거 — 그 그룹만 M_DIFF 와 같은 라벨을 받는 것을 막는다.
        for i, (a, b) in enumerate(zip(ks, shuffled)):
            if a == b:
                j = (i + 1) % len(ks)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return dict(zip(ks, shuffled))


def _compute_diff_rows(texts, golds, problems, spec, *, rng, uids) -> list[dict]:
    r"""M_DIFF / M_DIFF_RAND / M_DIFF0 의 행. answer_total = 최종 \boxed 의 gold 정오,
    meta_val = W·방향 크레딧, meta_defined = 크레딧이 정의된 행.

    ★라벨은 **배치의 uid 그룹**에서 나온다(math_diff.loo_agreement): 그룹 롤아웃 중 **이 행
    자신을 뺀** 나머지의 답 동의도를 세 버킷으로 이산화한 값. 오프라인 라벨표도, 얼어붙은
    채점기도, parquet 의 추가 컬럼도 필요 없다.
    ★M_DIFF_RAND 는 라벨을 **도너 그룹**(_permute_group_donors)의 LOO 동의도에서 만든다 —
    라벨 분포는 그대로고 그룹↔라벨 대응만 깨진다.
    ★`loo_pass_rate`/`stated_idx` 는 **지표 전용**이다(gold 파생) — 크레딧 계산은 읽지 않는다.
    ★이 팔의 meta_val 은 **그룹 중심화를 타지 않는다** — verl_sdc._compute_math_arm_stash 가
    별도 스태시 키(`diff`/`diff_spans`)로 싣고 `_math_add_diff_meta_advantage` 가 그대로 얹는다.
    """
    from src.training import math_diff as _mdf   # noqa: PLC0415  (순환 import 방지 — math_diff 가 이 모듈을 읽는다)

    n = len(texts)
    rows = [_mdf.parse_diff_row(t, g, p) for t, g, p in zip(texts, golds, problems)]
    keys = [str(u) for u in uids] if uids is not None else [norm_problem(r["problem"]) for r in rows]
    if len(keys) != n:
        raise RuntimeError(f"[MATH][DIFF] uids 길이 {len(keys)} != 행 {n}")
    by_key: dict = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    donors = (_permute_group_donors(keys, rng or random.Random(0))
              if spec["meta_term"] == "diff_shuffled" else None)
    if donors is not None:
        n_self = sum(1 for k, d in donors.items() if k == d)
        print(f"[MATH][DIFF_RAND] uid 그룹 도너 순열: n_groups={len(donors)} n_self={n_self}",
              flush=True)
    # ★M_DIFF0 은 가중치만 0 이다 — 프롬프트·행 계산·텔레메트리는 M_DIFF 와 바이트 동일하고,
    #   메타 스팬에 얹히는 값만 사라진다(어드밴티지 함수가 전 행 0 이면 no-op).
    w = 0.0 if spec["meta_term"] == "diff_zero" else diff_weight()
    for k, idxs in by_key.items():
        group = [rows[i] for i in idxs]
        label_rows = ([rows[i] for i in by_key[donors[k]]] if donors is not None else group)
        for local_i, i in enumerate(idxs):
            r = rows[i]
            f = _mdf.diff_row_flags(r)
            r.update(f)
            # ★라벨 자리: M_DIFF 는 자기 그룹에서 자기를 빼고(LOO), RAND 는 도너 그룹에서
            #   같은 자리(길이가 다르면 나머지 연산)를 뺀다 — 두 경우 모두 «한 행을 뺀 동의도».
            li = local_i % max(1, len(label_rows))
            agree = _mdf.loo_agreement(li, label_rows)
            credit, defined = _mdf.diff_row_credit(
                r, label_rows, self_idx=li) if label_rows else (0.0, False)
            met = _mdf.diff_row_metrics(r, group, self_idx=local_i)
            r["uid"] = k
            r["group_id"] = k
            r["best_decision"] = None
            r["loo_agreement"] = agree
            lbl_bucket = _mdf.agree_bucket(agree)
            r["agree_bucket"] = lbl_bucket
            # ★F2(0914): 텔레메트리가 STATED 버킷 옆에서 LABEL 버킷 분포를 나란히 읽을 수 있게
            #   별도 이름으로도 싣는다(math_diff.label_bucket_histogram/entropy 가 이 키를 읽는다).
            #   `agree_bucket` 과 같은 값이지만, 라벨이 **정의된** 경우에만 채운다(None 이면
            #   히스토그램·엔트로피가 그 행을 «못 쟀다»로 건너뛴다 — bucket_histogram 과 같은 규약).
            r["label_bucket"] = lbl_bucket if lbl_bucket in _mdf.BUCKETS else None
            r["stated_idx"] = met["stated_idx"]
            r["loo_pass_rate"] = met["loo_pass_rate"]     # ★지표 전용(gold 파생)
            r["diff_credit"] = float(credit)
            r["diff_defined"] = int(defined)
            r["judge"] = r["diff_credit"]     # SAMPLE 로그·judge_match 가 읽는 이름 유지
            r["meta_val"] = w * r["diff_credit"]
            r["meta_defined"] = int(defined)
            r["answer_total"] = float(r["r_corr"])
    return rows


def meta_char_spans(row: Mapping) -> list[tuple[int, int]]:
    """메타 항이 얹힐 문자 구간(첫 블록만 — parse_meta 가 첫 블록을 채점하므로 같은 구간)."""
    s, e = row.get("meta_start"), row.get("meta_end")
    if not _cdr._bool01(row.get("emitted", 0)) or s is None or e is None or e <= s:
        return []
    return [(int(s), int(e))]


def decision_char_spans(row: Mapping) -> list[tuple[int, int]]:
    """SL 항이 얹힐 문자 구간 — 메타 블록 안 `decision:` 줄의 **결정 단어**(verify/redirect)
    딱 그만큼. 블록에 decision 줄이 없으면 빈 리스트(그 행은 SL 에서 빠진다).
    ★오프셋은 전체 응답 텍스트 기준이다(meta_start/meta_end 가 이미 절대 오프셋이고 raw 는
    meta_start 에서 시작한다) — verl_sdc._char_to_tok 이 그대로 토큰 구간으로 사상한다."""
    s, e = row.get("meta_start"), row.get("meta_end")
    if not _cdr._bool01(row.get("emitted", 0)) or s is None or e is None or e <= s:
        return []
    if row.get("decision") not in _OPPOSITE:
        return []
    m = _cdr._DECISION.search(row.get("text") or "", int(s), int(e))
    if not m:
        return []
    return [(int(m.start(1)), int(m.end(1)))]


def annotate_sl_rows(rows: Sequence[Mapping], *, step) -> Sequence[Mapping]:
    r"""★M_RETRY_SL 의 자기지도(self-distilled judgment) 항을 행에 채운다.

    라벨은 **그 행 자신의 첫 답 정오**다(교사 없음): first_correct==0 → " redirect",
    아니면 " verify". 정책이 실제로 표집한 결정 토큰 위에만 얹는다.

    왜 어드밴티지 주입이 CE 인가 — REINFORCE 항등식. 목표 CE 의 기울기는
        ∇ CE = −∇ log p(target)
    인데 우리는 verl 0.7.1 의 actor loss(`ppo_loss`)를 건드리지 않고 어드밴티지만 만질 수
    있다. 정책 기울기 항은 A_t·∇ log p(sampled_t) 이므로, **표집된 토큰에 한정한** CE 의
    REINFORCE 형태는
        A_t = +W  (sampled == target)      → ∇ 는 +W·∇ log p(sampled)   = −W·∇CE (그 토큰)
        A_t = −W  (sampled != target)      → ∇ 는 −W·∇ log p(sampled)
    다. 두 번째 줄이 이 팔의 핵심이다: 정책이 redirect 를 **한 번도 표집하지 않으므로**
    +W 로 redirect 를 밀어 올릴 기회 자체가 없고, 「첫 답이 틀렸는데 verify 를 냈다」는 행에
    −W 를 얹어 p(verify) 를 내리면 softmax 가 그 질량을 나머지(=redirect)로 돌려준다.
    즉 p(redirect) 는 **간접적으로** 올라간다.

    ★그룹 중심화를 하지 않는다(verl_sdc._math_add_decision_sl_advantage). 이건 지도 신호이지
    상대 비교가 아니다 — 한 그룹이 전부 verify(=지금 실측)면 중심화가 신호를 정확히 0 으로
    지워 팔이 M_RETRY 와 바이트 동일한 무효 레버가 된다.

    행 조건: 비강제(forced_redirect=0) ∧ 발화 ∧ 단일 블록 ∧ decision 줄 존재. 강제 행은
    프롬프트가 달라 결정이 정책 판단이 아니므로 제외한다(판단 항과 같은 규약).
    """
    w = sl_step_weight(step)
    for r in rows:
        spans = decision_char_spans(r)
        # ★0914: 잘린 행은 첫 답이 없어 자기 라벨이 «redirect» 로 오염되므로 SL 에서 제외한다.
        ok = bool(spans) and not _cdr._bool01(r.get("forced_redirect", 0)) \
            and not _cdr._bool01(r.get("truncated", 0)) \
            and not int(r.get("multi_block", 0) or 0)
        tgt = "verify" if _cdr._bool01(r.get("first_correct", 0)) else "redirect"
        r["sl_target"] = tgt if ok else None
        r["sl_defined"] = int(ok)
        r["sl_match"] = int(ok and r.get("decision") == tgt)
        r["sl_val"] = ((w if r["sl_match"] else -w) if ok else 0.0)
        r["sl_spans"] = spans if ok else []
    return rows


def sl_telemetry(rows: Sequence[Mapping]) -> dict:
    """SL 항의 두 지표 — 얹힌 행 수와 «표집한 결정이 자기라벨과 같았나» 비율.
    sl_match_rate 가 1.0 에 붙어 있으면 SL 이 밀 것이 없다(=first_correct 가 전부 1 이거나
    이미 다 맞히고 있다), 0 에 붙어 있으면 −W 만 계속 얹히고 있다는 뜻이다."""
    defined = [r for r in rows if int(r.get("sl_defined", 0) or 0)]
    return {
        "sl_rows": len(defined),
        "sl_match_rate": (sum(int(r.get("sl_match", 0)) for r in defined) / len(defined))
                         if defined else float("nan"),
    }


# ── 텔레메트리·중단 ────────────────────────────────────────────────────────────
ABORT_RULES = {
    # ★재시도 팔만 워밍업(step≤5): 첫 답 뒤 블록을 내는 형식은 초기 정책이 아직 못 지킬 수 있다 —
    #   다른 메타 팔은 워밍업 없음(기존 계약 불변). `min_step_arms` 가 없으면 min_step 은 모든 팔에.
    # ★M_AGREE 도 M_RETRY 와 같은 첫답→판단→재시도 구조라 워밍업을 공유한다(_RETRY_LIKE_ARMS).
    "emit_rate":        {"op": "<", "thr": 0.2, "meta_arms_only": True,
                         "min_step": 5, "min_step_arms": _RETRY_LIKE_ARMS | _CRIT_ARMS | _DIS_ARMS | _DIFF_ARMS,
                         "why": "발화가 무너지면 메타 스팬 항이 얹힐 토큰이 없다"},
    "boxed_in_meta":    {"op": ">", "thr": 0.02, "meta_arms_only": False,
                         "why": "메타가 답을 담으면 메타 보상이 정답 보상의 사본이 된다"},
    # ★감사 1: 이 통계는 «최빈 메타 문장 하나의 점유율»(바닥 1/n_emitted)이라 0.05 는 n_emitted≤20
    #   이면 무조건 발화하는 규칙이었다. Countdown 과 같은 0.5, 표본 30 미만이면 «못 쟀다»(missing).
    "boilerplate_rate": {"op": ">", "thr": 0.5, "meta_arms_only": False, "min_n": ("n_emitted", 30),
                         "why": "최빈 메타 문장이 발화 행의 절반을 넘으면 판단이 아니라 상투구다"},
    # ★감사 6: 블록 둘 이상은 형식 위반(첫 블록만 채점되므로 나머지는 공짜 토큰).
    "multi_block_rate":  {"op": ">", "thr": 0.10, "meta_arms_only": False,
                         "why": "메타 블록이 둘 이상인 행이 10% 를 넘으면 형식이 무너진 것이다"},
    # ★감사 7: 사전등록 «acc < M_G0 − 1pp». 문턱은 런처가 MATH_ACC_FLOOR 로 준다 — 미설정이면
    #   이 규칙은 참여하지 않는다(check_abort 가 건너뛴다). 설정되면 3-스텝 연속 규칙에 참여.
    "acc":              {"op": "<", "thr": None, "env_thr": "MATH_ACC_FLOOR", "meta_arms_only": False,
                         "why": "정확도가 M_G0 − 1pp 아래로 3 스텝 연속이면 메타 항이 해롭다"},
    # ★수정 3(재시도 팔 전용). redirect 가 사라지면 «verify 만 쓰고 +1 챙기기»로 붕괴한 것이다 —
    #   워밍업(step≤5)은 봐준다(초기 정책의 redirect 율이 낮을 수 있다). trunc 는 두 번째 시도가
    #   RESP_LEN 에 잘려 최종 \boxed 를 못 내는 사고 — 판단 항이 아니라 길이가 결과를 정한다.
    #   ★redirect_rate 의 분모는 **결정이 파싱된 행**이다(0914 검증 ④) — 전체 행이 분모면 발화가
    #   낮은 초기에 redirect 가 «죽은 것처럼» 보여 잘못 중단한다. 전체 행 분모는 redirect_rate_all_rows.
    # ★SL 팔(M_RETRY_SL)만 min_step 15: SL 항은 «−W 로 p(verify) 를 내려 redirect 질량을
    #   간접적으로 키우는» 우회로라 step 5 안에 redirect 가 2% 를 넘길 보장이 없다. 5 로 두면
    #   SL 이 일하기 전에 팔을 죽인다(그 판정은 SL 의 실패가 아니라 «못 기다렸다»가 된다).
    "redirect_rate":    {"op": "<", "thr": 0.02, "meta_arms_only": True, "arms": _RETRY_LIKE_ARMS, "min_step": 5,
                         "min_step_by_arm": {"M_RETRY_SL": 15},
                         "why": "redirect 가 결정된 행의 2% 아래면 재시도 탐색이 죽었다(verify 단일 전략 붕괴)"},
    # ★0914 수리: 문턱 0.15→0.25 + min_step 3. 두 번째 시도가 응답을 늘리므로(RESP_LEN 6144 기본)
    #   smoke step 1 에서 정상 정책도 .131 을 찍었다 — 초기 워밍업 없이 0.15 로는 오탐 중단이 난다.
    "trunc_rate":       {"op": ">", "thr": 0.25, "meta_arms_only": False, "arms": _RETRY_LIKE_ARMS,
                         "min_step": 3,
                         "why": "잘린 행이 25% 를 넘으면(3 스텝 뒤에도) 두 번째 시도가 길이에 막혀 판단 항이 오염된다"},
    # ★M_AGREE 전용(사전등록 확장 0914b). agreement: 줄 자체가 없으면 모니터링 항이 정의 자체가
    #   안 된다 — «값이 나쁘다»가 아니라 «형식 자체를 안 지킨다»는 신호라 emit_rate 와 별개로 잰다.
    # ★M_CRIT 전용(사전등록 수정 6). 누출된 비평은 «비평의 값»이 아니라 «답을 다시 보여 준 값»을
    #   재게 만든다 — 30% 를 넘으면 이 팔이 재는 것이 더 이상 비평이 아니다(step≤3 워밍업).
    "leak_rate":        {"op": ">", "thr": 0.3, "meta_arms_only": True, "arms": _CRIT_ARMS,
                         "min_step": 3,
                         "why": "비평이 답을 흘린 행이 30% 를 넘으면 IG 가 비평 내용이 아니라 답 누출을 잰다"},
    # ★항이 정의된 행이 30% 아래면 메타 스팬에 얹힐 크레딧이 거의 없다(무효 레버와 같은 상태).
    "crit_defined_rate": {"op": "<", "thr": 0.3, "meta_arms_only": True, "arms": _CRIT_ARMS,
                         "min_step": 5,
                         "why": "정보이득 항이 정의된 행이 30% 아래면 메타 스팬이 사실상 보상을 안 받는다"},
    "agree_line_rate":  {"op": "<", "thr": 0.8, "meta_arms_only": True, "arms": _AGREE_ARMS, "min_step": 5,
                         "why": "agreement: 줄이 발화 행의 80% 아래면 모니터링 항이 정의될 자리 자체가 없다"},
    # ★M_DIS 전용(0914d). 이 팔은 블록·커밋이 **형식이 아니라 신호의 전달 수단**이다 — 블록이
    #   없거나 `Commit: n` 이 없으면 판단 자체가 관측되지 않는다. 그래서 문턱이 다른 팔의 발화
    #   규칙(0.2)보다 훨씬 높다(0.8). 워밍업 step≤3 — 초기 정책이 형식을 잡는 데 몇 스텝 준다.
    #   ★generic `emit_rate`(thr 0.2) 는 이 팔에서도 살아 있지만 항상 이 규칙보다 늦게 걸린다.
    "dis_emit_rate":    {"op": "<", "thr": 0.8, "meta_arms_only": True, "arms": _DIS_ARMS, "min_step": 3,
                         "why": "메타 블록이 행의 80% 아래면 진단을 아예 안 쓰는 것이다(신호가 관측되지 않는다)"},
    "commit_parsed":    {"op": "<", "thr": 0.8, "meta_arms_only": True, "arms": _DIS_ARMS, "min_step": 3,
                         "why": "`Commit: n` 이 행의 80% 아래로 파싱되면 판단 크레딧이 붙을 자리가 없다"},
    # ★크레딧이 정의된 행이 30% 아래면 메타 스팬이 사실상 보상을 안 받는다(무효 레버와 같은 상태).
    #   crit_defined_rate 와 같은 문턱·같은 이유.
    "dis_defined_rate": {"op": "<", "thr": 0.3, "meta_arms_only": True, "arms": _DIS_ARMS, "min_step": 5,
                         "why": "판단 크레딧이 정의된 행이 30% 아래면 메타 스팬이 사실상 보상을 안 받는다"},
    # ★M_DIFF 전용(0914e). 블록·버킷은 이 팔에서 **형식이 아니라 판단의 전달 수단**이다 —
    #   블록이 없거나 버킷이 안 파싱되면 판단 자체가 관측되지 않는다(M_DIS 와 같은 문턱 0.8,
    #   워밍업 step≤3 — 초기 정책이 형식을 잡는 데 몇 스텝 준다).
    "diff_emit_rate":   {"op": "<", "thr": 0.8, "meta_arms_only": True, "arms": _DIFF_ARMS, "min_step": 3,
                         "why": "메타 블록이 행의 80% 아래면 난이도를 아예 안 말하는 것이다"},
    "bucket_parsed":    {"op": "<", "thr": 0.8, "meta_arms_only": True, "arms": _DIFF_ARMS, "min_step": 3,
                         "why": "`difficulty: easy|medium|hard` 가 행의 80% 아래로 파싱되면 판단 크레딧이 붙을 자리가 없다"},
    # ★버킷 붕괴: 한 버킷이 90% 를 넘으면 «전부 medium» 같은 상수 정책으로 굳은 것이다 —
    #   크레딧은 여전히 붙지만(문제의 90% 가 그 버킷이면 +1 을 받는다) 판단은 사라졌다.
    #   워밍업 step≤5(초기 몇 스텝의 쏠림은 아직 정책이 아니다).
    "bucket_max_share": {"op": ">", "thr": 0.90, "meta_arms_only": True, "arms": _DIFF_ARMS, "min_step": 5,
                         "why": "한 버킷이 90% 를 넘으면 난이도 판단이 상수로 붕괴한 것이다"},
    # ★F2(0914): 이건 STATED 가 아니라 **LABEL**(동의도 이산화) 쪽 붕괴다 — 정책이 buckets 을
    #   골고루 말해도(bucket_max_share 통과) 라벨 자체가 한쪽으로 몰려 있으면(2604.24070 의
    #   이봉 분포와 반대로, 이 배치의 문제들이 우연히 전부 같은 난이도면) 그 크레딧은 «맞혀도
    #   배울 것이 없는» 상태다 — 데이터가 애초에 가르칠 것이 없는 런을 죽인다(정책의 붕괴가
    #   아니라 데이터의 붕괴). 문턱은 bucket_max_share 보다 빡빡하다(.95) — STATED 축은 정책이
    #   고칠 수 있지만 LABEL 축은 그 배치의 문제 구성이 정하므로 더 관대하게 보되, 한쪽으로
    #   완전히 쏠리면(.95+) 막는다. 워밍업 step≤5(bucket_max_share 와 같다).
    "label_bucket_max_share": {"op": ">", "thr": 0.95, "meta_arms_only": True, "arms": _DIFF_ARMS,
                              "min_step": 5,
                              "why": "라벨(동의도) 버킷이 95% 를 넘게 한쪽이면 이 배치의 크레딧은 정보가 없다"},
    # ★크레딧이 정의된 행이 30% 아래면 메타 스팬이 사실상 보상을 안 받는다(crit/dis 와 같은 문턱).
    "diff_defined_rate": {"op": "<", "thr": 0.3, "meta_arms_only": True, "arms": _DIFF_ARMS, "min_step": 5,
                          "why": "방향 크레딧이 정의된 행이 30% 아래면 메타 스팬이 사실상 보상을 안 받는다"},
}


def telemetry(rows: Sequence[Mapping], *, arm: str, step) -> dict:
    n = max(1, len(rows))
    emitted = [r for r in rows if _cdr._bool01(r.get("emitted", 0))]
    n_dec = sum(1 for r in emitted if r.get("decision") in _OPPOSITE)
    n_lab = sum(1 for r in emitted if r.get("best_decision") in _OPPOSITE and r.get("decision") in _OPPOSITE)
    bp = _cdr.boilerplate_rate(rows, form="math")
    rep = {
        "step": int(step), "arm": arm, "n_rows": len(rows),
        "acc": sum(int(r["r_corr"]) for r in rows) / n,
        "emit_rate": len(emitted) / n,
        "n_blocks_mean": sum(int(r.get("n_blocks", 0)) for r in rows) / n,
        "multi_block_rate": sum(1 for r in rows if int(r.get("n_blocks", 0)) > 1) / n,
        "decision_rate": n_dec / max(1, len(emitted)),
        # 라벨·결정이 둘 다 있는 발화 행 중 일치 비율(없으면 NaN — 0 으로 읽히면 안 된다)
        "judge_match": (sum(1 for r in emitted if r.get("judge", 0) > 0) / n_lab
                        if n_lab else float("nan")),
        "boxed_in_meta": sum(int(r.get("boxed_in_meta", 0)) for r in rows) / n,
        "boilerplate_rate": bp.get("boilerplate_rate"),
        "n_emitted": bp.get("n_emitted", 0),
        "len_mean": sum(int(r.get("n_chars", 0)) for r in rows) / n,
        "meta_val_abs_mean": sum(abs(float(r.get("meta_val", 0.0))) for r in rows) / n,
    }
    if arm in _RETRY_ARMS:
        rep.update(retry_telemetry(rows))
        if arm in _SL_ARMS:
            rep.update(sl_telemetry(rows))
    elif arm in _AGREE_ARMS:
        rep.update(agree_telemetry(rows))
    elif arm in _CRIT_ARMS:
        rep.update(crit_telemetry(rows))
    elif arm in _DIS_ARMS:
        rep.update(dis_telemetry(rows))
    elif arm in _DIFF_ARMS:
        rep.update(diff_telemetry(rows))
    return rep


def retry_telemetry(rows: Sequence[Mapping]) -> dict:
    """M_RETRY 세 곡선의 재료 — 학습 중(여기)과 held-out(scripts/local/math_retry_eval.py)이
    **같은 정의**를 쓴다. 조건부 비율의 분모가 0 이면 NaN(0 으로 읽히면 안 된다).

    ★0914 forced-redirect: redirect_rate(중단 규칙이 읽는 지표)와 judgment_acc 는 강제 행이
    섞이면 «판단이 살아있다»고 오판할 수 있다(강제 행은 항상 redirect 라 분자를 인플레이트한다)
    — 그래서 `dec`(중단 규칙용) 자체를 비강제 행으로만 구성한다. forced_frac/forced_rescue_rate/
    forced_first_acc 는 새 키(스펙 지시, 기존 키는 전부 보존). ★B4(0914): judgment_acc_unforced
    는 judgment_acc 와 바이트 동일한 중복이었다(`dec` 가 이미 비강제 행만 담는다) — 제거."""
    n = max(1, len(rows))
    is_forced = lambda r: bool(_cdr._bool01(r.get("forced_redirect", 0)))
    unforced_rows = [r for r in rows if not is_forced(r)]
    forced_rows = [r for r in rows if is_forced(r)]
    # ★redirect_rate 의 분모는 «비강제 ∧ 결정이 파싱된 행» — 강제 행은 판단 신호가 아니므로 전부 배제.
    dec = [r for r in unforced_rows if r.get("decision") in _OPPOSITE]
    wrong = [r for r in dec if not _cdr._bool01(r.get("first_correct", 0))]
    right = [r for r in dec if _cdr._bool01(r.get("first_correct", 0))]
    _rate = lambda xs, pred: (sum(1 for r in xs if pred(r)) / len(xs)) if xs else float("nan")
    is_red = lambda r: r.get("decision") == "redirect"
    out = {
        "first_acc": sum(int(r.get("first_correct", 0)) for r in rows) / n,
        "final_acc": sum(int(r.get("final_correct", 0)) for r in rows) / n,
        # ★분모 = 비강제·결정이 파싱된 행(중단 규칙이 읽는 값). 분모 0 → NaN(missing). 전체 행 분모는 참고용.
        "redirect_rate": _rate(dec, is_red),
        "redirect_rate_all_rows": sum(1 for r in unforced_rows if is_red(r)) / max(1, len(unforced_rows)),
        "redirect_rate_given_wrong": _rate(wrong, is_red),
        "redirect_rate_given_right": _rate(right, is_red),
        # 판단 정확도 = «틀렸으면 redirect, 맞았으면 verify» 를 맞힌 비율(결정 있는 행 중, 비강제)
        "judgment_acc": _rate(dec, lambda r: is_red(r) != bool(_cdr._bool01(r.get("first_correct", 0)))),
        "second_attempt_rate": sum(int(r.get("has_second_attempt", 0)) for r in rows) / n,
        # ★redirect 가 결정된(비강제) 행 중 실제로 두 번째 시도를 냈는지 — redirect_rate 와 짝지어
        #   읽는다: redirect 는 뛰는데 이 값이 낮으면 «결정만 하고 실행은 안 함»이 드러난다.
        "second_attempt_rate_given_redirect": _rate(
            [r for r in dec if is_red(r)], lambda r: bool(_cdr._bool01(r.get("has_second_attempt", 0)))),
        # ★B6(0914): 중단 규칙이 읽는 trunc_rate 는 비강제 행만 — 강제 행은 F 실측·구제율을
        #   보려고 일부러 붙인 것이라 잘림도 «판단 오염»의 신호가 아니다. 전체 행은 참고용으로 남긴다.
        "trunc_rate": sum(int(r.get("truncated", 0)) for r in unforced_rows) / max(1, len(unforced_rows)),
        "trunc_rate_all_rows": sum(int(r.get("truncated", 0)) for r in rows) / n,
        "n_decided": len(dec),
        # ★강제 행 전용 지표 — F 실측치와 «강제해도 최종 정답이 구제되는가»를 본다.
        "forced_frac": len(forced_rows) / n,
        "forced_first_acc": _rate(forced_rows, lambda r: bool(_cdr._bool01(r.get("first_correct", 0)))),
        "forced_rescue_rate": _rate(forced_rows, lambda r: (not _cdr._bool01(r.get("first_correct", 0)))
                                    and _cdr._bool01(r.get("final_correct", 0))),
    }
    # ★0914 후속: within-problem 선택성(retry_metrics) — group_pass_rate 가 붙은(build_math_parquet
    #   --forced_from_rollouts) 행이 하나라도 있으면 그것으로 mixed 판정(오프라인 K 샘플 기준),
    #   없으면 (구 parquet·비강제 팔) 스킵한다 — 배치 하나엔 프롬프트당 롤아웃이 몇 안 돼 배치-로컬
    #   first_correct 로는 mixed 여부가 잡음투성이다. 키는 uid(같은 프롬프트의 배치 내 롤아웃 묶음).
    pr = {str(r.get("uid")): r["group_pass_rate"] for r in rows if r.get("group_pass_rate") is not None}
    if pr:
        for r in rows:
            r.setdefault("group_id", r.get("uid"))
        out.update(_rmet.all_metrics(rows, group_key="uid", pass_rate=pr))
    return out


def crit_telemetry(rows: Sequence[Mapping]) -> dict:
    """M_CRIT 지표 — 학습(여기)과 held-out(scripts/local/math_critique_eval.py)이 같은 정의를 쓴다.

    crit_rows            비평 문장이 파싱된 행 수(발화·단일 블록·비어 있지 않음; 누출 포함)
    crit_defined_rate    정보이득 항이 **정의된** 행 비율(= meta_defined; 누출·형제 없음·채점 실패 제외)
    leak_rate            비평이 답을 흘린 비율(분모 = 비평이 파싱된 행) ← 프롬프트의 금지가 지켜지는가
    ig_mean/ig_donor_mean/ig_delta_mean   정보이득과 그 내용 대조(분모 = 항이 정의된 행)
    crit_words_mean      비평 단어 수 평균(짧으면 형식 붕괴, 길면 비평 자리에서 다시 푼 것)
    ★조건부 비율의 분모가 0 이면 NaN(«못 쟀다») — 0 으로 읽히면 중단 규칙이 오작동한다.
    """
    n = max(1, len(rows))
    parsed = [r for r in rows if bool((r.get("critique") or "").strip())
              and _cdr._bool01(r.get("emitted", 0)) and not int(r.get("multi_block", 0) or 0)]
    defined = [r for r in rows if int(r.get("meta_defined", 0) or 0)]
    _mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {
        "crit_rows": len(parsed),
        "crit_defined_rate": len(defined) / n,
        "leak_rate": (sum(int(_cdr._bool01(r.get("leaked", 0))) for r in parsed) / len(parsed))
                     if parsed else float("nan"),
        "ig_mean": _mean([float(r["ig"]) for r in defined if _cdr._finite(r.get("ig"))]),
        "ig_donor_mean": _mean([float(r["ig_donor"]) for r in defined
                                if _cdr._finite(r.get("ig_donor"))]),
        "ig_delta_mean": _mean([float(r["ig_delta"]) for r in defined
                                if _cdr._finite(r.get("ig_delta"))]),
        "crit_words_mean": _mean([float(r.get("crit_words", 0)) for r in parsed]),
        # ★`first_acc` 는 여기서 내지 않는다 — 기본 텔레메트리의 `acc`(= r_corr = 첫 답 정오)와
        #   같은 값이고, 이 키가 있으면 format_tel 이 «재시도 팔» 줄을 찍으려다 없는 키를 읽는다.
        # ★프롬프트가 금지한 «두 번째 시도»를 실제로 쓴 비율 — 형식 위반의 조기 경보(중단 규칙 아님).
        "second_attempt_rate": sum(int(_cdr._bool01(r.get("has_second_attempt", 0))) for r in rows) / n,
    }


def dis_telemetry(rows: Sequence[Mapping]) -> dict:
    r"""M_DIS 지표 — 학습(여기)과 held-out(scripts/local/math_dis_eval.py)이 같은 정의를 쓴다.

    dis_rows                 블록이 있는 행 수
    dis_emit_rate            블록이 있는 행 비율(중단 규칙이 읽는 값 — generic emit_rate 와 같은 값)
    commit_parsed            `Commit: n`(1..N_CAND)이 파싱된 행 비율
    dis_defined_rate         자기증류 크레딧이 **정의된** 행 비율(= meta_defined)
    dis_credit_mean          크레딧 평균(분모 = 정의된 행) — +1 쪽으로 쏠리면 다수답 순응,
                             −1 쪽이면 소수 선택. 0 근처가 «판단이 갈리고 있다»는 상태다.
    commit_is_minority_rate  커밋이 **네 후보의 다수답과 다른** 행 비율(분모 = 커밋 파싱 행)
    cites_two_plus_rate      후보를 둘 이상 인용한 행 비율(분모 = 블록이 있는 행)
    diag_words_mean          진단 단어 수 평균(분모 = 블록이 있는 행)
    final_acc                최종 \boxed 의 gold 정오(= generic acc 와 같은 값; 이름을 같이 낸다)
    cand_acc_mean            **후보들의** gold 정오 평균 — 진단 이전의 출발선(지표 전용)
    commit_cand_correct_rate 커밋한 후보가 실제로 정답이었던 비율(분모 = 커밋 파싱 ∧ cand_correct
                             가 실린 행). ★지표 전용 — 보상은 이 값을 읽지 않는다(gold 는 답 스팬에만).
    ★조건부 비율의 분모가 0 이면 NaN(«못 쟀다») — 0 으로 읽히면 중단 규칙이 오작동한다.
    """
    n = max(1, len(rows))
    emitted = [r for r in rows if int(_cdr._bool01(r.get("has_meta", r.get("emitted", 0))))]
    committed = [r for r in rows if r.get("commit") is not None]
    defined = [r for r in rows if int(r.get("dis_defined", r.get("meta_defined", 0)) or 0)]
    with_cc = [r for r in committed if r.get("cand_correct")]
    _mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {
        "dis_rows": len(emitted),
        "dis_emit_rate": len(emitted) / n,
        "commit_parsed": len(committed) / n,
        "dis_defined_rate": len(defined) / n,
        "dis_credit_mean": _mean([float(r.get("dis_credit", 0.0)) for r in defined]),
        "commit_is_minority_rate": _mean([float(int(r.get("commit_is_minority", 0)))
                                          for r in committed]),
        "cites_two_plus_rate": _mean([float(int(r.get("cites_two_plus", 0))) for r in emitted]),
        "diag_words_mean": _mean([float(r.get("diag_words", 0)) for r in emitted]),
        "final_acc": sum(int(r.get("r_corr", 0)) for r in rows) / n,
        "cand_acc_mean": _mean([_mean([float(x) for x in r["cand_correct"]])
                                for r in rows if r.get("cand_correct")]),
        "commit_cand_correct_rate": _mean([
            float(r["cand_correct"][int(r["commit"]) - 1])
            for r in with_cc if 1 <= int(r["commit"]) <= len(r["cand_correct"])]),
    }


def diff_telemetry(rows: Sequence[Mapping]) -> dict:
    r"""M_DIFF 지표 — 학습(여기)과 held-out(scripts/local/math_diff_eval.py)이 같은 정의를 쓴다.

    diff_rows              블록이 있는 행 수
    diff_emit_rate         블록이 있는 행 비율(중단 규칙이 읽는 값)
    meta_first_rate        블록이 응답 **맨 앞**에서 시작한 행 비율(분모 = 블록이 있는 행) —
                           «풀기 전에 판단했는가». 뒤로 밀리면 이 팔의 주장이 사라진다.
    bucket_parsed          `difficulty: easy|medium|hard` 가 파싱된 행 비율
    has_why_rate           `why:` 문장이 있는 행 비율(분모 = 블록이 있는 행)
    bucket_easy/medium/hard  버킷별 **비율**(분모 = 버킷이 파싱된 행)
    bucket_max_share       최빈 버킷의 점유율 — .90 을 넘으면 버킷 붕괴(ABORT_RULES)
    bucket_entropy         버킷 분포 엔트로피(nats, 최대 ln3≈1.099)
    label_bucket_easy/medium/hard  ★LABEL(동의도 이산화) 쪽 버킷별 비율(분모 = 라벨이 정의된 행).
                           2604.24070: 4B 정책의 자기 롤아웃 분포는 이봉일 수 있고, 라벨이
                           한쪽으로 몰리면 그 크레딧은 정보가 없다 — STATED 옆에 LABEL 을 둬야
                           그것이 보인다.
    label_bucket_max_share LABEL 최빈 버킷 점유율(ABORT_RULES — 라벨, STATED 아님)
    label_bucket_entropy   LABEL 버킷 분포 엔트로피(nats)
    label_bucket_{b}_n     LABEL 버킷별 **개수**(format_tel 의 `lbl=e/m/h=` 줄이 읽는다 — 비율이
                           아니라 개수라 n 이 작을 때 «못 쟀다»와 «0개」가 헷갈리지 않는다)
    diff_defined_rate      방향 크레딧이 **정의된** 행 비율(= meta_defined)
    diff_credit_mean       크레딧 평균(분모 = 정의된 행)
    stated_vs_agree_spearman   말한 버킷 vs (1 − LOO 동의도) 순위 상관 — 이 팔의 **신호**
    stated_vs_truepass_spearman 말한 버킷 vs (1 − 실제 LOO pass rate) — ★지표 전용(gold 파생)
    final_acc              최종 \boxed 의 gold 정오(= generic acc 와 같은 값 — 이 팔은 한 번만
                           푼다). ★`first_acc` 라는 키는 **두지 않는다** — format_tel 이 그 키의
                           존재로 «재시도 팔»을 판별하므로, 두면 없는 재시도 지표를 찍다 죽는다.
    ★조건부 비율의 분모가 0 이면 NaN(«못 쟀다») — 0 으로 읽히면 중단 규칙이 오작동한다.
    """
    from src.training import math_diff as _mdf   # noqa: PLC0415

    n = max(1, len(rows))
    emitted = [r for r in rows if int(_cdr._bool01(r.get("has_meta", r.get("emitted", 0))))]
    parsed = [r for r in rows if r.get("bucket") in _mdf.BUCKETS]
    defined = [r for r in rows if int(r.get("diff_defined", r.get("meta_defined", 0)) or 0)]
    hist = _mdf.bucket_histogram(rows)
    n_b = max(1, sum(hist.values()))
    lbl_hist = _mdf.label_bucket_histogram(rows)
    n_lbl = max(1, sum(lbl_hist.values()))
    _mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    acc = sum(int(r.get("r_corr", 0)) for r in rows) / n
    out = {
        "diff_rows": len(emitted),
        "diff_emit_rate": len(emitted) / n,
        "meta_first_rate": _mean([float(int(r.get("meta_first", 0))) for r in emitted]),
        "bucket_parsed": len(parsed) / n,
        "has_why_rate": _mean([float(int(bool(str(r.get("why") or "").strip()))) for r in emitted]),
        "bucket_max_share": (max(hist.values()) / n_b) if sum(hist.values()) else float("nan"),
        "bucket_entropy": _mdf.bucket_entropy(rows),
        "diff_defined_rate": len(defined) / n,
        "diff_credit_mean": _mean([float(r.get("diff_credit", 0.0)) for r in defined]),
        "stated_vs_agree_spearman": _mdf.stated_vs_agree_spearman(rows),
        "stated_vs_truepass_spearman": _mdf.stated_vs_truepass_spearman(rows),
        "agree_mean": _mean([float(r["loo_agreement"]) for r in rows
                             if r.get("loo_agreement") is not None]),
        "final_acc": acc,
    }
    out.update({f"bucket_{b}": (hist[b] / n_b) if sum(hist.values()) else float("nan")
                for b in _mdf.BUCKETS})
    out["label_bucket_max_share"] = (max(lbl_hist.values()) / n_lbl) if sum(lbl_hist.values()) else float("nan")
    out["label_bucket_entropy"] = _mdf.label_bucket_entropy(rows)
    out.update({f"label_bucket_{b}": (lbl_hist[b] / n_lbl) if sum(lbl_hist.values()) else float("nan")
                for b in _mdf.BUCKETS})
    out.update({f"label_bucket_{b}_n": lbl_hist[b] for b in _mdf.BUCKETS})
    return out


def agree_core_metrics(rows: Sequence[Mapping]) -> dict:
    """M_AGREE 전용 지표 — 학습 텔레메트리(agree_telemetry, uid 그룹)와 held-out 평가
    (math_retry_eval.py, group_id 그룹)가 같은 정의를 공유한다(agree_auc_wrong_mixed·
    agree_calibration_bins 만 eval 쪽에서 추가로 얹는다 — 배치 하나엔 mixed 판정에 쓸
    문제당 샘플이 부족하다)."""
    n = max(1, len(rows))
    is_forced = lambda r: bool(_cdr._bool01(r.get("forced_redirect", 0)))
    unforced = [r for r in rows if not is_forced(r)]
    dec = [r for r in unforced if r.get("decision") in _OPPOSITE]
    ap = [r for r in rows if r.get("agree_pred") is not None]
    at_rows = [r for r in unforced if r.get("agree_true") is not None]
    both = [r for r in rows if r.get("agree_pred") is not None and r.get("agree_true") is not None]
    _rate = lambda xs, pred: (sum(1 for r in xs if pred(r)) / len(xs)) if xs else float("nan")
    # ★agree_auc_wrong: 라벨 = 첫 답 오답(양성), 점수 = 1 − agree_pred — «자기 답이 틀렸다고
    #   예측할수록(형제와 안 맞을 거라 예측할수록) 실제로 틀렸는가»를 잰다. 결정이 파싱된·
    #   비강제 행만(스펙 지시 "decided unforced rows").
    auc_rows = [r for r in dec if r.get("agree_pred") is not None]
    labels = [0 if _cdr._bool01(r.get("first_correct", 0)) else 1 for r in auc_rows]
    scores = [1.0 - float(r["agree_pred"]) for r in auc_rows]
    minority_dec = [r for r in dec if r.get("agree_true") is not None and float(r["agree_true"]) < 0.5]
    majority_dec = [r for r in dec if r.get("agree_true") is not None and float(r["agree_true"]) >= 0.5]
    return {
        "agree_pred_mean": (sum(float(r["agree_pred"]) for r in ap) / len(ap)) if ap else float("nan"),
        "agree_true_mean": (sum(float(r["agree_true"]) for r in at_rows) / len(at_rows)) if at_rows else float("nan"),
        "agree_mae": (sum(abs(float(r["agree_pred"]) - float(r["agree_true"])) for r in both) / len(both))
                    if both else float("nan"),
        "agree_auc_wrong": _rmet._auc_binary(labels, scores) if auc_rows else float("nan"),
        "minority_rate": _rate(at_rows, lambda r: float(r["agree_true"]) < 0.5),
        "redirect_rate_given_minority": _rate(minority_dec, lambda r: r.get("decision") == "redirect"),
        "redirect_rate_given_majority": _rate(majority_dec, lambda r: r.get("decision") == "redirect"),
        # ★수정 4 abort 규칙: agreement: 줄 자체가 있는가(값이 무효라도) — 형식 준수 신호.
        "agree_line_rate": sum(int(_cdr._bool01(r.get("agree_line", 0))) for r in rows) / n,
    }


def agree_calibration_bins(rows: Sequence[Mapping], n_bins: int = 5) -> list[dict]:
    """agree_pred(예측) vs agree_true(실측) 5-bin 보정 곡선 — held-out 평가 전용(스펙 지시).
    둘 다 정의된 행만; 빈 bin 은 n=0/NaN."""
    xs = [r for r in rows if r.get("agree_pred") is not None and r.get("agree_true") is not None]
    buckets: list[list] = [[] for _ in range(n_bins)]
    for r in xs:
        idx = min(n_bins - 1, max(0, int(float(r["agree_pred"]) * n_bins)))
        buckets[idx].append(r)
    out = []
    for i, bx in enumerate(buckets):
        lo, hi = i / n_bins, (i + 1) / n_bins
        n = len(bx)
        out.append({
            "bin": f"[{lo:.1f},{hi:.1f})", "n": n,
            "pred_mean": (sum(float(r["agree_pred"]) for r in bx) / n) if n else float("nan"),
            "true_mean": (sum(float(r["agree_true"]) for r in bx) / n) if n else float("nan"),
        })
    return out


def agree_auc_wrong_mixed(rows: Sequence[Mapping], *, group_key: str = "group_id") -> float:
    """agree_auc_wrong 을 **혼합 문제**(0<first_pass_rate<1, 같은 group_key 안)로만 제한한
    값 — held-out 평가 전용(스펙 지시). 배치 텔레메트리(agree_core_metrics)는 배치 하나에
    문제당 샘플이 몇 안 돼 mixed 판정이 잡음투성이라 이 제한판을 쓰지 않는다."""
    rates = _rmet.group_first_pass_rate(rows, group_key=group_key)
    mixed = {k for k, v in rates.items() if 0.0 < v < 1.0}
    is_forced = lambda r: bool(_cdr._bool01(r.get("forced_redirect", 0)))
    xs = [r for r in rows if str(r.get(group_key)) in mixed and r.get("agree_pred") is not None
         and not is_forced(r) and r.get("decision") in _OPPOSITE]
    if not xs:
        return float("nan")
    labels = [0 if _cdr._bool01(r.get("first_correct", 0)) else 1 for r in xs]
    scores = [1.0 - float(r["agree_pred"]) for r in xs]
    return _rmet._auc_binary(labels, scores)


def agree_telemetry(rows: Sequence[Mapping]) -> dict:
    """M_AGREE 텔레메트리 = retry_telemetry(같은 첫답/판단/재시도 구조) + agree_core_metrics
    (스펙 지시 "telemetry appended to retry keys")."""
    out = dict(retry_telemetry(rows))
    out.update(agree_core_metrics(rows))
    return out


def format_tel(rep: Mapping) -> str:
    def _f(x):
        try:
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "nan"
    return (f"[MATH][TEL] step={rep['step']} arm={rep['arm']} acc={_f(rep['acc'])} "
            f"emit={_f(rep['emit_rate'])} n_blocks_mean={_f(rep['n_blocks_mean'])} "
            f"multi_block={_f(rep.get('multi_block_rate'))} "
            f"decision_rate={_f(rep['decision_rate'])} judge_match={_f(rep['judge_match'])} "
            f"boxed_in_meta={_f(rep['boxed_in_meta'])} len_mean={_f(rep['len_mean'])} "
            f"boilerplate={_f(rep['boilerplate_rate'])} n_emitted={rep.get('n_emitted', 0)} "
            f"meta_val_abs_mean={_f(rep.get('meta_val_abs_mean'))}"
            # ★수정(revision) 팔: 습관률·구제/탈선·사행 감시 지표가 이 줄에 실린다.
            + (f" | rev_rate={_f(rep.get('revision_rate_batch'))} "
               f"rev_member={_f(rep.get('rev_member_rate'))} "
               f"rev_save={_f(rep.get('rev_save'))} rev_derail={_f(rep.get('rev_derail'))} "
               f"rev_shift={_f(rep.get('rev_shift_mean'))} "
               f"rev_credit_rows={_f(rep.get('rev_credit_rows'))} "
               # ★스킵 회계 — credit_rows=0 인 스텝이 왜 0 인지 이 줄만 보고 답한다.
               f"skip[state={_f(rep.get('rev_skipped_state'))} "
               f"multi={_f(rep.get('rev_skipped_multi'))} "
               f"nobox={_f(rep.get('rev_skip_nobox'))} "
               f"unrevised={_f(rep.get('rev_skip_unrevised'))} "
               f"nogold={_f(rep.get('rev_skip_nogold'))} "
               f"anchor={_f(rep.get('rev_skip_anchor'))} "
               f"zone={_f(rep.get('rev_skip_zone'))} "
               f"tok={_f(rep.get('rev_skip_tok'))} "
               f"pmi_nan={_f(rep.get('rev_skip_pmi_nan'))}] "
               f"first_correct={_f(rep.get('first_correct_mean'))}"
               # ★혼합 팔(M_REV_PMI_CF): 두 항이 각각 얼마나 기여하는지 — 한쪽이 0 이면
               #   그 팔은 순수 팔과 바이트 동일한 무효 레버다.
               + (f" rows_pmi={_f(rep.get('rev_credit_rows_pmi'))} "
                  f"rows_cf={_f(rep.get('rev_credit_rows_cf'))} "
                  f"pmi_mean={_f(rep.get('rev_pmi_mean'))} "
                  f"cf_mean={_f(rep.get('rev_cf_mean'))}"
                  if rep.get("rev_credit_rows_cf") is not None
                  and rep.get("rev_credit_rows_pmi") is not None else "")
               if "revision_rate_batch" in rep else "")
            + (f" | first_acc={_f(rep['first_acc'])} final_acc={_f(rep['final_acc'])} "
               f"redirect={_f(rep['redirect_rate'])} "
               f"redirect_all_rows={_f(rep.get('redirect_rate_all_rows'))} "
               f"redirect|wrong={_f(rep['redirect_rate_given_wrong'])} "
               f"redirect|right={_f(rep['redirect_rate_given_right'])} "
               f"judgment_acc={_f(rep['judgment_acc'])} "
               f"second_attempt={_f(rep['second_attempt_rate'])} "
               f"second_attempt|redirect={_f(rep.get('second_attempt_rate_given_redirect'))} "
               f"trunc={_f(rep['trunc_rate'])} n_decided={rep.get('n_decided', 0)} "
               f"forced_frac={_f(rep.get('forced_frac'))} "
               f"forced_first_acc={_f(rep.get('forced_first_acc'))} "
               f"forced_rescue_rate={_f(rep.get('forced_rescue_rate'))}"
               if "first_acc" in rep else "")
            + (f" | selectivity_mixed={_f(rep['selectivity_mixed'])} "
               f"auc_mixed={_f(rep['mean_within_problem_auc'])} "
               f"judgment_acc_mixed={_f(rep['judgment_acc_mixed'])}"
               if "selectivity_mixed" in rep else "")
            + (f" | agree_pred={_f(rep.get('agree_pred_mean'))} agree_true={_f(rep.get('agree_true_mean'))} "
               f"agree_mae={_f(rep.get('agree_mae'))} agree_auc_wrong={_f(rep.get('agree_auc_wrong'))} "
               f"minority_rate={_f(rep.get('minority_rate'))} "
               f"redirect|minority={_f(rep.get('redirect_rate_given_minority'))} "
               f"redirect|majority={_f(rep.get('redirect_rate_given_majority'))} "
               f"agree_line_rate={_f(rep.get('agree_line_rate'))}"
               if "agree_pred_mean" in rep else "")
            # ★M_CRIT: 비평 행 수·정의율·누출율과 정보이득 3종(ig / ig_donor / 그 차).
            + (f" | crit_rows={rep.get('crit_rows', 0)} "
               f"crit_defined={_f(rep.get('crit_defined_rate'))} "
               f"leak={_f(rep.get('leak_rate'))} ig={_f(rep.get('ig_mean'))} "
               f"ig_donor={_f(rep.get('ig_donor_mean'))} ig_delta={_f(rep.get('ig_delta_mean'))} "
               f"crit_words={_f(rep.get('crit_words_mean'))}"
               if "crit_rows" in rep else "")
            # ★M_DIS: 블록·커밋의 형식 두 줄과, 판단 크레딧 세 줄(정의율·평균·소수 선택률),
            #   그리고 지표 전용 두 줄(후보 정오 평균 / 커밋한 후보가 실제 정답이었나).
            + (f" | dis_rows={rep.get('dis_rows', 0)} dis_emit={_f(rep.get('dis_emit_rate'))} "
               f"commit_parsed={_f(rep.get('commit_parsed'))} "
               f"dis_defined={_f(rep.get('dis_defined_rate'))} "
               f"dis_credit={_f(rep.get('dis_credit_mean'))} "
               f"commit_minority={_f(rep.get('commit_is_minority_rate'))} "
               f"cites2={_f(rep.get('cites_two_plus_rate'))} "
               f"diag_words={_f(rep.get('diag_words_mean'))} "
               f"final_acc={_f(rep.get('final_acc'))} "
               f"cand_acc={_f(rep.get('cand_acc_mean'))} "
               f"commit_cand_correct={_f(rep.get('commit_cand_correct_rate'))}"
               if "dis_rows" in rep else "")
            # ★M_DIFF: 형식 세 줄(발화·메타가 맨 앞인가·버킷 파싱)과 버킷 분포(히스토그램·
            #   최빈 점유율·엔트로피), 크레딧 두 줄, 그리고 상관 두 줄(신호 / ★지표 전용 gold 판).
            + (f" | diff_rows={rep.get('diff_rows', 0)} diff_emit={_f(rep.get('diff_emit_rate'))} "
               f"meta_first={_f(rep.get('meta_first_rate'))} "
               f"bucket_parsed={_f(rep.get('bucket_parsed'))} why={_f(rep.get('has_why_rate'))} "
               f"buckets={_f(rep.get('bucket_easy'))}/{_f(rep.get('bucket_medium'))}/"
               f"{_f(rep.get('bucket_hard'))} max_share={_f(rep.get('bucket_max_share'))} "
               f"H={_f(rep.get('bucket_entropy'))} "
               f"lbl=e/m/h={rep.get('label_bucket_easy_n', 0)}"
               f"/{rep.get('label_bucket_medium_n', 0)}"
               f"/{rep.get('label_bucket_hard_n', 0)} "
               f"H={_f(rep.get('label_bucket_entropy'))} "
               f"diff_defined={_f(rep.get('diff_defined_rate'))} "
               f"diff_credit={_f(rep.get('diff_credit_mean'))} "
               f"rho_agree={_f(rep.get('stated_vs_agree_spearman'))} "
               f"rho_truepass={_f(rep.get('stated_vs_truepass_spearman'))} "
               f"final_acc={_f(rep.get('final_acc'))}"
               if "diff_rows" in rep else "")
            # ★M_RETRY_SL: SL 항이 몇 행에 얹혔고(sl_rows) 그 중 몇이 자기라벨과 일치했나
            #   (sl_match_rate). 1.0 에 붙으면 밀 것이 없고, 0 에 붙으면 −W 만 계속 얹힌다.
            + (f" | sl_rows={rep.get('sl_rows', 0)} sl_match={_f(rep.get('sl_match_rate'))}"
               if "sl_rows" in rep else ""))


def check_abort(rep: Mapping, *, arm: str) -> list[dict]:
    """위반 목록(빈 리스트 = 통과). 지표가 NaN 이면 «못 쟀다» — 죽이지 않고 missing 으로."""
    spec = require_arm(arm)
    out = []
    for name, rule in ABORT_RULES.items():
        if rule["meta_arms_only"] and not spec.get("require_meta", False):
            continue
        if rule.get("arms") is not None and arm not in rule["arms"]:
            continue                      # 팔 전용 규칙(재시도 팔) — 다른 팔엔 missing 도 찍지 않는다
        # ★min_step_by_arm: 같은 규칙을 팔마다 다른 워밍업으로 — SL 팔은 15(위 주석).
        _min_step = rule.get("min_step_by_arm", {}).get(arm, rule.get("min_step"))
        if (_min_step is not None
                and (rule.get("min_step_arms") is None or arm in rule["min_step_arms"])
                and int(rep.get("step", 0) or 0) <= int(_min_step)):
            continue                      # 워밍업(min_step_arms 가 있으면 그 팔에만)
        thr = rule["thr"]
        if rule.get("env_thr"):
            ev = os.environ.get(rule["env_thr"])
            if ev is None or ev == "":
                continue                      # 문턱 미설정 → 규칙 불참(감사 7)
            thr = float(ev)
        v = rep.get(name)
        if v is None or not _cdr._finite(v):
            out.append({"metric": name, "status": "missing", "value": v})
            continue
        if rule.get("min_n"):
            nkey, nmin = rule["min_n"]
            if int(rep.get(nkey, 0) or 0) < nmin:
                out.append({"metric": name, "status": "missing", "value": float(v),
                            "note": f"{nkey}={rep.get(nkey)}<{nmin}"})
                continue
        bad = (float(v) < thr) if rule["op"] == "<" else (float(v) > thr)
        if bad:
            out.append({"metric": name, "status": "abort", "value": float(v),
                        "thr": thr, "why": rule["why"]})
    return out


def update_streak(streak: dict, key: str, hits: Sequence[Mapping], *, patience: int | None = None) -> bool:
    """연속 위반 카운터 — patience(기본 countdown_rewards.get_abort_patience(), 3)에 닿으면 True.
    호출자(verl_sdc)가 True 를 받으면 `_CountdownAbort` 를 던진다(rc 75 경로 재사용)."""
    if patience is None:
        patience = _cdr.get_abort_patience()
    if hits:
        streak[key] = streak.get(key, 0) + 1
    else:
        streak[key] = 0
    return streak.get(key, 0) >= patience


# ── non_tensor_batch 컬럼 읽기(flat → extra_info 폴백) ────────────────────────
_NT_COL_MISSING = object()


def nt_col(nt: Mapping, name: str, default=_NT_COL_MISSING) -> list:
    """★verl 0.7.1 async 경로는 flat 컬럼을 gen_batch 로 pop 해 버린다 — `extra_info`
    안의 같은 값으로 폴백한다(COUNTDOWN `_col` 과 같은 이유). 둘 다 있고 어긋나면 즉사.

    ★0914 forced_redirect: `default` 를 주면 컬럼이(flat·extra_info 둘 다) 통째로 없을 때
    조용히 [default]*len(batch) 를 돌린다 — 구 parquet(강제탐색 이전)·비-RETRY 팔과의 호환.
    len 은 uid(항상 존재해야 하는 컬럼)로 잡는다. `default` 미지정이면 기존과 동일하게 즉사."""
    v = nt.get(name, None)
    ei = nt.get("extra_info", None)
    alt = None
    if ei is not None:
        try:
            cand = [(e or {}).get(name, None) for e in list(ei)]
            if not all(x is None for x in cand):
                alt = cand
        except Exception:
            alt = None
    if v is None:
        if alt is None:
            if default is not _NT_COL_MISSING:
                uid = nt.get("uid", None)
                n = len(list(uid)) if uid is not None else 0
                return [default] * n
            raise RuntimeError(
                f"[MATH] non_tensor_batch 에 '{name}' 컬럼이 없다(extra_info 폴백도 실패). "
                f"사용 가능한 키: {sorted(nt.keys())}. scripts/local/build_math_parquet.py 로 "
                "빌드한 parquet 인지 확인하라.")
        return list(alt)
    v = list(v)
    if alt is not None:
        bad = sum(1 for a, b in zip(alt, v) if str(a) != str(b))
        if bad:
            raise RuntimeError(
                f"[MATH] '{name}' 이 flat 컬럼과 extra_info 에서 다르다({bad}/{len(v)} 행) — "
                "어느 쪽으로 학습했는지 사후 확정이 불가능해지므로 즉사한다.")
    return v
