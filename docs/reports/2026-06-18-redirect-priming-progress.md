# Redirect-Priming v2 — 진행 결론 보고서 (2026-06-18)

**프로젝트**: metacognition-math (ctsd-phase-c). **북극성**: 유용한 메타인지(redirect)를 강화해
수학 정확도를 올린다 — calibration·form·likelihood는 부분 목표일 뿐, 정확도 인과가 목표.

---

## 1. 무엇을 알아냈나 (v1·v2 A/B 결론)

- **v1 (메타 죽은 베이스라인) step300 1030-eval = overall 0.651** (gsm8k 0.893 / math500 0.447 /
  aime 0.029). e4_baseline 0.786에 **−0.135 미달**, 특히 어려운 문제(math500/aime)에서 크게 뒤짐.
- **메타는 켜지면 정확도를 깎았다**: 코드가 이미 `acc_with 0.71 < acc_without 0.81 (meta net HARMFUL)`을
  측정해뒀음. s3b 라이브: PMI≈0.04, `verify_execution≈−0.66`(메타가 "검산"을 말만 하고 실행 안 함).
- **forming 붕괴 메커니즘**: RL은 메타에 **비용(길이·기회비용)은 있고 효용(PMI)은 없으니 합리적으로
  잘라낸다.** v1 wf 0, v2(forming 처방)도 wf 0.66→0.05로 붕괴. 처방은 절벽을 step80→145로 늦췄을 뿐.
- **두 보상 다 실패**: 반사실(acc_with−acc_without)은 sparse(~95% 0)·느림·leak guard 50–75% ungrade로
  기각됐고, 그 대체인 PMI(우도)는 정답과 무관·길이 인플레·net-harmful.

**핵심 결론**: 메타가 죽는 건 *내용이 hollow*(정답을 실제로 안 바꿈)라서다. 처방·floor로 떠받치는 건
미봉책. 진짜 해법 = **메타를 기능하게 만들어 RL이 살려둘 이유(정답 개선)를 주는 것**.

## 2. 무엇을 하기로 했나 (Redirect-Priming v2 설계)

실패 롤아웃에서 redirect가 **정답을 인과적으로 뒤집은 사례만** 골라 SFT로 주입 → RL은 그 redirect가
정답을 도울 때만 보상 → 효과를 **R−B′**(matched 2차 시도 대비)로 측정. 5라운드 ultracode 적대검토로
spec REV-6 확정(확정결함 25→30→18→18→13, critical 4→3→3→2→1→해소). 키스톤:
- **`<|switch|>` 토큰**: redirect를 이산화 → 차단가능(ablation)·계수가능·보상 게이팅.
- **반사실 정답이득 보상**(PMI 폐기): redirect가 정답을 도울 때만 +; 불필요 redirect는 −.
- **R−B′ 추정량**: 2차 시도 affordance를 통제(B′=평문 재시도 허용+switch만 −inf 차단) → 자기일관성·
  off-policy 퇴화 교란 제거.
- **PRE-GATE 7종**(PG0 수율~PG6 RL-headroom): GPU 쓰기 전 경험적 STOP 게이트(특히 PG1 분리성·PG0 수율).

## 3. 무엇을 만들었나 (구현 ④단계)

GPU 불요 핵심 로직 전부 TDD 구현 + **108+ 테스트 통과**, 커밋 다수:
- `switch_ban_processor`(−inf), `redirect_behavior_detector`(행동≠토큰, LLM judge primary),
  `cf_stats`(방향성+검정력 McNemar·parse·degeneracy gate), `rml_verdict`(R−B′ 판정),
  `redirect_cf.redirect_cf_rmeta`(반사실 보상, C1-C5 게이트), `redirect_tripwires`(C7 gap-gaming halt),
  `harvest` 로직(lower-CI+B′arm 채택), `segment_loss_mask`(prompt+prefix 마스킹), config(T10), PG0 드라이버.
- **★ultracode 의도부합 점검 2회전이 "비인과 redirect를 성공으로 점수하는 북극성 역전" 버그(탐지기가
  토큰에 반응·방향성없는 통계·보상이 emit게이트 누락 등 24개)를 본실험 GPU 전에 잡아 수정.**

## 4. 지금 어디 (⑤ PRE-GATE)

- **PG0(수율 파일럿)** H100 노드 가동 시도 중 — basicvc Standard **용량 고갈**(좌비 Running 잡 + v2 점유)로
  2회 무로그 즉시 실패, 재제출(pg0-yield-3)해 용량 대기. PG0가 GO/STOP을 내면 본실험 진입 여부 결정.
- **v2(s3b-v2-resume-1)**: 메타 죽은 채 step~234 순항. step300 eval은 v1 0.651과 유사할 가능성↑(uninformative).

## 5. 다음
PG0 GO → T7(`<|switch|>` 토큰 체인)+최소 prime → PG1(분리성) → T9 라이브 통합(cf_prefix_agent/verl_sdc) →
PG2/4/5/6 → **모든 PG 통과 시에만 autoresearch로 본실험**. PG0/PG1 실패 시 = 현 설계 무효, 재설계.
