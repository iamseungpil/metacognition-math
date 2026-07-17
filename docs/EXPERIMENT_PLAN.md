# EXPERIMENT_PLAN — RQ3 매치드 래더 전방 계획표 (2026-07-17 신설)

이 문서는 **앞으로 할 일의 단일 계획표**다. 과거 이력은
`docs/redesign/EXPERIMENT_LOG.md`(상태원장, §11이 최신)에 있고, 여기는
"논문이 서려면 무엇이 더 필요한가"를 리뷰어 관점(soundness/AC 렌즈)으로
역산해 실험·분석·한계명시로 배정한 표만 둔다. 판정 규약은 불변:
**모든 최종 판정은 gs300 held-out 1030(16k, avg@8, math_verify)에서만** —
in-training val594는 모니터링 전용이다. 커밋 시점 기준 클러스터는
재할당 후 VC 미복원("VC does not exist", 0715 22:41~)으로 차단 중이며
재발사는 자동루프가 담당한다.

## 1. 핵심 claim 3개 × 필요 증거 × 현재 상태

| # | claim | 필요한 증거 | 현재 보유 | 판정 |
|---|---|---|---|---|
| C1 | meta-SFT init이 matched base를 held-out 정확도에서 이긴다 (RQ1 = B2−B0) | B0·B2 gs300 완주 → held-out 1030 페어드 비교 + **corpus 통제**(동일 코퍼스 meta-stripped twin) + 다중 시드 | B0 gs300 완주(단 held-in val594 숫자뿐, geometry −.152/precalc −.081/omni −.587 도메인 파탄 포함). B2는 gs150 선점사·완전 재개점 gs140(HF)·재발사 대기. corpus confound 확정(gold 1,290행 vs RV 1,763행). 단일 시드 | **비교 자체가 아직 없음** — E5(eval)+E1(B1 twin)로 닫는다 |
| C2 | 메타 보상 패키지가 correctness-only RL 대비 추가 향상 (RQ2 = B3pkg−B2) | B3pkg·B2 gs300 held-out + 정규화 대칭 통제 | **B3pkg 증거 = 0 (gs0 미발사)**. C-1로 정규화 비대칭(B0/B2=std-norm, B3=mean-only) 구조 내장 — 수정 안 함(고치면 B2 전후반 갈라져 RQ1 오염). 논문 초안의 대응 숫자(+18.8pp 등)는 삭제된 instruct 세대 ckpt 산출물로 not-certifiable | **증거 0** — B3pkg 발사가 유일 해법, C-1은 E6+E2로 완화 |
| C3 | 향상이 in-reasoning 자기검증(메타 행동)에 인과 귀속된다 | pmi 격리 대조(B3pkg−B3-noPMI) + 행동-정확도 결합 증거 + (이상적으로) placebo/dedup arm·독립 judge | **전무.** 사전등록 shift-only는 §9에서 실패·폐기, B3-noPMI 보류·미발사, placebo/dedup arm은 config조차 없음, 32B judge 미실행 | E2(minus-one 대조)+E4(행동인과분석)로 부분 확보; 못 닫으면 주장을 "6-head 패키지 효과"로 강등 |

## 2. 실험 계획표 (기존 4-arm + 보강 E1–E6)

기준 비용: 300스텝 1암 ≈ 220 GPU-h (H100×4, ~55h wall). 기존 확정 잡(B2
재개 ≈117 GPU-h, B3pkg 220 GPU-h)은 아래 신규 예산과 별도로 이미 큐에 있다.
신규 파일은 전부 **추가만**(src/·configs/·h100std_*.yaml 기존 파일 byte-동일
유지) → tarball 재패키징 → 새 CODE_TAR_REVISION으로 제출.

### 2.1 기존 4-arm (확정 잡)

| arm | init / config | 명령 델타 | 상태 (0717) |
|---|---|---|---|
| B0 | b0_gold_sft(공개 HF gold 1,290행) + VANILLA_GRPO (`base_matched_grpo_h100_4x4k`) | — | **gs300 완주**, 재제출 금지. held-in val594: gsm8k .768 / algebra .658 / counting .886 / geometry −.152 / precalc −.081 / omni −.587 |
| B2 | b23_rv_unmasked_sft(RV 1,763행) + 동일 VANILLA_GRPO | B0 대비 init 경로만 상이 | gs150 선점사, **완전 재개점 gs140(HF)**, 재발사 대기(자동루프) |
| B3pkg | 동일 meta-SFT + TRIOBJ_DCPO_V4 풀패키지 (w_meta .8 / w_format .35 / w_emit .1 / w_cal .3 / len .08 / trunc_open .3 / w_over 0 / rmeta=pmi_shift) | `--config-name=triobj_dcpo_v4_stage3b_h100_4x4k` + `++algorithm.dcpo_rmeta_source=pmi_shift` | **미발사(gs0)** — 논문 핵심, VC 복원 즉시 최우선 발사 |
| B3-noPMI | 풀패키지 − pmi (w_meta=0) | b3pkg 런처 + `++algorithm.dcpo_w_meta=0.0` | **보류**(b3pkg 우선, 사용자 결정 0715) — E2로 승격 예정 |

### 2.2 보강 실험 E1–E6

| ID | 목적 (닫는 약점) | init / config | 명령 델타 | GPU-h | 우선순위 | 판정 규칙 | 상태 |
|---|---|---|---|---|---|---|---|
| **E5** held-out 1030 최종판정 | 판정 절차 고정 — val594(보상-오염 지표) 대신 논문 수치의 유일 원천 확립 (C1·C2 공통) | gs300 ckpt를 HF에서 snapshot_download → `scripts/eval_vllm_1030.py` (gsm8k+math500+aime2024, 16k, temp .7, num_samples 8, tp 4, seed 42) | 오버라이드 금지·완전 고정 커맨드; 집계 avg@8 + truncation% + emission rate + Brier; 문제단위 클러스터 부트스트랩 10,000회(벤치 층화) | 암당 ~20 (B0/B2/B3pkg 필수 ≈60) | **P0** | 95% CI가 0을 제외할 때만 "유의" 기술 | 각 arm gs300 도달 즉시 수행 (ckpt HF 내구 저장 — 선점 무관 재시도 가능). P1 부속: SFT-init 3종 eval(+60)로 SFT vs RL 기여 분해 |
| **E1** B1 arm: meta-스트립 RV corpus + VANILLA_GRPO | **RQ1 corpus confound** — B2−B0은 corpus까지 다른 비교. B1은 동일 RV 1,763행에서 메타만 제거 → B2−B1 = meta 토큰 순수 효과 | 신규 `scripts/build_b1_rv_stripped_data.py` → `data/b1_rv_stripped_sft.parquet` → `configs/sft_b1_rv_stripped.yaml`(sft_b23 복사, 경로만) → `h100std_rq3_b1.yaml`(b2 복사) | 2단 스트립: ① meta 블록 정규식 제거(기존 build_base_rv_sft_data 로직 재사용) ② hollow-meta 평문 라인(confidence/assessment/action/decision/study_need 접두 + 고아 `<\|im_start\|>`) 제거. 빌드 게이트 assert: 행수 1,763 보존·메타 잔류 0·\boxed 답 불변·user 턴 byte-동일. RL은 B2와 동일 오버라이드, init 경로만 `models/b1_rv_stripped_sft` | ~252 (SFT 12 + RL 220 + eval 20) | **P0** | B2>B1 유의 → "meta 토큰 인과" 승격. B2≈B1>B0 → 주역은 RV prose(정직 기술 전환). B1≈B0 & B2>B0 → 토큰이 주역. B2<B1 → meta 토큰 순손해 보고 | 데이터빌드+SFT config는 CPU 작업 — VC 복원 대기 중 지금 수행 가능 |
| **E2** B3-noPMI 발사 | **패키지 귀속** — RQ2는 6-head 패키지 효과라 pmi_shift 단독 귀속 금지. minus-one 대조로 pmi 순기여 격리 (사전등록 shift-only는 §9 폐기 — 논문 §4.2 결정규칙을 minus-one 방향으로 재작성 필요) | 기존 `h100std_rq3_b3nopmi.yaml` 그대로(이미 존재·미발사) — 신규 수정 없음 | b3pkg 대비 `++algorithm.dcpo_w_meta=0.0` 하나 | ~240 (RL 220 + eval 20) | **P0** | B3pkg>noPMI 유의 → pmi_shift 순기여 확립. ≈ → pmi 단독 귀속 철회·패키지 효과로 기술. < → negative attribution도 보고 | B3pkg 완주 확인 후 발사(동일 VC 슬롯). B3pkg가 B2 대비 무효과면 정보가치 재평가 |
| **E4** 행동 인과 분석 (학습 없음) | "emission은 장식" 반론 — within-problem 비교로 난이도 confound 제거한 메커니즘 증거 (인과 확정 아님을 명시) + **H-2/M-1 caveat 정량 폐쇄** | E5 per-sample parquet(문제×8샘플: completion·is_correct·num_meta_blocks·meta_confidences·finish_reason). 대상 B2·B3pkg, B0=음성대조 | 신규 `scripts/analyze_meta_causality.py` — ① 문제내 paired emission 효과(emit vs no-emit, 클러스터 부트스트랩) ② wrong→right 전이(마지막 meta 블록 이전 prefix의 중간 후보답 기준) ③ 유령 confidence 감사(meta 밖 `confidence:` 검출률) + pmi dup/reversal 발생률 실측 ④ (옵션) gs100/200 종단 emission↔accuracy | 0 (옵션 ④만 +40) | **P0** | ①Δ>0 유의 ∧ ② emit군 전이율 상회 → "행동 메커니즘" 절 추가. null → 이득이 emission 경유 아님을 정직 기술. B0 emission>5%면 파서 오탐 조사 선행 | **지금 즉시 착수 가능(GPU 0)** — 스크립트 작성은 VC 차단과 무관 |
| **E3** 멀티시드 | 전 셀 단일 시드 — 문항 부트스트랩은 학습 stochasticity 미커버 | 헤드라인 대비쌍만: B0·B3pkg 각 +1 시드. 런처 시드 오버라이드만, 그 외 byte-동일 | trainer/rollout sampling seed + 데이터 셔플 시드 | ~480 (2런 440 + eval 40) | **P1** | 두 시드에서 B3pkg−B0 부호 동일 ∧ CI 중첩 → "2시드 재현"으로 caveat 교체. 부호 불일치 → 3rd 시드 또는 "단일시드 예비" 유지. 시드간 분산은 부트스트랩 CI와 별도 보고 | 미착수 — P0 완료 후 |
| **E6** C-1 비대칭 정량화 문서 | RQ2에 내장된 정규화 비대칭(B0/B1/B2=std-norm, B3=mean-only) — **수정 불가**(고치면 B2 전후반 분열) | wandb 로그 분석만 | 양 경로 advantage 스케일 실측 → limitation 절 정량 각주. E2(noPMI도 mean-only)가 RQ2 내부 대조를 정규화-동일로 만들어 부분 상쇄한다는 논리 병기 | 0 | **P2** | — (문서 작업) | VC 차단 중 지금 수행 가능 |

**발사 순서**(선점 환경 최적화): 지금(CPU) = E1 데이터빌드 + E4 스크립트 +
E6 → VC 복원 즉시 = E1 SFT(단시간·선점위험 최소) → RL 큐 = **B3pkg(논문
핵심) → B2 재개 → E2 → E1-RL → E3**. 각 gs300 도달 시마다 E5 즉시 수행.
P0 신규 합계 ≈ 550 GPU-h (기존 확정 잡 ~340 제외).

## 3. 약점 대응표 (리뷰어 렌즈)

### 3.1 구조적 reject-근거 (이대로 제출 시 즉사)

| 약점 | 대응 수단 | 배정 |
|---|---|---|
| 증거 기반 부재: 채워진 결과 전부가 0714 감사에서 not-certifiable 판정·삭제된 instruct 세대 ckpt 산출물 | **실험** — B2 재개·B3pkg 발사 → gs300 → E5 페어드 eval로 전량 교체. 그 전엔 instruct 숫자를 "pilot, decertified" 명시 강등 | E5 + 확정 잡 |
| 인과 주장 무근거: 제목 "causally helps"에 대응하는 분해 arm 전멸/보류 | **실험** — E2(minus-one) + E4(행동 메커니즘). 최소 방어선: 제목·abstract를 package-level로 강등 | E2·E4 |
| RQ2 정규화 비대칭(C-1): 비교에 구조 내장, 수정 불가 결정 | **분석+한계명시** — E6 정량 각주 + E2가 RQ2 내부 대조를 정규화-동일 조건으로 만듦 | E6·E2 |
| method 서술-설계 모순: "byte-audited 같은 데이터 twin" 주장 vs 실제 corpus confound(1,290 vs 1,763행) | **실험** — E1(B1 twin)이 원래 twin 설계를 실현. 불가 시 method twin 문단 삭제 + RQ1을 "meta-SFT 레시피 효과"로 재정의 | E1 |
| 전 arm 단일 시드 headline | **실험(부분)** — E3 헤드라인 쌍만 2시드. 나머지는 한계명시 + 효과크기 큰 셀에 주장 한정 | E3 |

### 3.2 표현/정리 수준 (문구 수정으로 해소)

| 지적 | 대응 |
|---|---|
| head 수 표기 불일치(논문 "seven-head" vs 현행 6-head, w_over=0) | 한계명시 아님 — 논문 문구 통일 |
| RQ 넘버링 3중 충돌(논문 RQ1–5 / 현행 계획 RQ1–2(+격리) / 사이트 RQ2–3) | 하나로 통일, 세대 표기 병기 |
| limitation 누락: H-2(유령 confidence 파서), M-1(pmi 가드 OFF), save_freq 비대칭(b3=5 vs b0/b2=10), w_cal=.3의 gold→Brier 지도가 "gold는 correctness에만" 헌법과 긴장 | 전부 내부 감사에 이미 존재 — 논문 limitation 절에 반영. H-2·M-1은 E4-③에서 발생률 실측으로 정량화 |
| discussion 미이행 약속(raw-base 참조행, 다중 시드, placebo bound, 토큰예산 통제열) | 이행분(E3·E5 부속)은 반영, 나머지는 future work로 명시 이동 |
| val594 숫자 혼용 위험 | val594=모니터링 전용 1줄 명시 (판정 규칙은 이미 명문화됨) |

## 4. 설명 사이트(metacog-explainer.pages.dev) 불일치 목록

사이트는 ~0706 T1/pre-audit 스냅샷이다. **사이트 수정은 이 커밋 범위 밖** —
갱신 필요항목만 기록한다.

1. **arm 구조 stale**: 현행 4-arm 래더(B0/B2/B3pkg/B3-noPMI) 대신 구세대
   pmishift-vs-matched-base 2-arm + pending B/G/S/P 분해를 표시. "B0", "B2",
   "b3pkg", "TRIOBJ_DCPO_V4", "VANILLA_GRPO", "Qwen3-8B-Base" 전부 미등장.
2. **삭제된 ckpt 인용**: footer의 `pmishift/gs300`·`pmishift_1030_v2`는 0714
   감사로 HF에서 전삭제·not-certifiable 판정된 산출물 — 제거 또는
   "decertified" 주석 필요.
3. **RQ 넘버링 충돌**: 사이트 "RQ3"=난이도 층화, 현행 RQ3=4-arm 래더;
   사이트 "RQ2"=구 B/G/S/P 분해 — 현행 정의로 재넘버링.
4. **상태 stale**: RQ1 6/6(+18.8pp 등)을 라이브 결과로 표시, RQ2 "PENDING".
   현실: B0 gs300 완주(held-in만), B2 재발사 대기, B3pkg gs0, 클러스터 차단
   — 미반영.
5. **0716 3중감사 caveat 전부 누락**: C-1 정규화 비대칭, corpus confound,
   w_cal-gold 긴장, H-2, M-1, 단일 시드. (패키지 귀속 caveat만 부분 존재.)
6. **레시피/판정규칙 부재**: v2 레시피(temp1.0/top_k−1/resp8192/lr1e-6/
   300steps), B3pkg head 가중치, "판정은 gs300 held-out 1030 avg@8 16k에서만"
   프레이밍 없음.
7. (확인됨) 우려했던 stale "+0.042"는 사이트에 없음 — lineage의 CF Δ+0.040은
   0624 확정치와 일치, 문제는 결과 레이어 전체의 세대 불일치다.

## 5. 공정 준수사항

기존 암의 보상·학습 정의 무변경(E1·E3는 v2 매치드 레시피 재사용, E2는
기제정 스펙 발사만). 신규 파일 추가 후 tarball 재패키징 → 새 release asset
id로 CODE_TAR_REVISION 갱신(구 파일 byte-동일 감사 필수). src/·configs/·
h100std_*.yaml 기존 파일 수정 금지. 커밋은 로컬만, 토큰은 .env 외 기록 금지.
