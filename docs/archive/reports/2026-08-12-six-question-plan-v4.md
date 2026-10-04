# 여섯 질문 검정 계획 v4 — 최종 편집본 (2026-08-12 사전등록)

## ① 요약: 여섯 질문의 커버리지

이 계획은 "전(前)-행동 복원 사다리"로 여섯 질문을 연다. **Q1(SFT2 범인?)** 은 N4(SFT2-init gs0 eval, C-017/C-018 의 네 번째 칸)와 E0′(EXP-0812c 잠정 인용)으로 관측을 확보하고, 개입 검정(SFT1-init RL·언마스크 SFT2)은 W5 조건부로 이월한다. **Q2(DCPO 경로 범인?)** 는 W0~W4 어디서도 열리지 않으며(N2 b3null 은 좌표 1점일 뿐 Q2 가 아님 — E-173), 유일한 계기는 W5 무조건 후보인 bfmt 분해 사다리(E-174 지정)다. **Q3(pmi_shift 범인?)** 은 T2b(전제복원 w_meta=0 짝팔)가 열되, 이중 조건부다 — T2(또는 T2e) 성공 그리고 검정력 패키지(MDE ≤ 기대 효과) 승인; 미충족이면 방향성 관측으로 강등되고 Q3 은 이 계획 계보에서 미결로 남는다. **Q4(메타 내용 범인?)** 는 N1 강제-계속(R vs N′ vs Nc, 층화·검정력 명세 포함)이 주장 수위 조정 역할로 연다. **Q5(전-행동 부재가 상류 원인?)** 는 이 계획 안에서 복원-가능성 절반만 닫는다 — T2(b4shift, 전제복원 풀패키지)가 "RL 레버로 복원 가능하고 파괴적이지 않은가"를, T1(b4inj, uid-수준 부분 주입 ρ=0.5)이 하니스 강제 하 구조 대조를 검정하며, 인과 절반("복원된 전제 위에서 pmi_shift/cal 이 기여하는가")은 T2b 로만 열린다. **Q6(보상 헤드 기능?)** 은 E1(ECE 재인용, 귀속 불가 서술)의 관측 절반과 §6 처리표(R_cal 결함 11종, ⑥ 참조)에 T3(b4cal, 수리된 R_cal — winner 존재 시 조건부)의 개입 절반을 더해 커버한다 — T2·T1 전멸 + 2차 재기준선 실패 경로에서는 Q6 개입 검정이 W5 로 이월된다.

---

## ② 실험 표

| 이름 | 여는 질문 / 역할 | GPU | 학습 | kill / 반려 기준 | 판정식 |
|---|---|---|---|---|---|
| **N0** b3shf gs300 내구 보존 | N3 의 전제 (자산 조치) | 0 | 아니오 | — | — (보존 목적은 N3 전제로 축소; E4 입력 역할 삭제) |
| **E0′** EXP-0812c 인용 | Q1 위치축 — N4 교차검증 전 provisional | 0 | 아니오 | — | 잠정 표기 인용 |
| **E1** ECE 재인용 | Q6 관측 절반 | 0 | 아니오 | — | 귀속 불가 서술로만 인용 |
| **E2** instruct 승자 prefix 문턱 산출 | T2 형상화 스케일 분모 후보 | 0~4 | 아니오 | — | 산출값 = T2 형상화 스케일 분모 후보 |
| **E3** init 롤아웃 적격률 + 억제 우회율 프로브 | T2 문턱·조기 kill 기준선 + **T1 설계 반려 게이트** | 4 | 아니오 | **반려 문턱(사전등록)**: pass-1 억제 하 내용 시그니처율 >10% 또는 any-flag Δ >5pp ⇒ W3 발사 전 T1 설계 반려(사용자 결정) | 문턱 후보 {8,12,16,24} 적격률([prefix≥문턱 AND 발화]) 실측 → 적격률 ≥30% 최대 문턱을 T 로 채택, 전부 미달 시 T=8. 우회율은 표본 ≥500행, per-row logit_bias={151669, close}=−100, C-033 배터리(닫는태그 잔존·decision: 필드·내용 시그니처·degeneracy_flags·비억제 대조 TOST) 채점. eval-측 스크립트, 정본 무수정 |
| **E4′** run-level 잡음 감사 (재설계) | σ_run 유효 추정기 존재 여부 판정 | 0 | 아니오 | — | 잡 로그 감사: 독립 재실행(resume 배제)이 같은 gs 산출한 지점 실존 판정. b3sh/b3shf 쌍 사용 금지(w_format 처치효과 혼입, E-152). 유효 지점 존재 ∧ 비교쌍 ≥3 ⇒ σ_run 하한·σ_total²=σ_eval²+σ_run². 비교쌍 ≤2 ⇒ 산출하되 '실측 성공' 불인정, 경계 승격 병행. **부재(기본 기대) ⇒ 폴백 무조건 발동**: 전 게이트 "CI=eval 잡음만" 명기 + 경계 사례 사용자 승격 |
| **E5** len_cost git 직독 | C-029 모순 해소 → T2 노브 확정 | 0 | 아니오 | — | 분기문에 병기: len_cost 는 코드 직접성 1위 후보이나 C-030 길이 5분위 분해와 불일치(Q4 −4.62·Q5 소멸) — len_cost=0 채택은 승자 패리티 원칙이지 −2pp 주범 제거 기대 아님 |
| **E6** 레거시 앵커 재채점 | CONF 전제 — 채점기 동일성 | 0 (CPU) | 아니오 | — | v3 유지 |
| **N4** SFT2-init gs0 eval | Q1 — C-017/C-018 네 번째 칸 | 4 | 아니오 | — | v3 유지 |
| **N1** 강제-계속 R/N′/Nc | **Q4** (주장 수위 조정 — 발사 차단 아님) | 4 | 아니오 | MDE > 0.034 인 채 비유의 ⇒ "분기 미발동" 자동 결론 금지, 사용자 결정 승격 | 층화 1차 = 사전-처치 level(L1-2 vs L4-5). 파일럿(첫 50문항×8)으로 ICC→MDE 산출, 층별 목표 n 을 MDE ≤ 0.034(PG0 hard 효과)로 사전 배정. 메타 유형(verify/redirect/기타)×난이도 구성 로깅. NEGATIVE 스코프: "이 수확 분포(붕괴 레짐 meta-first 자기수확)의 메타 내용에 한정" + degeneracy_flags 병기 의무 |
| **N2** b3null gs303 eval | 좌표 1점 + 첫 줄 검사 (**Q2 아님** — E-173) | 4 | 아니오 | — | v3 판정식 유지 — 양 분기 귀속 금지 |
| **N3** b3shf gs300 eval | Q6 보조 좌표 | 4 | 아니오 | — | v3 유지 |
| **T2** b4shift — 전제복원 풀패키지(cal-off) + 연속 형상화 floor | **Q5 복원-가능성 절반** | 4 (1노드 80G4) | **예** | 조기 무정보 kill: gs50 적격밀도 < max(2×E3, 5%) ∧ prefix_len_median 상승 추세 부재 ⇒ gs100 관찰 후 중단, T1 승격. kill-1 = 패키지 판정(포렌식 국소화). kill-2 = 발화<0.5(엔트로피 교란 미발동 조건): **구성 수준 NEGATIVE**(floor 0.05·선형 형상·T=채택값·단일 시드) — 동일 구성 재발사 금지, 금지 ≠ 원리 확정, 원리 승격은 W5 변주 재현 후만. 공통 kill: discard>20%·분리 skip 카운터 | 성공 = gs300 prefix 중앙값 ≥T ∧ 발화 ≥0.8 ∧ Δ(vs b2p) 문항-대응 클러스터 부트스트랩 CI 하한 > −1.84pp. 성공의 Q5 매핑 = 복원-가능성 절반만(인과 절반 미결). "prefix 복원 + 정확도 b3s 동등" = NO 방향 약한 증거 + 패키지 교란(단독 귀속 불가) 의무 병기. Δ(b4shift−b3s)>0 유의 = "전제복원 풀패키지+cal-off 효과"로만 명명. **보류(정확도 실패 ∧ 엔트로피>밴드) ⇒ T2e 발동 + T1/T3 규칙 전환** |
| **T2e** (조건부) T2 + entropy_coeff=0 | C-028 대응 레버 — T2 보류 시에만 | 4 | **예** | T2 와 동일 | T2 판정식 재적용 + "엔트로피 레버 조건부" 명명 의무. 생존 시 T1 의 Δ 기준선 승계·T3 winner 후보. 이것도 보류/실패 ⇒ Q5 는 "이 기질·이 레버 집합으로 판정 불능" 종결 |
| **T1** b4inj — T2 구성 + uid-수준 부분 주입(ρ=0.5) | **Q5** + 훈련-내 구조 대조 | 4 | **예** | 공통 kill + 1-step 스모크 6항(그룹 동질성 ∈{0,1}·커버리지 ρ±0.05 / fmt classify / region 라우팅 눈검사 / 런처 가드 / discard 1스텝 / C-033 배터리) 불통과 시 미발사 | **주 판정 = Δ방출률(T1−T2)** — 동일 gs300 하니스-없는 eval·동일 문항·문항-대응 클러스터 부트스트랩. 밴드는 T2 실측 후·T1 발사 전 수치 고정·부속 문서 파일(기본 템플릿: 성공 = Δ CI 하한 >0 ∧ Δ점추정 ≥+5pp / PARTIAL = CI 하한 >0 / NEGATIVE = 그 외). 절대 밴드(≥10%/<1%)·자기 궤적은 부지표(단 T2 보류 시 폴백 1차 복귀, 'Δ 부재 상태의 절대 판정' 명기). <1% NEGATIVE 는 비주입 그룹 floor_grant_rate 궤적 0 이면 판정 불능 강등. 정확도 Δ(T1−T2) = 주입 총효과, 난이도별 Δ 분해 부지표 사전등록(PG0 방향 사전확률). Δ(주입−자유 그룹) 훈련 궤적 = 구조 대조 부지표. 성공 밴드 −1.84pp·L4-5 기울기·엔트로피 교란 분기 공통 적용 |
| **T3** b4cal — 수리된 R_cal (winner+0.3) | **Q6 조건부(승자 존재 시)** + 요구② | 4 | **예** | 공통 kill. 발사 게이트 = 조건부 재기준선: 1차 b2p / 2차 b3s / 둘 다 실패 시 취소 | v3 유지: δ_cal A-vs-A 밴드·0.88 점유율 병기·조건부 재기준선 |
| **T2b** 전제복원 w_meta=0 짝팔 (W5 이중 조건부) | **Q3 + Q5 인과 절반** | 4 | 예 | 발사 조건 미충족(검정력 패키지 미승인) ⇒ "방향성 관측(비판정)" 강등 | C-031 충돌 명시·해소 논거 사전등록(estimand 구분: 채점 안 되는 상태 vs 전제복원 레짐에서 채점되는 상태) + 재확인 계수기 +1 원장 기록 + codex 게이트 상정. 발사 전 검정력 패키지 의무(문항 풀 확장·다중 디코딩 시드, MDE ≤ 기대 효과 밴드, 계산서 첨부). 전제: T2(/T2e) 성공 — 없으면 C-031 재구매 |
| **W5 기타 후보** bfmt 분해 사다리(무조건·Q2 유일 계기, ~130 GPU-h) / SFT1-init RL·언마스크 SFT2(Q1 개입) / kill-2 원리승격 변주 / same-config σ_run 복제(2발사×100스텝) / KL 앵커 팔 | Q1·Q2·σ_run·kill-2 처분 | 가변 | 혼합 | — | 우선순위만 사용자 결정 |

**게이트 공통 명세**: 전 비열등·우열 게이트는 동일 held-out 문항 집합(MATH500 n=500, avg@8)의 문항-대응 차이의 문항-클러스터 부트스트랩 CI 로 통일(독립 2표본 금지·n 사전 고정). 전 문턱 −1.84pp. 승자 타이브레이크: 차이 < max(1.05pp, 3σ_total 또는 σ_run 미실측 시 3σ_eval+경계승격) 이면 비강제 방출률 우선. 엔트로피 교란 분기(전 T 팔): gs150/200/250 체크·NFKC 병기·보류 명명.

---

## ③ 웨이브 순서와 의존성

| 웨이브 | 내용 | 동시 GPU | 의존성·게이트 |
|---|---|---|---|
| **W0** | N0 보존(최우선) → E0′·E1(인용) · E5·E6(GPU 0) · E2(0~4) · E3+억제 프로브(4) | ≤8 | E3 프로브 = T1 반려 게이트. N0 은 N3 의 전제 |
| **W1** | 순차: N4(4) → N2(4) → N3(4) · N1(4) 은 승인 후 삽입 · 말미 E4′(0) | ≤8 | N3 은 N0 에 의존. E4′ 기본 결론 = 폴백 발동 |
| **W2** | **A18 머지 확인** → T2 학습(4, 1노드 80G4 — 앵커 패리티) | 4 | A18 머지 전 어떤 T 팔도 발사 금지(현 코드로 ack 통과 불가). 발사 전 검사는 OmegaConf.create() 래핑 resolved config 로 validate()(plain dict 금지). 잔여 쿼터로 W1 eval 병행 가능 |
| **W2′** | (조건부) T2e — T2 보류 확정 시에만 | 4 | 사전 승인 조건부 팔 — 트리거에 묶인 계획 내 분기, W5 임의 후보 아님 |
| **W3** | T1: Δ 밴드 파일 → 1-step 스모크(1노드) → 학습(4) | 4 | 발사 전 게이트 넷 전부: A18 머지 + E3 억제 프로브 통과 + N1 층화 게이트 처분(MDE 미달 시 사용자 결정 + 승인문에 "검정력 부족 상태의 균일 주입" 명기) + T2 실측 후 Δ 밴드 파일. A6 승인 선행 |
| **W4** | T3(4) | 4 | winner 게이트(조건부 재기준선: 1차 b2p / 2차 b3s). winner 부재 경로 = ⑤ 처분 |
| **W5 후보** | bfmt(무조건) / T2b(검정력 패키지 조건부) / SFT1-init·언마스크 SFT2 / kill-2 원리승격 변주 / same-config σ_run 복제 / KL 앵커 팔 | — | 우선순위만 사용자 결정. T2b 는 이중 조건부(T2/T2e 성공 ∧ 검정력 승인) |

**슬롯 규율**: "GPU 미결 ≤2 + 학습 팔 ≤1"(T2e 는 T2 종료 후라 위반 아님). 16 GPU 동시 학습은 이 계획에 없음 — 필요 시 별도 승인 + 비교 가능성 훼손을 ⑤에 명기하는 조건.

**학습 팔 공통 구성**: init·300스텝·G8 diff·w_format≥0.35·meta_floor=0.05·`++algorithm.dcpo_w_cal=0.0`(T1/T2)·KNOBS 등재+테스트. 세계크기 = 앵커 동일 1노드×4 GPU(sku 80G4-H100, nnodes=1/n_gpus=4; v3 표의 16 은 산술 오류 정정, G8 매니페스트에 세계크기 열 신설). `dcpo_format_replace` 는 발사 노브 제외("config 기본값 true 확인" 항목으로 강등). 전 T 팔 발사 명령에 동일 4-이름 ack 리터럴 `++algorithm.dcpo_ack_load_bearing='[dcpo_meta_floor,dcpo_rmeta_source,trainer.val_before_train,dcpo_pmishift_min_prefix_tokens]'`.

---

## ④ 승인 필요 차분 목록 (정본 코드 변경 — 사전 승인 대상)

공통 필수 칸: KNOBS.yaml 등재 + `test_knob_registry.py` 통과 + ack 리스트 정합.

| # | 파일 | 차분 | 형식 | 쓰는 팔 |
|---|---|---|---|---|
| A1 | `verl_sdc.py:1626` 부근(split_first_meta — :1636 오기 정정) | `dcpo_pmishift_min_prefix_tokens`(기본 0=byte-identical) — 채점 하드게이트 + A2 형상화 스케일 공유 단일 노브 + `skip_short_prefix` | 배선 | T1·T2·T2e |
| A2 | `verl_sdc.py:406-408` + `dcpo_region.py:1331-1348` | 연속 형상화 floor(기본 0=기존 동작) + `floor_grant_rate`·`prefix_len_median` | 배선 | T1·T2·T2e |
| A3 | `verl_sdc.py:1597,1646` | `dcpo_pmishift_dup_metric` | 순수 추가 | T1·T2·T2e |
| A4 | `_decoy_utils.py:132` + `verl_sdc.py:1654` | length-matched decoy + `skip_decoy` 승격 | 교체 | T1·T2·T2e |
| A5 | `verl_sdc.py:1712` 부근 | `ref_vs_actor_pmi_gap` | 순수 추가 | 전 학습 팔 |
| A6 (재작성) | `verl_sdc.py:2496-2519` 스텁 완성 + `meta_inject.py` 재사용 | `_force_inject_rollout` 를 generate_sequences 랩(`_bci` :2623 선례)으로 완성: uid-수준 ρ 태깅(그룹 동질) · pass-1 per-row logit_bias={open 151669, close}=−100(:2977-3081 채널·누출 단언 재사용, bad_words 폐기) · 캡/스플라이스/seed = `first_boxed_token_idx`/`splice_prefix`/`build_inject_segment(MARKER_ONLY)`(병렬 구현 금지) · response_mask 3분절 · floor 스프레드 분모에서 loss-mask 0 토큰 제외 1행 보정 · 랩 설치 순서 = CF fmt_replace 랩 최외곽 · launch-lint 배제 = {`sdc_force_inject_conf`, legacy `sdc_force_inject`} 동시 점검 · `pass1_close_tag_rate`·`pass1_signature_rate`·`injected_group_fraction`·`spont_meta_first_rate` 생산 · 승인 문서에 "랩 완성 vs agent-loop 신설" 비교 단락 + A.3 MARKER_ONLY(+5pp) 선례 인용 | 배선 | T1 |
| A8~A13 | (v3 유지) | cal 수리 패키지 + conf_* 생산자 | — | T3 |
| A14 | N1 패키지 | v3 전항 + 층별 목표 n·ICC 파일럿 MDE 계산서 + 메타 유형 분류(verify/redirect/기타)·유형×난이도 로깅 + cf_prefix 참조 경로 `configs/archive/cf_prefix_agent.yaml` 갱신(+`verl_sdc.py:2606` 주석 정정) | 배선 | N1 |
| A15 | `freeze_run_manifest.py:66` | 채점기 해시 (v3 유지) | 배선 | 전 eval |
| A16 | 로깅 생산자 매핑표 | v3 유지 + A6 신규 스칼라 편입 | 배선 | 전 학습 팔 |
| A17 (위치 명세) | `core/KNOBS.yaml` | `dcpo_pmishift_min_prefix_tokens` 는 최상위 `load_bearing:` 섹션(:80)에 `status: live` 로 등재(rule (3) missing/extra 대칭 검사와 ack 4-이름 리터럴 정확 일치) · 나머지 신규 노브 3종(dup_metric·decoy_length_matched·reversal_min_magnitude)은 live:/default_only: 등재·ack 미포함 · set_at 행번호 갱신 · Phase-2 byte-identical 조항 개정 | 문서 정정 | 전 팔 |
| A18 (신규·**전 T 팔 발사의 선행 조건**) | `src/training/knob_registry.py:168` + `tests/test_knob_registry.py` | ack 읽기를 문자열-제외 시퀀스 덕타이핑으로 교체(`if acked is not None and not isinstance(acked, str): acked_set = set(str(x) for x in acked)`) + OmegaConf DictConfig/ListConfig 로 validate() 통과/실패 회귀 테스트 — 현 코드는 ListConfig 에서 항상 빈 집합이라 어떤 ack 값도 통과 불가(KNOWN OPEN GATE 라 역사상 미관측) | 버그 수정 | 전 학습 팔 |

발사 명령 노브(`dcpo_w_cal=0.0`·ack 4-이름 리터럴·T2e 의 `entropy_coeff=0.0`)는 G8 매니페스트 대조 칸(ack 열 + 세계크기 열 신설). E3 억제 프로브는 eval-측 스크립트로 정본 무수정(승인 불요, 산출물은 원장 기록).

---

## ⑤ 수용된 한계 · 닫힌 문 · 처분

- **T2 의 최빈 선례 경로 = 보류**: 최근접 선례 b3s 가 엔트로피 밴드(0.20→5.41/5.88, C-028)와 정확도 밴드(−2.00pp [−3.40,−0.62], C-030)를 둘 다 위반 — T2 도 보류로 끝날 사전확률이 높다. 이를 감추지 않고 T2e 조건부 팔·T1 폴백 규칙·T3 처분으로 사전 배선했다. T2e 까지 보류면 Q5 는 이 레버 집합에서 판정 불능으로 종결.
- **Q2 는 W0~W4 에서 열리지 않는다** (E-173 의 강제·bfmt 만).
- **Q3 은 이중 조건부**: T2(/T2e) 성공 ∧ T2b 검정력 패키지 승인. 미승인 시 T2b 는 방향성 관측으로 강등, Q3 은 이 계획 계보에서 미결(C-031 재구매 금지의 대가).
- **요구②(pmi_shift·cal 동시 동작)의 검정은 W4 조건부** — T2·T2e·T1 전멸 + 2차 재기준선 실패 시 이 계획 안에서 미검정 종료, cal 동시 동작 검정은 후속 계획 이월. 그 경로에서 Q6 은 E1 관측 + ⑥ 처리표로만 남는다(개입 검정 W5 이월).
- **σ_run 은 사실상 미계상일 공산**: E4′ 의 기본 기대는 유효 추정기 부재 — 전 게이트는 "CI=eval 잡음만" 명기 + 경계 사용자 승격으로 운용. 정공법(same-config 복제)은 W5. 단일 학습 시드(C-021) 자체는 해소 불가.
- **T1 억제의 잔여면**: 닫는 태그까지 logit_bias 로 막아도 다중 토큰 자연어 우회(`decision:`·내용 시그니처)는 토큰 억제 불가 — 검출(E3 프로브 반려 게이트 + 스모크 C-033 배터리)로만 대응, 반려 문턱 이하의 잔여 오염은 Δ(T1−T2) 해석에 명기. 억제 디코딩의 경미한 off-policy 편향(v3 명기)도 유지.
- **T1 균일 주입의 PG0 긴장**: 난이도 조건화는 N1 층화 게이트 뒤에 있고 그 게이트는 MDE 미달 시 사용자 승격으로만 보호 — hard 층 역효과 위험(PG0 hard −0.034)은 난이도별 Δ 분해 부지표와 `inj_redirect_on_hard_rate` 로깅으로 사후 분해 가능하게 남긴다(사전 차단 아님).
- **T2 kill-1 은 패키지 판정 / kill-2 는 구성 수준** — 원리 승격은 W5 변주 재현 후에만. seed 토큰 불변식은 A6 1행 보정으로 해소(한계 아님으로 이동).
- **E5 분기 잔여 위험**(len_cost=0.08 판명 시 유지) + "채택은 패리티 원칙, 손상 제거 기대 아님" 프레이밍 유지.
- **N1 estimand 한계**: 동결 정책 ≠ end-to-end·body-prefix off-policy·수확 분포 조건화 스코프(NEGATIVE 는 붕괴 레짐 meta-first 자기수확 분포에 한정).
- v3 §7 나머지 전부 유지: Q1 개입 W5·conf 2값·anchor EMA·TRIOBJ_META_V1 기각·0811 원장 밴드 문구 재상속 금지·판정문 규율.

---

## ⑥ R_cal 결함 목록과 각 수정의 대응

전체 11종 처리표는 v3 §4 를 변경 없이 상속한다(본 v4 에서 무수정). 본 계획 문면에 명시된 결함과 대응은 다음과 같다.

| 결함 (감사① 11결함 중 명시분) | 증상 / 근거 | 대응 |
|---|---|---|
| 침묵=만점 (conf 부재 시 R_cal=0) | `dcpo_region.py:951-954` — 발화가 죽은 b3p 의 R_cal 궤적 −0.21→−0.016 이 "최고 개선"으로 보이는 아티팩트 | A8~A13 cal 수리 패키지(T3 수리된 R_cal) + `conf_*` 생산자·cal 발화율 로깅(A16) — E1 인용은 귀속 불가 서술로 제한 |
| conf 준-상수 교란 (정확도 상승만으로 R_cal 기계적 개선) | conf 두 값(0.75/0.88)뿐·시도 수준 정보 0·슬롯 RNG 지배 | T3 판정에 δ_cal A-vs-A 밴드 + 0.88 점유율 병기 의무 + conf 분산 로깅 스칼라 |
| 출처불일치 | 감사① 적발 | A8~A13 수리 패키지 (v3 §4 처리표 상속) |
| membership 부재 | 감사① 적발 | A8~A13 수리 패키지 (v3 §4 처리표 상속) |
| anchor 자기잠금 | 감사① 적발 — anchor EMA 한계는 ⑤에 수용된 한계로 병기 | A8~A13 수리 패키지 (v3 §4 처리표 상속) |
| floor 무보호 CONF | 감사① 적발 | A8~A13 수리 패키지 + meta_floor=0.05 유지(발화 유지 필수) — 단 floor 는 "채널만 열고 내용은 채점 안 함"이므로 내용 채점은 T3 R_cal 수리로 |
| (나머지 결함 5종) | v3 §4 처리표에 항목별 등재 | v3 §4 처리표 상속 — 변경 없음 |

**학습 중 관측 계획(Q6 의 상시 계기)**: 전 학습 팔에 wandb 스칼라 — `skip_short_prefix`, cal 발화율, conf 분산, `ref_vs_actor_pmi_gap`(A5), 비강제 방출률(`spont_meta_first_rate`), `floor_grant_rate`·`prefix_len_median`(A2), `pass1_close_tag_rate`·`pass1_signature_rate`·`injected_group_fraction`(A6), `inj_redirect_on_hard_rate`·유형×난이도 구성(A14) — 생산자 매핑은 A16 이 고정한다. Q6 의 개입 검정 절반은 T3(winner 존재 시 조건부)이며, winner 부재 + 2차 재기준선 실패 경로에서는 E1 관측 + 본 처리표만 남고 개입 검정은 W5 로 이월된다(⑤ 참조).