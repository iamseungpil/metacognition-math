# 사전등록 — Countdown 고정 자리 2×2 (cd8, 로컬 H100 GPU 0~2)

작성 2026-09-04 17:40 · 학습 결과 0건 시점. 자 표(2단계)는 계산 중이며, 그 결과에 따라 조건부 팔(MR)만 추가된다.
1파도 결과(`docs/RESULTS_cd7.md`): 스텝 50 held-out N0 0.708 · A 0.634 · SC 0.642 — 메타 요구가 −7pp, 내용 채점은 되돌리지 못함.

## 1. 질문 (2×2)

| | 처음부터 출발(자율) | 같은 자리 출발(고정, 배치 절반) |
|---|---|---|
| 메타 항 없음 | A(있음), N0(있음) | **M0** |
| 정답표 타이밍 + 살아 있는 새 길 | **FT** | **MT** |
| 자 표 합격 자 | — | MR-x (합격 시에만) |

- Q1 (M0 − A): 같은 자리 그룹으로 신용 배분만 바꿔도 메타 요구의 비용(−7pp)이 사라지거나 뒤집히는가.
- Q2 (MT − M0): 정답표 항이 고정 자리 위에서 값을 더하는가.
- Q3 (FT − A): 정답표 항이 자율 출발에서도 값을 더하는가(고정 자리 없이).
- Q4 (MR − M0, 조건부): 합격한 내부 자가 고정 자리에서 값을 더하는가.

## 2. 설계

- 모델 Qwen3-4B, SFT 없음, 4수, 프롬프트 new, thinking off. 100스텝, 64 프롬프트 × 8 롤아웃, lr 1e-6, 체크포인트 5스텝, 판정 30·50·100.
- 고정 자리 데이터 `data/sites_v1/mixed_train_v2.parquet`: 같은 자리 안에서 학습 전 정책의 16개 결과가 갈리는 자리 981개 + 일반 문제 981개. 자리 행은 assistant 차례에 프리픽스를 넣고 `continue_final_message` 로 이어 쓴다(렌더링 바이트 일치 검증, `test_countdown_site_prompt_render.py`).
- 항 정의(`countdown_rewards`): timing = 죽은 계열에서 redirect +1 / 산 계열에서 redirect −1 / 죽은 계열에서 비redirect −1 / 시도 없음 0, 무게 0.5·w. live_new = 메타 뒤 첫 수가 «해 있음 ∧ 안 가본 쌍» ∧ 이행, 무게 1.0·w. w 는 0→20스텝 선형. 공통 corr 1.0 · format 0.35 · meta_floor 0.02.
- 팔 서명은 `arm_signature`, 데이터 파일은 `[run_arm]` 헤더에 찍힌다.

## 3. 판정 지표와 해상도

- **1차: 판정 자리 1,000개(문제 125, 훈련과 문제 단위 분리)에서 메타 허용 모드 16개 이어쓰기 성공률.** 모든 팔이 같은 자리를 받는 짝지은 비교. 학습 전 정책 0.496, A 스텝 50 정책 0.565. 자리 단위 부트스트랩 CI. 이 지표의 팔 간 표준오차는 약 1pp(1,000자리 × 16).
- 2차: held-out 500×8 정답률(«언제»까지 전이됐나). 잡음 참고치: A 두 씨앗 스텝 30 차이 0.7pp, 이전 라운드 σ_run 2.9pp.
- 3차(기제): 죽은 자리에서 redirect 비율, 산 자리에서 redirect 비율, 메타 뒤 새 쌍 시도율, 메타 위치.

## 4. 판정 밴드 (결과 전 고정)

| 결과 | 판정 |
|---|---|
| M0 − A ≥ +3pp (1차) 그리고 held-out 에서 N0 대비 격차가 절반 이하로 줄어듦 | 신용 배분이 메타 요구 비용을 상쇄 → 고정 자리 채택 |
| MT − M0 ≥ +3pp (1차) 그리고 죽은 자리 redirect 비율 상승·산 자리 하락 | 정답표 타이밍 항 채택 |
| FT ≈ A, MT > M0 | 정답표 항은 고정 자리에서만 산다(신용 배분이 전제) |
| 전부 A 와 ±3pp 안 | Countdown 에서 메타 학습 라운드 종료. 기계는 그대로 수학 이식 |
| 어느 팔이든 held-out 이 N0 보다 5pp 이상 낮은 채 100스텝 | 메타 요구 자체를 재검토(발화 강제 없는 팔) |

## 5. 무효 조건·중단 규칙
- 자리 행 필터링 0(`data.filter_overlong_prompts=false` 확인), `[COUNTDOWN][WIRED] n_site_rows` 가 배치의 40~60%.
- 중단 규칙은 1파도 기본값 유지(emit_rate<0.2, boilerplate, answer_leak, arith_in_meta 0.02, confidence_mean 0.8). 1파도에서 SC 가 60~78스텝에 반복 근접했으므로 발동 시 그 스텝까지의 판정으로 읽는다.
- 2스텝 스모크(M0, 혼합 배치)에서 자리 행의 r_corr 가 프리픽스+응답으로 채점되는지, 템플릿 패치가 Ray 워커에서 작동하는지 확인 후 발사.

## 6. 자원
GPU 0·1·2 각 한 팔(M0·MT·FT). 판정 자리 이어쓰기는 사이드카가 판정 지점마다 큐에 넣는다(`gen_continuations.py --sites sites_judge --modes meta --k 16`). MR 은 자 표 합격 시 첫 팔 완료 후 발사.

## 7. 개정 — timing2 / OPT_MT2 (2026-09-06)

`docs/RESULTS_cd7.md`("OPT_MT v3 s1(pre-fix) 스텝 100"): `timing` 항은 `family_dead is
None`(아직 시도가 없다 — 판정 불가)일 때 항상 0 이라, 정책이 스텝 100 께 응답 첫
토큰부터 redirect 메타를 ~100%로 내는 무효 레버를 팠다(비용 0, 운 좋으면 +1). 신규 항
`timing2`(=`r_timing2`)는 이 한 칸만 고친다: `family_dead is None ∧ redirect → −1`
(계열 생존 중 redirect 와 같은 벌), `family_dead is None ∧ ¬redirect → 0`(변화 없음).
나머지 다섯 칸은 `timing` 과 바이트 동일. `timing`/`OPT_MT` 는 손대지 않는다 — 새 팔
`OPT_MT2`(=OPT_MT 의 timing→timing2 치환, 그 외 동일)로만 이 수정을 싣는다.

판정: 1차/2차/3차 지표는 위 §3 과 동일선상(같은 자리표, 같은 held-out). Positive =
같은 자리(사이트) 정합 지표가 OPT_MT s1 스텝 50 (0.647) − 1pp 이상이면서, 스텝 100의
`frac_meta_first`(응답 첫 토큰 메타 비율)가 0.2 미만 **그리고** held-out 발화율이 0.5
미만으로 유지될 때. 어느 한쪽이 깨지면(정합 하락 또는 여전히 첫 토큰 메타 지배) 착취가
형태만 바꿨다고 읽고 기각.

## §8. OPT_CF — 반사실 쌍둥이 (2026-09-07)

동기(`docs/RESULTS_cd7.md` "같은 자리 인과 검사", 09-07 01:10). 학습 전 정책에서
스스로 낸 메모는 nometa 대비 같은 자리 성공률을 바꾸지 못한다(짝지은 자리별 차이
−0.001, 95% CI [−0.007, +0.005] — 0 을 포함). FT/MT/OPT_T/OPT_MT(오라클 타이밍
정합)와 OPD/OPT_OPDC(힌트 교사 KL)는 전부 "이 메모가 옳은 형태였는가"만 재고
"이 메모가 이 문제에서 **실제로 정답률을 바꿨는가**"는 안 잰다. OPT_CF 는 그것을
직접 잰다.

**정의** (`src/training/countdown_rewards.py`): 같은 자리(site_id)를 메타를
허가한 프롬프트(main, `PROMPT_VARIANTS["opt"]`)와 메타 문장 자체가 없는 프롬프트
(twin, `PROMPT_VARIANTS["plain"]`, 반사실 기준선)로 **둘 다** 한 배치에 태운다
(`scripts/local/build_cf_twins.py` → `mixed_train_v3c_cf_opt.parquet`, main 234 ·
twin 234 · normal 239 = 707행). 배치 전체에서 `cf_key`(=site_id)로 main/twin 을
짝짓고(`cf_center_rows`, GRPO 그룹이 아니라 배치 단위 — main 과 twin 은 서로 다른
프롬프트라 다른 uid 를 갖는다), main 행 중 메타를 낸 행에

```
r_cf_meta = clip(corr_main − mean(corr, twin 그룹(같은 cf_key))), −1, +1) × W_CF   (W_CF=1.0, warmup)
```

를 준다. twin 행 자체·일반 롤아웃·미발화 main·같은 배치에 twin 이 없는 main 은
전부 0(비교 불가를 조용히 숨기지 않고 무처치로 둔다). `ARM_SPECS["OPT_CF"]` =
`terms=("corr","format","cf_meta")`, `require_meta=False`, `meta_form="new"`,
`data_hint="mixed_cf"`.

**셔플 배선.** `configs/countdown_6arm.yaml` 의 `data.shuffle` 기본값은 **True**
— verl 은 매 에폭 데이터셋 자체를 섞은 뒤 `train_batch_size`(64)로 순서대로
자르므로, `build_cf_twins.py` 가 만든 (main,twin) 인접 순서가 셔플로 깨져 짝을
못 찾는 main 행이 늘어날 수 있다. `scripts/local/run_arm.sh` 는 `ARM=OPT_CF` 일
때만 `data.shuffle=false` 를 강제한다(다른 팔은 무영향). 쌍은 항상 짝수 인덱스에서
시작하고 배치 크기(64)도 짝수라, 셔플이 꺼져 있으면 어떤 쌍도 배치 경계에 걸리지
않는다 — 트레이너 쪽에서 그룹별로 재배열하는 대안(설계 §의 폴백)은 필요 없었다.

**판정.**
- 1차(주 지표) = 같은 자리 성공률(§3 과 같은 지표·같은 자리표). Positive:
  OPT_MT v3c s2 (스텝 30/50 = 0.599/0.622) 와 OPT(스텝 30/50 = 0.545/0.572) 를
  둘 다 앞선다(같은 판정선 규약 — §4 의 +3pp 밴드를 그대로 적용).
- 기제 = 학습 텔레메트리 `mean(corr, main 발화 행) − mean(corr, twin 행)` 이 학습
  중 0 위로 올라가는가(§8 정의상 cf_meta 의 무가중 원값과 같다) — 오르지 않으면
  "메모가 결과를 바꾼다"는 전제 자체가 이 정책에 없다는 뜻이고, cf_meta 항은
  구조적으로 보상을 줄 재료가 없다(OPD 가 "메타를 그만 낸다"로 수렴했던 실패와
  같은 종류의 붕괴를 의심).
- 무효화: held-out 정답률이 스텝 50 에서 N0 −5pp 밑으로 떨어지면 중단(§4 규칙과
  동일). `[COUNTDOWN][WIRED]` 의 `cf_main_scored`/`cf_pairs_found` 로 배선이
  실제로 걸렸는지(무효 레버 아님) 매 스텝 확인 — `cf_pairs_found` 가 배치의
  site 쌍 수 대비 낮으면(셔플 재발 등) 1차 지표를 신뢰하지 않는다.

## 9. 개정(2026-09-07 17:00) — 오염 제거 후 재실험 (E-133/E-134)
- **E-133 평가 오염**: 자리(sites_v1)·결합 SFT 가 held-out val 문제에서 채굴됐음이 확인됨(§ RESULTS 09-07 16:5x). 자리 계열의 모든 held-out 비교를 무효화하고, **train 세트 롤아웃**(`eval/train1500_{gs0,A}_new`)에서 `sites_v4` 를 재채굴한다. 산출물 전부에 대해 val 과의 (nums,target) 교집합 0 을 스크립트로 강제한다.
- **E-134 대조군 프롬프트 불일치**: 런처가 `MIXED_DATA` 지정 시 opt 팔에 `_opt` 를 붙이지 않던 결함을 수정. v3/v3c 세대 OPT_MT 씨앗 1~4·OPT_M 결과는 «학습 강제 / 평가 허용」 불일치 조건으로 재분류하고 본 비교에서 제외한다.
- **재실험 팔(모두 SLIM=8-bit AdamW, opt 프롬프트, sites_v4)**: ① 결합 SFT v4(힌트 이어쓰기, 자리 이득 게이트) → ② 그 위에 OPT_CF(판정, 씨앗 1·2), OPT_M(대조: 결과 보상만), OPT_MT(비교: 정답표 타이밍). 기준선 N0/OPT 는 기존 값을 그대로 쓴다(train 세트만 학습, 오염 없음).
- **판정(불변)**: 1차 같은 자리(sites_v4 judge, 문제 단위 분리) 성공률, 2차 held-out 500(이제 전 팔 미노출), 3차 기제(발화 선택성·redirect·새 쌍·잘림). 긍정 = held-out 에서 N0 (0.708@50) 를 씨앗 2개 평균으로 넘고 OPT_M 대조군보다 +2pp 이상.

## §11. OPT_OPDG — 교사 게이트 증류 (2026-09-07)

동기. 힌트 교사 증류는 두 번 실패했다. **OPT_OPD**(단측 벌, 절대 기준 0)는
「메타를 그만 낸다」로만 수렴했다 — 메타를 낼 때마다 기대 보상이 음수라 발화율이
7% → 0.8% 로 무너졌다(15스텝, `docs/RESULTS_cd7.md` "그룹 중심화 OPD").
**OPT_OPDC**(그룹 중심화)는 그 부호 문제를 고쳤지만 **굶었다** — 자율 출발 정책의
발화율이 ~6% 라 GRPO 그룹당 채점 가능한 메타가 2개 미만이었고, `opd_center_rows`
의 「채점 대상 2 미만 → 전원 0」 규약이 거의 모든 그룹에서 걸렸다.

두 실패에 각각 대응하는 변경이 이번에 둘 다 갖춰졌다.
- **굶음(OPT_OPDC)** ← **결합 SFT 체크포인트에서 출발**한다. 발화율 기준선이
  50%+ 라 그룹당 채점 메타가 2개 이상인 그룹이 다수가 된다(항이 실제로 분산을
  갖는다). 자리 절반이 섞인 `data_hint="mixed"` 도 같은 방향으로 돕는다.
- **잘못된 방향으로의 증류(OPT_OPD, 그리고 OPDC 에도 남아 있던 결함)** ←
  **교사 신뢰도 게이트**. 두 팔 모두 「힌트 교사와 가까운가」만 재고, 교사가 그
  자리에서 아무것도 안 하는 것보다 나은지는 묻지 않았다. 게이트는 그것을 묻는다.

**정의.** `ARM_SPECS["OPT_OPDG"]` = `terms=("corr","format","opd_meta_c")`,
`meta_form="new"`, `require_meta=False`, `data_hint="mixed"`, `opd_gate=True`.
보상식은 OPT_OPDC 와 **바이트 동일**하다 — 달라지는 것은 「어느 행이 센터링의
채점 대상인가」뿐이다(`arm_signature` 에 `|opdgate=on` 이 박혀 두 팔이 로그에서
갈린다).

**게이트**(`scripts/local/build_gate_sites.py`, 자리 단위):

```
hint_rate   = mean(r_corr | mode=="hint",   그 site_id)
nometa_rate = mean(r_corr | mode=="nometa", 그 site_id)
gate = 1  iff  (hint_rate − nometa_rate) >= τ  AND  hint_rate >= min_hint
```

기본값 **τ = 0.10, min_hint = 0.25**. 게이트 결과는 mixed parquet 의 행마다
`extra_info.opd_gate ∈ {0,1}`(+ 진단용 `hint_rate`/`nometa_rate`)로 얹히고,
자리가 없는 일반 행은 `opd_gate=0, hint_rate=nometa_rate=0.0` 이다.
`verl_sdc._compute_countdown_arm_stash` 가 그 마스크를 `opd_center_rows(gate=…)`
로 넘긴다: 센터링 평균은 게이트 통과 행만으로 잡고, 게이트 밖 행은 항상 0(=순수
결과 GRPO), 그룹 안 게이트 통과 행이 2 미만이면 그 그룹은 전원 0(기존 규약 상속).
배치 게이트 행 수는 `[COUNTDOWN][WIRED]` 의 `opdg_gated_rows=` 로 매 스텝 찍힌다.

**v1 프로브**(v4 산출물이 아직 없어 v1 유사물로 사전 점검, 0907):
`mixed_train_v3c_opt` × `hint_gs0` × `conts_train_gs0` 에서 게이트 통과 자리는
**423/2844 = 14.9%**, `hint_rate − nometa_rate` 의 십분위는 p10 −0.0625 ·
p20~p70 0.0000 · p80 +0.0625 · p90 +0.2500 · p100 +1.0000(중앙값 0 — 힌트가
대부분의 자리에서 아무것도 바꾸지 않고, 소수의 자리에서만 크게 이긴다). τ=0.10 은
그 소수 꼬리를 정확히 집어내는 값으로 보인다(p90 이 +0.25 이므로 τ 를 0.05 로
낮추면 게이트가 급격히 넓어지고, 0.25 로 올리면 10% 아래로 좁아진다).

**판정.** §8(OPT_CF)과 같은 지표 세 개를 그대로 쓴다.
- 1차 = 같은 자리(sites_v4 judge) 성공률. Positive = 같은 초기 모델의 **OPT_M**
  대조군(결과 보상만)을 §4 의 +3pp 밴드로 앞선다.
- 2차 = held-out 500 정답률(전 팔 미노출) — N0 대비 하락 없음.
- 3차 = 기제(발화 선택성·redirect·새 쌍·잘림).
- **기제 요건(추가)**: 새 첫수(novel-first-move) 비율과 죽은 자리 redirect 비율이
  **OPT_M 대조군 대비 상승**해야 한다. 오르지 않으면 1차가 양성이어도 「증류가
  행동을 바꿨다」고 주장하지 않는다(게이트가 아니라 잡음일 수 있다).

**무효화 규칙.** 발사 전 `build_gate_sites.py` 요약에서 게이트 통과 자리가 전체의
**15% 미만 또는 85% 초과**면 τ 를 재조정하고 나서 실험을 건다(너무 좁으면 항이
다시 굶고, 너무 넓으면 게이트가 아무것도 안 거른 OPT_OPDC 의 재탕이다). 학습 중
`opdg_gated_rows` 가 0 이면 즉시 중단 — 배선 없는 선언이다. §4 의 held-out 하락
중단 규칙(N0 −5pp)도 그대로 적용한다.

## §10. OPT_CFG — 반사실 쌍둥이, 그룹 내부판 (2026-09-07)

동기(§8 실측 교란). OPT_CF 의 twin(plain 프롬프트) 기준선이 그 자체로 부풀려
있다 — 학습 전 정책이 같은 500 held-out 문제에서 `plain` 0.426 vs `opt` 0.352
(7.4pp)를 낸다. 즉 cf_meta 는 "메모가 없어서 못 푼다"가 아니라 "twin 이 plain
프롬프트라서 더 잘 푼다"를 일부 섞어서 재고, 그만큼 발화에 불리하게 편향된다.

**정의** (`src/training/countdown_rewards.py`): 프롬프트를 아예 바꾸지 않는다.
같은 GRPO 그룹(=같은 opt 프롬프트, uid 하나) 안에서 실제로 메타를 낸 롤아웃
(E, emitted==1)과 안 낸 롤아웃(NE, emitted==0)을 갈라, E 의 corr 을 그 그룹 NE
평균 corr 과 비교한다(`cf_group_rows`):

```
r_cf_group = clip(corr_E − mean(corr, 그룹 내 NE)), −1, +1) × W_CF   (W_CF=1.0, warmup)
```

그룹 안에 NE·E 둘 다 없으면(전원 발화 또는 전원 무발화) 그룹 전체 0. NE 행 자체는
항상 0(비교 기준일 뿐 처치 대상 아님). `ARM_SPECS["OPT_CFG"]` =
`terms=("corr","format","cf_group")`, `require_meta=False`, `meta_form="new"`,
`data_hint="mixed"` — OPT_M/OPT_MT 와 같은 `${MIXED_DATA}_opt.parquet` 로
라우팅된다(twin 데이터 불필요, 셔플 오버라이드도 불필요).

**판정.** §8 과 동일선상 — 같은 자리 성공률(1차)·held-out(2차)·기제(3차)를 같은
자리표·같은 판정선으로 본다. Positive 조건은 §8 의 OPT_CF 판정을 그대로 적용
(OPT_MT/OPT 를 앞서고 held-out 이 N0 −5pp 밑으로 떨어지지 않을 것). 기제는
`mean(corr, E) − mean(corr, NE)`(그룹 내부, `[COUNTDOWN][WIRED]` 의
`cfg_scored`/`cfg_groups_ok`)가 학습 중 0 위로 올라가는가.

**알려진 한계.** 그룹 안에서 "누가 메타를 냈는가"는 정책 자신의 선택이다 —
emit/no-emit 이 **내생적**(endogenous)이라, 모델이 이미 풀 수 있다고 느낀
문제·서브그룹에서만 메타를 낼 수도 있다(선택 편향이 인과 추정을 오염시킬 수
있다). OPT_CF(서로 다른 프롬프트 쌍둥이)는 이 내생성 문제가 없는 대신 §8 의
7.4pp 프롬프트 교란이 있다 — 두 팔을 서로 대체하지 않고 **상호 보조적 이차
확인**으로 함께 유지한다(어느 한쪽만 양성이면 결론을 유보).

## §12. OPT_VTR / OPT_VTRW — 온라인 검증 게이트 + "언제" 보상 (2026-09-08)

**동기.** OPT_OPDG(§11)의 교사 신뢰도 게이트는 **오프라인**이다 — 학습 시작 전
고정 힌트 교사 롤아웃 한 번으로 자리별 게이트를 계산해 parquet 에 얹어 두고,
100스텝 내내 그 값을 그대로 쓴다. 그런데 발화 습관·능력 둘 다 학습 중 바뀐다
(§9/§11 실측: 결합 SFT 출발 발화율 50%+ 가 RL 중 침식하는 경향, `docs/
RESULTS_cd7.md` 여러 절). 오프라인 게이트가 "그때"는 맞았어도 "지금"은 틀릴 수
있다 — 정책이 이미 그 자리를 스스로 풀 수 있게 됐는데도 옛 게이트가 계속 증류를
켜 두면, 이제는 필요 없는 교사 의존을 강화한다(또는 그 반대). OPT_VTR 은 게이트를
**매 배치 온라인**으로 다시 잰다.

**정의.** `ARM_SPECS["OPT_VTR"]` = `terms=("corr","format","opd_meta_c")`,
`require_meta=False`, `meta_form="new"`, `data_hint="mixed_vtr"`,
`opd_gate=True`, `vtr_online_gate=True`. 보상식은 OPT_OPDG 와 **바이트
동일**하다(`opd_meta_c`) — 달라지는 것은 게이트를 **어떻게 재는가**뿐이다.

**힌트 twin(K=4).** `scripts/local/build_cf_twins.py --mode hint` 가
`mixed_train_v4_gate_opt.parquet`(§11 의 오프라인 `opd_gate` 폴백을 이미 포함한
판)에서 자리(site)마다 **1 main + 4 힌트 twin**을 만든다. main 행은 opt 프롬프트
(메타 허가) 그대로 — `extra_info.vtr_role="main"`. 힌트 twin 행은 시스템 메시지를
그대로 두고(OPT_CF 의 twin 처럼 plain 으로 바꾸지 않는다 — §10 이 지적한 7.4pp
프롬프트 교란을 피한다), user 메시지의 `Target: N` 뒤에 `countdown_inv.
inv_hint_prompt` 와 같은 한 줄(`Hint: one valid solution is {witness}.`)을 끼운
사본 4개 — `extra_info.vtr_role="twin"`, `vtr_key=site_id`,
`vtr_hint_idx∈{0,1,2,3}`. 출력은
`data/sites_v4/mixed_train_v4_vtr_opt.parquet`(sites_v1 은 건드리지 않는다).

**온라인 게이트**(`countdown_rewards.vtr_batch_gate`, 배치 단위 순수 함수).
자리(`vtr_key`)마다:

```
hint_corrs   = corr(그 배치에서 뽑힌 그 자리의 힌트 twin 롤아웃 전부, K=4 그룹)
nometa_corrs = corr(그 배치에서 뽑힌 그 자리의 main 그룹 중 무발화 롤아웃)
```

`hint_corrs`·`nometa_corrs` 가 **둘 다** 1개 이상이면 **온라인**:
`gate = 1  iff  mean(hint_corrs) − mean(nometa_corrs) >= tau`(τ=`VTR_TAU`,
기본 0.10, 환경변수로 오버라이드). 어느 한쪽이라도 이 배치에 없으면(예: 발화율이
올라가 그 그룹이 전원 발화 — 무발화 표본이 없다) **오프라인 폴백**:
`extra_info.opd_gate`(§11 산출) 을 그대로 쓴다. 온라인/폴백 두 경우 다 없는
자리(그 배치에 아예 없음)는 게이트 표에 안 실린다. 게이트가 매긴 값은 main 행의
`opd_gate` 를 덮어써 `opd_center_rows(gate=…)` 가 그대로 읽는다(§11 의 배선을
재사용) — 힌트 twin 행 자신은 별도 GRPO 그룹(다른 uid)이라 이 센터링의 대상이
되지 않는다.

**"when" 보상(OPT_VTRW).** `ARM_SPECS["OPT_VTRW"]` = OPT_VTR + `terms`에 `"when"`
추가. 게이트 증류(`opd_meta_c`)는 "교사와 얼마나 가까운가"만 재므로, "죽은
계열에서 실제로 갈아탔는가"를 직접 상벌하는 `r_timing`(FT/MT, §2) 과 같은
오라클(`family_dead`)을 재사용해 다음을 준다(발화한 site 행에서만, 발화 강제
없음):

```
(redirect ∧ family_dead)      -> +w
(continue ∧ ¬family_dead)     -> +w   (continue = ¬redirect, verify/무결정 포함)
그 외 정합 불일치(redirect∧생존, continue∧죽음) -> −w
무발화                          -> 0    (절대 발화를 요구하지 않는다)
family_dead 판정불가(시도 없음)   -> 0
```

w=`VTR_WHEN_W`(기본 0.2, 환경변수 오버라이드). 텔레메트리:
`[COUNTDOWN][WIRED]` 에 `vtr_twins`(배치의 힌트 twin 행 수)·`vtr_sites`(배치에
등장한 자리 수)·`vtr_gated_online`/`vtr_gated_fallback`(게이트 통과 자리 수,
출처별)·`vtr_gate_rate`(통과 자리 / 등장 자리)·`when_scored`(OPT_VTRW, family_dead
판정 가능했던 발화 행 수)·`when_match_rate`(그중 정합 비율).

**판정.** §11(OPT_OPDG)과 같은 세 지표(같은 자리 성공률·held-out·기제)에 두 조건이
더 붙는다:

- **1차 통과 조건(강화)**: held-out 정답률이 N0(8-bit AdamW 기준선) 을 넘고,
  **또한** OPT_OPDG 의 (같은 초기 모델·같은 씨앗 조건) 오프라인 게이트 결과를
  **+2pp 이상** 앞선다. OPT_OPDG 대비 우위가 없으면 온라인 재계산의 추가 비용
  (아래 §토큰 비용)이 정당화되지 않는다.
- **기제(추가)**: **사이트 unstick 개선** — 오프라인 게이트가 "막혀 있었다"(N-스텝
  연속 게이트 미통과) 판정한 자리 중, 온라인 게이트가 학습 중 최초로 통과로
  전환되는 비율이 **0보다 커야** 한다(폴백만 계속 쓰이고 온라인 전환이 0건이면
  "온라인"이라는 처치 자체가 무효 레버 — `vtr_gated_online` 이 매 스텝 0에
  머무르면 즉시 재검토).
- OPT_VTRW 추가 판정: `when_match_rate` 가 학습 중 상승 추세(적어도 하락하지
  않음) — 하락하면 "언제" 항이 오히려 잘못된 타이밍을 강화하고 있다는 신호.

**무효화.** §11 의 두 규칙(발사 전 게이트 통과 자리 15~85% 확인, `opdg_gated_rows`
0 이면 즉시 중단)을 그대로 물려받는다(온라인 게이트 값이 `opd_gate` 를 덮어써도
그 컬럼을 읽는 하위 배선은 동일). 추가로 `vtr_gate_rate` 가 매 스텝 0(온라인·
폴백 둘 다 전멸)이면 "온라인 검증"이라는 처치가 배선만 있고 실효가 없는 것이므로
중단한다. `data.shuffle=false` 강제(OPT_CF·아래 비용 절 참조)가 안 걸려 있으면
1차 지표를 신뢰하지 않는다.

**twin 비용 추정(K=4).** §8(OPT_CF)의 블록은 (main, twin) 2행/자리였다. OPT_VTR
은 (main, twin×4) = 5행/자리 — §8 대비 자리당 2.5배. `mixed_train_v4_gate_opt.
parquet` 기준 자리 수가 §11 v1 프로브와 비슷한 규모(수천)라면, 배치의 site 쪽
행 수가 (기존 main-only 대비) 최대 5배까지 부풀 수 있다 — 정상 배치의 "자리
절반·일반 절반"(§5 무효화 규칙) 균형을 유지하려면 정상(normal) 행을 반복해
채우는 기존 관례(`build_cf_twins.py` 의 need/reps 로직)를 그대로 물려받되,
블록 크기가 5 로 커진 만큼 정상 행 반복 배수도 커진다 — 발사 전
`[COUNTDOWN][WIRED]` 의 `n_site_rows`(배치의 40~60% 여야 함, §5)로 실측 확인이
필수다. GPU 비용 측면에서는 힌트 twin 도 일반 롤아웃과 똑같이 롤아웃·채점을
거치므로(빈 자리표시자가 아니다), 자리당 롤아웃 수가 5배 느는 만큼 그 자리를
포함한 배치의 유효 처리량이 준다 — §8 과 같은 `data.shuffle=false` 오버라이드가
필수인 이유도 같다(블록 인접성이 깨지면 같은 배치 안에서 main/twin 을 못 찾아
온라인 게이트가 폴백으로만 돌게 된다).
