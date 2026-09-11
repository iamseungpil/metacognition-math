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

**§12 수정(09-08 리뷰, 발사 전).** ① 힌트 twin 은 자리당 **행 1개**(`N_HINT_TWINS=1`)로
줄인다 — verl 이 행마다 `rollout.n=8` 개를 뽑으므로 힌트 이어쓰기는 **K=8**(원안 K=4 보다
많다)이고, 자리당 롤아웃은 main 8 + twin 8 = 2배(원안 4행이면 5배). ② twin 행은
`_countdown_mask_twin_advantages` 가 GRPO 어드밴티지를 0 으로 지워 **정책 손실에서 제외**한다
— 특권 정보는 교사 증류(opd_meta_c)로만 들어간다는 주장을 지키기 위함. 위 «twin 비용
추정(K=4)» 문단의 5배 수치는 이 수정으로 2배로 읽는다.

**§12-b 전 구간 게이트 증류(09-08, 베이스 출발 팔 전용 knob `OPD_FULL_SPAN=1`, 계보 접미사 `_fs`).**
실측: 베이스 출발 OPT_OPDG 스텝 1~2 에서 `opd_c_groups>=2 = 2/64` — 발화 7% × 게이트 17% 라
메타 구간만 재는 증류는 그룹당 채점 행이 2 미만이라 신호가 0 이다. 수정: 게이트 통과 자리
(`opd_gate=1`, 프리픽스 있음)는 메타 발화와 무관하게 **프리픽스 직후 이어쓰기 앞
`OPD_FULL_SPAN_TOK`(기본 256) 토큰**을 상태 힌트 교사와의 KL 구간으로 삼는다(상태=프리픽스).
그룹 8행 전부가 채점 대상이 되므로 중심화 조건(≥2)이 항상 성립한다. 해석: HDPO 식 특권
자기증류를 «정답 힌트·전패 프롬프트» 대신 «상태 힌트·검증된 결정 자리»에 건다. 판정선은 §12 와
같다. 메타 구간 전용 판(knob 0)은 SFT4 출발 팔(`v4_OPT_OPDG_s1`)에서 그대로 유지한다.

**§12-c 밀집(토큰별) OPD 어드밴티지(09-08, knob `OPD_DENSE=1`, 계보 접미사 `_dn`; `OPD_DENSE_C`=0.5,
`OPD_DENSE_CLIP`=2.0).** 실측: §12-b 전 구간 판도 게이트 자리(스텝당 32행 = 64그룹 중 4그룹)에서만
**스칼라** 중심화 보상을 주므로 배치의 6% 에만 신호가 닿고, 그마저 GRPO 가 시퀀스 합 → 그룹 정규화로
뭉갠다. 수정: 게이트 자리의 응답 토큰(롤아웃이 실제 뽑은 id, 앞 256개)마다
`adv[i,t] += C · clip(lp_teacher_t − lp_student_t, ±2)` 를 GRPO 어드밴티지 **위에** 더한다 —
Thinking Machines OPD 의 토큰별 역KL 어드밴티지 정의 그대로(교사 = 같은 모델 + 상태 힌트). 게이트 밖
행·트윈 행은 0. 사용자 지시(09-08 11:0x «RL 로 베이스라인을 꼭 이겨야 한다»)에 따라 베이스 출발 3팔
(OPDG·VTR·VTRW)에 기본 적용하고, 스칼라 판(`_fs`)은 절제로 남긴다. 판정선 §12 동일: 깨끗한 held-out
매치드 스텝에서 N0 8-bit 초과 ∧ 게이트 없는 절제 +2pp ∧ 막힌 자리 이득 > 0.

## §13 검산 사다리 — 메모 칸 점수 세 방식 (2026-09-09, RL 만·SFT 없음)
**동기(0908 실측).** 메모 팔의 적자는 전부 «메모 낸 표본」(메모 뒤 정답률 .12, 훈련 불변)이고 메모 없는
표본은 N0 와 같다. N0 오답의 73% 는 «값이 목표와 다른 식을 박스에 넣은 거짓 주장」, pass@8 .952.
→ 중반 «갈아타기」 대신 **답 직전 자기 주장 감시(check)** 를 유일한 메타 행동으로 두고, 메모 칸 점수
방식만 바꿔 나란히 잰다. 첫 수를 나열하는 plan 은 **메타인지가 아니라 계산의 전진**이라 제외(0909).
**프롬프트** `chk` = plain + 한 줄 허가(`<check> expr = value ✓/✗ </check>`; 정확하지 않은 식을 박스할
때는 ✗ 로 표시). 메타 블록 없음(구 문법 세금 7pp 회피).
**답 칸(공통)** corr(+1) · format · **fclaim**(박스 식이 오답인데 ✗ 표시 없음 → −0.5).
**메모 칸** TAG0: 없음 / FIXED_CHK: 형식 맞는 check 존재 시 +0.5(내용 무관, Trust-but-Verify 류) /
EVC_CHK: ✗ 로 표시한 식이 정말 틀렸고 최종 박스가 그 식과 다르며 정답일 때만 +0.5.
**판정(held-out 500×8, 매치드 스텝 30/50/100; N0 .640/.724/.809).** TAG0−N0 = 세금, FIXED−TAG0 = 고정
보너스 값, EVC−FIXED = 효과 검증 값. 성공 = EVC > N0 ∧ EVC > FIXED ∧ EVC > TAG0+2pp. 기제: 거짓 주장률
(N0 .73 of wrong) ↓, «잡고 고침」 비율, 같은 문제 짝 비교(check 有/無) ≥ 0. 무효: chk 프로브에서 세금
> 3pp(문법 재설계), 세 팔 s30 모두 < .60.
**프로브(학습 없음)** 베이스·N0 s100 을 chk 프롬프트로 채점 — 세금과 «지시만으로 거짓 주장이 줄고
정확도가 오르는가」(RL 여지의 하한).

**§13-b 영역 분할(09-09, knob `CHK_REGION=1`, 접미사 `_rg`) — 1단계 기본값.** 메모 칸 항(chk_fixed/chk_evc)을
시퀀스 스칼라에서 빼고, 그룹 중심화(Dr.GRPO, /std 없음)한 값을 **<check> 구간 토큰에만** 어드밴티지로
더한다. 답 칸(corr/format/fclaim)만 GRPO 시퀀스 보상. 이유: 스칼라 합이면 GRPO 가 메모 점수를 전 토큰에
뿌려 «메모 부분과 정답을 다르게 채점」이 어드밴티지 수준에서 성립하지 않는다(사용자 지적 0909 — 한계로
남기지 말 것). 스칼라 판은 절제로 뒤에 돌린다. 거짓 주장 지름길(전부 ✗ 표시 후 아무 식 박스)은 정답
보상 손실로 억제되며 «오답 중 ✗ 표시율」과 정확도를 함께 감시한다. 응답 예산 2048 은 N0 와 매치드
조건이라 유지(2560 은 절제).

## §14 R4 EVCA — 효과-검증 크레딧 증폭 (2026-09-10, OPRD 식 «목표가 아니라 방향»)

**동기.** §13 의 여섯 시도(CFG·OPDG·VTR·VTRW·EVC) 는 전부 **새 보상 채널**을 추가했고, 그때마다 게이밍
구멍이 생겼다(EVC: «✗ 잡고 아무 식이나 박스»가 대조군 TAG0 보다 4.5pp 나쁨, E-137 재채점 확정).
유일한 양성인 **EVCM**(R3, 09-10 08:00 기록: s30 +0.3pp / s50 +0.9pp vs TAG0)은 정반대로 갔다 — 새
보너스 없이, 이미 있는 정답 보상의 크레딧이 `<check>` 토큰까지 갈지 말지(0/1)만 검증기로 걸렀다.
arXiv:2609.08798(OPRD)의 «교사를 목표로 끌어당기지 말고, 검증기가 지지하는 학생 자신의 그래디언트
성분만 증폭하라»를 이 자리에 그대로 대입한다 — 우리는 별도 교사 모델이 없으므로 **검증기 자체**(check_row
의 chk_solved/over_claim 판정)가 «지지된 방향」을 정의한다.

**정의(`countdown_rewards.py`, `ARM_SPECS["EVCA_CHK"]`).** EVCM 과 항·데이터·프롬프트 전부 바이트
동일(`terms=("corr","format","fclaim")`, 보너스 0). 차이는 `chk_mask` 값 하나뿐 — EVCM 은 `True`(이진
통과/차단), EVCA 는 `"amplify"`. `verl_sdc._countdown_add_check_region_advantage` 가 `<check>` 구간
토큰의 **기존** GRPO 어드밴티지에 배율을 곱한다: `chk_solved`(✗ 로 잡은 식이 정말 틀렸고 다른 식을
박스해 정답) 행은 `CHK_AMP`(환경변수, 기본 1.5) 배, `over_claim`(허위 경보) 행은 1.0(그대로, EVCM 과
동일하게 벌은 유지), 나머지는 0.0(지움, EVCM 과 동일). **`CHK_AMP=1.0` 이면 EVCA 는 EVCM 과 바이트
동일** — 이것이 회귀 안전망이다.

**최소 구현인 이유.** 새 보상 항 없음, 새 데이터 없음, 새 배선 없음 — 기존 §13-c(EVCM) 마스크 경로의
분기 하나(스칼라 0/1 → 배율)만 바꾼다. 테스트(`tests/test_check_terms.py::test_evca_*`) 2건이 (a)
EVCA 스펙이 EVCM 의 상위호환임을, (b) 배율 0/1/>1 세 경우가 각각 지움/그대로/증폭으로 정확히
갈리는지를 검증한다.

**판정.** EVCM 과 같은 씨앗·같은 데이터로 스텝 30 부터 짝지어 비교한다(§13 판정 지표 그대로: held-out
500×8, N0 .640/.724/.809 기준). **1차** = EVCA s30 ≥ EVCM s30 + 1pp. 통과하면 s50 까지 계속하고
CHK_AMP 를 2.0~3.0 으로 올린 씨앗도 하나 더 건다(`chk_solved` 가 배치당 0.1~0.4% 로 극히 드물어
1.5 배로는 신호가 안 보일 수 있다 — 이 경우도 유효한 음성 결과로 기록). **무효** = EVCM 과 통계적으로
구분 안 되는 채 100스텝(증폭이 드문 신호라 실질적으로 EVCM 과 같다는 뜻) 또는 `chk_solved` 배치당
발생이 0 인 스텝이 10 연속(기아, E-136 류 — 이 경우 컬럼 부재가 아니라 표본 요동이므로 killer 는 걸지
않는다).

**자원.** GPU 2·3 에 §13 나머지 팔(FIXED_CHK·PERSIST_CHK)이 아직 돌고, EVCM(R3)이 s100 완주를 앞두고
있다 — EVCA 는 그중 하나가 끝나 카드가 비는 대로, **EVCM 의 s100 완주 곡선이 나온 뒤에** 20스텝
스모크로 시작한다(§13 관행: 스모크 → s30 짝비교 → 승격/폐기). R5(FIXED 완주)·R6(절제·씨앗2)·R7(수학
이식) 보다 앞선다 — 구현 비용이 사실상 0이고, 지금까지 유일한 양성 결과를 직접 강화하는 시도이기
때문이다.

**§14 부기 — 절제 보충 코퍼스 (2026-09-10, 데이터만·발사 없음).** 결합 SFT
(`coupling_sft_v4.parquet`, 463행)는 «막힌 자리에서 메타→다른 수→정답» 양성만 담아
"언제 안 해야 하는지"를 한 행도 가르치지 않는다. `scripts/local/build_restraint_sft.py`
가 `conts_v4/base_train.parquet` 의 `mode=="meta"` 이어쓰기(=site 원본 system 프롬프트와
같은 조건)에서 건강 자리(`family_dead==0`) 두 종류를 뽑는다: **restraint**(메타 미발현
∧ 정답, 780행/416사이트 — `wrong_prefix`=site prefix)와 **decorative**(건강 자리에서
굳이 낸 검산 블록, 300행 — `wrong_prefix` 를 `</meta>` 까지 확장해 장식 블록을 통째로
loss-mask). 둘 다 `scenario="redirect"` 로 기존 `sft.py::_should_mask_prefix` 경로를
그대로 쓴다(sft.py 무변경). 스키마는 결합 코퍼스 14컬럼 + `kind` → `concat` 가능,
섞으면 coupling 30.0% / restraint 50.6% / decorative 19.4%. 한계: 기존 SFT 에 음의
손실이 없어 decorative 는 장식 블록의 확률을 **깎지 못하고** 올리지 않을 뿐이며, 그
학습 구간의 93%가 60자 미만(사실상 `\boxed{}` 한 줄)이라 실효가 작다 —
restraint 만 먼저 섞는 것이 기본. 출력 =
`/hdd_data/seungpil/scratch/data/sites_v4/restraint_sft_v4.parquet`,
테스트 = `tests/test_build_restraint_sft.py` (19건). 아직 어떤 SFT/RL 도 발사하지 않았다.

## §15 R8 DPO_CHK — 구성된 쌍 위의 직접 선호 목적함수 (2026-09-10, 구현 노트·발사 아님)

지금까지 검산 사다리는 전부 스칼라 보상 항(EVC — 정확도 해침) 아니면 기존 크레딧의
마스크/배율(EVCM·EVCA·TAG0)이었다. **두 궤적을 한 손실 안에서 직접 맞대는** 목적함수는
이 프로젝트에서 한 번도 안 돌았다 — 수학 단계 R18b(`archive/docs_pre_rq3/PLAN.md` "A.1
contrastive-on-natural-meta — FAIL")가 자연 발생 그룹에서 쌍을 찾다 굶은 것이 전부다.
`scripts/local/chk_pair_probe.py` 재측정도 같은 결론이다: 자연 GRPO 그룹(8롤아웃) 중
`chk_solved` 와 `over_claim`/미해결이 **함께** 있는 그룹은 **0.4~5.0%**. 그래서 쌍을 자연
그룹에 기대지 않고 **같은 자리(site)의 K개 이어쓰기에서 구성**한다(OPT_CF 쌍둥이·
`gen_continuations.py` 와 같은 관행). 만든 것: `scripts/local/build_check_pairs.py`
(자리별 chosen/rejected 1쌍, 분류는 `countdown_rewards.check_row` 재사용, 양쪽이 실제로
존재할 때만 방출, 쌍 비율이 굶으면 하드 실패 + 무작위 5쌍 대장 기록), `src/training/
dpo_check.py`(표준 DPO 손실 + 단독 스텝, 참조 logp 는 계산하지 않고 기존 OPD/PMI 경로
`trainer._compute_ref_log_prob` 산출물을 받는다), 시험 `src/training/tests/
test_dpo_check.py` 15건 전부 통과. **현 시점 디스크에 있는 conts_v1/v4 이어쓰기에는
`<check>` 가 한 건도 없다**(전부 check 프롬프트 변형 이전 생성분) — 즉 실제 쌍 코퍼스는
아직 없고, 스크립트는 그 상태에서 설계대로 크게 실패한다. ⚠️**트레이너 배선은 안 했다**:
verl PPO/GRPO 스텝에 연결돼 있지 않고 ARM_SPECS 에도 없다. 발사 가능한 팔이 아니며,
판정선·성공 기준은 여기서 정하지 않는다(사전등록이 아니라 구현 기록이다).

## §16 R4c FIXEDA_CHK — FIXED_CHK(형식 보너스) + EVCA(증폭) 결합 (2026-09-11)

**동기(09-11 01:30 실측).** 평가 파이프라인 수리 뒤 확보한 FIXED_CHK 스텝 50 held-out
이 TAG0 대비 **+4.0pp** — 지금까지 나온 모든 팔 중 대조군 우위가 가장 크다(EVCM
+0.9pp 보다 큼). FIXED_CHK 는 검산 사다리에서 가장 단순한 보상(형식만 맞는 `<check>`
존재 시 +W_CHK, 내용 무관)인데, 가장 정교한 팔들보다 앞섰다. `<check>` 사용률이
98~99%(EVCM 은 23~27% 로 선택적)라는 점에서 EVCM 과는 다른 기전으로 보인다.

**결합 배선 수리(선행 필요, 0911).** `verl_sdc._countdown_add_check_region_advantage`
에 이전엔 몰랐던 상호작용 버그가 있었다: chk_mask 배율(scale≠1) 이 걸리면 `continue`
로 곧장 다음 행으로 넘어가, 같은 행의 그룹 중심화 보너스(`c`, chk_fixed 등) 덧셈이
**조용히 스킵**됐다. EVCM/EVCA/EVCAS 는 `terms` 에 chk_fixed 류가 없어 `c` 가 항상
0 이라 지금까지 발사된 어떤 팔에서도 관측된 적 없는 잠재 버그였다. 배율과 덧셈을
독립된 단계로 분리해 같은 행에서 **둘 다** 적용되도록 고쳤다(회귀 없음 — 기존
테스트 18개 그대로 통과, 결합 검증 테스트 1건 추가:
`test_scale_and_additive_bonus_compose_on_same_row`).

**정의.** `ARM_SPECS["FIXEDA_CHK"]` = `terms=("corr","format","fclaim","chk_fixed")`
(FIXED_CHK 와 바이트 동일) + `chk_mask="amplify"`(EVCA 와 바이트 동일). 새 항 없음 —
이미 검증된 두 기전을 합쳤을 뿐이다.

**판정.** §14 EVCA 와 같은 판정 지표(held-out 500×8, 짝지은 스텝). 1차 =
FIXEDA s50 ≥ FIXED_CHK s50 + 1pp(증폭이 형식 보너스 위에 추가 값을 더하는가).
2차 = N0 와의 격차가 FIXED_CHK 단독보다 좁혀지는가.

**자원·순서.** 새 GPU 요구 없음 — EVCA 판정 뒤(§14 큐 규율 그대로) 다음 순번으로
큐에 넣는다. 아직 제출하지 않았다(사전등록 시점 = 구현 완료·미발사).

## §17 R4d NOSURR_CHK — "정직한 포기" 무료 통행증 막기 (2026-09-11)

**동기(09-11 실측).** FIXED_CHK 오답의 70.1% 가 "스스로 ✗ 로 표시한 최종식을 그대로
제출"하는 패턴이다(응답 예산은 남아 있음). `over_claim` 은 이 경우를 안 잡는다 —
정의상 자기 최종식이 이미 flagged_bad 에 있으면 "확신에 찬 오답"이 아니라 "정직한
오답"으로 분류돼 fclaim 도 안 받는다. 즉 지금까지 이 패턴은 완전히 무료였다.

**정의.** `check_row` 에 `honest_flag`(최종식을 스스로 ✗ 로 표시하고 그대로 제출) 필드
추가. `verl_sdc._mark_honest_surrender_early(rows, groups, frac=0.6)` 가 그룹(같은
문제 8롤아웃) 동료가 실제로 쓴 최대 응답 길이의 60% 미만만 쓰고 끝난 honest_flag 행을
`honest_surrender_early` 로 표시한다(절대 길이·config 무관, 그룹 상대적). `ARM_SPECS
["NOSURR_CHK"]` = EVCM 과 항 동일(새 보상 없음) + `chk_mask="amplify_nosurr"`: chk_solved
는 CHK_AMP 배 증폭(EVCA 와 동일), `over_claim` **그리고** `honest_surrender_early` 둘 다
CHK_AMP_NEG 배 증폭(EVCAS 의 대칭판을 확장 — 새 벌점이 아니라 기존 어드밴티지 증폭).

**테스트.** `tests/test_check_terms.py` 4건 추가(honest_flag 판정, 그룹 상대 계산,
배율 선택, ARM_SPECS 등록) — 전부 통과, 회귀 없음(기존 3건 실패는 무관, 사전 확인됨).

**판정.** §14 EVCA 와 같은 지표. 1차 = NOSURR s30 ≥ EVCM s30 + 1pp. 기제 = 학습 중
`honest_surrender_early` 비율이 스텝이 갈수록 줄어드는가(EVCM 의 keep_nonzero 감소
패턴과 같은 방향인지). 아직 발사 전 — 스모크(20스텝) 부터.

## §18 R4e LENBONUS_CHK — 길이 confound 대조군 (2026-09-11)

**동기.** §16 에서 확보한 FIXED_CHK 의 우위(TAG0 대비 +4.0pp, 지금까지 최대)를 그대로
"check 형식 보너스가 유효하다"로 해석하기 전에, 더 단순한 대안 가설을 배제해야 한다:
`chk_fixed` 는 `<check>` 태그가 **있기만 하면** 내용과 무관하게 +W_CHK 를 준다 —
이는 사실상 "체크섹션 하나를 더 쓰라"는 신호이고, 그 결과 응답이 구조적으로 길어진다.
FIXED_CHK 의 우위가 check 내용(자기 검산이라는 행동)이 아니라 **단순 응답 길이 증가**
자체의 부산물일 가능성을 이 실험으로 격리한다.

**정의.** `ARM_SPECS["LENBONUS_CHK"]` = `terms=("corr","format","fclaim","len_bonus")` —
chk 내용 항(`chk_fixed`/`chk_evc`/`chk_persist`/`chk_solved`) 은 전혀 넣지 않는다.
프롬프트·학습/평가 데이터는 FIXED_CHK 와 완전히 동일(`prompt_variant="chk"` 재사용,
새 파이프라인 없음). `len_bonus` 는 응답 문자 길이(`_resp_char_len`, NOSURR_CHK 작업 때
이미 전 행에 무조건 계산되도록 배선됨)가 `LEN_BONUS_CHARS`(기본 800, 환경변수로 조절)
이상이면 1.0, 아니면 0.0 — `<check>` 존재 여부·내용을 전혀 보지 않는다.

**테스트.** `tests/test_check_terms.py` 3건 추가: (1) LENBONUS_CHK 에 check-내용 항이
전혀 없고 len_bonus 만 있는지, (2) `arm_reward` 로 임계값 경계(799자 vs 800자)에서
len_bonus 가 0→1 로 바뀌는지, (3) check 태그를 정확히 썼어도 짧으면 여전히 0 인지
(내용이 아니라 순수 길이만 본다는 것을 재확인). 전부 통과, 기존 3건 무관 실패
(`test_arm_specs_match_spec_table` 등, LENBONUS_CHK 이전부터 존재) 외 회귀 없음.

**판정.** §16 FIXEDA_CHK 큐 뒤(신규 GPU 요구 없음, 자원 경합 시 순번 대기). held-out
500×8, 짝지은 스텝(s30/s50) 비교.
- LENBONUS s50 이 FIXED_CHK s50 과 통계적으로 구분 안 됨(±1pp 이내) → **confound 확인**:
  FIXED_CHK 의 우위는 길이 부산물이지 check 내용과 무관. 이후 "check 형식 보너스"
  계열(FIXED_CHK, FIXEDA_CHK 포함)의 해석을 재검토해야 한다.
- LENBONUS s50 이 TAG0 수준(N0 대비 우위 없음 또는 FIXED_CHK 대비 유의하게 낮음) →
  check 내용 자체(자기 검산이라는 행동 형태)가 유효하다는 뜻 — FIXED_CHK/FIXEDA_CHK
  계열의 우위는 진짜다.

아직 발사 전 — 스모크(20스텝) 부터.

## §19 R6 내용-진리 자 + PL(계획) 팔 발사 — 검산에서 메타 내용으로 (2026-09-11)

**동기(사용자 지시).** "체크는 좀 이상한 것 같다, 전혀 메타가 아니다." 실제로
`FIXED_CHK`(현 최고)의 보상은 check 태그의 **내용을 전혀 안 본다**(형식만 맞으면 지급).
메타인지를 키우려는 실험에서 1등이 «내용 무관 보상»이라는 건 재고 있는 대상이 메타인지가
아닐 수 있다는 신호다. 그래서 계측을 내용 쪽으로 돌린다.

**새 자 6종(`scripts/local/meta_content_rulers.py`).** cd7 자 표(09-05)의 7종은 전부
«메타가 끼친 영향»(PMI-shift·move_kl·dCont…)이나 «내부 확신»(OSD)이었고, **메타가 말하는
내용이 사실인지** 잰 자는 하나도 없었다. 여기서 채우는 축: `names_move`(구체적 첫수 지목) ·
`move_novel`(아직 안 써본 수) · `move_live`(**오라클 완전열거 기준 아직 해로 가는 수**) ·
`followed`(약속 이행) · `claim_true_rate`(블록 안 산술 주장 검증) · `localizes`(오류 위치).

**실측(학습 없음·기존 롤아웃 재분석).**

| 팔/프롬프트 | names_move | move_novel | **move_live** | followed | claim_true | 어휘 고유도 | held-out acc |
|---|---|---|---|---|---|---|---|
| Qwen3-4B 베이스(opt) | .008 | .000 | **.000** | .008 | — | 1.00 | .352 |
| OPT s50(메타 허용, 보상 없음) | .007 | .000 | **.000** | .007 | — | 1.00 | .688 |
| OPT_MT s50(타이밍 보상) | .156 | .128 | **.037** | .138 | .94 | 1.00 | — |
| FIXED_CHK s100(검산) | .996 | .105 | **.049** | .832 | .65 | 0.60 | .742 |
| **베이스 + p3 프롬프트(next 슬롯)** | **.983** | **.966** | **.542** | **.944** | .93 | 0.99 | **.098** |

세 가지가 한꺼번에 확정된다.
1. **다양성은 고칠 대상이 아니다.** 메타 팔의 어휘 고유도가 이미 1.00(전부 서로 다름)이다.
   자기희귀성·KL 기반 다양성 보상은 이미 최대치인 값을 올리려는 것이 된다 — 기각.
2. **비어 있는 건 진리성이다.** `move_live`가 베이스·OPT 0%, 타이밍 보상 3.7%, 검산 4.9%.
   즉 지금까지 어떤 팔도 «아직 해로 가는 수»를 지목한 적이 거의 없다. 7종 자가 전멸한 것도
   당연하다 — 잴 내용 자체가 없었다.
3. **형식 하나로 내용이 생긴다.** p3(`next:` 슬롯 강제)만으로 `move_live`가 0%→**54.2%**.
   학습 없이, 베이스 모델에서. 단 정확도는 .098로 붕괴한다(plain .426).

**발사: PL 팔.** `ARM_SPECS["PL"]` = `_COMMON + ("plan",)`, `plan` = `1[next 가 해를
살린다(완전열거)] × 1[실제로 이행]`. 0902에 구현·오프라인 A/B 검증(«치환으로 확인된 유일한
내용 신호», 막힘 자리 +8.1pp, 2시드)까지 끝났는데 **한 번도 발사된 적이 없다**(라운드가
OPT/MT/CF/OPD/검산으로 흘렀다). 배선 점검 완료: `countdown_rewards.plan_next()`(오라클),
`verl_sdc.py:1346` 행 필드, `cd/plan_{ok,followed,hit}_rate` 텔레메트리, `_4num_p3` 데이터
전부 살아 있다.

**왜 지금 p3 인가.** OPD 계열은 신호 밀도 3~15%에서 굶어 죽었다(«항이 사실상 굶는다»).
p3 는 `move_live` 54%로 **유일하게 밀도가 충분한 형식**이다. 대신 정확도 출발선이 .098로
낮다 — 그래서 이 팔의 1차 질문은 «대조군을 이기는가»가 아니라 **«내용이 좋아지면 정확도가
따라 오르는가»**(핵심 가설의 직접 검정)다.

**판정.** 20스텝 스모크 → 자동 승격(promote_smoke) → 100스텝. 게이트: s30 held-out
acc < 0.20 이면 중단(베이스 .098 대비 RL 이 사실상 아무것도 못 올린 경우만 거르는 파국
바닥). 기제 지표: `plan_hit_rate`(=plan_ok ∧ followed)가 스텝에 따라 오르는가, 그리고
그것이 held-out acc 와 같은 방향으로 움직이는가.

**후속(미구현).** p3 의 정확도 붕괴는 강제·중량 형식 탓일 가능성이 크다(발화 88%).
`opt` 의 가벼움 + p3 의 `next:` 슬롯만 결합한 경량 변형이 다음 후보다 — p3 의 내용
성질을 유지하면서 세금을 줄이는 것이 목표.
