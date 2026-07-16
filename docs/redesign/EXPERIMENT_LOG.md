# EXPERIMENT LOG — RQ3 매치드 래더 (Qwen3-8B-Base 재설계)

> rq3 재설계의 시행착오를 시간순으로 남긴다. 각 항목은 "무엇이 깨졌고, 어떻게
> 원인을 확정했고, 무엇으로 고쳤는가"를 기록한다. 진단 원칙과 게이트는
> `docs/CONSTITUTION.md`, 현행 레시피는 `docs/redesign/base_rl_recipe.md` 참조.
> 마지막 갱신: 2026-07-16.

## 1. v1 전멸 — base 엔트로피 붕괴 (2026-07-08)

v1 발사분(B0/B2/B3)이 전부 gs50–100에서 엔트로피 붕괴(entropy→0.01, clip→0.9,
response length 717→3582 폭주). **원인 = instruct 세대 하이퍼파라미터를 base
substrate에 그대로 복사**한 것: temp 0.6 + top_k 20, max_response 4096 캡,
GDPO advantage std-정규화. 인과 확정 근거: 동일 레시피의 instruct 쌍둥이 런은
300스텝 완주 → substrate 인과. (SFT는 무죄 — eos 불변식 정상, 코퍼스 4096 초과
0%, gs25까지는 정상이었음.) **처방 = v2 레시피**: temp 1.0 / top_k -1 /
top_p 1.0 / max_response 8192 / max_model_len·max_num_batched_tokens 10240 /
`norm_adv_by_std=false`(Dr.GRPO) / logprob micro-bs 2. SimpleRL-Zoo·Dr.GRPO·
DAPO 문헌의 base-RL 권고와 일치.

## 2. pmi_shift 불발 — 하류 가설 기각, 상류(SFT emission)로 추적 (2026-07-10)

B3 계열에서 pmi_shift가 불발(pmishift_attempted 0.03, n_save 0). 보상 스케일 /
anchor_norm / len_cost가 원인이라는 하류 가설을 먼저 검증 — 전부 **기각**
(anchor_norm은 불발 신호를 증폭하는 증상 증폭기일 뿐, 원인 아님). 진짜 원인은
상류: **base meta-SFT가 under-trained** — 코퍼스는 100% meta인데 RL 시작 시
emission이 ~52%뿐이고 RL이 40%까지 침식(예전 instruct 세대는 99% emission으로
pmi_shift가 정상 발화했던 것과 대조). 메타 보상은 재료(meta 블록)가 없으면
정의상 불발이다.

## 3. 근본 원인 = sft.py wrong_prefix segment-mask → unmask fix (2026-07-10)

emission 저조의 근인을 `src/training/sft.py`에서 확정: **wrong_prefix
segment-mask가 응답 시작 부분을 학습에서 제외**해, 모델이 "응답을 meta로
시작하는 습관" 자체를 못 배움(emission 38%). **fix = b23 parquet의
`wrong_prefix` 필드를 비워 whole-response 학습**(unmask) →
`data/b23_rv_unmasked_sft.parquet`. emission 38% → **92%**.

## 4. B0 baseline 데이터 교체 — 오염된 matched_clean → 공개 HF gold (2026-07-10)

B0(no-meta) SFT 데이터를 기존 `v8_base_matched_clean` 계열에서 **공개 HF gold**
로 교체: gsm8k 637 + MATH 653 = 1,290행(`data/b0_gold_sft.parquet`), RV 문제
부분집합, 정답은 math_verify로 검증. → `models/b0_gold_sft`.

## 5. rq3 3-arm 발사 + B3 gs25 게이트 통과 (2026-07-11)

`h100std_rq3_b0/b2/b3.yaml` 발사 (init = `models/b0_gold_sft` /
`models/b23_rv_unmasked_sft`). **B3 gs25 게이트 통과 확정**:
meta_emit **0.89** (≥0.8) · pmishift_attempted **0.40** (≥0.3) ·
n_save **7** (>0) · acc_with **0.70** ≫ acc_without **0.28**.

## 6. 운영 교훈 (Standard 선점 클러스터)

- **선점/긴 validation을 행(hang)으로 오진하지 말 것.** Standard tier는 선점이
  잦고, in-training validation(594문제)은 ~1시간 걸리며, wandb는 단조 step
  거부로 조용할 수 있다 — 셋 다 "죽은 것처럼" 보인다. **진짜 행 판별법 =
  stdout을 재다운로드해서 줄 수가 불변인지 확인.**
- 진짜 행도 1회 실재했음: **gs40 체크포인트 저장 직후 vLLM 데드락** 1회.
- wandb는 `WANDB_RUN_ID` 고정 + `WANDB_RESUME=allow`로 run 파편화 방지
  (v1에서 18조각으로 파편화됐던 문제의 해결).
- save_freq: B0=5, B2/B3=10. test_freq=25. ckpt는 HF model repo
  `iamseungpil/metacot-h200-triobj-dcpo-v3` 경유 릴레이.

## 7. 예비 결과 (2026-07-12 현재 — 단일 시드·진행 중·**미확정**)

- **RQ1 (B2−B0)**: 매칭 val 3점 **+0.151(gs25) / +0.164(gs50) / +0.189(gs75)**,
  9개 데이터셋 전부 양성.
- **RQ2 (B3−B2)**: gs25 **+0.042** 한 점. 어려운 과목 집중 —
  intermediate_algebra +0.125, counting +0.089, precalculus +0.081;
  쉬운 gsm8k -0.02.
- **B3 meta emission이 RL 중 0.89 → 0.54로 침식 중.** answer 스팬만
  correctness를 받는 region-split의 구조적 압력으로 해석. acc_with ≫
  acc_without은 유지(행동 자체는 건재) — 관찰 지속.
- val은 in-training 594문제(verl_val_meta_mix), reward +1/−1 스케일이므로
  acc = (r+1)/2. 최종 판정(gs300 held-out 1030) 전까지 모든 숫자 PRELIMINARY.

## 8. 열린 항목

- [ ] B3 meta emission 침식(0.89→0.54) 계속 관찰 — 바닥 확인 / emission-floor
      개입 판단.
- [ ] **B2-R arm** 추가 (region-split, meta advantage=0) — RQ2의 순수 격리
      (현행 RQ2=B3−B2는 "region-split 구조 + pmi_shift" 패키지 차이).
- [ ] gs300 완주 → held-out 1030 최종 판정.

## 9. ★실패와 정정 (2026-07-12): B3를 pmi_shift 단독으로 스트립한 게 잘못 — RQ2 음성

**증상**: RQ2(B3−B2)가 gs25 **+0.042** → gs75 **−0.120**으로 뒤집힘.
B3 held-out이 9개 데이터셋 전부 저하(gs25 0.624 → gs75 0.526), 반면 B2는
상승(0.582 → 0.646). held-out meta_structure도 +0.089 → −0.092.

**근인 (수치로 확정)**: SFT 실패가 아니라 **RL 중 메타 형식 붕괴**.
`dcpo/wellformed_rate` 추이 — gs1~30 **~0.40 안정**(정점 0.426@gs30) →
gs40 0.295 → gs50 0.102 → gs60 0.055 → gs75 **0.016**. emit도 0.88(gs1-30)
→ 0.57(gs75). pmi_shift n_save 4~8(gs1-30) → 0~3(gs50+), rmeta_mean ~0.
즉 **SFT는 잘 됐고(gs30까지 안정) RL이 gs40부터 메타를 파괴**.

**진짜 원인 = 설계 이탈**. 예전 T1-승리 pmishift 런(`archive/launchers_pre_rq3/
h100std_pmishift.yaml`)은 triobj **풀 패키지**(w_meta 0.8 + w_format 0.35 +
trunc_open_penalty 0.3 + w_emit 0.1 + w_cal 0.3 + len_cost 0.08), **w_over만 0**.
그런데 rq3-b3(civil-eagle 등)은 "순수 pmi_shift 격리"를 위해 **w_format·
trunc_open_penalty·w_emit·w_cal·len_cost 5개를 전부 0으로 스트립**하고
pmi_shift만 남김. w_format(0.35)·trunc_open_penalty(0.3)·w_emit(0.1)이 바로
RL 중 wellformed 메타를 붙잡아주던 비계인데 그걸 제거 → pmi_shift 단독으로는
형식 유지 실패 → wellformed 붕괴 → pmi_shift가 계산할 belief-shift가 없어
불발(n_save→0) → 자기강화 붕괴(gs40 티핑포인트) → held-out 저하.

instruct 성공 대비: 예전 att 0.52~0.66·n_save 8~11·rmeta +1.0~1.2로 신호가
2배 강하고 지속됨(패키지가 형식 유지) vs base rq3-b3 att 0.33→0.08·n_save
4→0. **substrate 차이 + 형식 비계 제거의 복합.**

**방법론 교훈**: "깔끔한 단일변수 격리"를 위해 한 번에 5개 head를 끈 것이
오류. 성공 레시피 재현이 목표였다면 패키지를 유지하고 한 변수만 바꿔야 했음.
w_emit을 "form-not-behavior 함정"이라며 능동적으로 제거한 런처 주석이 특히
잘못된 판단.

**정정 (rq3-b3pkg)**: B3를 예전 pmishift와 **동일한 풀 패키지**로 재구성
(dcpo_rmeta_source=pmi_shift + config 기본 head 전부, w_over만 0). base
substrate + b23 SFT init + v2 붕괴수정 레시피는 유지. 손상된 gs60 ckpt에서
resume하지 않고 **gs0부터 새로** 시작(WANDB_RUN_ID=rq3-b3pkg, ckpt config_name
rq3_b3pkg). 기존 rq3-b3(pmi-only 실패, gs1~75)는 이 기록의 근거로 보존.
B0/B2는 정상이므로 불변. 검증 질문: "형식 비계를 살리면 pmi_shift가 base에서도
유지·발화하는가"(gs30까지 잘 됐다는 데이터가 성공 가능성을 뒷받침).

## 10. 격리 arm 설계 정정 (2026-07-12): B2-R(전부-off) → B3-noPMI(pmi만 제거)

RQ2 = B3pkg − B2 는 두 요인(메타 패키지 + region-split 라우팅)이 섞여 있어
순수 격리 arm이 필요. 처음엔 **B2-R**(region-split + 메타 head 전부 0)로 라우팅을
격리하려 했으나 — 메타 스팬에 advantage가 0이라 실패한 pmi-only처럼 **메타가
붕괴**(퇴화 컨트롤). 사용자 지적으로 교체.

**B3-noPMI**로 대체: B3pkg 풀 패키지에서 **w_meta(pmi)=0 하나만** 제거, form
비계(w_format 0.35·w_emit 0.1·w_cal 0.3·len 0.08·trunc 0.3)는 유지. 그래서
메타가 살아있고(붕괴 없음), **B3pkg − B3-noPMI = pmi_shift belief-shift 보상의
순수 한계 기여**(논문 핵심 메커니즘)를 깨끗이 측정. rmeta_source=pmi_shift는
single_turn 롤아웃 매치용(w_meta=0이 pmi advantage를 0으로).

구현 함정 2개 기록: ①B2-R이 rmeta_source 미오버라이드→기본 cf_group→반사실
agent-loop 유발(취소·수정). ②격리는 "변수 하나만 바꾸기"가 철칙 — 원래 B3
실패도 한 번에 5개 head를 끈 데서 왔음. **최종 4-arm**: B0(gold+vanilla)·
B2(meta+vanilla)·B3pkg(meta+풀패키지)·B3-noPMI(meta+패키지−pmi). 잡:
absolute-mallard·fair-vulture·sunny-camel·sterling-firefly.

## §11 0713–0716: 관측성 수정·fresh 재시작·유지보수 정지 (상태원장)

### 4-arm 상태 (2026-07-16 기준)

| arm | init | mode | amlt exp | wandb run | gs | 상태 |
|---|---|---|---|---|---|---|
| B0 | models/b0_gold_sft | VANILLA_GRPO | elegant-walleye 계열 | rq3-b0-2 | **gs300** | **완료** — 재제출 금지 |
| B2 | models/b23_rv_unmasked_sft | VANILLA_GRPO | first-hawk (retry_001~007) | rq3-b2-2 | gs150 (HF 안전) | 2026-07-16 12:52 UTC **영구 failed** — 재제출 시 resume_mode=auto로 gs150부터 |
| B3pkg | models/b23_rv_unmasked_sft | TRIOBJ_DCPO_V4 풀 패키지 | (미발사) | rq3-b3pkg-2 | gs0 | **미발사** — h100std_rq3_b3.yaml 준비완료, CODE_TAR_REVISION=476703893 |
| B3-noPMI | models/b23_rv_unmasked_sft | 풀 패키지 − pmi | (보류) | rq3-b3nopmi-2 | gs0 | **보류** — b3pkg 우선 (사용자 결정 2026-07-15) |

B3pkg head 구성: w_meta 0.8 / w_format 0.35 / w_emit 0.1 / w_cal 0.3 /
len_cost 0.08 / trunc_open 0.3 / w_over 0 / rmeta_source=pmi_shift.
B0 최종 held-in val(594, reward +1/−1 스케일): gsm8k 0.768 · algebra 0.658 ·
counting 0.886 · geometry −0.152 · precalc −0.081 · omni −0.587.

### 0713 무출력 사망 → 0714 감사 → 전면 fresh 재시작

2026-07-13 b3pkg의 "무출력 pass 사망" 연쇄는 코드 버그가 아니라 Standard 티어
콜드스타트 선점으로 확정(interactive 노드 SSH 손실행으로 gs1 도달·에러 0 확증);
9회의 헛다리(gpu_mem·tee 등) 끝에 관측성 수정(라인버퍼 출력·save_freq 단축)으로
발판 구간을 줄였다. 2026-07-14 4-arm 전면 감사에서 8개 수정(optimizer state
저장, anchor-EMA resume 불변성, gs0 콜드스타트 가드/RGS, 최종-push 이식,
save_freq, grader/절단 가드, bootstrap Xet)을 적용했고, 이전 단일-시드 런은
certifiable하지 않다고 판정 → **HF `checkpoints/rq3_*` 전부 삭제 + 전 arm
fresh 재시작(gs0)**. 따라서 RGS가 `rq3_b3pkg`에서 아무것도 못 찾는 것은
정상이며 fresh gs0가 올바른 동작이다.

### 유지보수 정지 (2026-07-16)

MSR GCR 전체 클러스터가 B200(Bonete) 재할당 + NVLINK 펌웨어로 정지: 2026-07-16
16:00 UTC(목 9am PT) 전 잡 강제취소 + 클러스터 오프라인, 복구 ETA **2026-07-17
00:00 UTC**(목 5pm PT) + 신규 할당(랩 GPU delegate 확인 필요 — 아직 연락 없음).
2026-07-15 22:41 UTC부터 모든 `amlt run`이 "(UserError) The virtual cluster
does not exist ... in the same cloud"로 실패하는데 이는 **유지보수/재할당
컨트롤플레인 문제이지 로컬 설정 문제가 아니다** (읽기 계열 `amlt target info`는
정상: H100 쿼터 272/560, 8/1120). 별도 메일(A100 Palisades/MSRRESRCHVC 유지보수
7/20–24 + 월간 OS 이미지 업그레이드)은 다른 VC·다른 GPU·다음 주 건으로 무관.
**GPU/VC를 갈아타지 말 것** — b0/b2가 이미 msrresrchbasicvc H100 Standard에서
돌았으므로 b3만 옮기면 매치드 래더가 깨진다.

### 0716 구현 감사 — 알려진 caveat (보상 코드는 불변 유지, 사용자 결정)

- **C-1 (가장 중요)**: `norm_adv_by_std_in_grpo=false`가 VANILLA 경로에서
  조용히 무시됨 — verl 0.7.1의 GDPO 분기가 이 플래그를 core
  `compute_advantage`로 전달하지 않아(우리 쪽 전달점
  `src/training/verl_sdc.py:4454~4492`은 region 경로만 유효) **B0/B2는 실제로
  group-std 정규화로 학습됐고** B3(region 경로)만 진짜 mean-only다. RQ1은
  내부적으로 일관되지만 RQ2에는 정규화-방식 비대칭이 실려 limitation으로
  명시해야 하며, "std-norm이 v1 붕괴 원인"이라던 §1의 믿음도 수정된다(B0가
  std-norm 하에서 300스텝 완주). 복구 후 노드 conda-pack verl 검증 한 줄:
  `python -c "import inspect, verl.trainer.ppo.ray_trainer as m; print(inspect.getsource(m.compute_advantage))"`.
- **H-2**: confidence 파서(`src/training/rewards.py` `_parse_confidence` 계열,
  소비처 `src/training/dcpo_region.py:950`)가 meta 블록만이 아니라 롤아웃
  전체를 스캔해 수학 본문의 확률-단어 숫자가 w_cal 타깃을 오염시킬 수 있다
  (해석 caveat only).
- **M-1**: pmi_shift 세이프가드(dup-thresh, `reversal_min_magnitude` —
  `src/training/dcpo_pmi_shift.py:68`)가 기본 OFF이고 어떤 config도 켜지 않는다
  (해석 caveat only).
- **RQ1-corpus**: RQ1은 순수 "meta-init" 효과가 아니라 코퍼스-수준 SFT 비교다
  (B0 gold-explanation 1,290행 vs B2 RV 1,763행).
- **RQ2-package**: RQ2는 6-head 패키지 효과(gold→confidence calibration인
  w_cal=0.3 포함)이며 절대 pmi_shift 단독에 귀속하지 말 것.
- **save_freq 비대칭**: b3=5 vs b0/b2=10 (콜드스타트 발판 단축 목적).

### 운영 규칙 (신규 인력 필독)

제출은 배치만(`amlt run <yaml> -y`) — interactive `-i`는 Standard opportunistic
풀에서 노드를 못 잡는다. amlt 바이너리는
`/home/v-seungplee/miniconda3/envs/amlt/bin`(PATH export 필수), 제출 직전 반드시
`set -a; source .env; set +a`. 선점 진단: `amlt log list <exp>`에서 retry_NNN이
있고 running이면 선점 중 — **그대로 둘 것, 취소 금지**(취소하면 노드가 글로벌
풀로 새어나감). 진짜 hang 시그니처 = 재다운로드해도 stdout 라인 수 불변 +
retry 증가 없음 + 다음 HF ckpt 부재. wandb run id는 -2 접미사(rq3-b0-2 등,
프로젝트 gistdslab/metacot-dcpo-v4). HF ckpt 배치: 모델 repo
`iamseungpil/metacot-h200-triobj-dcpo-v3`, 경로
`checkpoints/rq3_<arm>/global_step_N`. **완료된 arm은 재제출 금지**(resume→즉시
종료→sleep 86400이 노드를 점거). arm 비교는 val-aux correctness만(val-core는
보상-혼합 shaped라 arm마다 다름). 최종 RQ 판정은 매치드 gs(또는 gs300)의
held-out 1030에서만 — in-training val 594가 아님.
