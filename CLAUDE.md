# Meta-CoT: Metacognitive Chain-of-Thought for Math Reasoning

## Goal
Train models that externalize metacognitive reasoning via `<|meta|>` tokens,
enabling self-assessment, error correction, and calibrated confidence.
**Core metric: Meta-CoT models must outperform Base SFT on math benchmarks.**

### Intent (north-star, 2026-06-07)
**메타인지 행동을 강화해서 성능(정확도)을 올린다.** 이를 위해:
1. **metacot를 정의한다** — *언제* 어떤 메타인지를 해야 하는지, 그리고 *무엇이 좋은
   메타인지 습관*인지를 명시적으로 규정한 메타인지 프로토콜. (예: 막혔을 때 가정 점검,
   답 직전 검산, 불확실하면 접근 전환 — 그리고 이것들을 *유용할 때만* 한다.)
2. **그 metacot를 강화하는 RL 방법을 개발한다** — 단순히 confidence 숫자를 맞추는
   calibration이 아니라, **좋은 메타인지 행동이 실제로 정답률을 높일 때 그 행동을 보상**하는
   RL. 즉 메타인지는 목적이 아니라 정확도를 끌어올리는 *수단*이며, RL은 "유용한 메타인지"를
   선택적으로 키운다.

핵심 구분: **calibration(자기 confidence를 정답률에 맞추기)은 부분 목표일 뿐**이고,
최종 목표는 **메타인지 행동 → 정확도 향상**이다. confidence 정렬은 "유용한 메타인지"의
한 신호(언제 검산/전환할지 판단)일 때만 가치가 있다.

## 현행 상태 (2026-10-06, cd9 수정 60) — 이 절이 우선한다
- **의도(고정)**: «특정 언어 패턴인 메타 인지를 내부 신호를 활용해 강화하려는 셀프 디스틸레이션» — 첫 `\boxed` 뒤 «Wait—» 되짚어 틀린 답을 선택적으로 고침. 정확도가 목표, 버릇은 수단, 칭찬 핵심 = 내부 신호(PMI 계열), 쌓아 나가기.
- **방법(SPONT_PFX, hc2 데이터, Qwen3-4B-Instruct-2507)**: 고정 앞부분 뒤 이어쓰기 8개, 결과 GRPO + 교사(모두 얼린 자기 자신, 문맥만 다름):
  눈 가린 나 = 리셋 증류 `PFX_DISTILL`(문제 + «전에 X», 풀이 가림; 가산, `PFX_DISTILL_ROWS=wrong|all`) · 어제의 나 = 유지 `PFX_KEEP` · 대조 = CH `PFX_FORK=ch`(성공/실패 형제, 곱) · 끈 = `KL_COEF`.
- **핵심 결과**: 리셋 = 반복 꼬리·닻 제거(엄격 균등 대비 +10%p), 관대는 균등 수준, 씨앗 번짐; 고치기↔지키기 맞바꿈 선. **오라클 쪽지(10-06)**: «틀렸다» 쪽지만으로 고침 .026 → .216(망침 ≈ 0) = 병목은 알아채기. 이어쓰기 안 Y≠X 는 오답 정밀도 .89, Y≠X 뒤 결정은 이미 84% 고침.
- **계획(수정 60)**: 1 바탕 굳히기(재현·위약·리셋 모든 행·세 교사·표준 시험) → 2 말 단위 배분(c_t = 눈 가린 나 − 어제의 나) → 3 앞부분 OPD(라운드 연습지 → 온라인 두 단계) → 4 굳히기(씨앗·AIME/HMMT/MATH-500·두 번째 모델).
- 사전등록 맨 위(수정 60·59b·59·58c…) · 원장 `docs/RESULTS_cd9.md` 맨 아래 · 코드 `mc/`(mc + tests ≤ 5,000줄, 현재 4,907) · 실행 스크립트 `/hdd_data/seungpil/scratch/analysis/u_checks_0925/a56_hard_class.sh` · GPU 2·3만, git/wandb 없음 · 검토는 Fable 쓰지 않음(본 에이전트·Opus).

## 현행 상태 (2026-09-24, cd9 수정 18) — 10-06 수정 60 절로 대체됨
- **의도(고정)**: 특정 언어 패턴인 메타인지(첫 `\boxed` 뒤 스스로 되짚어 틀린 답을 고치는 말)를 **모델 자신의 믿음 이동(PMI) 내부 신호**로 강화하는
  셀프 디스틸레이션. 정확도가 목표, 버릇은 수단. **칭찬의 핵심은 PMI — 다른 신호로 대체 금지**(결함은 PMI 를 고쳐 해결, MC 다시 쓰기는 검증 전용).
- **PMI(수정 18)**: 로그 단위, 되짚기 구간 안 새 답을 쓰기 전 자리에서 **방향**(정답 − 대안 평균 이동; 칭찬·벌)과 **크기**(첫 답에서 멀어짐; 말뿐인 되짚기 걸러내기).
  틀린 첫 답: 방향 ±, 맞은 첫 답: 망침만 벌(데드밴드) + 긴 말뿐 되짚기 비용. 바탕 = 행 전체 GRPO. 가지치기 금지(가중치로).
- **순서**: V1′(구별력·길이 통제) → V2′(실제 미래 대조) → 스텝1 부호표 → 1-B·1-P(gold) → 위약·질량 대조 → 20스텝 확증 → 2단계(다수결).
- 사전등록 `docs/PREREGISTRATION_cd9_math_judgment.md` 수정 18 · 원장 `docs/RESULTS_cd9.md` · 코드 `mc/`(5,000줄 이하) · GPU 2(+5 임시), git/wandb 없음.

## 현행 상태 (2026-09-18, cd9 수정 6) — 0924 수정 18 절로 대체됨
- **축**: «자발적 답 수정 행위의 증폭»(0917 관측: 진술 기반 메타 신호는 문제 내 짝비교에서
  전부 소멸, 자발적 답 수정만 유일하게 통과). `docs/RESULTS_cd9.md` "0917" 절, 사전등록 수정 6.
- **크레딧**: 행 내부 반사실(같은 행 첫 답 대 마지막 답의 다수결 일치 차) — 형제 평균 대비
  아님. 크레딧 구역 = 첫 박스 끝 → 마지막 박스.
- **PMI 두 자리**: 첫 박스 직후 / 마지막 \boxed 직전.
- **앵커 세 팔**: gold · 다수결 · 조합.
- **네 팔 50스텝 계획**: 결과만 / 결과+정오개선 크레딧 / 결과+PMI gold / 결과+PMI 조합.
- **멈춤 규칙**: 정밀도 <.6, 첫 답 정확도 하락(사행), 20스텝 수정률 무변화.
- 아래 0914 절은 이 절로 대체됨(우선순위 1~3 은 폐기, 새 축으로 교체).

## 현행 상태 (2026-09-14, cd9) — 0918 절로 대체됨
- **의도**: 정책이 자기 풀이의 상태를 스스로 읽고(모니터링), 그 판단으로 다시 볼지·어디를 볼지
  정하며(통제), 그 판단의 옳고 그름을 정책 자신의 롤아웃만으로 채점해 학습한다(자기 증류).
  정확도는 결과이지 목표가 아니다. (사전등록 수정 5)
- **우선순위 1**: M_DIST(정오 마스크 건 분포 이동 보상) — 관문 G2 «분포 수준에서 own ≠ donor».
- **우선순위 2**: 자기 인용 자리 반사실(이행 게이트 + 무작위 자리 대조, 관문 «인용 자리 > 무작위 자리»).
- ⛔**우선순위 3(형제 중심 거리 = 판단 라벨)은 기각됨**(0914 20:40) — G1 관문(Level 5 문제별
  AUC ≥ .65) 미달, 실측 .511/.551/.569로 우연 수준. math500 소표본(.724/.794, 7문제)은 재현
  실패. `docs/RESULTS_cd9.md` §G1 형제 기하학 관문 참조.
- **그 밖**: M_RETRY(+강제 redirect)·M_AGREE 는 통제 축, DeepScaleR 코퍼스는 성능 축 대조로 완주.
  관문 없는 팔은 올리지 않는다.
- **무대: 수학**(MATH-500/AIME25/HMMT25) · **정책: Qwen3-4B-Instruct-2507** · Countdown 종료.
- 평가 예산 `EVAL_MAX_TOKENS=8192`, 디렉터리 `math500_8k` / `math500_retry_8k`.
- **사전등록 `docs/PREREGISTRATION_cd9_math_judgment.md`**(banner 수정 1~5) · 원장 `docs/RESULTS_cd9.md`.
- 런처 `scripts/local/run_math_arm.sh ARM SEED [STEPS]`(`DATA_TRAIN`/`DATA_VAL`/`RESP_LEN`), ARM 은
  `src.training.math_meta.MATH_ARM_SPECS`. 중단은 rc 75 + 큐 aborted/. GPU **2·3만**, git/wandb 없음.

## Key Tokens
- 모든 토큰(GitHub PAT / HuggingFace / WandB)은 **.env에만** 둔다 —
  `set -a; source .env; set +a` 로 로드 (`GH_TOKEN`, `HF_TOKEN`,
  `WANDB_API_KEY`). 코드·yaml·문서 어디에도 하드코딩 금지.
  (repo: iamseungpil/metacognition-math, dataset: iamseungpil/metacot)
- 주의: 이전 버전의 이 파일이 라이브 토큰을 평문으로 담은 채 커밋 이력에
  존재한다 — 세 토큰 모두 회전(rotate)하고 git filter-repo/BFG로 이력에서
  제거해야 한다.
- TRAPI scope: api://trapi/.default (endpoint: trapi.research.microsoft.com/gcr/shared)

## Compute (구세대 참고용 — 현행은 위 «현행 상태» 절)

**현재(0904) 실제로 도는 곳: 로컬 H100×8 박스, GPU 0~3.** amlt/클러스터가 아니다.
`scripts/local/`의 GPU 큐(`gpu_queue.py`)가 워커·잡을 관리하고, 체크포인트·데이터·큐
전부 `/hdd_data`에만 둔다(`/scratch` 없음 — amlt 세대의 관례가 이 머신에는 적용 안 됨).
실행 방법은 `scripts/local/README.md`, 아키텍처는 `ARCHITECTURE.md` (c). 아래 두 절
(amlt VC 제약·현행 런처)은 **amlt 클러스터 복구 시에만** 유효하다 — 지금은 아무 잡도
그리로 안 간다.

### amlt VC 제약 (0727 기준 — 클러스터 복구 시에만 참조)

**msrresrchbasicvc** — H100/H200/A100/MI300X 보유. **0726 05:49부터 우리 신원의 신규 제출을
전부 거부**한다(`UserError: The virtual cluster does not exist`). 1-CPU echo 잡도 같은 메시지를
받으므로 용량이나 SKU 문제가 아니다. ⚠️0727의 "`GroupPolicy: e9deff52-...`에 멤버십을 받으면
된다"는 진단은 **0728에 철회**됐다 — 그 값은 제출자의 AAD object id이지 가입 가능한 정책이 아니다
(VC의 `groupPolicies` 17개 중 우리 것도 그 사용자 것도 없고, `expand-sku` 프로브는 모든 id에
대해 실패한다). 확정된 것은 **서비스측 권한 판정**이라는 사실뿐이다: amlt **11.9.1·11.14.2·
11.16.0 세 버전 모두** 동일한 서버 에러(클라이언트·yaml·SKU 별칭 배제), 제출하는 VC ARM id는
정상 잡과 바이트 동일, ARM 읽기(쿼터 GET)는 지금도 성공. 유력 가설(미확인)은 0716 GCR 재할당 때
우리 신원이 새 allocation으로 이관되지 않았고 구 경로가 0726 05:49에 폐기됐다는 것.
요청서 = `archive/incidents_pre_rq3/2026-07-26-basicvc-submission-block-escalation.md`(correlation ID 포함).
차단 이전에 진입한 잡은 영향 없이 계속 돈다. 이 VC에는 **Premium SLA가 없다**(Standard/Basic만).
Standard 티어라 선점이 잦으므로 ckpt 릴레이/resume 배선은 여전히 필수다.

**msrresrchvc** — A100/CPU/MI200만. **H100 없음.** 결정적 제약:
| | 쿼터 | 사용자 한도 |
|---|---|---|
| A100 80GB (`NC_A100_v4`/`NDAMv4`) | Premium 4/4, Standard 0/0, Basic 0/32 | **1 GPU** |
| A100 40GB (`NDv4`) | Premium 381/384 | 12 GPU |
사용자 한도가 **1장**이라 2·4-GPU 잡은 제출은 수락되고 **영원히 스케줄되지 않는다**(21시간 대기
관측). 1-GPU는 3분 만에 붙고 ~3.5h 후 선점된다. 8B SFT2는 1-GPU에서 ~7.7h가 필요하므로
**선점창 안에 완주할 수 없고**, HF로 미는 체크포인트는 weights-only(DeepSpeed 옵티마이저 ~96GB
제외)라 크로스노드 resume도 불가하다. 다GPU가 필요하면 40GB 계열이 유일한 길이다.

- Image: mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x
- Conda env: /scratch/conda_envs/simplerl (conda-pack)
- AMLT project: skilldiscovery2
- **클러스터 복구 시 재개할 런처**(그 외 루트의 `h100std_rq3_*`, `h100std_sft_*`,
  `a100g1_*`, `a100g2_*`는 은퇴 — 아무것도 지금 돌고 있지 않다):
  - SFT2 쌍: `h100std_sft_b0p2_rvfull.yaml`(컨트롤) / `h100std_sft_b2p2_rvfull.yaml`(메타)
  - RL: `h100std_rq3v2f_{b0p,b2p,b3p}.yaml` — 이 3종이 현재 도는 arm
    (solid-gibbon / hip-hound / pure-stag)
  - ⛔ **a100 판은 0803에 `archive/launchers_retired_0803/`으로 은퇴**. `a100_rq3v2f_*`
    3종은 ckpt_dir·HF repo_id·config_name·path_in_repo가 살아있는 arm과 바이트 동일이고
    `push_ckpts_to_hf.py --keep 1`의 `delete_folder`가 **도는 arm의 유일한 재개 상태를
    지운다**. 절대 제출 금지. 사유는 그 폴더의 README 참조.
  - **basicvc 복구 시 순서**: `h100std_sft_{b0p2,b2p2}_rvfull.yaml`(SFT2 쌍) → 두 산출물이
    HF에 착지한 뒤 `h100std_rq3v2f_{b0p,b2p,b3p}.yaml`(RL). 이 5종은 0727에 a100 판에서
    복제해 두었고 target/sku/tier만 다르다(`msrresrchbasicvc` / `80G4-H100` / Standard —
    basicvc엔 Premium SLA가 없다).
  - ⛔**`h100std_rq3v2_{b2p,b3p}.yaml`(f 없는 구 lineage)는 쓰지 말 것** — `b2p2_rvseg_sft`
    (E-093에서 위장된 시나리오 필터로 폐기된 378행 init)를 스테이징한다. RQ2 부록 전용이며
    **복제 결과로 보고 금지**.
- ⛔런처 yaml 편집 시 **`\`로 끝나는 줄 다음에 주석/빈 줄을 두지 말 것** — bash가 명령을 그
  지점에서 끊는다. `bash -n`은 통과시키므로 `tests/test_launcher_yaml_lint.py`가 지킨다(E-125).

## Data (HuggingFace: datasets/iamseungpil/metacot)
SFT inputs (current = **RQ3v2 think-on** matched ladder — 2단 SFT 스택):
- b0p arm: data/b0on_v8base_strict_sft.parquet → models/b0p_v8base_strict_sft
  (init Qwen3-8B-Base, 3ep lr 1e-5) — meta 제거된 matched base
- b2p/b3p arm: **SFT1** data/b2on_v8meta_strict_sft.parquet → b2p_v8meta_strict_sft
  (init Qwen3-8B-Base, 3ep lr 1e-5) → **SFT2** data/rv_redirect_verify_functional.parquet
  (raw 1,763행, E-093/094/095 수리판) → **models/b2p2_rvfull_eb16_sft** = 현행 RL init
  (3ep lr 1e-5). ⚠️구 378행 판(`b2p2_rvseg_sft2.parquet`→`models/b2p2_rvseg_sft`)은 은퇴.
  - ⛔**0812 발견(EXP-0812c)**: SFT2 는 wrong_prefix 를 **전 행**(verify 1,209 포함)에서
    손실-마스크해 "메타 앞 추론 생산"을 한 행도 가르치지 않았다 → RL step 1 부터
    온폴리시 97.2%가 meta-first. **수리**: ①`sft.py:_should_mask_prefix` — redirect(와
    scenario 무필드 레거시)만 마스크 ②수리 코퍼스 `data/rvfull_verify_unmasked.parquet`
    (HF metacot-rv) ③재빌드 출력 `models/b2p3_vunmask_sft` (h100std_sft_b2p3_vunmask.yaml).

SFT inputs (retired = RQ3 think-off 세대, 부록으로만):
- data/b0_gold_sft.parquet → models/b0_gold_sft (구 B0 init) — 공개 HF gold,
  gsm8k 637 + MATH 653 = 1,290행 (RV 문제 부분집합, 정답 math_verify 검증)
- data/b23_rv_unmasked_sft.parquet → 구 B2/B3 init — RV redirect-verify 1,763행,
  wrong_prefix 비움(whole-response 학습; base meta emission 38% → 92%).
  ⚠️ 이 경로는 **HF에 존재하지 않는다**(로컬 data/ 전용). 현행 init은 위 b2p2_rvseg_sft.

SFT inputs (pre-rq3 = v8 series, instruct 세대):
- data/v8_meta_inside_think.parquet → checkpoints/v8_meta_inside_E20a (Meta SFT)
- data/v8_meta_inside_strict.parquet → v8_meta_inside_strict_sft (cold start for all RL)
- data/v8_base_matched_clean.parquet, data/v8_base_matched_strict.parquet (Base SFT counterparts)
- base_sft.parquet (top-level): 4,996 chains, meta stripped (legacy Base SFT)

RL inputs:
- data/verl_train_redirect.parquet (R5, OPD, ROD-PT all use this — configs/meta_*_h100_4x4k.yaml)
- pulled via scripts/pull_parquets.py at job start

Code snapshot:
- code_snapshots/metacognition.tar.gz — all training yamls hf_hub_download + extractall('/scratch')
  before bootstrap. Push via tarball after every code change.

NOTE: Earlier draft mentioned metacot_v2_trapi.parquet — that file does NOT exist on HF.
The v8 series replaced it.

## Current Results (구세대: rq3 매치드 래더 — PRELIMINARY, 단일 시드·진행 중·미확정)
- RQ1(B2−B0): 매칭 val 3점 +0.151(gs25) / +0.164(gs50) / +0.189(gs75),
  9개 데이터셋 전부 양성.
- RQ2(B3−B2): gs25 +0.042 한 점 — 어려운 과목 집중(int_algebra +0.125,
  counting +0.089, precalculus +0.081; 쉬운 gsm8k -0.02).
- B3 gs25 게이트 통과(emit 0.89 · attempted 0.40 · n_save 7 ·
  acc_with 0.70 / without 0.28). 단 meta emission이 RL 중 0.89→0.54 침식 중
  (answer 스팬만 correctness 받는 구조적 압력; 행동은 건재) — 관찰 중.
- ⚠️ 위 숫자는 in-training val(594문제, greedy) 기준이며 gs300 held-out 1030
  최종 판정 전이다. 모든 숫자 PRELIMINARY 취급.

### Pre-rq3 (instruct 세대) 결과 — 보존
- AIME overconfidence: 97% → 14% (calibration success), AIME ECE: 0.870 → 0.610
- 초기: Meta-CoT accuracy < Base SFT (MATH 56.7% vs 76.7%;
  meta overhead 56% of tokens, 31% truncation)
- 최종(T1, instruct pmishift vs matched-base): held-out 6/6 셀 유의 승리
  (단일 시드, triobj 패키지 효과)

## Autoresearch Loop (until Meta-CoT > Base SFT)
1. Critic: analyze why Base > Meta, classify error types
2. Planner: hypothesize fix (SFT format, RL reward, token length)
3. Implementer: code + run experiment
4. Eval: 1,030 problems (GSM8K 500 + MATH 500 + AIME 30), max_tokens=4096
5. Repeat until Meta-CoT accuracy ≥ Base SFT

## Code Structure
- src/training/verl_sdc.py — **메인 RL 트레이너** (entry: `python -m
  src.training.verl_sdc`; VANILLA_GRPO + TRIOBJ_DCPO_V4)
- src/training/dcpo_region.py — advantage 조성 (region-split:
  correctness→answer 스팬, pmi_shift→meta 스팬)
- src/training/dcpo_pmi_shift.py — PMI-shift 보상
- src/training/sft.py — SFT training (wrong_prefix segment-mask)
- src/training/rewards.py — reward functions
- src/training/grpo_v2.py — pre-rq3 세대 GRPO variant (아카이브 취급 —
  메인라인 아님; 현행 메인 트레이너는 verl_sdc.py)
- src/eval/eval_hf.py — HF generate eval (legacy; 채점은 math_verify로)
- src/curriculum/rag.py — Meta-guided curriculum learning (FAISS + sentence-transformers)
- src/metacot/prompt_v2.py — V2 prompt (diverse confidence, error→fix)
