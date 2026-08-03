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

## Key Tokens
- 모든 토큰(GitHub PAT / HuggingFace / WandB)은 **.env에만** 둔다 —
  `set -a; source .env; set +a` 로 로드 (`GH_TOKEN`, `HF_TOKEN`,
  `WANDB_API_KEY`). 코드·yaml·문서 어디에도 하드코딩 금지.
  (repo: iamseungpil/metacognition-math, dataset: iamseungpil/metacot)
- 주의: 이전 버전의 이 파일이 라이브 토큰을 평문으로 담은 채 커밋 이력에
  존재한다 — 세 토큰 모두 회전(rotate)하고 git filter-repo/BFG로 이력에서
  제거해야 한다.
- TRAPI scope: api://trapi/.default (endpoint: trapi.research.microsoft.com/gcr/shared)

## Compute (0727 기준 — 두 VC 모두 제약이 있고, 그 제약이 실험 설계를 결정한다)

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
- **현행 런처**(그 외 루트의 `h100std_rq3_*`, `h100std_sft_*`, `a100g1_*`, `a100g2_*`는 은퇴):
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
  (init Qwen3-8B-Base, 3ep lr 1e-5) → **SFT2** data/b2p2_rvseg_sft2.parquet →
  **models/b2p2_rvseg_sft** = RL init for BOTH b2p and b3p (2ep lr 2e-6 light top-up)
  - ⚠️ SFT2 데이터는 rv_redirect_verify_functional(1,763행)의 378행 부분집합.
    E-093에서 배제 근인 확정: `think-closed` 조건이 위장된 시나리오 필터라
    redirect가 554→67(1/8.3)로 기아. 유효 학습량 ≈T1의 1/35. **재구축 대기**.

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

## Current Results (rq3 매치드 래더 — PRELIMINARY, 단일 시드·진행 중·미확정)
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
