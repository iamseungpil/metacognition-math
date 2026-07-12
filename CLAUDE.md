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

## Compute
- Cluster: msrresrchbasicvc (H100 × 4, **Standard** tier — 선점(preemption) 잦음,
  ckpt 릴레이/resume 배선 필수)
- Image: mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x
- Conda env: /scratch/conda_envs/simplerl (conda-pack)
- AMLT project: skilldiscovery2
- YAML: h100std_rq3_b0/b2/b3.yaml (RL), h100std_sft_b0_gold.yaml /
  h100std_sft_b23_unmasked.yaml (SFT)
- (pre-rq3 세대: msrresrchvc Premium A100×4, metacognition_premium.yaml,
  env grpo — 아카이브 세대 기록용)

## Data (HuggingFace: datasets/iamseungpil/metacot)
SFT inputs (current = rq3 matched ladder):
- data/b0_gold_sft.parquet → models/b0_gold_sft (B0 init) — 공개 HF gold,
  gsm8k 637 + MATH 653 = 1,290행 (RV 문제 부분집합, 정답 math_verify 검증)
- data/b23_rv_unmasked_sft.parquet → models/b23_rv_unmasked_sft (B2/B3 init) —
  RV redirect-verify 1,763행, wrong_prefix 필드 비움(whole-response 학습;
  이 unmask fix로 base meta emission 38% → 92%)

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
