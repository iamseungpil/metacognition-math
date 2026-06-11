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

## Key Tokens (DO NOT COMMIT TO GIT)
- GitHub: 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 (repo: iamseungpil/metacognition-math)
- HuggingFace: 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 (dataset: iamseungpil/metacot)
- WandB: 2f4e627868f1f9dad10bcb1a14fbf96817e6baa9
- TRAPI scope: api://trapi/.default (endpoint: trapi.research.microsoft.com/gcr/shared)

## Compute
- Cluster: msrresrchvc (Premium A100 80GB × 4)
- AMLT project: skilldiscovery2
- YAML: metacognition_premium.yaml (max_run 14 days = 1209600s)
- Conda env: grpo (torch 2.6, trl 0.19.1, transformers 4.52.3)

## Data (HuggingFace: datasets/iamseungpil/metacot)
SFT inputs (current = v8 series):
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

## Current Results
- AIME overconfidence: 97% → 14% (calibration success)
- AIME ECE: 0.870 → 0.610
- BUT: Meta-CoT accuracy < Base SFT (MATH 56.7% vs 76.7%)
- Root cause: meta overhead 56% of tokens, 31% truncation

## Autoresearch Loop (until Meta-CoT > Base SFT)
1. Critic: analyze why Base > Meta, classify error types
2. Planner: hypothesize fix (SFT format, RL reward, token length)
3. Implementer: code + run experiment
4. Eval: 1,030 problems (GSM8K 500 + MATH 500 + AIME 30), max_tokens=4096
5. Repeat until Meta-CoT accuracy ≥ Base SFT

## Code Structure
- src/training/sft.py — SFT training
- src/training/grpo_v2.py — GRPO with modular rewards (E1-E7)
- src/training/rewards.py — 7 reward functions
- src/eval/eval_hf.py — HF generate eval (max_tokens=4096)
- src/curriculum/rag.py — Meta-guided curriculum learning (FAISS + sentence-transformers)
- src/metacot/prompt_v2.py — V2 prompt (diverse confidence, error→fix)
