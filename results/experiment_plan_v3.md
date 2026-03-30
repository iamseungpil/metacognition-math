# Meta-CoT Experiment Plan v3: Beat Base SFT

**Date**: 2026-03-30
**Author**: Seungpil Lee
**Project**: metacognition-math (Qwen3-8B)

## Core Objective

Meta-CoT must outperform Base SFT on math accuracy.
Calibration is secondary value — accuracy is the primary metric.
The autoresearch loop continues until Meta-CoT >= Base SFT on overall accuracy.

## Current Results (baseline)

| Model | GSM8K | MATH-500 | AIME2024 | Overall | ECE (MATH) | Meta Blocks |
|-------|-------|----------|----------|---------|------------|-------------|
| Base SFT | 90.0% | **70.0%** | 6.7% | **55.6%** | N/A | 0 |
| Meta SFT (V2) | 90.0% | 63.3% | 10.0% | 54.4% | 0.368 | 3.6 |
| GRPO E1 (correct only) | 90.0% | — | — | — | — | — |
| GRPO E3 (format+meta+doubt) | 86.7% | 66.7% | 0.0% | 51.1% | 0.328 | 4.9 |
| GRPO E5 (stepwise trajectory) | 86.7% | 63.3% | 3.3% | 51.1% | 0.350 | 4.8 |
| GRPO E6 (probe calibration) | *in progress* | | | | | |

**Gap**: Meta SFT trails Base SFT by 1.2%p overall (54.4% vs 55.6%).
**Root cause**: Meta tokens consume ~56% of completion space; 31% of completions truncated at 1024 tokens.

## Autoresearch Loop

```
While Meta-CoT < Base SFT:
  1. Critic: analyze why Base > Meta, classify error types
  2. Planner: hypothesize fix (SFT format, RL reward, token length, data source)
  3. Implementer: code + run experiment
  4. Eval: 1,030 problems (GSM8K 500 + MATH 500 + AIME 30), max_tokens=4096
  5. Repeat until Meta-CoT accuracy >= Base SFT
```

## Hypotheses (priority order)

| ID | Hypothesis | Rationale | Expected Gain |
|----|-----------|-----------|---------------|
| H1 | max_tokens=4096 fixes truncation | 31% of Meta completions truncated at 1024; removing this bottleneck directly recovers lost accuracy | +10%p |
| H2 | Difficulty-adaptive meta (skip meta on easy problems) | Easy problems (GSM8K) don't benefit from meta overhead; conditional meta preserves accuracy on easy while helping hard | +3-5%p |
| H3 | Verification-only meta (post-answer check, minimal overhead) | Pre-solve meta planning wastes tokens; post-answer verification catches errors with fewer tokens | +2-4%p |
| H4 | gpt-5.4 data instead of gpt-5.4-mini | Higher-quality teacher chains may produce better reasoning patterns and more accurate meta-assessments | +3-5%p |
| H5 | Longer GRPO training (1000 steps) | Current 200-step GRPO may be under-trained; reward was still improving at termination | +2-3%p |
| H6 | Stepwise reward bonus for correct verification | Reward meta blocks that successfully identify and correct errors, not just meta presence | +1-3%p |

## Phase A: Current Training (IN PROGRESS)

| Task | Status | Notes |
|------|--------|-------|
| V2 Meta SFT | DONE | 4,996 chains, Qwen3-8B |
| Base SFT | DONE | 4,996 chains, meta stripped |
| GRPO E3 (format+meta+doubt, 200 step) | DONE | Best calibration (ECE 0.328) |
| GRPO E5 (stepwise trajectory, 200 step) | DONE | Best AIME ECE (0.826) |
| GRPO E6 (probe calibration) | IN PROGRESS | SimpleProbe AUROC 0.953 |
| V2+E3 GRPO (500 step, mixed data) | NEXT | Extended training with GSM8K+MATH |

## Phase B: Large-scale Eval (max_tokens=4096)

4 GPU parallel evaluation, 1,030 problems total.

| GPU | Model | Benchmark |
|-----|-------|-----------|
| GPU 0 | Base SFT | GSM8K 500 + MATH 500 + AIME 30 |
| GPU 1 | V2 Meta SFT | GSM8K 500 + MATH 500 + AIME 30 |
| GPU 2 | V2+E3 GRPO | GSM8K 500 + MATH 500 + AIME 30 |
| GPU 3 | V2+E7 GRPO | GSM8K 500 + MATH 500 + AIME 30 |

**Key change from v2**: max_tokens raised from 1024 to 4096 (H1).
This alone should recover most of the 31% truncation-induced accuracy loss.

**Metrics collected**:
- Accuracy per benchmark (with bootstrap 95% CI)
- ECE per benchmark
- Selective abstention curve (accuracy vs coverage at conf thresholds)
- Confidence distribution histogram
- Full completion text saved for qualitative analysis

## Phase C: Autoresearch Improvement Loop

Iterate through H1-H6 until Meta-CoT >= Base SFT.

### C1: Truncation fix (H1)
- Already applied: eval with max_tokens=4096
- If Meta SFT accuracy improves to >= Base SFT at 4096 tokens, truncation was the sole cause
- If gap persists, move to C2

### C2: Difficulty-adaptive meta (H2)
- Classify problems by difficulty (GSM8K=easy, MATH=medium, AIME=hard)
- For easy problems: skip meta blocks entirely (or use single-line meta)
- For medium/hard: full meta-CoT with verification
- Implementation: conditional prompt template or difficulty classifier head

### C3: Verification-only meta (H3)
- Remove pre-solve planning meta blocks
- Keep only post-answer verification meta: "Is my answer correct? Let me check..."
- Reduces meta token overhead from ~250 to ~80 tokens

### C4: Teacher upgrade (H4)
- Regenerate 4,996 chains using gpt-5.4 (full model, not mini)
- Higher teacher quality may produce more informative meta-reasoning
- Re-run SFT + best GRPO config

### C5: Extended GRPO (H5)
- Train for 1000 steps instead of 200
- Monitor for reward saturation and accuracy plateau
- Early stop if accuracy degrades

### C6: Verification reward (H6)
- Add reward component: +bonus when meta block identifies an error AND the final answer is correct
- Encourages productive (not decorative) meta-cognition

## Phase D: Curriculum Learning (after Phase C succeeds)

Only proceed after Meta-CoT >= Base SFT is achieved.

1. **Meta-guided weakness diagnosis**: Use meta blocks to identify problem types where the model fails
2. **RAG retrieval**: FAISS + sentence-transformers to find similar problems from NuminaMath/GSM8K
3. **Targeted SFT**: Fine-tune on retrieved weak-area problems
4. **Re-evaluate**: Measure improvement on previously-failed problem types
5. **Iterate**: Repeat until AIME performance improves

## Success Criteria

| Milestone | Condition | Priority |
|-----------|-----------|----------|
| M1: Parity | Meta-CoT overall accuracy >= Base SFT (55.6%) | **PRIMARY** |
| M2: MATH win | Meta-CoT MATH-500 >= Base SFT (70.0%) | HIGH |
| M3: Calibration | Meta-CoT ECE < 0.30 on MATH-500 | MEDIUM |
| M4: Abstention | Meta-CoT conf>=0.7 subset accuracy > Base SFT overall | MEDIUM |
| M5: AIME | Meta-CoT AIME accuracy > 10% | LOW |

## Known Risks

1. **Token overhead is fundamental**: Even at 4096 tokens, meta blocks may slow reasoning
2. **Confidence diversity**: 71% of predictions have conf>0.95; RL may not break this pattern
3. **Small eval variance**: 30 AIME problems = high variance (1 problem = 3.3%p)
4. **AIME OOD**: Training data lacks competition-level problems
5. **Compute budget**: Each GRPO run = ~5 hours on 4×A100; budget limits iteration count
