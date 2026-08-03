# Meta-CoT: Metacognitive Chain-of-Thought for Math Reasoning

**Author**: Seungpil Lee  **Date**: 2026-03-28  **Project**: metacognition-math

## Executive Summary

| Model | GSM8K | MATH-500 | AIME2024 | Overall | ECE (MATH) | Meta Blocks |
|-------|-------|---------|----------|---------|------------|-------------|
| Base SFT | 90.0% | **70.0%** | 6.7% | **55.6%** | N/A | 0 |
| Meta SFT | 90.0% | 63.3% | **10.0%** | 54.4% | 0.368 | 3.6 |
| GRPO E1 (correct only) | 90.0%* | — | — | — | — | — |
| GRPO E3 (format+meta+doubt) | 86.7% | 66.7% | 0.0% | 51.1% | **0.328** | **4.9** |
| GRPO E5 (stepwise trajectory) | 86.7% | 63.3% | 3.3% | 51.1% | 0.350 | 4.8 |
| GRPO E6 (probe calibration) | *in progress* | | | | | |

Meta-CoT SFT successfully teaches self-reflective reasoning (3.6 meta blocks/problem), and GRPO increases meta usage (+46%). However, accuracy degrades 2-6%p due to meta tokens consuming completion space. Calibration improves on MATH (ECE 0.368→0.328) but not on AIME. Confidence diversity remains the key bottleneck (71% of predictions have conf>0.95).

## 1. Implementation

### 1.1 Overview

Meta-CoT is a training framework where models externalize metacognitive reasoning via `<|meta|>` special tokens. The model performs self-Q&A before, during, and after solving math problems:

```
<|meta|>
Q: Can I solve this problem?
A: This is a modular arithmetic problem. probability 0.83.
Q: What should I watch out for?
A: Check all residues systematically.
<|/meta|>
[solution steps]
<|meta|>
Q: Is my approach correct?
A: Verified. confidence 0.95.
<|/meta|>
\boxed{answer}
```

### 1.2 Training Pipeline

```
Stage 1: Data Generation
  GPT-5.4 → 7,371 Meta-CoT chains (MATH + NuminaMath)

Stage 2: SFT
  Qwen3-8B + Meta-CoT data → Meta SFT model
  Qwen3-8B + math-only data → Base SFT model (control)

Stage 3: GRPO with modular rewards
  Reward functions (independently normalized via GDPO):
    R1: correctness_reward (+1/-1, math-verify sympy)
    R2: format_reward (\boxed{} bonus)
    R3: meta_quality_reward (meta block presence/quality)
    R4: calibration_reward (group-based Brier + Rewarding Doubt)
    R5: stepwise_trajectory_reward (confidence: low→high)
    R6: probe_calibration_reward (binary correctness Brier)
```

### 1.3 Infrastructure

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-8B (8.2B params) |
| Training | TRL GRPOTrainer + DeepSpeed ZeRO-3 |
| Hardware | 4× A100 80GB |
| Framework | torch 2.6.0, trl 0.19.1, transformers 4.52.3 |
| Answer verification | math-verify 0.9.0 (sympy-based) |
| Speed | ~42s/step (HF generate, no vLLM) |
| Loss type | dr_grpo (no length bias) |

## 2. Experimental Results

### 2.1 E1 (Correctness Only): RL degrades accuracy without format reward

GRPO E1 used only correctness_reward (+1/-1). Result: GSM8K accuracy dropped from 96% (previous eval) to 90%, and reward trend was flat throughout 200 steps.

**Root cause**: Model often omits `\boxed{}` format → math-verify cannot extract answer → correct solutions receive -1 reward → model learns wrong signal.

**Key finding**: format_reward is essential for math GRPO. Without it, the model loses answer formatting.

### 2.2 E3 (Format + Meta + Doubt): Meta increases but accuracy still drops

| Metric | Meta SFT | E3 | Change |
|--------|---------|-----|--------|
| MATH accuracy | 63.3% | 66.7% | +3.4%p |
| MATH ECE | 0.368 | **0.328** | **-10.9%** |
| Meta blocks (GSM8K) | 3.8 | **4.9** | **+29%** |
| Meta blocks (AIME) | 2.6 | **3.6** | **+38%** |

**Positive**: Meta blocks increased significantly (+29-38%), and MATH ECE improved (0.368→0.328). The model does more self-reflection after GRPO.

**Negative**: Overall accuracy still below Base SFT (51.1% vs 55.6%). AIME dropped to 0%.

**Training dynamics**: Reward improved from 0.31 to 0.76 (+145%) over 200 steps. format_reward reached 1.0 (100% boxed compliance).

### 2.3 E5 (Stepwise Trajectory): Overconfidence penalty for AIME

E5 added stepwise_trajectory_reward which penalizes starting with high confidence and rewards confidence that increases monotonically during reasoning.

| Metric | E3 | E5 | Better |
|--------|-----|-----|--------|
| AIME ECE | 0.901 | **0.826** | E5 |
| MATH ECE | **0.328** | 0.350 | E3 |
| Meta AIME | **3.6** | 3.1 | E3 |

Stepwise reward improved AIME calibration (ECE 0.901→0.826) — the model is slightly less overconfident on hard problems. However, MATH calibration was better with E3's group doubt.

### 2.4 Qualitative Analysis: Why meta hurts accuracy

**Case 1: Base correct, Meta wrong (7/90 problems)**
- Meta SFT completions include 3-4 meta blocks consuming ~250 tokens
- With max_completion_length=1024, only ~774 tokens remain for solution
- Complex problems require all 1024 tokens → meta-augmented solutions get truncated

**Case 2: Meta correct, Base wrong (6/90 problems)**
- Meta successfully identifies problem type ("divisibility-and-digit-pattern problem")
- Meta identifies watch-out points ("check all residues systematically")
- These hints guide the model to correct solution strategies

**Confidence analysis**:
- conf > 0.95: 71% of predictions (all models)
- conf > 0.9 bucket: 61-64% actually correct (overconfident)
- conf < 0.7 bucket: 0% correct (calibration partially works for low confidence)
- conf 0.7-0.9: 0% correct (problematic middle zone)

### 2.5 Root Cause: Confidence diversity bottleneck

The SFT training data contains confidence values predominantly in 0.8-0.99 range. The model learned to output "probability 0.99" as a default pattern regardless of actual difficulty. GRPO rewards (Brier score, Rewarding Doubt) cannot break this pattern because:

1. Nearly all completions have similar confidence → similar calibration reward → no advantage difference
2. Binary correctness provides coarse signal (only 0 or 1, not continuous p̂)
3. Group accuracy with num_gen=4 is noisy (0%, 25%, 50%, 75%, 100%)

## 3. Proposed Improvements

### 3.1 Hidden-State Probe Reward (E6, in progress)

**Problem**: Text-based confidence is always ~0.99, providing no calibration gradient.

**Solution**: Use SimpleCorrectnessProbe (AUROC 0.953) on hidden states to get continuous p̂. R_probe = -(stated_conf - p̂)². This forces stated confidence to match the model's internal belief.

**Expected Effect**: Confidence diversity should increase because probe p̂ varies per problem, creating different targets for different problems.

### 3.2 Longer Completion Length

**Problem**: Meta blocks consume 250+ tokens, leaving insufficient space for complex solutions.

**Solution**: Increase max_completion_length from 1024 to 2048 (already applied in E5+).

### 3.3 Self-Curation Learning (Future)

**Problem**: Model fails on AIME because training data is mid-difficulty (pass_rate 25-75%).

**Solution**: Use meta's self-diagnosis to identify weak areas → retrieve similar problems from rollout DB → targeted training.

## 4. Limitations

- **Small eval set**: 30 problems per benchmark may have high variance
- **Completion truncation**: eval saved only 500 chars of completion, limiting qualitative analysis (fixed)
- **Single seed**: No multiple runs for statistical significance
- **AIME is out-of-distribution**: Training data doesn't include competition-level problems

## 5. Conclusion

Meta-CoT successfully teaches models to perform self-reflective reasoning (+29-38% more meta blocks after GRPO). The framework partially improves calibration (MATH ECE -10.9%). However, three key challenges remain: (1) confidence diversity is insufficient (71% > 0.95), (2) meta tokens reduce solution space causing accuracy drops, and (3) hard problems (AIME) require fundamentally different training data.

## 6. Next Experiments

### E6: Probe Calibration Reward (in progress)
- **Tests**: Does binary Brier score improve confidence diversity?
- **Config**: probe_calibration_reward weight=1.5, GDPO enabled
- **Expected**: Confidence distribution should spread below 0.9 for wrong answers

### E7: Real Probe Reward (hidden states)
- **Tests**: Does probe p̂ provide better calibration signal than binary?
- **Config**: Forward pass through training model, SimpleProbe on hidden states
- **Expected**: ECE improvement on all benchmarks, not just MATH

### E8: Self-Curation Learning
- **Tests**: Can meta-guided curriculum improve AIME performance?
- **Config**: AIME rollouts → meta diagnosis → RAG retrieval → targeted SFT
- **Expected**: AIME accuracy improvement from meta-identified weak areas
