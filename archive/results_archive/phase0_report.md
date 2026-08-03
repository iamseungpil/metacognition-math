# Phase 0: Baseline Evaluation Report

## Setup
- **Models**: Meta SFT (Qwen3-8B, full FT) vs GRPO E1 (200 step, correctness only)
- **Benchmark**: GSM8K 50 problems
- **Eval**: HF generate, temperature=0.7, 1 sample per problem

## Results

| Model | GSM8K Acc | ECE | Meta Blocks | Conf Rate |
|-------|----------|-----|-------------|-----------|
| Meta SFT | **96.0%** | **0.055** | 3.6 | 100% |
| GRPO E1 | 90.0% | 0.110 | 4.0 | 100% |

## Key Findings

### 1. GRPO E1 degraded accuracy (-6%p)
- Correctness-only reward without format_reward caused the model to lose `\boxed{}` usage
- Math-verify couldn't extract answers → correct solutions got -1 reward
- Model learned wrong signal: "don't use boxed format" instead of "solve correctly"

### 2. ECE doubled (0.055 → 0.110)
- No calibration reward → RL disrupted the SFT-learned confidence patterns
- Confidence became less aligned with actual accuracy

### 3. Meta reasoning survived RL (3.6 → 4.0 blocks)
- **Positive**: RL did NOT kill metacognitive self-Q&A
- Model continued to generate pre/mid/post meta blocks
- This means meta quality can be improved with proper rewards

## Conclusions
- E1 (correctness only) is harmful without format_reward
- Calibration requires explicit reward (group doubt / Rewarding Doubt)
- Meta reasoning is robust to RL — good foundation for Phase 1

## Next Steps (Phase 1)
- Add format_reward (\boxed{} bonus) to prevent answer format degradation
- Add group_doubt reward for calibration improvement
- Focus on hard problems (pass_rate < 25%) where meta is most needed
- Increase num_gen=8 for better group-based signals
