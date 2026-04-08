# Phase 2: 3-Way Comparison Report

## Setup
- **Base SFT**: Qwen3-8B + math-only SFT (2371 samples, no meta)
- **Meta SFT**: Qwen3-8B + meta-CoT SFT (7371 samples, with <|meta|>)
- **GRPO E3**: Meta SFT + GRPO 200 step (correctness + format + meta + doubt)
- **Eval**: GSM8K 30, MATH-500 30, AIME2024 30

## Results

| Benchmark | Base SFT | Meta SFT | GRPO E3 |
|-----------|---------|---------|---------|
| GSM8K     | 90.0%   | 90.0%   | 86.7%   |
| MATH-500  | **70.0%** | 63.3% | 66.7%   |
| AIME2024  | 6.7%    | 10.0%   | 0.0%    |
| OVERALL   | **55.6%** | 54.4% | 51.1%  |

| Metric | Base SFT | Meta SFT | GRPO E3 |
|--------|---------|---------|---------|
| ECE (MATH) | N/A | 0.368 | **0.328** |
| Meta blocks (GSM8K) | 0 | 3.8 | **4.9** |
| Meta blocks (AIME) | 0 | 2.6 | **3.6** |

## Key Findings

### 1. Meta-CoT SFT hurts MATH accuracy (-6.7%p vs Base SFT)
- Base SFT MATH 70% > Meta SFT MATH 63.3%
- Meta tokens consume completion tokens → less space for solution
- max_completion_length=1024 is insufficient for meta + solution

### 2. GRPO E3 partially recovers MATH (66.7% vs Meta SFT 63.3%)
- E3 MATH > Meta SFT MATH (+3.4%p)
- But still below Base SFT MATH (70%)

### 3. GRPO E3 improves MATH calibration (ECE 0.368 → 0.328)
- Group doubt reward working for MATH difficulty
- But not for AIME (ECE worsened)

### 4. Meta blocks increase with RL (as intended)
- GSM8K: 3.8 → 4.9 (+29%)
- AIME: 2.6 → 3.6 (+38%)

### 5. AIME catastrophic: E3 = 0%
- RL on medium-difficulty data doesn't transfer to hard problems
- AIME requires fundamentally different skills

## Root Cause Analysis
- **Meta tokens eat into solution space** (1024 tokens shared)
- **Confidence always ~0.99** → calibration reward can't differentiate
- **AIME is out-of-distribution** for our training data (pass_rate 25-75%)

## Next Steps
1. Increase max_completion_length to 2048 (meta + solution space)
2. E5 with stepwise trajectory (confidence: low → high)
3. Focus on MATH (where calibration showed improvement)
4. Consider separate AIME experiment with harder data
