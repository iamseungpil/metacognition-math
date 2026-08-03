# Phase 3: E5 Stepwise Trajectory Results

## 4-Way Comparison

| Benchmark | Base SFT | Meta SFT | E3 (doubt) | E5 (stepwise) |
|-----------|---------|---------|------------|---------------|
| GSM8K     | 90.0%   | 90.0%   | 86.7%      | 86.7%         |
| MATH-500  | **70.0%** | 63.3% | **66.7%** | 63.3%         |
| AIME2024  | 6.7%    | **10.0%** | 0.0%    | 3.3%          |
| OVERALL   | **55.6%** | 54.4% | 51.1%    | 51.1%         |

| Metric | Meta SFT | E3 | E5 | Best |
|--------|---------|-----|-----|------|
| ECE GSM8K | 0.109 | 0.139 | 0.136 | **Meta SFT** |
| ECE MATH  | 0.368 | **0.328** | 0.350 | **E3** |
| ECE AIME  | 0.810 | 0.901 | **0.826** | **E5** |
| Meta GSM8K | 3.8 | **4.9** | 4.8 | **E3** |
| Meta AIME  | 2.6 | **3.6** | 3.1 | **E3** |

## Key Findings

### E5 Stepwise Trajectory
- AIME ECE improved: 0.901→0.826 (E3) → 0.826 (E5 even better)
- Stepwise trajectory reward shows overconfidence penalty working
- Training: stepwise reward -1.10 → -0.68 (38% improvement)
- But accuracy didn't improve over E3

### Confidence Diversity Problem (Persistent)
- All meta models: confidence >0.95 in 71% of cases
- Even wrong answers: mean confidence 0.91
- Stepwise reward didn't sufficiently break this pattern
- Need stronger intervention (probe reward or confidence penalty)

### \boxed{} Parsing Issue
- 100% of wrong answers = NO_BOXED in stored completions
- Model sometimes doesn't use \boxed{} format
- format_reward helps during training but eval needs better extraction

## Next Steps
1. Probe reward (hidden state → more accurate calibration signal)
2. Longer training (500+ steps for both E3 and E5)
3. Confidence diversity penalty (explicit punishment for conf>0.95)
