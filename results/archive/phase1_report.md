# Phase 1: E3 (Format + Meta + Doubt) Evaluation Report

## Setup
- **Training**: GRPO E3, 200 step, Full FT, ZeRO-3, ~40s/step
- **Rewards**: correctness (+1/-1), format (\boxed bonus), meta_quality, group_doubt (Brier + log)
- **GDPO**: per-reward normalization enabled
- **Eval**: GSM8K 30, MATH-500 30, AIME2024 30

## Results

| Benchmark | Meta SFT | GRPO E3 | Change |
|-----------|---------|---------|--------|
| GSM8K     | 90.0%   | 86.7%   | -3.3%  |
| MATH-500  | 66.7%   | 63.3%   | -3.4%  |
| AIME2024  | 6.7%    | 6.7%    | ±0%    |
| OVERALL   | 54.4%   | 52.2%   | -2.2%  |

| Metric | Meta SFT | GRPO E3 | Change |
|--------|---------|---------|--------|
| ECE (GSM8K) | 0.108 | 0.139 | worse |
| ECE (MATH)  | 0.336 | 0.379 | worse |
| ECE (AIME)  | 0.849 | 0.853 | same  |
| Meta blocks (GSM8K) | 3.6 | **5.2** | **+44%** |
| Meta blocks (MATH)  | 3.6 | **4.4** | **+22%** |
| Meta blocks (AIME)  | 2.6 | **3.8** | **+46%** |

## Key Findings

### 1. Meta reasoning increased significantly
- GRPO E3 produces 22-46% more meta blocks than SFT
- Largest increase on hardest benchmark (AIME: +46%)
- meta_quality_reward successfully incentivizes self-reflection

### 2. Accuracy slightly degraded (-2-3%)
- Not catastrophic, but E3 didn't improve accuracy
- Possible cause: 200 steps insufficient, or reward design needs tuning
- format_reward helped (100% boxed usage during training)

### 3. Calibration NOT improved
- ECE stayed same or slightly worse across all benchmarks
- Group doubt reward didn't translate to better calibration
- Hypothesis: sequence-level reward too coarse for per-block calibration

### 4. Critical observation: AIME meta blocks increased but ECE didn't improve
- Model does MORE metacognition on hard problems (good!)
- But the metacognition isn't ACCURATE (bad!)
- → Need stepwise per-block calibration reward

## Conclusions
- Meta quantity improved ✅ (meta_quality_reward works)
- Meta quality (calibration) NOT improved ❌ (group doubt insufficient)
- Next: stepwise meta training (Phase 3a) for per-block calibration

## Training Dynamics
- Reward trend: 0.31 → 0.76 (+145% over 200 steps)
- format_reward: 0→1.0 (100% boxed compliance)
- calibration_reward: -0.72→-0.41 (improved during training but not in eval)
