# Meta-CoT Experiment Plan v4: Autoresearch Loop

**Date**: 2026-03-30
**Author**: Seungpil Lee (with autoresearch reporter agent)
**Project**: metacognition-math (Qwen3-8B)

---

## Executive Summary

Meta-CoT trails Base SFT by **3.3%p overall** (55.6% vs 58.9%) and **20%p on MATH-500** (56.7% vs 76.7%). Calibration is proven (AIME ECE 0.870 -> 0.610, overconfidence 97% -> 14%), but accuracy remains the bottleneck. The current V2+E3 GRPO training is at step ~250/500 with checkpoint-200 saved. Training dynamics show healthy KL (0.13-0.25) and oscillating loss around zero, consistent with the confirmed-normal TRL GRPO behavior at beta~0.001.

**Primary goal**: Close the accuracy gap. Meta-CoT overall accuracy >= Base SFT (58.9%).
**Secondary goal**: Preserve calibration gains (ECE < 0.35 on MATH).

The autoresearch loop will systematically test H1-H6 until parity is achieved.

---

## Current Training Status: GRPO E3 (500 steps)

### Configuration
- **Model**: Qwen3-8B Meta SFT + LoRA (rank 32) -> Full FT GRPO
- **Rewards**: correctness + format + meta_quality + group_doubt (GDPO)
- **Data**: 1,565 problems filtered by pass rate 10-90%
- **Settings**: batch=4/GPU, num_gen=16, LR=5e-6, beta=0.001, temp=1.0
- **Compute**: 4x A100 80GB, DDP, ~2 min/step

### Training Metrics Trajectory

| Step | Loss | KL Divergence | Notes |
|------|------|---------------|-------|
| 10 | -0.013 | 0.000 | Warmup, KL not yet diverging |
| 90 | 0.079 | 0.255 | KL spike, model exploring |
| 150 | -0.002 | 0.131 | KL stabilizing |
| 200 | -0.048 | 0.149 | Checkpoint saved |
| ~250 | — | — | Currently training |

### Training Dynamics Analysis

1. **Loss oscillation around zero**: Confirmed normal for TRL GRPO with beta=0.001 and num_iterations=1. Not a training failure.
2. **KL trajectory**: Rose to 0.255 at step 90, then settled to ~0.14-0.15 range. This indicates the model initially explored broadly, then converged on a policy close to the reference. Healthy behavior.
3. **No signs of collapse**: KL staying bounded (<0.3) means no mode collapse. The model is learning within a stable regime.
4. **Concern**: KL may be too low, suggesting the model is not deviating enough from the SFT policy to fix accuracy problems. If eval shows no improvement, longer training (H5) or stronger reward signal may be needed.

### Checkpoints Available
| Checkpoint | Status | Notes |
|------------|--------|-------|
| `checkpoints/qwen3_meta_sft` | Ready | Base Meta SFT model |
| `checkpoints/base_sft` | Ready | Control (no meta) |
| `checkpoints/simple_probe_qwen3/best_probe.pt` | Ready | AUROC 0.953 |
| GRPO checkpoint-200 | Ready for eval | Mid-training snapshot |
| GRPO checkpoint-500 | Pending (~2.5 hrs) | Final checkpoint |

---

## Phase B: Large-Scale Eval Plan

### Status: BLOCKED on GRPO completion

Checkpoint-200 is ready for preliminary eval while awaiting the full 500-step model. The eval plan uses 4 GPUs in parallel on 1,030 problems with max_tokens=4096 (H1 applied).

### Eval Matrix

| GPU | Model | Benchmarks | max_tokens |
|-----|-------|------------|------------|
| GPU 0 | Base SFT | GSM8K-500 + MATH-500 + AIME-30 | 4096 |
| GPU 1 | V2 Meta SFT | GSM8K-500 + MATH-500 + AIME-30 | 4096 |
| GPU 2 | GRPO E3 ckpt-200 | GSM8K-500 + MATH-500 + AIME-30 | 4096 |
| GPU 3 | GRPO E3 ckpt-500 | GSM8K-500 + MATH-500 + AIME-30 | 4096 |

### Eval Command Template
```bash
python src/eval/eval_hf.py \
  --model_path <MODEL_PATH> \
  --benchmarks gsm8k math500 aime2024 \
  --max_problems 500 \  # 500 for GSM8K/MATH, 30 for AIME
  --max_tokens 4096
```

### Metrics to Collect
- Accuracy per benchmark (with bootstrap 95% CI)
- ECE per benchmark
- Truncation rate (% of completions hitting max_tokens)
- Meta block count and token overhead
- Selective abstention curve (accuracy vs coverage at conf thresholds)
- Confidence distribution histogram
- Full completion text saved for qualitative analysis

---

## Autoresearch Hypotheses: Priority Ranking

### Decision Framework

The gap is **3.3%p overall, 20%p on MATH**. The hypotheses are ranked by expected impact and implementation cost.

### Tier 1: Quick wins (test first)

| Rank | ID | Hypothesis | Expected Gain | Cost | Time |
|------|-----|-----------|---------------|------|------|
| 1 | H1 | max_tokens=4096 (already in Phase B eval) | +5-10%p | Zero (config change) | 0 hrs |
| 2 | H5 | Longer GRPO (500 steps, already running) | +2-5%p | Already committed | 0 hrs |

### Tier 2: Medium effort (test if Tier 1 insufficient)

| Rank | ID | Hypothesis | Expected Gain | Cost | Time |
|------|-----|-----------|---------------|------|------|
| 3 | H2 | Difficulty-adaptive meta | +3-5%p | Prompt template change | 2 hrs |
| 4 | H3 | Verification-only meta | +2-4%p | New SFT data format | 4 hrs |

### Tier 3: High effort (last resort)

| Rank | ID | Hypothesis | Expected Gain | Cost | Time |
|------|-----|-----------|---------------|------|------|
| 5 | H4 | gpt-5.4 data (not mini) | +3-5%p | TRAPI calls + SFT retrain | 8+ hrs |
| 6 | H6 | Verification reward bonus | +1-3%p | New reward function | 4 hrs |

---

## Decision Tree: Post-Eval Actions

```
Phase B eval results arrive
│
├── IF Meta-CoT (4096 tokens) overall >= 58.9% (Base SFT)
│   ├── SUCCESS: M1 achieved
│   ├── Report results, check M2-M5 milestones
│   └── Proceed to Phase D (curriculum learning)
│
├── IF Meta-CoT (4096) overall = 55-58.8% (close, gap < 4%p)
│   ├── Truncation was partial cause
│   ├── Try H2: difficulty-adaptive meta
│   │   ├── Skip meta on GSM8K (easy), full meta on MATH/AIME
│   │   └── Expected: GSM8K accuracy recovers to ~93%, MATH unchanged
│   ├── If still short: try H3 (verification-only)
│   └── If still short: try H4 (better teacher data)
│
├── IF Meta-CoT (4096) overall = 50-55% (large gap, >= 4%p)
│   ├── Truncation was NOT the main cause
│   ├── Root cause is likely meta token overhead or data quality
│   ├── Try H3 first: verification-only meta (minimal overhead)
│   ├── Then H4: regenerate data with gpt-5.4 (quality boost)
│   └── Then H2: difficulty-adaptive meta
│
└── IF Meta-CoT (4096) overall < 50% (regression)
    ├── GRPO training damaged the model
    ├── Fall back to V2 Meta SFT (no GRPO) with 4096 tokens
    ├── If SFT alone < 50% at 4096: fundamental data quality issue
    └── Must try H4 (better teacher) as first priority
```

---

## Training Dynamics Interpretation Guide

### What to watch for in remaining 250 steps (step 250-500)

| Signal | Healthy | Concerning | Action if Concerning |
|--------|---------|------------|---------------------|
| KL | 0.1-0.5, slowly rising | KL > 1.0 or KL < 0.05 | Reduce LR / increase beta |
| Loss | Oscillates near 0 | Monotonically negative | Check for reward hacking |
| Reward std | > 0.1 | < 0.05 (collapsed) | Reward signal too weak |
| Grad norm | < 10 | > 100 (spikes) | Reduce LR |

### Expected checkpoint-500 behavior
- KL should be 0.15-0.40 (moderate divergence from SFT)
- If KL < 0.10: model barely changed, GRPO too conservative -> need H5 (more steps) or higher LR
- If KL > 0.50: model significantly different, check if accuracy improved or collapsed

---

## Success Milestones (unchanged from v3)

| Milestone | Condition | Priority | Status |
|-----------|-----------|----------|--------|
| M1: Parity | Meta-CoT overall >= Base SFT (58.9%) | **PRIMARY** | Pending |
| M2: MATH win | Meta-CoT MATH-500 >= Base SFT (76.7%) | HIGH | Pending |
| M3: Calibration | Meta-CoT ECE < 0.30 on MATH-500 | MEDIUM | Near (0.328) |
| M4: Abstention | conf>=0.7 subset accuracy > Base SFT overall | MEDIUM | Achieved (60.5% > 58.9%) |
| M5: AIME | Meta-CoT AIME accuracy > 10% | LOW | Pending |

---

## Timeline Estimates

| Phase | Task | ETA | Dependencies |
|-------|------|-----|-------------|
| A (current) | GRPO E3 completes (500 steps) | +2.5 hrs | Running |
| B.1 | Eval ckpt-200 (preliminary) | +3 hrs | Can start now |
| B.2 | Eval ckpt-500 (full) | +5.5 hrs | Phase A done |
| C.1 | H1 result analysis (from B eval) | +6 hrs | Phase B done |
| C.2 | H2 implementation (if needed) | +8 hrs | After C.1 decision |
| C.3 | H3 implementation (if needed) | +12 hrs | After C.2 decision |
| C.4 | H4 data regeneration (if needed) | +20 hrs | Last resort |

**Estimated total time to M1**: 6-20 hours depending on how many hypotheses are needed.

---

## Known Risks (updated)

1. **Token overhead is fundamental**: Even at 4096 tokens, meta blocks consume computation budget that could be used for reasoning
2. **KL too conservative**: Current KL ~0.15 may mean GRPO is not changing the model enough to fix accuracy
3. **Small eval variance**: 30 AIME problems = high variance (1 problem = 3.3%p); 500 MATH problems will be more reliable
4. **GRPO reward hacking**: group_doubt reward may incentivize low confidence rather than accurate confidence
5. **Compute budget**: Each GRPO run = ~5 hours on 4xA100; each full eval = ~2 hours on 4 GPUs
6. **gpt-5.4-mini data ceiling**: V2 data from weaker teacher; may need gpt-5.4 full to match Base SFT accuracy

---

## Appendix: Result Tables Reference

### Phase 4 Report (30 problems/benchmark, max_tokens=1024)

| Model | GSM8K | MATH-500 | AIME | Overall | ECE(AIME) |
|-------|-------|---------|------|---------|-----------|
| Base SFT | 93.3% | **76.7%** | 6.7% | **58.9%** | N/A |
| V1 E5 (best accuracy) | 93.3% | 63.3% | 10.0% | 55.6% | 0.819 |
| V2 SFT | 90.0% | 56.7% | 6.7% | 51.1% | 0.712 |
| V2 E7 (best calibration) | 83.3% | 50.0% | 3.3% | 45.6% | **0.610** |

### GRPO E3 Results (30 problems/benchmark, max_tokens=1024)

| Model | GSM8K | MATH-500 | AIME | Overall | ECE(MATH) |
|-------|-------|---------|------|---------|-----------|
| GRPO E3 (200 step) | 86.7% | 66.7% | 0.0% | 51.1% | 0.328 |
