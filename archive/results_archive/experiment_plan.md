# Prioritized Experiment Plan (2026-03-29)

## Situation Summary

| Model | MATH Acc | AIME Acc | AIME ECE | AIME Overconf | Meta Overhead |
|-------|----------|----------|----------|---------------|---------------|
| Base SFT | **76.7%** | 6.7% | N/A | N/A | 0% |
| V1 Meta SFT | 60.0% | 3.3% | 0.870 | 97% | ~56% |
| V2 SFT | 56.7% | 6.7% | 0.712 | 36% | ~40% |
| V2 E7 (stepwise) | 50.0% | 3.3% | **0.610** | **14%** | ~40% |

**Core tension**: Calibration improves monotonically (ECE 0.870 -> 0.610), but accuracy drops monotonically (76.7% -> 50.0%). The paper story requires demonstrating that calibration gains are not merely a side-effect of the model becoming worse at math.

**Root causes identified**:
1. Meta token overhead consumes 40-60% of completion budget
2. 31% of MATH errors are truncation at max_length=2048
3. V2 data from gpt-5.4-mini is weaker than V1 data from gpt-5.4
4. Confidence diversity bottleneck (71% of predictions have conf > 0.95 in V1)

---

## Phase A: Complete and Evaluate V2+E3 (Days 1-2)

### A1. V2+E3 Training Completion
- **Status**: 500 steps, in progress
- **Action**: Let it run to completion (500 steps is likely sufficient given E3 reached good reward in 200 steps)

### A2. V2+E3 Evaluation
**Eval on all 3 benchmarks (GSM8K, MATH-500, AIME) with max_length=4096**

Key metrics to collect:
1. Accuracy (GSM8K, MATH-500, AIME)
2. ECE (all 3 benchmarks)
3. Overconfidence rate on wrong AIME answers
4. Confidence distribution histogram (bins: <0.3, 0.3-0.5, 0.5-0.7, 0.7-0.9, >0.9)
5. Truncation rate (% of completions hitting max_length)
6. Meta block count and average token length
7. conf < 0.5 on wrong answers (target: > 30%)

### A3. Decision Gate after A2

| Outcome | Signal | Next Step |
|---------|--------|-----------|
| **Win** | MATH acc >= 60% AND AIME ECE <= 0.65 | Go to Phase B (scale up) |
| **Calibration win, accuracy loss** | AIME ECE <= 0.65 BUT MATH acc < 55% | Go to Phase C (fix accuracy) |
| **No improvement** | AIME ECE > 0.70 OR MATH acc < 50% | Go to Phase D (fallback) |

---

## Phase B: Scale Up (if V2+E3 wins) -- Days 3-5

### B1. Fix Truncation with max_length=4096 Eval
If A2 was done at 2048, re-evaluate V2+E3 at max_length=4096. This alone should recover ~31% of MATH errors (based on truncation analysis). This is the single highest-ROI intervention.

### B2. Higher-Quality V2+ Data with gpt-5.4
Generate 10K+ Meta-CoT chains using gpt-5.4 (not mini) via TRAPI:
- Same V2 prompt template (boxed format, diverse confidence, short meta blocks)
- Include harder problems (AIME-level from NuminaMath competition subset)
- Target: conf distribution with mean ~0.65, std > 0.2

**Decision**: If V2+ SFT with gpt-5.4 data recovers MATH accuracy to >= 70% while maintaining ECE gains, this becomes the new baseline for all subsequent GRPO.

### B3. V2+ SFT + E3 GRPO (Best Configuration)
Train on the higher-quality V2+ data, then apply E3 reward (format + meta + group doubt).
- Expected: best of both worlds -- accuracy from better data, calibration from GRPO

### B4. Larger Eval (100 problems per benchmark)
Current n=30 is high variance. Before any paper claims, expand to n=100:
- GSM8K: random 100 from test set
- MATH-500: random 100 (stratified by difficulty level)
- AIME: use all available 2024 problems + 2023 for more signal

---

## Phase C: Fix Accuracy Gap (if calibration wins but accuracy drops) -- Days 3-7

### C1. max_length=4096 (Quick Fix)
First, just increase max_length. If 31% of MATH errors are truncation, this should recover ~10%p on MATH without any retraining. Run eval-only.

**Decision**: If MATH acc recovers to >= 65% at 4096, stop here and proceed to B2.

### C2. Post-Verification-Only Meta (V3 Approach)
This directly addresses Question 6: "verification-only meta."

Design: The model solves normally, then appends a single meta block at the end:
```
[full solution using all available tokens]
\boxed{answer}
<|meta|>
Q: Am I confident in this answer?
A: [confidence statement with score]
<|/meta|>
```

**Advantages**:
- Zero token overhead for the solution itself
- Meta block is ~30 tokens (not 250)
- The model "verifies" rather than "plans" -- simpler task

**Implementation**:
1. Generate V3 data: take V2 data, move all meta blocks to after \boxed{}
2. SFT on V3 data
3. GRPO with E3 reward (only the final confidence matters)

**Decision**: If V3 achieves MATH acc >= 72% (close to base) AND AIME ECE <= 0.75, this is the winning approach. It sacrifices mid-solution metacognition but preserves accuracy.

### C3. Hybrid: Full Meta for Easy, Short Meta for Hard
Use meta-predicted difficulty to decide meta verbosity:
- If initial confidence > 0.8 (easy): full meta-CoT (enriches reasoning)
- If initial confidence < 0.5 (hard): minimal meta, save tokens for solution

This requires a two-pass or adaptive generation strategy. More complex to implement but could be the best of both worlds.

### C4. LoRA Rank Increase
Current LoRA rank=32. If the model struggles to simultaneously maintain math skill and learn calibration, increasing to rank=64 or rank=128 may help. Low cost to try.

---

## Phase D: Fallback (if V2+E3 does not improve) -- Days 3-7

### D1. Diagnose Why E3 Failed on V2
Compare E3 reward curves on V1-SFT vs V2-SFT:
- Is reward variance sufficient? (V2 data may already be well-calibrated, leaving no room for GRPO)
- Is the group-doubt signal noisy? (num_gen=4 gives only 5 possible group accuracies)

### D2. Probe Reward (E6/E7)
Use SimpleCorrectnessProbe (AUROC 0.953) as the calibration reward:
- R_probe = -(stated_conf - probe_p_hat)^2
- This provides a continuous calibration target per problem
- Expected to break the confidence diversity bottleneck

**Implementation priority**: This was already planned as E6/E7. If E3 fails, accelerate this.

### D3. Increase num_gen to 16
More rollouts per problem = smoother group accuracy signal:
- num_gen=4: possible group acc = {0, 0.25, 0.5, 0.75, 1.0}
- num_gen=16: possible group acc = {0, 0.0625, 0.125, ..., 1.0}
- Better gradient for calibration reward

Trade-off: 4x slower per step. May need to reduce batch size.

### D4. Direct Preference Optimization on Confidence
Create preference pairs:
- Preferred: solution with confidence close to actual correctness
- Rejected: solution with overconfident wrong answer
- Use DPO/KTO instead of GRPO for calibration

---

## Phase E: Paper-Ready Experiments -- Days 7-14

### E1. Minimum Viable Paper Result
The paper needs to show:

| Claim | Required Evidence | Current Status |
|-------|------------------|----------------|
| Meta-CoT teaches self-reflection | Meta block count increases after SFT/GRPO | DONE (3.6 -> 4.9 blocks) |
| Calibration improves | ECE reduction on AIME | DONE (0.870 -> 0.610) |
| Overconfidence drops | Overconf rate on wrong AIME | DONE (97% -> 14%) |
| Accuracy is not sacrificed | MATH acc within 5%p of base | **NOT MET** (76.7 -> 50.0) |
| Selective abstention works | "I don't know" on hard problems improves F1 | NOT TESTED |

**Critical gap**: Accuracy parity. Without it, the reviewer will say "the model just got worse and less confident about being worse." The paper cannot survive this critique.

### E2. Selective Abstention Experiment (New, High Priority)
This may be the strongest angle for the paper. Instead of treating confidence as a continuous score, evaluate the model's ability to abstain:

Protocol:
1. Set confidence threshold tau (e.g., 0.5, 0.6, 0.7)
2. Model answers only when confidence >= tau
3. Measure: accuracy on answered questions vs coverage (% answered)
4. Plot: accuracy-coverage curve for each model

**Expected result**: V2+GRPO should dominate -- higher accuracy at every coverage level because it correctly abstains on problems it cannot solve. Base SFT cannot abstain (no confidence). V1 Meta SFT abstains poorly (overconfident on everything).

This reframes the story from "accuracy drops" to "the model makes fewer confident mistakes."

### E3. Cost-Benefit Analysis of Meta Tokens
Show that meta overhead is worth it per-token:
- accuracy_per_token = accuracy / avg_completion_tokens
- calibrated_accuracy = accuracy * (1 - ECE)
- Plot these for all models

### E4. Qualitative Examples for Paper
Curate 3-5 compelling examples:
1. Base wrong + Meta correct (meta self-correction saves the day)
2. Base correct + Meta wrong but low-confidence (meta knows it failed)
3. AIME problem where V2+GRPO says "confidence 0.3" and is indeed wrong
4. GSM8K problem where all models are correct but meta is well-calibrated

### E5. Statistical Significance
- Bootstrap confidence intervals on all metrics (n=1000 resamples from n=100 eval)
- McNemar's test for accuracy differences
- Paired t-test or Wilcoxon for ECE differences

---

## Priority Ranking (What to Do First)

| Priority | Experiment | Rationale | Days |
|----------|-----------|-----------|------|
| **P0** | A2: Eval V2+E3 at max_length=4096 | Need data to make decisions | 1 |
| **P1** | C1: Re-eval existing models at max_length=4096 | May recover 10%p MATH for free | 0.5 |
| **P1** | E2: Selective abstention on existing models | Could be the paper's main result | 1 |
| **P2** | C2: V3 verification-only meta data + SFT | Fix accuracy gap structurally | 2 |
| **P2** | B2: Generate V2+ data with gpt-5.4 | Higher quality + more data | 1 |
| **P3** | B3: V2+ SFT + E3 GRPO | Best possible model | 2 |
| **P3** | D2: Probe reward (E6/E7) | Break confidence diversity ceiling | 2 |
| **P4** | B4: Large-scale eval (n=100) | Statistical rigor for paper | 1 |
| **P4** | E4+E5: Qualitative + significance | Paper polish | 2 |

---

## Decision Tree Summary

```
V2+E3 finishes
  |
  +-- Eval at max_length=4096
       |
       +-- MATH acc >= 60% AND ECE <= 0.65
       |     |
       |     +-- Generate V2+ data (gpt-5.4, 10K chains)
       |     +-- V2+ SFT + E3 GRPO = flagship model
       |     +-- Selective abstention experiment
       |     +-- Scale eval to n=100
       |     +-- Write paper
       |
       +-- MATH acc < 55% BUT ECE <= 0.65
       |     |
       |     +-- Try V3 verification-only meta (C2)
       |     +-- If V3 acc >= 72%: use V3 + GRPO as flagship
       |     +-- If V3 acc < 72%: reframe paper around abstention (E2)
       |     +-- Selective abstention is CRITICAL here
       |
       +-- ECE > 0.70 (GRPO did not help calibration)
             |
             +-- Diagnose (D1): reward variance, training curves
             +-- Try probe reward (D2) with continuous p-hat target
             +-- Try DPO on confidence pairs (D4) as alternative to GRPO
```

---

## Minimum Viable Paper

If we cannot close the accuracy gap, the paper can still work with this framing:

**Title**: "Meta-CoT: Teaching Language Models to Know What They Don't Know"

**Core claim**: Meta-CoT enables selective abstention -- the model achieves higher accuracy on problems it chooses to answer, and correctly identifies problems it will get wrong.

**Required results**:
1. AIME overconfidence: 97% -> 14% (already have this)
2. Selective abstention curve showing V2+GRPO dominates base at all thresholds
3. At least ONE configuration where accuracy is within 5%p of base (V3 or max_length=4096)
4. Qualitative examples of self-aware reasoning
5. n >= 100 eval with bootstrap CIs

**Nice to have**:
- Accuracy parity with base SFT
- Probe reward (E7) showing internal belief alignment
- Self-curation learning (E8)

The selective abstention angle (E2) is the single most important missing experiment. It directly operationalizes "models that know what they don't know" in a way reviewers will find compelling.
