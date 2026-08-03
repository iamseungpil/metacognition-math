# Autoresearch Configuration: Meta-CoT vs Base SFT

**Created**: 2026-03-30
**Project**: metacognition-math (Qwen3-8B)

---

## Goal

Meta-CoT overall accuracy >= Base SFT overall accuracy on 1,030 problems
(GSM8K-500 + MATH-500 + AIME-30)

## Success Metric

```
metric: overall_accuracy
target: >= 58.9% (Base SFT baseline)
direction: maximize
secondary_constraint: ECE(MATH) < 0.35 (preserve calibration)
```

## Iteration Limit

- **Max rounds**: 5
- **Per-round budget**: ~5 hours (training) + ~2 hours (eval) = ~7 hours
- **Total budget**: ~35 hours max
- **Early stop**: If M1 achieved, stop and report

## Evaluation Protocol

```
benchmarks: [gsm8k, math500, aime2024]
max_problems: [500, 500, 30]
max_tokens: 4096
temperature: 0.0 (greedy)
eval_script: src/eval/eval_hf.py
metrics: [accuracy, ece, truncation_rate, meta_block_count, confidence_distribution]
bootstrap_ci: 95% (1000 resamples)
```

## Base SFT Reference Scores (to beat)

| Benchmark | Base SFT Accuracy (n=30) | Pending (n=500/30) |
|-----------|-------------------------|-------------------|
| GSM8K | 93.3% | Eval at 4096 tokens pending |
| MATH-500 | 76.7% | Eval at 4096 tokens pending |
| AIME-2024 | 6.7% | Eval at 4096 tokens pending |
| **Overall** | **58.9%** | **Pending** |

---

## Hypotheses (ordered by priority)

### Round 0: H1 + H5 (already applied/in-progress)

**H1: max_tokens = 4096**
- What to change: Eval config `max_tokens` from 1024 to 4096
- Rationale: 31% of Meta-CoT completions were truncated at 1024 tokens; meta blocks consume ~56% of token budget. Increasing the ceiling removes this bottleneck.
- Expected impact: +5-10%p overall (recovers truncated completions)
- How to verify: Compare truncation rate at 1024 vs 4096. If truncation drops to <5% and accuracy rises proportionally, H1 is confirmed.
- Implementation: Already done in Phase B eval config (zero cost)
- Status: READY TO EVAL

**H5: Longer GRPO (500 steps)**
- What to change: GRPO training from 200 to 500 steps
- Rationale: At step 200, KL was only 0.149 suggesting the model had not diverged enough from SFT. More steps allow further policy optimization.
- Expected impact: +2-5%p overall
- How to verify: Compare ckpt-200 vs ckpt-500 accuracy. If ckpt-500 > ckpt-200, training is still improving.
- Implementation: Already running (ETA ~2.5 hrs)
- Status: IN PROGRESS

### Round 1: H2 (if H1+H5 insufficient)

**H2: Difficulty-adaptive meta**
- What to change: Modify prompt template to skip meta blocks on easy problems (GSM8K difficulty), use single-line meta on medium (MATH L1-3), and full meta only on hard (MATH L4-5, AIME).
- Rationale: Easy problems (GSM8K) don't benefit from metacognitive overhead. Skipping meta preserves token budget for the actual solution, recovering accuracy on easy problems while maintaining meta benefits on hard ones.
- Expected impact: +3-5%p overall (mainly from GSM8K accuracy recovery)
- How to verify: GSM8K accuracy should rise to ~93% (matching Base SFT) while MATH/AIME accuracy remains stable or improves.
- Implementation:
  1. Create difficulty classifier: `if benchmark == "gsm8k": skip_meta = True`
  2. Modify eval_hf.py to conditionally include meta instruction in prompt
  3. No retraining needed (inference-time change)
- Time estimate: 2 hours

### Round 2: H3 (if H2 insufficient)

**H3: Verification-only meta**
- What to change: Restructure meta-CoT to remove pre-solve planning blocks. Keep only post-answer verification: a single meta block after the boxed answer that checks the solution.
- Rationale: Pre-solve meta (planning, difficulty assessment) wastes tokens on overhead that doesn't improve the solution. Post-answer verification catches errors with fewer tokens (~80 vs ~250 tokens overhead).
- Expected impact: +2-4%p overall
- How to verify: Meta token overhead drops from ~56% to ~15%. Accuracy should improve proportionally. Error-catching rate in verification blocks should be > 0%.
- Implementation:
  1. Modify prompt_v2.py to produce verification-only format
  2. Generate new SFT data (4,996 chains) with verification-only template
  3. Retrain SFT (3 hours)
  4. Optionally apply GRPO E3 (5 hours)
- Time estimate: 8 hours (data + SFT + eval)

### Round 3: H4 (if H3 insufficient)

**H4: gpt-5.4 teacher data (not mini)**
- What to change: Regenerate all 4,996 Meta-CoT chains using gpt-5.4 (full model) instead of gpt-5.4-mini via TRAPI.
- Rationale: Phase 4 report showed V2 data (gpt-5.4-mini) has worse accuracy than V1 data (gpt-5.4, 7,371 chains). The weaker teacher produces lower-quality reasoning chains that the student cannot overcome.
- Expected impact: +3-5%p overall (V1 vs V2 gap was ~5%p on MATH)
- How to verify: Compare SFT accuracy with gpt-5.4-mini vs gpt-5.4 data, same format. If gpt-5.4 data produces higher accuracy, teacher quality was the bottleneck.
- Implementation:
  1. Run TRAPI generation with gpt-5.4 (rate limit: ~100 req/min)
  2. Generate 5,000+ chains (~2 hours)
  3. Retrain SFT (3 hours)
  4. Apply best GRPO config (5 hours)
- Time estimate: 10+ hours
- Risk: TRAPI quota/cost

### Round 4: H6 (if H4 insufficient)

**H6: Verification reward bonus**
- What to change: Add a new reward component that gives +1.0 bonus when a meta verification block correctly identifies an error AND the final answer changes to be correct.
- Rationale: Current meta blocks may be decorative (always saying "looks correct"). Rewarding productive error-catching incentivizes meta blocks that actually improve accuracy.
- Expected impact: +1-3%p overall
- How to verify: Measure "productive meta rate" (% of meta blocks that change the answer). This should increase from ~0% to ~10%+.
- Implementation:
  1. Add `verification_bonus_reward()` to rewards.py
  2. Detect error-identification patterns in meta blocks
  3. Train GRPO with added reward (5 hours)
- Time estimate: 7 hours

---

## Autoresearch Loop Protocol

```
for round in range(5):
    # 1. CRITIC: Analyze current eval results
    analyze_errors(eval_results)
    classify_failure_modes()  # truncation, overhead, wrong_answer, format_error

    # 2. PLANNER: Select next hypothesis
    hypothesis = select_next_hypothesis(priority_order, already_tried)

    # 3. IMPLEMENTER: Apply the change
    implement(hypothesis)

    # 4. EVAL: Run full 1,030-problem eval
    new_results = eval_hf(model, benchmarks, max_tokens=4096)

    # 5. DECIDE: Keep or discard
    if new_results.overall_accuracy >= 58.9%:
        print("M1 ACHIEVED")
        break
    elif new_results.overall_accuracy > best_so_far:
        best_so_far = new_results
        keep(hypothesis)
    else:
        discard(hypothesis)
        revert()
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/training/grpo_v2.py` | GRPO training (E1-E4 modes, GDPO) |
| `src/training/rewards.py` | 7 reward functions |
| `src/eval/eval_hf.py` | HF generate eval pipeline |
| `src/metacot/prompt_v2.py` | V2 prompt template |
| `checkpoints/qwen3_meta_sft` | Base Meta SFT model |
| `checkpoints/base_sft` | Control model (no meta) |
| `results/experiment_plan_v4.md` | Full experiment plan |

---

## Abort Conditions

- If after 5 rounds Meta-CoT still < Base SFT: the meta-CoT approach may be fundamentally token-inefficient for this model size. Consider larger model (14B+) or abandon meta-CoT for accuracy tasks.
- If any round causes > 10%p regression: revert immediately, investigate.
- If GRPO training shows KL > 2.0: stop training, reduce LR or increase beta.
