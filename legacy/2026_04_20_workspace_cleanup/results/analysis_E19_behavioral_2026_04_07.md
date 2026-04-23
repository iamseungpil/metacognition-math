# E19 Meta-CoT Behavioral Analysis (2026-04-07)

## 1. Executive Summary

E19 SFT (Qwen3-8B + V6 clean 6329 samples, 3 epochs) achieves **75.0% accuracy** (+7.9pp over base_sft 67.1%). However, metacognitive blocks (`<|meta|>`) are used in only 15.9% of responses. A critical structural finding: Qwen3's pretrained `<think>` habit generates empty `<think></think>` blocks in 100% of responses, and all meta blocks appear OUTSIDE this structure.

**Key finding**: The model has learned metacognitive *vocabulary* (78.3% of meta responses mention switching) but not metacognitive *behavior* (only 2.2% show structural method change). This is **behavioral mimicry**: surface reproduction of meta text without genuine strategy shifts.

## 2. Quantitative Results

### 2.1 Accuracy

| Model | Overall | GSM8K | MATH500 |
|---|---|---|---|
| base_sft | 67.1% | — | — |
| **E19** (3ep, lr=2e-6) | **75.0%** | 87.8% | 48.6% |
| **E19c** (3ep, lr=5e-6) | **75.2%** | 88.7% | 47.4% |

**Honest interpretation**: The +7.9pp gain comes from the full training recipe (6329 samples including 3500 straight-solve). We **cannot attribute this gain to metacognition** without an ablation training on straight-only data.

### 2.2 Meta Emission

| Benchmark | Total | Has Meta | Rate |
|---|---|---|---|
| GSM8K | 1030 | 12 | 1.2% |
| MATH500 | 500 | 232 | **46.4%** |
| **Overall** | 1530 | 244 | 15.9% |

The model selectively uses meta on harder problems — this is rational behavior, not a bug.

### 2.3 Behavioral Classification (244 meta responses)

| Behavior | Count | Rate | Success |
|---|---|---|---|
| Route switch (keyword) | 191 | 78.3% | 39.8% correct |
| Verification | 38 | 15.6% | 50.0% correct |
| Overconfidence correction | 24 | 9.8% | — |
| Decomposition | 14 | 5.7% | 57.1% correct |
| Confidence-only | 27 | 11.1% | — |

### 2.4 Structural Switch vs Keyword Switch

| Metric | Value |
|---|---|
| Keyword "switch" mentions | 78.3% (191/244) |
| Structural method change (heuristic) | **2.2%** (5/232 MATH500) |
| Switch + correct | 0/5 |
| Pre-meta non-theatrical (>50 chars) | 98.0% |
| Post-meta substantial (>100 chars) | 100% |

**Gap**: 78.3% say they switch, 2.2% actually do. The model reproduces metacognitive language more readily than it executes genuine strategy shifts.

### 2.5 Verify Effect (Counterintuitive)

| Condition | n | Accuracy |
|---|---|---|
| Meta + verify | 38 | 39.5% |
| Meta + no verify | 206 | 45.8% |
| **Delta** | | **-6.4pp** |

**Honest interpretation**: This likely reflects **double selection bias** — verification is triggered on the hardest problems where the model is least confident. Small sample (n=38) makes this estimate noisy. **Cannot claim verification causally hurts.**

### 2.6 Confidence Analysis (MATH500)

| Condition | Meta used |
|---|---|
| Correct problems (n=243) | 41.2% |
| Wrong problems (n=257) | 51.4% |

Meta is used 1.2x more on wrong problems. Consistent with: meta triggered by difficulty, not meta causing errors.

## 3. Structural Finding: `<think>` vs `<|meta|>`

| Observation | Value |
|---|---|
| Empty `<think></think>` in response | **100%** |
| `<think>` in training data | **0%** |
| Meta inside `<think>` | **0/244** |
| Meta outside `<think>` | 244/244 |

**Root cause**: Qwen3-8B has a strong pretrained habit to generate `<think>` blocks. Our 6329 SFT samples (with 0 `<think>` tags) could not override this. The model generates empty think, then proceeds with direct solving. Meta blocks appear as secondary, optional additions.

**Implication**: Training data must integrate `<|meta|>` within the `<think>` paradigm, not fight it.

## 4. Honest Claims (Codex-Reviewed)

### Safe to claim:
1. "E19 training recipe improves accuracy by 7.9pp over base SFT." (direct measurement)
2. "Model selectively emits meta on harder problems (46.4% MATH500 vs 1.2% GSM8K)."
3. "78.3% of meta responses describe switching; structural heuristic detects 2.2% — behavioral mimicry."
4. "Raw accuracy comparisons between meta/no-meta are confounded by difficulty."

### Require hedging:
5. "Negative verify effect (-6.4pp) likely reflects selection into hardest problems." (small n)
6. "Some accuracy gain may stem from inference-time metacognition." (ambiguous attribution)

### Must not claim:
- "Metacognition improves accuracy by 7.9pp." (conflates recipe with mechanism)
- "Verification hurts performance." (confounded)
- "Model switches strategies 39.8% of the time." (keyword ≠ actual)

## 5. Comparison with behavior-uncertainty Findings

| Finding | behavior-uncertainty | E19 Meta-CoT |
|---|---|---|
| Verification as evaluation mechanism | Effective on countdown | **Confounded** (negative raw effect, likely selection bias) |
| Entropy reduction after metacognition | Measured (before/after windows) | **Not measured** (token-level entropy analysis needed) |
| All-strategies best OOD | Confirmed (all_strategies=30%) | **Partially** (meta on hard problems = rational) |
| Habit internalization via SFT | Strong (98-100% habit markers) | **Partial** (78% keyword, 2% structural) |

**Key gap**: behavior-uncertainty measures token-level entropy changes around habit markers. We have not done this for `<|meta|>`. This would reveal whether meta blocks actually change the model's internal state or are just surface text.

## 6. Next Steps (의도/가설/검증)

### 6.1 Data Format Fix

**의도**: Align training data with Qwen3's natural `<think>` paradigm.

**가설**: Wrapping reasoning in `<think>` and placing `<|meta|>` between think blocks will increase meta emission from 16% to >50% and improve structural switching.

**검증**: Re-SFT with transformed data → eval → compare meta emission rate, switch rate, accuracy.

### 6.2 Token-Level Entropy Analysis

**의도**: Measure whether `<|meta|>` blocks actually change the model's internal state.

**가설**: If meta is functional, entropy should drop after meta blocks on correct problems (uncertainty resolved). If meta is decorative, entropy pattern should not change.

**검증**: Extract per-token logprobs using vLLM, compute before/after entropy windows around `<|meta|>` markers, split by correctness.

### 6.3 Straight-Only Ablation

**의도**: Isolate metacognition's contribution to accuracy.

**가설**: If meta contributes, E19 (meta+straight) should outperform straight-only model.

**검증**: Train on straight-only data, eval, compare accuracy difference.
