# Meta-CoT Experiment Plan v5: Fixing Meta Execution

**Date**: 2026-04-04
**Author**: Seungpil Lee
**Project**: metacognition-math (Qwen3-8B)
**Predecessor**: experiment_plan_v4.md (accuracy-focused autoresearch loop)

---

## 1. Current State Summary

### What works

1. **Parseable meta**: The SFT-trained model emits well-formed `<|meta|>...<|/meta|>` blocks containing confidence, diagnosis, and strategy statements. Coverage reaches 88.9% on `all_sft`.
2. **Calibration (ECE reduction)**: E8 achieves ECE 0.322 with only 3.6% wrong-high-confidence, the best overconfidence suppression among all conditions.
3. **Overconfidence suppression (E8)**: `overconfidence_penalty_reward` + `confidence_revision_reward` successfully pushed wrong-high-conf from 80.7% (verify_sft) down to 3.6% (E8).
4. **Accuracy near parity (E9)**: E9 reaches 41.1% overall (base SFT: 42.2%), the closest Meta-CoT model to parity, driven by GSM8K 90.0%.

### What does not work

1. **Verify execution**: Verification is same-route repetition, not independent falsification. 15/53 wrong answers in E9 are overconfident-verify-failed: the model says "verify" then re-derives the same wrong answer (e.g., AIME 540 re-uses identical extremal intuition; GSM8K restart miscounts time identically).
2. **Redirect execution**: Diagnosis appears but the solve body continues the original route. 0/4 diagnosis cases in E10 led to actual recovery. `effective_redirection_reward` checks `bool(solve_tail.strip())` for recovery, so any non-empty tail passes regardless of whether it contains a route switch.
3. **Repeated intervention**: Hard problems get exactly one meta block. No ongoing controller loop is observed. The model treats meta as a one-shot annotation rather than an iterative self-monitor.
4. **Full controller (E10)**: Only 5.6% confidence coverage. 53 out of 58 wrong answers have no meta signal at all. The model learned to omit confidence to avoid calibration penalties.

### Quantitative reference (n=90 pilot, 30 per benchmark)

| Model | Acc | AIME | GSM8K | MATH500 | conf_cov | ECE | wrong_high_conf |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_sft | 42.2 | 13.3 | 80.0 | 33.3 | 0.0 | n/a | n/a |
| all_sft | 33.3 | 6.7 | 63.3 | 30.0 | 88.9 | 0.398 | 51.7 |
| E8 | 38.9 | 10.0 | 80.0 | 26.7 | 23.3 | 0.322 | 3.6 |
| E9 | 41.1 | 6.7 | 90.0 | 26.7 | 91.1 | 0.301 | 50.9 |
| E9b | 40.0 | 6.7 | 83.3 | 30.0 | 0.0 | n/a | n/a |
| E10 | 35.6 | 6.7 | 66.7 | 33.3 | 5.6 | 0.340 | 0.0 |

---

## 2. Root Cause Analysis

Three code-level root causes explain why meta detection succeeds but meta execution fails.

### Root Cause 1: `effective_verification_reward` rewards same-route repetition

**Location**: `rewards.py` line 1016-1035, specifically `_has_effective_verification_signal` (line 119-125).

The function checks for word patterns ("substitute", "plug back", "recalculat") in the solve tail after the last meta block. The problem is that restating the same derivation with verification keywords counts as effective verification. There is no check for **structural independence** -- whether the verification method differs from the original solve route.

**Consequence**: The model learns to append "let me substitute back..." followed by the same calculation. This gets rewarded at +0.8 for correct answers, reinforcing decorative verification.

### Root Cause 2: `effective_redirection_reward` accepts any non-empty tail as recovery

**Location**: `rewards.py` line 1038-1063, specifically line 1051: `has_tail_recovery = bool(solve_tail.strip())`.

The maximum reward (+1.0 for correct, +0.2 for wrong) requires `has_conflict and has_switch and has_drop and has_tail_recovery`. But `has_tail_recovery` is satisfied by any non-empty text after the last meta block -- including the original solve continuation that ignores the announced strategy switch.

**Consequence**: The model announces "switch to parity-based analysis" in meta, then continues the original algebraic approach. Because the tail is non-empty, the redirection reward fires as if recovery occurred.

### Root Cause 3: `calibration_reward` returns 0.0 for no meta, making omission cost-free

**Location**: `rewards.py` line 656-710, specifically lines 679-680:

```python
if not blocks or all(b["confidence"] is None for b in blocks):
    rewards.append(0.0)
```

When the model emits no meta blocks or omits confidence, it receives 0.0 from calibration. Meanwhile, emitting confidence on wrong answers risks negative reward from both the Brier component and the log scoring rule. Rational reward optimization therefore favors omitting confidence entirely.

**Consequence**: E5 (conf_cov=1.1%), E9b (conf_cov=0.0%), and E10 (conf_cov=5.6%) all escaped calibration pressure by reducing meta emission. This makes calibration metrics uninterpretable and defeats the purpose of Meta-CoT.

---

## 3. Proposed Experiments

All three experiments start from the same SFT checkpoint: `qwen3_metacot_control_v5_all_sft`. This ensures the only variable is the reward composition.

### E9v2: Verify Quality

**Node**: EVAL (GPU 0-3, same as current eval lane)

**Intent**: Teach the model that verification must use an independent method, not repeat the same calculation. The current verify reward conflates "mentions verification keywords" with "actually performs a structurally different check."

**Hypothesis**: Adding `same_route_repetition_penalty` (-0.5 when the solve tail after a verify announcement shares >70% n-gram overlap with the pre-meta solve body) will reduce `overconfident_verify_failed` rate from 15/53 (28.3%, E9) to <5/53 (9.4%) without reducing overall accuracy by more than 2 percentage points.

**Mechanism**: The new penalty computes character trigram overlap between the pre-meta solve prefix and the post-meta solve tail. If overlap exceeds 70%, the verification is classified as same-route repetition. The existing `effective_verification_reward` remains unchanged -- the penalty is additive, creating a gradient between "verify with a new method (+0.8)" and "verify by restating (-0.5)."

**Verification**:
- **Primary metric**: `overconfident_verify_failed` count on 1,030-problem eval. Target: <10% of wrong answers (currently 28.3%).
- **Secondary metric**: `independent_method_rate` = fraction of verify completions where trigram overlap <70%. Target: >50%.
- **Falsification condition**: If `wrong_high_conf` stays >40% despite the penalty, the trigram heuristic is too weak to detect repetition and a stronger structural check (e.g., operation-type extraction) is needed.

**Reward composition**:

| Reward function | Weight | Source |
|---|---|---|
| correctness_reward | 3.0 | E9 |
| format_reward | 0.2 | E9 |
| correct_meta_reward | 0.3 | E9 |
| calibration_reward | 0.4 | E9 |
| confidence_revision_reward | 0.6 | E9 |
| overconfidence_penalty_reward | 1.0 | E9 |
| length_penalty_reward | 1.0 | E9 |
| effective_verification_reward | 0.9 | E9 |
| overconfidence_verify_reward | 0.9 | E9 |
| **same_route_repetition_penalty** | **0.5** | **NEW** |
| **confidence_omission_floor** | **0.5** | **NEW** |

---

### E9bv2: Redirect Execution

**Node**: TRAIN_B (GPU 0-3, dedicated training lane)

**Intent**: Ensure that redirect rewards require structural evidence of a route switch in the solve body, not just a non-empty tail. The current reward fires whenever the model writes anything after announcing a redirect, even if that "anything" continues the original approach.

**Hypothesis**: Replacing `bool(solve_tail.strip())` with `route_switch_evidence_reward` (+0.9 for structural method-keyword difference between pre-meta and post-meta, -0.3 for announced switch without evidence) will increase `diagnosis_with_recovery` rate from 0/4 (E10) to >50% on the AIME + MATH-hard slice.

**Mechanism**: The new reward extracts method keywords (algebra, geometry, combinatorics, number theory, substitution, casework, induction, parity, modular, generating function, etc.) from (a) the pre-meta solve body and (b) the post-meta solve tail. If the two keyword sets overlap <50%, the redirect is classified as a genuine route switch. If they overlap >80% and the meta announced a switch, the penalty fires.

**Verification**:
- **Primary metric**: `diagnosis_with_recovery / total_diagnosis` ratio on AIME (30) + MATH-hard (top-50 by difficulty) slice. Target: >50%.
- **Secondary metric**: Method keyword Jaccard distance between pre-meta prefix and post-meta tail. Target: mean distance >0.3 for redirect cases.
- **Falsification condition**: If models stop announcing redirects entirely (redirect_count drops from E9b's current level by >80%), the penalty weight is too harsh and should be reduced to 0.15.

**Reward composition**:

| Reward function | Weight | Source |
|---|---|---|
| correctness_reward | 3.0 | E9b |
| format_reward | 0.2 | E9b |
| correct_meta_reward | 0.3 | E9b |
| calibration_reward | 0.4 | E9b |
| confidence_revision_reward | 0.6 | E9b |
| overconfidence_penalty_reward | 1.0 | E9b |
| length_penalty_reward | 1.0 | E9b |
| effective_redirection_reward | 1.0 | E9b (modified: tail check replaced) |
| **route_switch_evidence_reward** | **0.6** | **NEW** |
| **confidence_omission_floor** | **0.5** | **NEW** |

---

### E10v2: Full Controller + Coverage Floor

**Node**: E8 (GPU 0-3, dedicated training lane)

**Intent**: Combine improved verify and redirect with mandatory meta emission. E10 failed because the model escaped calibration by omitting confidence. E10v2 adds a coverage floor that penalizes missing meta, forcing the model to emit confidence so that verify and redirect rewards can shape its behavior.

**Hypothesis**: `confidence_omission_floor` (-0.5 for completions with no parseable meta block) will increase confidence coverage from 5.6% (E10) to >60%, while `same_route_repetition_penalty` and `route_switch_evidence_reward` improve execution quality. Overall accuracy should remain >= 38% (within 4.2 percentage points of base SFT 42.2%).

**Mechanism**: Three new rewards are added on top of the E10 base:
1. `confidence_omission_floor`: Returns -0.5 when no meta block with parseable confidence exists, 0.0 otherwise. This makes omission costly without prescribing what confidence should be.
2. `same_route_repetition_penalty`: Same as E9v2. Penalizes decorative verify.
3. `route_switch_evidence_reward`: Same as E9bv2. Requires structural evidence for redirect.

**Verification**:
- **Primary metric**: `confidence_coverage > 60%` AND `accuracy >= 38%`. Both must hold simultaneously.
- **Secondary metric**: Failure mode distribution shift. `no_meta_signal` should drop from 53/58 wrong answers (E10) to <10/58.
- **Falsification condition**: If accuracy drops below 30%, the combined reward pressure from 16 reward functions is too strong and non-correctness weights should be uniformly scaled by 0.5x.

**Reward composition**:

| Reward function | Weight | Source |
|---|---|---|
| correctness_reward | 3.0 | E10 |
| format_reward | 0.2 | E10 |
| correct_meta_reward | 0.3 | E10 |
| calibration_reward | 0.4 | E10 |
| confidence_revision_reward | 0.6 | E10 |
| overconfidence_penalty_reward | 1.0 | E10 |
| length_penalty_reward | 1.0 | E10 |
| effective_verification_reward | 0.8 | E10 |
| effective_redirection_reward | 1.0 | E10 (modified: tail check replaced) |
| diagnosis_reward | 0.6 | E10 |
| decomposition_reward | 0.6 | E10 |
| anomaly_notice_reward | 0.4 | E10 |
| repeated_intervention_reward | 0.5 | E10 |
| overconfidence_verify_reward | 1.0 | E10 |
| **same_route_repetition_penalty** | **0.5** | **NEW** |
| **route_switch_evidence_reward** | **0.6** | **NEW** |
| **confidence_omission_floor** | **0.5** | **NEW** |

---

## 4. New Reward Functions (Exact Specifications)

### 4.1 `same_route_repetition_penalty`

```python
def same_route_repetition_penalty(completions, ground_truth=None, **kwargs):
    """Penalize verification that restates the same derivation.
    
    Computes character trigram overlap between the pre-meta solve body
    and the post-meta solve tail. If overlap > 0.7, returns -1.0.
    If overlap 0.5-0.7, returns -0.3. Otherwise 0.0.
    Only fires when verify intent is detected in meta.
    """
```

| Condition | Reward |
|---|---|
| No verify intent in meta | 0.0 |
| Verify intent + trigram overlap > 0.7 | -1.0 |
| Verify intent + trigram overlap 0.5-0.7 | -0.3 |
| Verify intent + trigram overlap < 0.5 | 0.0 |

Weight in E9v2: 0.5. Weight in E10v2: 0.5.

### 4.2 `route_switch_evidence_reward`

```python
def route_switch_evidence_reward(completions, ground_truth=None, **kwargs):
    """Reward structural method difference between pre-meta and post-meta.
    
    Extracts method keywords from pre-meta prefix and post-meta tail.
    Rewards genuine route switches; penalizes announced-but-not-executed switches.
    """
```

| Condition | Reward |
|---|---|
| No redirect signal in meta | 0.0 |
| Redirect signal + keyword Jaccard distance > 0.5 (genuine switch) + correct | +1.5 |
| Redirect signal + keyword Jaccard distance > 0.5 (genuine switch) + wrong | +0.3 |
| Redirect signal + keyword Jaccard distance < 0.2 (fake switch) | -0.5 |
| Redirect signal + keyword Jaccard distance 0.2-0.5 (ambiguous) | 0.0 |

Weight in E9bv2: 0.6. Weight in E10v2: 0.6.

### 4.3 `confidence_omission_floor`

```python
def confidence_omission_floor(completions, ground_truth=None, **kwargs):
    """Penalize completions that omit meta/confidence entirely.
    
    Returns -1.0 if no meta block with parseable confidence exists.
    Returns 0.0 otherwise.
    This prevents the model from escaping calibration pressure by omitting meta.
    """
```

| Condition | Reward |
|---|---|
| At least one meta block with parseable confidence | 0.0 |
| No meta block OR no parseable confidence in any block | -1.0 |

Weight in all v2 experiments: 0.5. Effective penalty for omission: -0.5.

### 4.4 Modification to `effective_redirection_reward` (E9bv2, E10v2 only)

Line 1051 changes from:
```python
has_tail_recovery = bool(solve_tail.strip())
```
to:
```python
has_tail_recovery = _has_route_switch_in_tail(solve_tail, text)
```

where `_has_route_switch_in_tail` checks for method-keyword divergence (same logic as `route_switch_evidence_reward` but returns bool). This is the **only modification** to an existing reward function. It is isolated to v2 experiments via a `strict_redirect=True` flag, leaving E9b/E10 behavior unchanged.

---

## 5. Execution Plan

### 5.1 Training

| Lane | Experiment | GPU allocation | Init checkpoint | Steps | Checkpoints |
|---|---|---|---|---|---|
| EVAL | E9v2 | 4x A100 (node EVAL) | qwen3_metacot_control_v5_all_sft | 300 | 100, 200, 300 |
| TRAIN_B | E9bv2 | 4x A100 (node TRAIN_B) | qwen3_metacot_control_v5_all_sft | 300 | 100, 200, 300 |
| E8 | E10v2 | 4x A100 (node E8) | qwen3_metacot_control_v5_all_sft | 300 | 100, 200, 300 |

All three start from the **same SFT init** to ensure the only variable is the reward composition.

### 5.2 Evaluation

After each final (step-300) checkpoint:
- Full 1,030-problem eval: GSM8K (500) + MATH-500 (500) + AIME 2024 (30)
- max_tokens = 4096
- Temperature = 0.0 (greedy) for reproducibility
- Save full completion text for qualitative failure analysis

Estimated time:
- Training: 300 steps x ~2 min/step = ~10 hours per lane (3 lanes in parallel = 10 hours wall time)
- Evaluation: 1,030 problems x ~90 sec/problem / 4 GPUs = ~6.4 hours per model (3 models sequential or parallel depending on node availability)
- Total: ~10 hours training + ~6-19 hours eval = **16-29 hours end-to-end**

### 5.3 Launch commands

```bash
# E9v2 (EVAL node)
accelerate launch --num_processes 4 --multi_gpu \
  src/training/grpo_v2.py --mode E9v2 --max_steps 300 \
  --model_path checkpoints/qwen3_metacot_control_v5_all_sft \
  --data mixed_train

# E9bv2 (TRAIN_B node)
accelerate launch --num_processes 4 --multi_gpu \
  src/training/grpo_v2.py --mode E9bv2 --max_steps 300 \
  --model_path checkpoints/qwen3_metacot_control_v5_all_sft \
  --data mixed_train

# E10v2 (E8 node)
accelerate launch --num_processes 4 --multi_gpu \
  src/training/grpo_v2.py --mode E10v2 --max_steps 300 \
  --model_path checkpoints/qwen3_metacot_control_v5_all_sft \
  --data mixed_train
```

---

## 6. Success Criteria

| Milestone | Condition | Priority | Depends on |
|---|---|---|---|
| M1: Coverage | E10v2 `conf_coverage` > 60% | **PRIMARY** | confidence_omission_floor |
| M2: Verify quality | E9v2 `overconfident_verify_failed` < 10% of wrong answers | HIGH | same_route_repetition_penalty |
| M3: Redirect execution | E9bv2 `diagnosis_with_recovery` > 50% | HIGH | route_switch_evidence_reward |
| M4: Accuracy parity | Any v2 model accuracy >= 40% (within 2.2%p of base_sft 42.2%) | MEDIUM | correctness weight dominance |
| M5: Calibration | ECE < 0.35 with coverage > 60% (both simultaneously) | MEDIUM | M1 prerequisite |

### Milestone dependencies

```
M4 (accuracy parity) ← independent, must not regress
M1 (coverage) ← enables M5
M2 (verify quality) ← improves accuracy on high-conf wrong
M3 (redirect execution) ← improves accuracy on diagnosis cases
M5 (calibration) ← requires M1 coverage to be interpretable
```

---

## 7. Decision Tree (Post-Eval)

```
1,030 eval results arrive for E9v2, E9bv2, E10v2
│
├── IF M1+M2+M3 all met
│   ├── SUCCESS: meta execution works
│   ├── Select best model by accuracy+calibration Pareto
│   ├── Proceed to RQ3: curriculum learning with RAG
│   └── diagnosis quality now sufficient for retrieval triggers
│
├── IF M1 met (coverage > 60%) but M2 not met (verify still repetitive)
│   ├── Trigram overlap heuristic too weak
│   ├── Action: upgrade to operation-type extraction (extract arithmetic ops,
│   │   algebraic manipulations, geometric constructions as structured features)
│   └── Re-run E9v2 with stronger structural independence check
│
├── IF M1 met but M3 not met (redirect still decorative)
│   ├── Keyword Jaccard too coarse
│   ├── Action: use sentence-transformer embedding cosine distance
│   │   between pre-meta and post-meta instead of keyword overlap
│   └── Re-run E9bv2 with embedding-based route switch detection
│
├── IF M1 not met (coverage still < 60%)
│   ├── confidence_omission_floor weight 0.5 is too low
│   ├── Action: increase weight to 0.8 and re-run E10v2
│   ├── If still not met at 0.8: the model may need SFT-level
│   │   conditioning on mandatory confidence emission
│   └── Consider re-generating SFT data with 100% confidence coverage
│
├── IF M4 not met (accuracy < 40% for all v2 models)
│   ├── Combined reward pressure too strong
│   ├── Action: scale all non-correctness weights by 0.5x
│   └── Re-run with correctness dominating the reward landscape
│
└── IF accuracy < 30% for any model
    ├── ABORT: reward composition is pathological
    ├── Fall back to best existing model (E9 at 41.1%)
    └── Re-examine whether 16-17 reward functions create conflicting gradients
```

---

## 8. Execution Gates

All three gates must pass before training begins. No exceptions.

### Gate 1: Unit tests for new reward functions

```bash
python -m pytest tests/test_rewards.py -k "same_route_repetition or route_switch_evidence or confidence_omission" -v
```

Required tests:
- `test_same_route_repetition_penalty_fires_on_identical_derivation`
- `test_same_route_repetition_penalty_zero_on_independent_method`
- `test_route_switch_evidence_reward_positive_on_genuine_switch`
- `test_route_switch_evidence_reward_negative_on_fake_switch`
- `test_confidence_omission_floor_negative_on_no_meta`
- `test_confidence_omission_floor_zero_on_meta_with_confidence`

### Gate 2: Compilation check

```bash
python -c "import py_compile; py_compile.compile('src/training/rewards.py', doraise=True)"
python -c "import py_compile; py_compile.compile('src/training/grpo_v2.py', doraise=True)"
```

### Gate 3: Smoke test with non-zero reward variance

```bash
python -c "
from src.training.rewards import same_route_repetition_penalty, route_switch_evidence_reward, confidence_omission_floor

# Case 1: repetitive verify
c1 = [{'role': 'assistant', 'content': '<|meta|>confidence: 0.9. Let me verify.<|/meta|>x=2, so 2+3=5. \\\\boxed{5}'}]
# Case 2: independent verify  
c2 = [{'role': 'assistant', 'content': '<|meta|>confidence: 0.9. Let me verify by substitution.<|/meta|>Plug x=2 into original: f(2)=4+6+1=11. Wait, different. \\\\boxed{11}'}]
# Case 3: no meta
c3 = [{'role': 'assistant', 'content': 'The answer is \\\\boxed{42}'}]

r1 = same_route_repetition_penalty([c1, c2])
r2 = confidence_omission_floor([c1, c3])
assert r1[0] != r1[1], f'Repetition penalty must differentiate: {r1}'
assert r2[0] != r2[1], f'Omission floor must differentiate: {r2}'
print('Gate 3 PASSED: reward variance confirmed')
"
```

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Trigram overlap is too blunt (false positives on legitimately similar verifications) | Medium | M2 fails | Fall back to operation-type extraction (decision tree branch 2) |
| Keyword Jaccard misclassifies same-domain switches as non-switches | Medium | M3 fails | Use embedding cosine distance as upgrade path |
| 17 reward functions in E10v2 create conflicting gradients | Medium | Accuracy collapse | Scale non-correctness weights by 0.5x (decision tree branch 5) |
| Coverage floor causes model to emit random confidence numbers | Low | M5 fails (high ECE) | Calibration reward already penalizes miscalibrated confidence |
| SFT init does not have enough capacity for independent verification | Low | M2 structurally impossible | Would require SFT data regeneration with explicit independent-method examples |

---

## 10. Relationship to Research Questions

| RQ | What this plan tests | Success signal |
|---|---|---|
| RQ1 (Meta-CoT representation) | Whether reward shaping can upgrade meta from "detection" to "execution" | M2 + M3: verify uses independent methods, redirect switches routes |
| RQ2 (Meta-RL behavior) | Whether targeted reward fixes close specific behavioral gaps | M1: coverage floor prevents escape; M4: accuracy maintained |
| RQ3 (Curriculum) | Gated behind M1+M2+M3 | Not tested in v5; proceeds only if meta execution works |

---

## Appendix A: Experiment Lineage

```
E8  (overconf suppression)     → E9  (+ verify)     → E9v2  (+ verify quality)
                                → E9b (+ redirect)   → E9bv2 (+ redirect quality)
                                → E9c (+ diagnosis)
                                → E10 (full controller) → E10v2 (+ coverage floor + quality)
```

## Appendix B: File Changes Required

| File | Change type | Description |
|---|---|---|
| `src/training/rewards.py` | ADD | 3 new functions: `same_route_repetition_penalty`, `route_switch_evidence_reward`, `confidence_omission_floor` |
| `src/training/rewards.py` | ADD | 1 helper: `_has_route_switch_in_tail` |
| `src/training/rewards.py` | ADD | 1 helper: `_method_keywords` (keyword extraction for route switch detection) |
| `src/training/grpo_v2.py` | ADD | 3 new mode entries: `E9v2`, `E9bv2`, `E10v2` in `reward_configs` dict |
| `src/training/grpo_v2.py` | MODIFY | `--mode` choices list: append `"E9v2", "E9bv2", "E10v2"` |
| `src/training/grpo_v2.py` | MODIFY | `use_gdpo` condition: append `"E9v2", "E9bv2", "E10v2"` |
| `tests/test_rewards.py` | ADD | 6 new test cases for 3 new reward functions |
| No other files are modified. | | |
