# Research Plan: Learning Models That Know What They Don't Know

## Research Question
Can we train a model to accurately recognize its own uncertainty during reasoning,
and does this metacognitive awareness improve math problem-solving performance at inference time?

## Core Hypothesis
If a model learns to accurately assess "how likely am I to solve this?" and
"which step am I uncertain about?", it can redirect its reasoning when stuck,
leading to higher accuracy compared to models without this self-awareness.

---

## Stage 1: Metacognitive SFT + RL (Current Focus)

### 1.1 Data Preparation

**Base SFT Data** (control):
- Source: Rollouts where model answered correctly
- Format: Standard CoT — step-by-step solution + \boxed{answer}
- No self-assessment, no uncertainty markers

**Metacognitive SFT Data** (treatment):
- Source: GPT-5.4 generates 3-phase solutions using rollout results + capability profile
- Format:
  ```
  Phase 1 — Pre-solve Assessment:
  "This is a modular arithmetic problem. My accuracy on this category is 36%.
   Key risk: I often miss cycle verification. I need to double-check periodicity."

  Phase 2 — Solve with Epistemic Awareness:
  "2^1≡2, 2^2≡4, 2^3≡8≡1 (mod 7).
   Wait, let me verify: 8=7+1, yes remainder 1. Confidence: high.
   100 = 33×3 + 1, so 2^100 ≡ 2^1 ≡ 2 (mod 7).
   Hmm, let me double-check the division: 33×3=99, 100-99=1. Correct.
   \boxed{2}"

  Phase 3 — Post-solve Reflection:
  "The cycle detection was correct. I should practice Euler's theorem
   for cases where the modulus is not prime."
  ```

### 1.2 Models to Compare

| Model | Training Data | Description |
|-------|-------------|-------------|
| Base SFT | Standard CoT solutions | Control — no self-awareness |
| Meta SFT | 3-phase metacognitive solutions | Self-aware but no RL signal |
| Meta RL-A | Meta SFT + GRPO(R_correct) | RL baseline — only correctness reward |
| Meta RL-B | Meta SFT + GRPO(R_correct + λ₁·R_calib) | Self-awareness accuracy rewarded |
| Meta RL-C | Meta SFT + GRPO(R_correct + λ₁·R_calib + λ₂·R_epistemic) | Self-correction behavior rewarded |
| Meta RL-D | Meta SFT + GRPO(R_calib only) | Pure self-awareness without correctness |

### 1.3 Reward Definitions

**R_correct**: Binary. 1 if final \boxed{answer} matches gold, 0 otherwise.

**R_calibration**: Measures how accurately the model knows its own ability.
```
c_text = model's stated confidence in the solution (extracted from text)
p_hat  = Gnosis probe's prediction from hidden states (ground truth self-knowledge)
R_calib = 1 - |c_text - p_hat|
```
High reward when model's self-assessment matches what its hidden states actually indicate.

**R_epistemic**: Measures whether self-correction attempts actually help.
```
If model expresses uncertainty AND changes approach:
  If final answer is correct: R_epistemic = 1.0  (correction worked)
  If final answer is wrong:   R_epistemic = 0.2  (good attempt)
If model expresses uncertainty but doesn't change approach:
  R_epistemic = 0.0  (recognized problem but didn't act)
If model doesn't express uncertainty:
  R_epistemic = 0.0  (no metacognitive behavior)
```

### 1.4 Gnosis Probe Role

- Architecture: Simple MLP on last-layer hidden states (already AUROC 0.9652)
- Purpose: Provides "ground truth" for what the model actually knows
- ECE calibration: Must verify probe outputs well-calibrated probabilities
- Temperature scaling post-hoc if needed
- During RL: probe is frozen, provides p̂ for R_calibration computation

### 1.5 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| MATH test pass@1 | Raw problem-solving ability |
| AIME pass@1 | Hard problem-solving ability |
| ECE | How well model's stated confidence matches reality |
| Epistemic token frequency | How often model expresses uncertainty |
| Self-correction success rate | When model says "wait" and changes approach, does accuracy improve? |
| Confidence-accuracy correlation | Does higher stated confidence → higher actual accuracy? |

### 1.6 Execution Plan

```
Step 1: Kill old processes, clean old data on compute
Step 2: Generate Metacognitive SFT data (GPT-5.4, this VM, new 3-phase format)
Step 3: Prepare Base SFT data (correct rollouts only, standard CoT)
Step 4: Train Base SFT model (compute, 4x A100)
Step 5: Train Metacognitive SFT model (compute, 4x A100)
Step 6: Verify Meta SFT model produces 3-phase output at inference
Step 7: Train/verify Gnosis probe + ECE calibration
Step 8: Run RL-A (R_correct only)
Step 9: Run RL-B (R_correct + R_calib)
Step 10: Run RL-C (R_correct + R_calib + R_epistemic)
Step 11: Run RL-D (R_calib only)
Step 12: Evaluate all 6 models on MATH/AIME
Step 13: Analyze results, compare to hypotheses
```

---

## Stage 2: Test-time Self-directed Learning (After Stage 1 succeeds)

**Question**: Can the metacognitive model improve at test time without additional training?
- Model recognizes "I'm weak at geometry" → changes reasoning strategy in real-time
- No gradient updates — purely inference-time adaptation

## Stage 3: Gnosis Re-training + Curriculum (After Stage 1,2)

**Question**: As the model improves through RL, does the probe need re-training?
- Probe was trained on pre-RL hidden states
- After RL, hidden state distribution shifts
- May need periodic probe re-training during RL (every N steps)

**Question**: Can the model design its own training curriculum?
- Model identifies weak categories → selects problems from data pool
- Fine-tune on selected problems → re-evaluate
- Compare self-directed curriculum vs random sampling vs difficulty-ordered

---

## Infrastructure

- Compute: 4x A100 80GB (Azure ML, tunnel: skilldiscovery)
- TRAPI: GPT-5.4 for data generation (this VM only, AzureCliCredential)
- WandB: gistdslab/metacot-math
- GitHub: iamseungpil/metacognition-math
- Base model: Qwen2.5-7B-Instruct
- Probe: SimpleCorrectnessProbe (AUROC 0.9652)
