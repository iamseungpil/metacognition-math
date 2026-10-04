# Asymmetric Counterfactual Meta-RL — Design Spec

**Date**: 2026-06-25  **Project**: metacognition-math (confidence-rv line)
**Status**: design approved (hierarchical gate+content), pending user spec review → writing-plans

---

## 1. Intent (north-star)

Train a model that **emits metacognition ONLY when it raises accuracy, and is actively suppressed when it would lower it** — i.e. learn the *timing* of meta from its *counterfactual outcome*, not from emitting behavior-shaped text.

Concrete goal: **raw accuracy ≥ base GRPO** (close the verified ~7-9pp gap) by (a) killing the derailment that costs accuracy and (b) keeping only the meta that genuinely saves wrong answers.

## 2. Why (verified findings driving this)

1. **Meta-CoT costs ~7-9pp vs base GRPO**, robustly, with *correct* grading (math_verify): CF 81 / 가산 83 / e21r 82 vs **base 90**. Holds on *matched* hard-inclusive data (base vs e21r) → it is the **meta, not the data difficulty**. (memory: math500-grader-broken-aime-premature-assertion-0625)
2. **The grader (eval_hf.check_correctness) is broken** — 26% false-neg on math500 even on base. ALL accuracy must be re-measured with math_verify.
3. **Failure mode = derailment, not absence of habits**: CF does MORE inline cognitive behaviors than base (verify 90% vs 8%) but *performatively* — correlated with its own errors (can't catch), and *over-verification derails* (CF 9065 chars wrong on a problem base solves in 1696). (memory: meta-fails-root-correlated-self-verification-0625)
4. **The meta token block is inert** (habits 0-3% inside it) — decoration, not the cause. Real behavior is inline.
5. **Root ceiling**: self-verification is correlated with the model's own errors (same distribution) → meta confirms blind spots. RL rewards the *shape* of meta, not an independent-checking *capability*.

**Therefore**: reward the *counterfactual outcome* of emitting meta (save vs derail), asymmetrically (penalize derail harder), gate emission on confidence, and (when emitted) reward only *independent* verification.

## 3. Design — 2-layer hierarchy

### Layer 1 — GATE (timing): asymmetric counterfactual
Per problem, estimate c₀ = P(correct | meta-OFF) and c₁ = P(correct | meta-ON) from k rollouts each. Reward the **emit decision** (the `<|meta|>`-open token), continuous asymmetric form:

```
R_gate = α·max(0, c₁ − c₀)          # SAVE (0→1): reward
       − β·max(0, c₀ − c₁)          # DERAIL (1→0): STRONG penalty, β > α
       − γ·1[c₀ ≈ c₁ ≈ 1]·overhead  # WASTE (1→1): small penalty for needless emission
       #  NEUTRAL (0→0): ~0
```
- **β > α** (asymmetric — derail hurts more than save helps): default β ≈ 2-3·α.
- **confidence-gated**: student self-confidence is a feature of the emit policy; high-confidence → suppress emission (derail-risk region). Reuses our calibration strength (AIME overconf 97→14%).
- **default-OFF / opt-in**: meta is not always-on; emitting must be net-positive to be learned.

### Layer 2 — CONTENT (quality): independence, conditional on emit
ONLY when the gate emits (and was net-positive), reward the meta *content* by whether it is a **genuine independent verification**, not performative re-assertion:
- **independent** = produces information the forward pass lacked: substitution-back, alternative method, dimensional/numeric plug-in. Re-derivation of the same path → credit 0.
- Reuse decoy-DiD / PMI machinery (does meta favor gold) **refined to require a DIFFERENT path** (penalize meta that merely restates the forward reasoning).
- Addresses the correlated-error ceiling at the content level.

### Composition (DCPO)
- `R_gate` → emit-decision token; `R_content` → meta tokens, **gated by Layer 1** (wrong-to-emit ⇒ content reward 0).
- Hierarchical: Layer 1 decides *whether*, Layer 2 scores *what* — only when whether=yes-and-correct.

## 4. Harmful-meta suppression (6 mechanisms)

1. **Continuous asymmetric counterfactual** (β>α) — primary; robust to k-noise vs hard quadrants.
2. **Confidence gate** — suppress emission where the model is likely already right (derail-risk).
3. **Length-aware waste penalty** — penalize meta that inflates tokens without changing correctness (the over-verification derail signature).
4. **Default-OFF / opt-in structure** — emission must earn its place.
5. **Margin + clip + emit-floor** — DERAIL needs a margin (c₀−c₁ > τ) to count (noise guard); β clipped and a small emit-floor to avoid total abstention collapse (cf_group gs50 precedent).
6. **Frontier-hard data** (see §5) — SAVE cases must exist or suppression → meta death.

## 5. Data (must be co-redesigned)

The counterfactual signal (SAVE/DERAIL) is dense only where the model is wrong ~half the time. Current rv data (in-train acc 0.89, 0% hard) is ~all WASTE → no signal.
- **Build a frontier-hard RL set**: problems with pass-rate ≈ 0.2-0.8 under the current model (the "hard-wrong-but-solvable" band; B.1: 76% of hard-wrong solvable at 16k). Source: MATH-500-frontier / hendrycks_math L4-5 / omni-math, filtered by measured pass-rate.
- **Preserve hard-math capability**: do NOT SFT-degrade the base; either mix hard into any priming or start RL from a strong base (base GRPO-class), not a weak easy-only SFT.
- **Grading = math_verify** everywhere (eval_hf.check_correctness is retired for measurement).

## 6. Risks & safeguards

| Risk | Safeguard |
|---|---|
| Total abstention collapse (cf_group) | emit-floor + bounded β + ensure real SAVEs in data |
| Counterfactual noise (small k) | continuous form + DERAIL margin τ + adequate k |
| Inert reward (gs190 trap) | production-parity TDD: assert R_gate/R_content actually route in `compose_dcpo_region_advantage`; unit tests on the 4 regimes |
| Signal sparsity on easy data | frontier-hard data (§5) is a hard prerequisite |
| Grader corruption | math_verify only |

## 7. Experiment design (autoresearch)

Arms (on frontier-hard data, math_verify grading):
- **A0**: CF baseline (current best, +0.040 held-out) — control.
- **A1**: asymmetric-CF, **gate-only** (Layer 1).
- **A2**: asymmetric-CF, **gate + content** (full hierarchical).

Decision metrics (held-out, math_verify):
- ★**derail rate (1→0)** ↓ (primary new metric — direct target).
- **emit selectivity** (not always-on; ≈ base's sparse use).
- **held-out Δ** (acc_with − acc_without) > CF +0.040.
- ★**absolute acc vs base GRPO** (the real goal — does the gap close?).
- secondary: epistemic preserved, length not inflated.

## 8. Success criteria

- **S1**: derail rate < CF's, with emit selective (not always-on).
- **S2**: held-out Δ ≥ +0.04 and net-positive on hard stratum.
- **S3** (north-star): absolute acc (math_verify) closes ≥half the gap to base GRPO on the frontier set, without regressing easy.

## 9. Open questions (resolve in plan)

- Exact α/β/γ ranges (sweep) and DERAIL margin τ.
- Independence detector for Layer 2: behavioral (substitute/alt-method markers + answer-change) vs decoy-DiD-different-path vs second-independent-solve.
- Confidence-gate mechanism: explicit confidence token threshold vs learned.
- Whether to keep the `<|meta|>` region (inert) or restructure to inline opt-in.
- k (rollouts per arm) for stable c₀/c₁.

## 10. Sequencing

frontier-hard data build (math_verify) → reward impl (gate+content, TDD, production-parity) → smoke (intent-check via karpathy iterate) → autoresearch A0/A1/A2 on H100 → held-out math_verify eval + derail-rate + absolute-vs-base.
