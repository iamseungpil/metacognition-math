# Survey: RLSD-family and Adjacent Methods for Self-Distilling Metacognitive Signals

**Date**: 2026-04-17
**Context**: Scoping the space around EAD ("self-distillation that preserves aligned epistemic shift at meta tokens")
**Coverage**: 12 core methods + 5 theoretical-adjacent threads, all arXiv IDs verified via WebSearch.

---

## Core RLSD-family methods

### RLSD — Self-Distilled RLVR (arXiv, 2026)

**Self-Distilled RLVR** (arXiv:2604.03128, Apr 2026)
- Intent: Combine RLVR directional signal with self-distillation magnitude signal for stable fine-grained updates.
- Method: Use environment reward to set the sign of each token update (reinforce vs. penalize). Use a privileged-teacher (same model, conditioned on ground truth or reference) as a *magnitude evaluator*: the teacher's token-level policy difference vs. student gives a per-token "evidence ratio" that modulates update size. Direction ← verifier, magnitude ← teacher.
- Signal scope: **token-level**
- Teacher type: **privileged (self-conditioned on reference)**
- Key eq: `∇J_t ∝ sign(r_env) · (log π_teacher(y_t|·) − log π_student(y_t|·)) · ∇ log π_student(y_t|·)`
- Limitation: Needs a privileged teacher pass per rollout (2× forward cost); only tested on Qwen3-VL-8B, unclear on pure-text long reasoning.
- Distinction: RLSD is the closest published antecedent. It provides the blueprint (direction/magnitude split, teacher-as-evaluator) but does **not** restrict updates to meta-spans or define a metacognitive objective — it treats every token uniformly under one global verifier.

There is no earlier canonical "RLSD" paper. Some earlier GitHub/blog usages call SDPO "RLSD," but the 2604.03128 paper is the first formal *Self-Distilled RLVR* formulation. Two adjacent formulations complete the canonical picture:
- **SDPO** (arXiv:2601.20802, "Reinforcement Learning via Self-Distillation"): feedback-conditioned self-teacher distilled back into policy.
- **OPSD** (arXiv:2602.04942, "Privileged Information Distillation for LMs"): RL with reverse-KL toward a PI-conditioned teacher.

### REDI — Reinforcement Distillation (arXiv:2505.24850, May 2025)
- Intent: Recover learning signal from *negative* distilled traces that SFT pipelines normally discard.
- Method: Two-stage offline pipeline. Stage 1: SFT on positive traces. Stage 2: asymmetric DPO-like loss that uses both positive and negative traces to shape the policy toward correct reasoning.
- Signal scope: **full trace (response-level)**
- Teacher type: **external (distilled teacher traces, e.g., DeepSeek-R1)**
- Key eq: `L_REDI = L_SFT(pos) + β · [ log π(pos) − λ log π(neg) ]` (stage-2 asymmetric)
- Limitation: Offline only; does not localize *which token* made the trace wrong.
- Distinction: Operates at response granularity with an external teacher; has no notion of meta-tokens or per-token epistemic state.

### Self-Rewarding LLMs (arXiv:2401.10020, Yuan et al., Jan 2024)
- Intent: Remove the human reward-model ceiling by using the model itself as judge.
- Method: The LLM generates responses and, via LLM-as-a-Judge prompting, scores them. Preference pairs from these self-scores feed Iterative DPO. Both instruction following and judgment quality co-improve.
- Signal scope: **response-level**
- Teacher type: **self (same-model judge)**
- Key eq: Iterative DPO on `(x, y_chosen, y_rejected)` with labels from `π_θ` acting as judge.
- Limitation: Judge saturates after ~3 iterations (later fixed by Meta-Rewarding).
- Distinction: Response-level preference, not token-level; no privileged teacher; no epistemic localization.

### Meta-Rewarding LLMs (arXiv:2407.19594, Wu et al., Jul 2024)
- Intent: Break self-rewarding saturation by making the model judge its own judgments.
- Method: Adds a third "meta-judge" role. The model plays actor → judge → meta-judge. Meta-judge preferences update the judge, which then supplies better actor rewards in iterative DPO.
- Signal scope: **response-level** (plus implicit judge-quality)
- Teacher type: **self (hierarchical)**
- Key eq: Same iterative DPO skeleton, with judge fine-tuned on meta-judge preference pairs.
- Limitation: Requires careful length control; risk of judge collusion with actor.
- Distinction: "Meta" here is about judging the *judge*, not about a metacognitive span inside the trace. Orthogonal to EAD.

### SPPO — Self-Play Preference Optimization (arXiv:2405.00675, Wu et al., May 2024)
- Intent: Frame alignment as a constant-sum two-player game seeking a Nash equilibrium of preferences, avoiding Bradley-Terry assumptions.
- Method: At each round, current policy plays against itself; a preference oracle (e.g., PairRM) labels pairs; policy is updated via a multiplicative-weights-style objective that provably approaches Nash.
- Signal scope: **response-level**
- Teacher type: **self (prior iterate)**
- Key eq: `π_{t+1}(y|x) ∝ π_t(y|x) · exp(η · (P(y ≻ · |x) − 1/2))`
- Limitation: Needs an external preference oracle; not token-localized.
- Distinction: Game-theoretic self-distillation with no meta-span notion.

### Iterative DPO (various, 2023–2024)
- Intent: Close the loop between pair-mining and policy update.
- Method: Sample `k` responses per prompt, rank with a reward model (or LLM judge), form `(chosen, rejected)` pairs, run DPO, repeat.
- Signal scope: **response-level**
- Teacher type: self / external reward model
- Key eq: `L_DPO = − log σ( β (log π/π_ref|chosen − log π/π_ref|rejected) )`, iterated.
- Limitation: No per-token credit; coarse signal that can eat epistemic hedging as collateral.
- Distinction: Baseline backbone used by Self-Rewarding, Meta-Rewarding, SPPO. Does not preserve epistemic shifts.

### RLCD — RL from Contrastive Distillation (arXiv:2307.12950, Yang et al., Jul 2023 → ICLR 2024)
- Intent: Generate preference pairs *without human labels* by prompting the same model with contrasting principles.
- Method: Two prompts (positive/negative) induce paired outputs; these form preference data to train a reward model, then PPO.
- Signal scope: **response-level**
- Teacher type: **contrastive (self via prompt conditioning)**
- Key eq: `(y⁺, y⁻) ∼ π(·|x, prompt⁺), π(·|x, prompt⁻)` → reward model → PPO.
- Limitation: Signal clarity depends entirely on prompt contrast quality.
- Distinction: This is the cleanest precedent for using *contrastive prompting* to synthesize preference pairs. It is response-level, unprivileged, and does not target meta-spans.

### DistiLLM / DistiLLM-2 (arXiv:2402.03898, 2503.07067, ICML 2024/2025)
- Intent: Make white-box LLM distillation stable and contrastive.
- Method: DistiLLM uses skew-KLD and adaptive student-generated-output blending. DistiLLM-2 adds a contrastive objective: *increase* likelihood of teacher answers, *decrease* likelihood of student's own mistakes.
- Signal scope: **token-level (full trace)**
- Teacher type: **external (larger model)**
- Key eq (DistiLLM-2): `L = KL(π_T ‖ π_S on y_T) − α · KL(π_T ‖ π_S on y_S)` (contrastive pairing).
- Limitation: Assumes a larger, separate teacher; no RL or verifier.
- Distinction: Token-level, but every token treated equally — no meta-span selection, no verifier-driven direction.

### SDPO — Self-Distillation Policy Optimization (arXiv:2601.20802, Jan 2026)
- Intent: Convert rich textual feedback (compiler errors, judge notes) into dense token-level signal without a reward model.
- Method: Run rollout → inject textual feedback into the context → re-run the same model to get a "feedback-conditioned" next-token distribution → distill that distribution back into the unconditioned policy.
- Signal scope: **token-level**
- Teacher type: **self (conditioned on feedback)**
- Key eq: `L_SDPO = KL( π_θ(·|s, feedback) ‖ π_θ(·|s) )` applied across the rollout.
- Limitation: Requires feedback source; conditioning prompt engineering is load-bearing; no meta-span localization.
- Distinction: Uses self-teacher for token-level magnitudes (like RLSD) but drives *direction* from conditioning rather than a verifier — and again, no meta-span scoping.

### ReST-EM (arXiv:2312.06585, Singh et al., Dec 2023; ICLR 2024 workshop)
- Intent: Scale self-training past human data using verifiable feedback (math/code).
- Method: E-step samples many completions, filters by correctness; M-step fine-tunes on the filtered set; repeat a few rounds.
- Signal scope: **full trace (filter-based)**
- Teacher type: **self (prior iterate)**
- Key eq: `π_{t+1} = arg min E_{(x,y) ∼ D_t^+}[−log π(y|x)]`, where `D_t^+ = {y ∼ π_t : R(y)=1}`.
- Limitation: Binary filter discards all partially-correct traces; no credit-assignment.
- Distinction: Coarsest possible signal; no meta-span awareness.

### MiniLLM (arXiv:2306.08543, Gu et al., 2023/2024)
- Intent: On-policy distillation from a larger teacher without mode-covering pathologies of forward-KL.
- Method: Replace forward-KL with *reverse-KL* and derive a policy-gradient estimator (PG-KL), so the student samples and learns to avoid low-prob teacher regions.
- Signal scope: **token-level (on-policy)**
- Teacher type: **external (larger model)**
- Key eq: `L = KL(π_S ‖ π_T)`, gradient estimated on-policy via PG.
- Limitation: Teacher has to be strictly larger; still response-wide signal weighting.
- Distinction: On-policy distillation precedent, but no verifier, no meta scoping.

### MATH-Shepherd (arXiv:2312.08935, Wang et al., ACL 2024)
- Intent: Build a *process* reward model for math without human step annotations.
- Method: For each intermediate step, run MCTS-like rollouts from it; label the step by empirical rate of reaching a correct final answer. Train PRM on these labels; use PRM for step-level PPO and for verification reranking.
- Signal scope: **span-level (per reasoning step)**
- Teacher type: **external (PRM)**
- Key eq: `r_step(s_i) = P(correct | complete from s_i)`, plugged into PPO advantage at step boundaries.
- Limitation: Expensive rollout tree for every step; step granularity is heuristic.
- Distinction: Span-level signal — a precedent for *not* treating all tokens equally — but span = reasoning step, not metacognitive commit; signal comes from MCTS rollouts, not a per-token Bayes factor.

---

## Theoretical-adjacent threads

### Proper scoring rules for LLM calibration (Brier / log-score)
Closest formal instantiation: **RLCR — Beyond Binary Rewards** (arXiv:2507.16806, Jul 2025). Replaces binary correctness reward with `R = 1{correct} − λ · Brier(confidence, correct)`, training the model to emit verbalized confidence that is both accurate and calibrated. Provides the proof that Brier (and log-score) are *strictly proper* scoring rules, so the unique optimum is truthful reporting.
- Relevance to EAD: Supplies the theoretical justification for using a proper-scoring-rule reward on commit-quality — if EAD uses Brier / log-score on a commit token, the optimum is guaranteed to preserve (not collapse) epistemic state.

### Token-level credit assignment / control-critical masking
Key references: *Ignore the KL Penalty! Boosting Exploration on Critical Tokens* (arXiv:2502.06533), *TEMPO — tree-structured credit* (arXiv:2509.18314), *GTPO/GRPO-S* (OpenReview 2025). Common finding: **only a small fraction of tokens are causally critical** for the final reward, and treating them specially (higher exploration, branch-gated TD, or masked loss) yields 3–8pp gains. This is the direct mechanistic argument for why a *meta-span mask* should help: it aligns the expensive learning signal with the causally decisive tokens.

### Privileged-teacher distillation (Li et al. teacher–student)
Primary reference: **Privileged Information Distillation for LMs** (arXiv:2602.04942). Introduces π-Distill (joint teacher-student objective, teacher conditioned on PI like ground truth or reference CoT) and OPSD (on-policy RL with reverse-KL to the PI-teacher). Core theoretical point: what matters is *informativeness* of the PI, and if teacher and student stay too close in KL, they collapse (optimization stalls). This is the analytical backbone RLSD builds on.

### Entropy-shaped loss (Four Habits PPO working note, EPO line)
EPO (arXiv:2509.22576) and Policy Split (arXiv:2604.11510) regularize policy entropy with *trajectory-level* and *phase-weighted* terms to avoid premature convergence and late chaotic exploration. The Four-Habits working note (local project) observed that free-text confidence tokens drop out under uniform PPO because the reward is identical with-or-without the hedge (same reward → tokens are overhead the policy deletes). The entropy-shaped loss family is the general class of remedies: prevent the policy from collapsing entropy *at the very spans where uncertainty matters*. EAD's meta-span mask is a structural variant: only penalize entropy drop *where epistemic state is supposed to live*.

### Commit-quality / structural constraint losses
The literature closest to "commit-quality loss" is process-reward (MATH-Shepherd, PRMs) and constrained-generation (CRANE, Const-o-T). No published method directly optimizes a *commit-quality* score on a designated meta-span with a proper scoring rule. The closest structural precedent is RLCR, but RLCR places calibration at the response level (a single scalar confidence), not at a span inside the trace.

---

## Focus-question answers

### 1. Canonical RLSD formulation — single paper or synthesis?

**Synthesis of three papers, one of which (Self-Distilled RLVR, 2604.03128) is the explicit "RLSD" name**. The family is:
- **Self-Distilled RLVR (RLSD, 2604.03128)** — direction from verifier, magnitude from self-teacher evidence ratio.
- **SDPO (2601.20802)** — feedback-conditioned self-teacher as magnitude source.
- **OPSD / π-Distill (2602.04942)** — privileged-teacher with reverse-KL RL.

All three appeared in Jan–Apr 2026 and share the core idea: "use a same-model teacher (privileged or feedback-conditioned) to produce dense token-level weights while keeping a clean directional signal." RLSD is the cleanest name; SDPO and OPSD are complementary reference points.

### 2. Signal placement — categorization

| Granularity | Methods |
|---|---|
| Full-trace / coarse filter | ReST-EM, REDI (offline), MiniLLM (response) |
| Response-level (pairwise) | Self-Rewarding, Meta-Rewarding, SPPO, Iterative DPO, RLCD, RLCR |
| Span / step-level | MATH-Shepherd (reasoning step), constrained generation (structural span) |
| Token-level | RLSD, SDPO, DistiLLM-2, MiniLLM (PG-KL), critical-token methods (KL-relax, TEMPO) |

No published method places the signal specifically on a **meta-span** (a designated metacognitive commit window inside the trace). The closest is MATH-Shepherd at the step boundary, but the step boundary is content-defined, not epistemic-role-defined.

### 3. Privileged teacher + contrastive pair combo

Exact combination (privileged-teacher AND contrastive-pair):
- **RLSD** (privileged teacher as magnitude, verifier sign as direction) — uses a *directional contrast* but not a paired contrastive loss.
- **DistiLLM-2** (contrastive: up teacher, down student-mistake) — but teacher is external, not privileged.
- **RLCD** (contrastive pairs) — but teacher is a prompt-conditioned contrast, not a privileged-information teacher.
- **π-Distill / OPSD** (privileged teacher) — no contrastive pair.
- **HDPO** (arXiv:2603.23871, "Hybrid Distillation Policy Optimization via Privileged Self-Distillation") — combines privileged self-distillation with a policy-optimization pair; closest hybrid found.

So the **cleanest published combo of (privileged teacher) + (contrastive pair)** is **HDPO (2603.23871)**, and the weaker combo of (privileged teacher) + (directional contrast with verifier) is **RLSD (2604.03128)**. No paper combines all three of {privileged teacher, contrastive pair, per-token Bayes-factor weight}.

### 4. Is "meta span + privileged teacher + per-token Bayes factor" already published?

**No — not as a single combination.**

Component coverage in the literature:
- meta-span / metacognitive localization: *partial* — step-level (MATH-Shepherd), critical-token (2502.06533), verbalized confidence spans (RLCR) — but none define a named "meta span" and mask the loss to it.
- privileged teacher: covered by π-Distill/OPSD (2602.04942), RLSD (2604.03128).
- per-token Bayes factor (teacher-vs-student evidence ratio on a span): RLSD (2604.03128) uses the teacher-student log-ratio as an update *magnitude*, which is functionally a Bayes factor, but applies it to **all** tokens, not a meta-span.

The precise triple {masked to meta-span} × {privileged teacher} × {per-token Bayes factor} is, to the best of this search, novel. RLSD is the closest precedent and differs only in scope (global vs. meta-span) and in what role the Bayes factor plays (magnitude only, vs. full update signed by calibration).

### 5. Theoretical basis for EAD's "EV alignment preservation, not answer imitation"

**Supporting evidence**:
- **RLCR (2507.16806)**: proves that strictly proper scoring rules (Brier, log-score) uniquely reward truthful probability reporting — imitation-only losses (MLE, reverse-KL to a confident teacher) do not preserve calibration.
- **"The paradox of LLM self-distillation" (2026 techtalk, referencing empirical findings)**: self-distillation *suppresses* epistemic verbalization because privileged teachers conditioned on the answer produce over-confident traces; forcing the student to imitate collapses hedging tokens.
- **Token-level uncertainty objective (2503.16511)**: shows epistemic uncertainty aligns with predictive loss; uniform MLE over-fits low-uncertainty regions and washes out aleatoric structure. Proposes masked MLE + self-distillation to *preserve* uncertainty on non-masked tokens. This is the closest published theoretical match to EAD's claim.
- **π-Distill (2602.04942)**: warns that teacher-student KL collapse kills learning — motivates keeping a *distributional gap* (i.e., preserving the student's own epistemic state) rather than matching.

**Potentially contradictory**:
- **MiniLLM (2306.08543)**: argues reverse-KL is preferred precisely because it *avoids* mode-covering over the teacher distribution — a different stance than EV-preservation but compatible (both reject naive forward-KL imitation).
- **Iterative DPO / SPPO**: implicitly treat alignment as response-level preference ordering; say nothing about token-level EV. Not contradictory, just orthogonal.

**Net**: EAD's claim has direct theoretical support from (a) proper scoring rules (RLCR), (b) token-level uncertainty preservation (2503.16511), and (c) the empirically observed self-distillation-kills-hedging phenomenon. No paper published in 2024–2026 *contradicts* the claim; several adjacent methods (MiniLLM, π-Distill) agree with the weaker version "imitation is insufficient."

---

## Summary table (all arXiv IDs verified)

| Method | arXiv | Scope | Teacher |
|---|---|---|---|
| Self-Distilled RLVR (RLSD) | 2604.03128 | token | privileged self |
| SDPO | 2601.20802 | token | feedback-conditioned self |
| REDI | 2505.24850 | response | external |
| Self-Rewarding | 2401.10020 | response | self-judge |
| Meta-Rewarding | 2407.19594 | response | self-judge + meta-judge |
| SPPO | 2405.00675 | response | self (prior iterate) |
| RLCD | 2307.12950 | response | contrastive-prompt self |
| DistiLLM-2 | 2503.07067 | token | external contrastive |
| MiniLLM | 2306.08543 | token | external |
| ReST-EM | 2312.06585 | trace-filter | self |
| MATH-Shepherd | 2312.08935 | step | external PRM |
| π-Distill / OPSD | 2602.04942 | token | privileged self |
| HDPO | 2603.23871 | token | privileged self + pair |
| RLCR | 2507.16806 | response + confidence | self (Brier reward) |
| Token-uncertainty objective | 2503.16511 | token mask | self |
| Critical-token KL relaxation | 2502.06533 | token | — (exploration) |
| EPO | 2509.22576 | trajectory entropy | — |

**Word count**: ~2,420.
