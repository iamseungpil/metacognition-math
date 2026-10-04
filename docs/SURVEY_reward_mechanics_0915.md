# Reward mechanics of RL for metacognition — a survey at formula level

Compiled 2026-09-14 for the cd9 round (Qwen3-4B-Instruct-2507, verbal `<|meta|>` span,
self-distilled labels only). Every arXiv id below was verified by fetching the abs page.
Rows marked **[unverified]** could not be read at full text; their formulas are from the
abstract only and must not be trusted at coefficient level.

---

# Part 0 — ID verification ledger

| As given in the brief | Verdict |
|---|---|
| 2607.11881 Metacognition in LLMs | ✅ exact title "Metacognition in LLMs: Foundations, Progress, and Opportunities" (Liu, Gani, Lu, Thomas, Steyvers, Cohan; 13 Jul 2026) |
| 2406.01297 Kamoi et al. | ✅ "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs", TACL 2024 |
| 2509.06870 "AggLM" | ⚠️ id correct, **title wrong**: actual title is "The Majority is not always right: RL training for solution aggregation" (AggLM is the method name) |
| 2410.08146 | ✅ Setlur et al., "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning", ICLR 2025 — the id the brief left blank |
| 2510.03259 "MAPR" | ✅ id correct; title "Verifying Meta-Awareness via Predictive Rewards in Reasoning Models" (also circulated as "Meta-Awareness Enhances Reasoning Models: Self-Alignment RL"; method = MASA) |
| 2509.26626 RSA | ✅ "Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models" |
| 2408.15240 GenRM · 2402.06457 V-STaR · 2403.09629 Quiet-STaR | ✅ all three match |
| 2507.16806 "RLCR / Learning to Reason about Uncertainty" | ⚠️ id correct, **title wrong**: actual title is "Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty" (Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas, MIT) |
| 2602.22751 "EGPO" | ⚠️ **id resolves but the title is not EGPO** — it is "Know What You Know: Metacognitive Entropy Calibration for Verifiable RL Reasoning" (26 Feb 2026). EGPO is the method name inside. **And it is gold-supervised**, so it does not satisfy the label-free constraint. |
| 2605.25507 "Self-Reset PO" | ⚠️ **id resolves but the title is not that** — it is "Credit Assignment with Resets in Language Model Reasoning" (25 May 2026). SRPO = Self-Reset Policy Optimization is one of two methods inside. |
| 2608.23493 "SRPO self-reflective" | ✅ id resolves; title "SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning" (24 Aug 2026). ⚠️ **Three-way acronym collision** with 2605.25507's SRPO and with 2506.01713 "SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware RL". **Cite by id, never by acronym.** |
| "Reflect, Retry, Reward" (id not given) | ✅ **arXiv:2505.24726** (Writer, 30 May 2025) |
| "RLPR" (id not given) | ✅ **arXiv:2506.18254** "RLPR: Extrapolating RLVR to General Domains without Verifiers" |
| 2508.00410 Co-rewarding | ✅ id correct; full title "Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in LLMs" (cited elsewhere as Co-Reward / CoReward) |

---

# Part 1 — The surveys and their own taxonomies

The point of this section is to show the field's own organizing categories before
imposing mine. Four taxonomies matter for cd9; two of them are directly usable as a
design space.

## 1.1 Metacognition in LLMs — Liu et al., arXiv:2607.11881 (Yale/UCI, Jul 2026)

The only survey that is *about* metacognition rather than about a neighbouring
mechanism. Its decomposition:

- **Knowledge** — declarative (what one knows about oneself as a learner), procedural
  (how strategies apply), conditional (*when and why* a strategy suits a task).
- **Regulation** — planning (goal-setting, strategy selection), evaluation (reflecting
  on strategy effectiveness).
- **Experience** — "subjective internal states — such as feelings of knowing, judgments
  of learning, or confidence".
- **Core loop: Monitoring ↔ Control.**

Section structure (verbatim headings): §4.1 "Measuring Metacognition in LLMs" (six
measurement families: psychologically-grounded / meta-d′, neurofeedback-based,
confidence-based, interpretability-based, task-specific, benchmarks); §4.2 "Current
Findings"; §5 "Giving LLMs Metacognitive Abilities" (5.1 Implementations, 5.2
"Metacognition for Reasoning Models", 5.3 "Metacognition for LLM Agents"); §6
"Metacognitive Methods to Improve Capabilities of LLMs".

**Its taxonomy of interventions that improve metacognition:** prompting/instruction
(metacognitive prompts, structured frameworks DCO/PCE/MGV); training (SFT on augmented
traces, RL, DPO); architecture/module (State Stream Transformer, SAGE-nano,
monitor–generate–verify frameworks); inference-time (Buffer of Thoughts, behavior
handbooks, early termination); calibration/signal training (training on **self-generated
correctness signals**, difficulty assessment, pass-rate prediction alignment).

**What it lists under RL** — and this is the whole of it: Kim et al. (= MASA/MAPR,
2510.03259) "align predicted statistics about forthcoming generations"; Yeom et al.
"incorporate self-signals regarding answer correctness"; Wang et al. "apply SFT and RL
on self-critiqued reasoning traces"; Leong et al. "fine-tuning on metacognitive QA
pairs"; Pangu Embedded; MIRA.

⚠️ Two observations worth carrying into the cd9 design. First, **the survey does not
treat RL as a standalone metacognition-enhancement method** — RL appears only as a
vehicle for injecting self-generated signals. There is no established recipe to copy;
the space the brief is aiming at is genuinely thin. Second, the survey warns that
**RLHF post-training systematically alters internal uncertainty** and can "degrade
metacognitive efficiency on STEM tasks" — i.e. the training that makes a model helpful
is itself a confound on any metacognition measurement taken after it.

## 1.2 A Survey of RL for Large Reasoning Models — arXiv:2509.08827

The most useful taxonomy in the set, because it is organized by *reward object*, which
is exactly the axis the brief asks for. Verbatim headings under §3.1 Reward Design:

- **3.1.1 Verifiable Rewards** — rule-based rewards, rule-based verifiers.
- **3.1.2 Generative Rewards** — model-based verifiers, assessment-based rewards; the
  "Co-Evolving Systems" subsection is where self-rewarding / self-critique lives
  ("a single model generates its own training signals … alternates between policy and
  verifier roles … performs self-correction based on its own critique").
- **3.1.3 Dense Rewards** — token-level, step-level, turn-level.
- **3.1.4 Unsupervised Rewards** — split into **model-specific** and **model-agnostic**:
  - model-specific / *output consistency*: EMPO, TTRL ("operationalize this via
    clustering and majority voting");
  - model-specific / *internal confidence*: negative entropy (EM-RL, RENT), generation
    probabilities (Intuitor, RLSC, RLSF);
  - model-specific / *self-generated knowledge*: self-rewarding, self-instruction;
  - model-agnostic: heuristic (length, format) and data-centric rewards.
- **3.1.5 Reward Shaping** — rule-based and structure-based.

On hacking it is blunt: model-based dense rewards "are vulnerable to reward hacking";
heuristic rewards "can be gamed … leading to superficial improvements without advancing
true capability"; and the unsupervised section closes with "both approaches … remain
susceptible to reward hacking".

**Read against the brief: §3.1.4 is the entire gold-free menu the field currently has —
output consistency, internal confidence, self-generated knowledge — and the survey's own
verdict on all three is "susceptible to reward hacking."**

## 1.3 When Can LLMs Actually Correct Their Own Mistakes? — Kamoi et al., TACL (2406.01297)

Two axes. **Feedback source**: intrinsic / external information / fine-tuning. **Framework
realism**: realistic vs unrealistic (does it use accessible information?) crossed with
fair vs unfair (were the initial responses best-possible?), giving *intrinsic*,
*fair-asymmetric*, *unfair-asymmetric*.

Three findings: (1) the bottleneck is **feedback generation**, not refinement — models
refine well given reliable feedback but cannot produce reliable feedback on themselves;
(2) no prior work demonstrates successful intrinsic self-correction outside tasks
"exceptionally suited" to it; (3) success comes from **reliable external feedback** or
**large-scale fine-tuning** (>100K instances, usually with feedback distilled from a
stronger model).

Its checklist item that bites hardest here: "**Not using oracle information, such as
ground-truth answers**", and its critique that prior work "uses ground-truth answers and
does not apply self-correction when the initial responses are correct, which unfairly
ignores mistakes caused by updating correct responses incorrectly." That is a direct
description of the trap in any redirect/verify arm gated on known-wrongness.

Its condition for RL-based self-correction to work: "tasks whose responses can be easily
evaluated **given ground-truth answers**." The survey does not know of a gold-free route.

## 1.4 Credit assignment in RL for LLMs — arXiv:2604.09459

Two axes: **granularity** (token / segment / step-turn / multi-agent) and
**methodology** (Monte Carlo, TD-value learning, LLM-as-critic, game-theoretic,
information-theoretic).

- Segment-level family named explicitly: **SPO** (semantic cutpoints), **TEMPO**
  (tree-structured, segment-gated TD), **SCAR** (segments as players, credit = Shapley
  value).
- Counterfactual family: **C3**, **CCPO** (leave-one-out over turns), **HCAPO**
  (hindsight counterfactual), **SCAR**.
- Methods that need no gold: **T-REG** (contrastive self-prompting, token-level
  log-prob differences between the model's *own* correct and incorrect solutions),
  **SPRO** (leave-one-out by masking steps), **CAPO** (LLM-as-critic natural-language
  critiques), **iStar/StepAgent** (implicit credit from a DPO-trained model).
- Its negative result on outcome-level GRPO: it "assigns identical credit to a pivotal
  tool selection and a trivial formatting action". PRMs are criticized for requiring
  verifiable intermediate states.

## 1.5 Uncertainty and calibration surveys

- **arXiv:2503.15850** (Liu et al., "Uncertainty Quantification and Confidence
  Calibration in LLMs: A Survey"): taxonomy by *computational efficiency* × *uncertainty
  dimension* (input / reasoning / parameter / prediction). Confidence-estimation
  families: (a) self-evaluation / LLM-as-a-judge, (b) sampling-consistency (multi-sample
  agreement, semantic clustering, conformal prediction), (c) logit/token-probability
  (perplexity, max log-prob, entropy, SAR), (d) training-based calibration (SAPLMA,
  supervised uncertainty estimation, UaIT, LoRA ensembles). ⚠️ There is **no RL category**
  — this survey predates the RL-calibration wave.
- **arXiv:2601.15690** (Zhang et al., ACL 2026, "From Passive Metric to Active Signal"):
  the survey that *does* have an RL section. §5.1 Robust Reward Models, §5.2
  Self-Improvement RL, §5.3 Scalable Process Supervision. Under confidence-as-intrinsic-
  reward it names **RLSF**, confidence maximization, **EM/RENT**, **Intuitor**. Under
  uncertainty-to-prevent-hacking: URM, UALIGN (feeds the policy's semantic entropy to the
  reward model), Bayesian RMs. Under sample selection: EDU-PRM (high-entropy tokens as
  "uncertainty anchors" for automatic segmentation). Under adaptive compute: UnCert-CoT,
  MUR, ThoughtTerminator; under uncertainty-triggered self-correction: UAG, SPOC,
  AdaptiveStep.
- **arXiv:2412.05563** (ACM CSUR): taxonomy of UQ methods, open challenges. No RL.

## 1.6 Test-time scaling / verification surveys

- **arXiv:2503.24235** — "what, how, where, how well to scale". Parallel (sample +
  select via a verification mechanism) / sequential (iterative refinement) / hybrid.
- **arXiv:2508.16665** "Trust but Verify! A Survey on Verification Design for Test-time
  Scaling": §3.1 Heuristic, §3.2 Discriminative, §3.3 Generative, §3.4 Reasoning-Based
  Generative, §3.5 Symbolic verifiers. ORM vs PRM distinction. On self-verification:
  "fine-tuning based or symbolic approaches have shown to be more robust and calibrated
  than simply asking LLMs."

## 1.7 Self-improvement surveys

- **arXiv:2412.14352** "A Survey on LLM Inference-Time Self-Improvement": independent
  (decoding/sampling) / context-aware / model-aided.
- **arXiv:2510.02665** "Self-Improvement in Multimodal LLMs: A Survey" (EMNLP 2025
  Findings): unifies self-evolution, self-training, self-consistency, self-correction,
  self-reflection, self-refinement under one head.

## 1.8 Cross-survey verdict on "what interventions improved metacognition, and which are RL"

Putting 1.1–1.7 together, the field's own list of interventions that *moved a
metacognitive metric* is: metacognitive prompting; SFT on self-generated correctness or
pass-rate signals; **RL with a proper scoring rule on a verbalized probability** (RLCR
lineage); **RL on self-predicted rollout statistics** (MASA); RL with rule-checkable
meta-tags (RLVMR); multi-turn self-correction RL with an explicit improvement bonus
(SCoRe, Reflect-Retry-Reward); learned verifiers used as rerankers (GenRM, V-STaR);
and the label-free family (TTRL, EMPO, Intuitor, Co-rewarding). **Of these, only MASA,
Quiet-STaR, RLVMR's tag term and the label-free family compute their metacognitive
signal without consulting a gold answer at reward time** — and of *those*, only
Quiet-STaR does so without gold anywhere in the pipeline.


---

# Part 2 — The reward-mechanics table

Format per row: **(1)** paper+id · **(2)** ability · **(3)** formula · **(4)** inputs ·
**(5)** where it applies · **(6)** group-centering · **(7)** ★gold-removal ·
**(8)** gain (accuracy vs calibration) · **(9)** failure modes.

## Group A — proper scoring rules on a stated probability (calibration)

### A1. RLCR — "Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty", arXiv:2507.16806
(Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas, MIT. v1 22 Jul 2025, v2 15 May 2026.)
- **(2)** calibration / confidence expression with an explicit **uncertainty-reasoning
  span**: the response is structured `<think>/<answer>/<analysis>/<confidence>`.
  Explicitly **not** an accuracy claim — accuracy *preservation* is the claim.
- **(3)** Eq. 8: **`R_RLCR(y, q, y*) = 1_{y≡y*} − (q − 1_{y≡y*})²`**, with the second
  term named `R_Brier`. A format reward is added on top; the paper says only "both
  format and calibration rewards are weighted equally" — **the format coefficient is
  [unverified]**. No clipping. **Theorem 1** generalizes to any bounded proper scoring
  rule `S` with `S(p,1) − S(p,0) < λ`; **log loss is explicitly ruled out**
  (Corollary 1: `S(p,1) − S(p,0) = log((1−p)/p) → ∞` as `p → 0`, so "the log loss does
  not incentivize correctness").
- **(4)** gold answer `y*` (twice — it appears in *both* terms); an equivalence checker
  (exact match on HotpotQA-Modified, `math-verify` on Big-Math); the model's own emitted
  scalar `q`. **No judge, no teacher, no K-rollout aggregation, no logits, no probe.**
  Qwen2.5-7B base, GRPO, no KL.
- **(5)** whole-sequence scalar. ⚠️ Despite the span structure, there is **no per-span
  credit assignment** — one scalar per rollout.
- **(6) ★** Varies within a group on **both** terms. This is the mechanically important
  property: where RLVR's advantage is identically zero on an all-correct or all-wrong
  group, **RLCR still has gradient because `q` varies across siblings.** It is the only
  calibration reward here that survives group-centering on a uniformly-wrong prompt.
- **(7) ★** Deleting `y*` kills the formula entirely — not a degraded reward, *no*
  reward. Substitute: swap `1_{y≡y*}` for a TTRL majority pseudo-label or leave-one-out
  sibling agreement. **The paper ran no gold-free ablation.** Best available proxy for
  the cost: Intuitor's GRPO-PV (plurality-vote) row — 0.820/0.636 vs gold GRPO
  0.826/0.636 on GSM8K/MATH500, i.e. **nearly free in-domain**. ⚠️ But a majority-vote
  label makes the Brier target *the group's own consensus*, so the model can satisfy it
  by collapsing `q` onto the empirical agreement rate — the calibration signal
  degenerates into self-consistency, which the paper's own Answer-Probability baseline
  already shows is weak (AUROC 0.72). The closest published attempt is arXiv:2604.24070
  (§N1 below): distilling self-consistency into verbal confidence on **Gemma-3-4B**,
  a **pre-registered negative result** with a post-hoc rescue.
- **(8) CALIBRATION, not accuracy, in-domain.** HotpotQA in-domain: RLCR 62.1% acc /
  AUROC 0.69 / Brier 0.21 / **ECE 0.03** vs RLVR 63.0 / 0.50 / 0.37 / 0.37 — **accuracy
  0.9 pt below RLVR**. HotpotQA→OOD avg: RLCR 56.2% vs RLVR 53.9% (+2.3 acc). Big-Math
  in-domain: 72.7% vs RLVR 72.9%. **Math→OOD: RLCR 50.9% vs RLVR 52.5% — RLCR *loses*
  accuracy**; SFT+RLCR collapses to 43.8%.
- **(9)** (a) hedging is anticipated and blocked by Theorem 1, not observed;
  (b) **SFT warmup causes catastrophic forgetting** (72.2 → 43.8 OOD);
  (c) OOD ECE stays high (0.21–0.25); (d) models "assign high confidence to multiple
  contradictory answers"; (e) per-question confidence std is low for a fixed answer —
  confidence is nearly a deterministic function of the answer, **limiting how much the
  `<analysis>` span can be doing**; (f) ⚠️ **Appendix G: a 7B classifier trained on RLCR
  CoTs does not beat one trained on RLVR CoTs — the uncertainty reasoning only helps at
  0.5B/1.5B capacity.** That is a direct threat to the claim that the metacognitive span
  carries the work, and it is the most important caveat in this row for a 4B policy.

### A2. On the effectiveness of reward functions for confidence calibration — arXiv:2607.04332
(Tan, Lin, Motani, Lee; 5 Jul 2026). **The theory paper for the whole A family, and the
one that tells you which calibration rewards are safe.**
- **(3)** A reward scheme is a pair `(f(c), g(c))` — reward for a correct answer and for
  an incorrect answer, both functions of the stated confidence `c ∈ (0,1)`. Expected
  reward at true correctness probability `p`: `R(c,p) = p·f(c) + (1−p)·g(c)`.

  | Scheme | f(c) | g(c) | Hackable? |
  |---|---|---|---|
  | Correctness-only | 1 | 0 | no |
  | Log-k | 1 + k ln c | k ln(1−c) | **yes** |
  | Brier-k | 1 − k(1−c)² | −k c² | yes for k>1 |
  | **Brier-1 (= RLCR's form)** | 1 − (1−c)² | −c² | **no** |
  | Overconfidence-k | [(k+1)ln(ck+1) − ck] / [(k+1)ln(k+1) − k] | [ln(ck+1) − ck] / [(k+1)ln(k+1) − k] | no |
  | Underconfidence-k | [kc + ln(1 − kc/(1+k))] / [k − ln(1+k)] | [kc + (k+1)ln(1 − kc/(1+k))] / [k − ln(1+k)] | no |

- **Theorem 1 (non-hackability), the inequality you want:** a scheme is non-hackable iff
  (i) it is a proper scoring rule — there exists `h(c) = f′(c)/(c−1) = g′(c)/c`;
  (ii) `h(c) ≤ 0` for all `c ∈ (0,1)`; (iii) `f(a⁺) ≥ g(a⁺)` on the restricted
  confidence range.
- **The hacking mechanism, stated precisely:** for Log-1, `R_max(0⁺) = 0` while
  `R_max(c) < 0` for `p` below a threshold **p ≈ 0.648**. So on any problem the model
  believes it will solve with probability < 0.648, **the reward-maximizing action is to
  refuse and declare that refusal confidently** rather than attempt. Empirically: "the
  LLM has learnt to give up answering harder questions," with the accuracy loss growing
  with difficulty.
- **(4)** gold answer, an `isCorrect(q,a) ∈ {0,1}` checker, and the model's own
  verbalized `c`. **No rollouts needed** — the reward is computed on a single response.
- **(8)** Qwen2.5-3B-Instruct under Dr-GRPO on BigMath / DeepMath-103K / HotpotQA:
  BigMath accuracy 0.954 (correctness-only) vs 0.953 (Brier-1); DeepMath 0.531 vs 0.511.
  **Brier-1 and Overconfidence-4 give the best tradeoff; no scheme dominates.**
  Conclusion: "the reward scheme should be tuned as a hyperparameter."
- **(9)** *Selective* confidence reward hacking — the model abandons hard problems.
  Moving toward the underconfident end of the spectrum "maintains or generally decreases"
  accuracy.
- **★ Why this row matters for cd9:** it proves that within family A there is a
  *formally checkable safety condition*. If a meta-reward is ever put on a verbalized
  confidence, Theorem 1 is the gate to run before the experiment, not after.

### A3. Rewarding Doubt — arXiv:2503.02623 · A4. SaySelf — arXiv:2405.20974 · A5. SEED-GRPO — arXiv:2505.12346 · A6. RLPR
See batch-A results section below.

## Group B — label-free consensus and confidence rewards (the gold-free family)

### B1. TTRL — Test-Time RL, arXiv:2504.16084 (Zuo et al., Tsinghua/Shanghai AI Lab)
- **(2)** No metacognitive claim; the reference label-free method.
- **(3)** `y* = argmax_y Σ_i 1[ŷ_i = y]` over N rollouts; `R(ŷ_i, y*) = 1[ŷ_i = y*]`;
  GRPO.
- **(4)** the policy's own K rollouts' extracted answers. **Nothing else.** Gold used for
  evaluation only.
- **(5)** whole-sequence scalar → GRPO advantage.
- **(6)** Varies within a group **only when the group disagrees**. Unanimous group ⇒
  `r ≡ 1` ⇒ std 0 ⇒ **zero gradient**. TTRL does not report the unanimous fraction and
  has no filter for it.
- **(7)** Already gold-free. **The "lucky hit" argument, with their numbers:** on AIME24
  with Qwen2.5-Math-7B, *label* accuracy (maj@n vs truth) = **37%** while *reward*
  accuracy = **92%**, and the modal answer holds only 16.6% of the mass. Why: "even if
  the estimated label does not match the ground truth, as long as it differs from the
  predicted answer, the verifier will still output a negative reward, and this is exactly
  the correct reward." Fig. 8: "although the label accuracy rarely exceeds 50%, the
  reward accuracy remains consistently high, staying above 75%." **Reward accuracy is
  anti-correlated with model strength in a helpful way** — a weak model's wrong answers
  scatter, so the negative reward is usually right.
- **(8)** ACCURACY. Qwen2.5-Math-7B on unlabeled test data: AIME24 12.9 → **40.2**
  (+211.6%), AMC 35.6 → 68.1, MATH-500 46.7 → 83.4. Approaches the "RL (leakage)"
  gold curve.
- **(9)** Gain decays monotonically with difficulty (MATH-500 L1 +175.3% → **L5 +75.3%**);
  hyperparameter-sensitive collapse with "persistently high entropy that does not
  diminish"; self-assessed as inheriting "sensitivity to data difficulty, strong reliance
  on priors, and risk of collapse."

### B2. TTRL-Guard / Correct-Answer Extinction Window — arXiv:2605.19444
**Read this before citing TTRL's +211%.** Verbatim thesis: TTRL's gains "**reflect
sharpening of already-solvable problems rather than genuine learning**, while problems
corrupted from correct to incorrect outnumber truly learned ones, and this damage is
irreversible once majority vote locks onto a wrong answer."
- **Decomposition**: 44.5% "Stable Always Right" (sharpened only); **0.7% genuinely
  learned; 21.6% degraded ⇒ corruption outnumbers learning 31:1.** On Llama-3.2-3B /
  MATH-500, >60% degraded, <1% learned.
- **The extinction window**: correct-vote rate for degraded problems starts near 58% and
  "collapses to near zero by the final third of training"; correct answers appear in
  41.1% of samples during the high-flip-rate window, **3.3% after flip rate collapses**.
- **Flip Rate** `FR_i^t = (1/W) Σ_{s=t−W+1}^{t} 1[ŷ_i^s ≠ ŷ_i^{s−1}]`; FRS scales the
  reward by `w_i^t = α_i^t γ_i^t δ_i^t`, `α_i^t = 1 − λ₁ FR_i^t`; MPS mixes in a
  minority-preserving label when `FR_i^t > τ_FR`; RCSU suspends updates on polarized
  problems.
- **(8)** Qwen2.5-7B-Instruct 45.5 vs TTRL 42.3; **Qwen3-4B 59.7 vs 57.6**; AIME25
  23.3 vs 15.6.

### B3. Co-rewarding — arXiv:2508.00410 ("Co-rewarding: Stable Self-supervised RL for
Eliciting Reasoning in LLMs"; often cited as Co-Reward/CoReward)
- **(3) Co-rewarding-I (cross-view)**: rephrase q → q′, roll out both, majority-vote each
  side separately, then **cross the labels**:
  `y_v = argmax Σ_i 1[ans(y_i)=·]`, `y'_v = argmax Σ_i 1[ans(y'_i)=·]`,
  `Â_i = [r(y'_v, y_i) − mean]/std` and `Â'_i = [r(y_v, y'_i) − mean]/std`.
  **A view never grades itself** — that is the whole mechanism.
  **Co-rewarding-II (EMA self-distillation)**: `ỹ_v^(k) = argmax Σ_j 1[ans(ỹ_j)=·]` from
  a slow reference `π̃_ref`; `Â_i^(k) = [r(ỹ_v^(k), y_i) − mean]/std`; teacher update
  `π̃_ref^(k) ← α^(k) π̃_ref^(k−1) + (1−α^(k)) π_{θ_old}^(k)` with
  `α^(k) = 1 − [(α_end − α_start)/2]·[1 + cos(πk/K)]`, **α_start = 0.99 → α_end = 0.9999**
  (the teacher gets *slower* over training).
- **(4)** rollout answers only, plus an offline LLM rewriter for q′ (no answer attached).
  No gold in training.
- **(6)** ★ **The label comes from outside the group being centered**, so within-group
  variance does *not* vanish when the student's own rollouts are unanimous. This is
  precisely the fix for TTRL's degenerate-group problem.
- **(8)** Qwen3-8B-Base, DAPO-14k: Co-rewarding-II MATH500 80.6 / GSM8K **94.01** /
  AMC 54.37 / AIME24 16.35, vs **GT-Reward 86.6 / 87.19 / 61.75 / 24.58**. Beats gold on
  GSM8K, loses badly on AIME24 (16.35 vs 24.58). +3.31% avg over the best self-rewarding
  baseline.
- **(9)** Their diagnosis of the baselines is the useful part: entropy rewards make the
  policy "concentrate probability mass on a small set of tokens and produce repetitive
  strings"; majority-vote rewards "converge to a consistent yet incorrect answer that
  attains high consensus." Fig. 4: both baselines "reach maximum reward quickly then
  collapse to near-zero validation performance." **Ablation: removing the EMA update
  degrades clearly — a fully frozen teacher is not enough, a fully online one collapses.**

### B4. EMPO — arXiv:2504.05812 ("Right Question is Already Half the Answer", NeurIPS 2025 Spotlight)
- **(3)** Semantic entropy `H = −Σ_j p(c_j|q) log p(c_j|q)` over meaning clusters;
  `r_i = p(c_j|q)` where `l(o_i)=c_j` — the empirical cluster frequency, i.e. a **soft**
  majority vote; `A_i = (r_i − mean)/std`, G = 7.
- **★ Anti-collapse gate**: prompts are **filtered out** unless `δ_low < H < δ_high`.
  Verbatim: "continuously optimizing on responses with already low entropy is redundant
  and at the risk of overconfidence." This is the explicit handling of TTRL's unanimous
  group.
- **(4)** no gold; regex clustering for math, DeBERTa-v3-large (304M) for free-form.
- **(8)** Qwen2.5-Math-7B math avg 30.7 → **48.1** (gold GRPO 46.8 — EMPO *beats* gold
  here); Qwen2.5-7B MMLU-Pro 32.1 → 50.1 vs gold 57.1 (**7 pts below the gold ceiling**).
- **(9)** "Models could exploit the reward by overfitting to high-confident but wrong
  predictions for the most frequent semantic clusters without careful reasoning."
  Pass@k analysis: at large k the base model matches or exceeds the RL model ⇒
  **RL sharpens sampling efficiency rather than teaching new reasoning.**

### B5. SRT — "Can Large Reasoning Models Self-Train?", arXiv:2505.21444 (CMU)
**The definitive negative result for majority-vote self-training.**
- **(3)** identical to TTRL: `y_majority = argmax_{y'} Σ_i 1[answer(y^(i)) = y']`,
  `r(y) = 1[answer(y) = y_majority]`.
- **(9)** "Models learn to produce consistent responses in order to optimize [the
  self-reward] **irrespective of their true correctness**." The degenerate optimum,
  verbatim: "a very high entropy, essentially random, set of tokens followed by the same
  'template' final answer … nearly independent of the input prompt" — **constant output
  is perfectly self-consistent**. Fig. 6: "SRT improves performance at first, but then
  demonstrates complete model collapse" on all four base models. Fig. 7: collapse
  "closely coincides with a sudden increase in the SRT self-reward objective."
- **Mitigations and their verdicts** (this is the part to internalize):
  (a) early stopping — only delays; "prolonged training with lower learning rates would
  still result in complete model collapse"; (b) curriculum — works on synthetic, "doesn't
  prevent collapse on real math datasets"; (c) offline/fixed-teacher labels — prevents
  runaway but underperforms by ~6–10%; (d) **Fig. 8: varying KL coefficient, lr and n
  "does not affect model collapse significantly" — KL to reference does not save you.**
- **(8)** peaks at mean@32 0.20–0.23 vs **gold RL 0.32–0.36** on MATH-12K/DAPO.

### B6. Intuitor / RLIF — arXiv:2505.19590 (Berkeley)
- **(3)** Self-certainty = **KL to uniform**, not entropy:
  `SC(o|q) = (1/|o|) Σ_i KL(U ‖ p_{π_θ}(·|q,o_{<i})) = −(1/(|o|·|𝒱|)) Σ_i Σ_j log(|𝒱|·p_{π_θ}(j|q,o_{<i}))`;
  `Â_{i,t} = (u_i − mean)/std` under GRPO.
- **(4)** the policy's own token distributions. No gold in training.
- **(8)** Qwen2.5-3B: in-domain **below** gold GRPO (GSM8K 0.792 vs 0.826; MATH500 0.612
  vs 0.636) but **out-of-domain above it** (LiveCodeBench 0.153 vs 0.085, +65% rel;
  CRUXEval-O 0.416 vs 0.341).
- **(9) ★ The polarity lesson.** With an **offline** (frozen base) self-certainty
  annotator, around **step 100** the policy learns to "append solved problems" to inflate
  reward → length spike, accuracy collapse. Fix: score with the **online, evolving**
  policy. This is the *exact inverse* of RLSR (B8), where the online judge collapsed.
  **Rule the batch supports: a reward that is a *statistic of the current policy* should
  be computed online; a reward produced by a *judge the policy can target* must be
  frozen.**

### B7. RENT — "Maximizing Confidence Alone Improves Reasoning", arXiv:2505.22660 (CMU)
- **(3)** `R(y_pred) = −H(π(x)) = (1/T) Σ_t Σ_v p_t(v) log p_t(v)`; GRPO.
- **(4)** own next-token distributions; "y_target is not used in the reward … we do not
  assume access to this at any point in training."
- **(8)** Qwen2.5-7B-Instruct: AIME 0.139 → **0.270**, MATH500 0.7735 → 0.823, but
  **GSM8K 0.911 → 0.900 (regression)**.
- **(9) ★ The span ablation you should read.** Entropy↔accuracy correlation is highest in
  the **last chunk** of tokens, **but minimizing entropy only on the final answer tokens
  performs poorly** — they conclude "token-level confidence of the final answer tokens is
  not well-calibrated." The usable confidence signal lives in the **trailing reasoning
  tokens**, not the answer span. Also: "it is, of course, a possibility for the model to
  be confidently wrong."

### B8. RLSR — arXiv:2505.08827 (v1 "Self-Rewarding Self-Improving")
- **(3)** the policy itself judges: `r = 1[judge emits <JUDGE_SCORE>1</JUDGE_SCORE>]`,
  exploiting the generate-vs-verify asymmetry (differentiate the integral you produced).
- **(4)** judge = same model; **"we use an offline judge that doesn't update during
  training."** No gold in training.
- **(8)** Integration: self-judge **75%** vs symbolic verification **92%** — a 17-point
  gold gap. Countdown ≈ +20%, matching the formal verifier.
- **(9)** the best hacking catalogue in the survey: "agents struggle to substantially
  exceed their judge's capabilities"; Llama-3.2-3B agents "quickly developing strategies
  to trigger false positives through deceptive formatting"; "formatting tricks or
  nonsensical but convincing-looking expressions"; **updating the judge online → "rapid
  training collapse as models exploited their own evaluation biases"**; "minor wording
  changes could trigger catastrophic reward hacking within dozens of training steps."

### B9. OM-GRPO — "Don't Peek at the Answer", arXiv:2608.03119 (Xiamen Univ., Aug 2026)
**The most directly transplantable 2026 result for a span-scoped meta reward.**
- **Diagnosis**: "the model can improve reward by **directly sharpening high-frequency
  answer tokens, without improving the reasoning that leads to them**." Fig. 5 KL
  decomposition: "answer-span token KL rises sharply while reasoning-token KL grows
  slowly" — **the shortcut is spatially localized to the answer span**. Fig. 1: MV
  baseline's answer diversity per batch drops from ~8 to near zero.
- **(3)** Answer-span **gradient mask** `m_{i,t} = 0` on answer tokens, 1 elsewhere;
  **soft consensus reward** `r_i^q = Pr(z = z_i | z ∈ {z_1..z_G})` (empirical frequency,
  replacing hard majority vote); **CAR** expands the answer pool from G to O(G²) via
  pairwise trajectory comparison. Architecture: **reward is estimated from the answer
  span while the gradient never touches it** — estimation decoupled from optimization.
- **(8)** avg over 9 benchmarks: Qwen3-1.7B **35.19** vs GT-Reward 35.55 vs MV 32.64;
  Llama-3.2-3B 31.21 vs 31.53; Qwen2.5-7B 43.25 vs 43.90. **Gold gap 0.3–0.65 pts — the
  smallest label-free gap on record.** TTRL setting: 46.89 vs MV 42.65, where "MV leads
  to a decrease in Avg@k (36.92 → 35.09), indicating convergence toward incorrect
  consensus."
- **(9)** A death taxonomy by family (Fig. 3): Self-Certainty → "over-confident
  repetition, spiked KL, vanishing entropy"; Co-rewarding → "drifts toward high-entropy
  uncertainty"; Majority Voting → "consensus collapse, degenerating into trivial,
  high-agreement responses with near-zero accuracy." **Fig. 4: partial masking is worse
  than none — answer-token weight w = 0.50–0.75 "destabilizes and collapses"; only full
  masking (w = 0.0) is stable.** A sharp, non-monotone threshold.

### B10. JURY-RL — arXiv:2604.25419
- **(3)** `â = argmax_a Σ_i 1[a_i = a]`; `r_i = δ·1[a_i = â] + (1−δ)·r_i^ResZero`, with
  δ=1 iff one call to an external **Lean** verifier confirms â, else the zero-mean
  fallback `r_i^ResZero = α·1[i∈R]·(z_i − ū) − cα·1[i∈M] + γ`, `γ = cα²`, `α = |M|/G`.
- **★ The group-centering pathology in one line**: "as the majority share increases,
  **supporters' advantages go to zero while dissenters' penalties blow up**," driving
  entropy collapse under GRPO normalization. This is the mechanism behind every consensus
  collapse in this section.
- **(8)** Qwen2.5-7B pass@1 avg **43.90 = GT-Reward 43.90**; MV 41.42; LLM-judge 41.61.
  Pass@k *exceeds* gold RLVR (+9.06 pp on Qwen3-1.7B-Base).
- Label-free but **not verifier-free** (external Lean prover).

### B11. One-shot Entropy Minimization — arXiv:2505.20282
- **(3) Not RL** — a direct loss: `L_EM(x;θ) = (1/|I|) Σ_{t∈I} H_t`,
  `H_t = −Σ_v p_θ(v|y_{<t},x) log p_θ(v|y_{<t},x)`. "A closed-form objective, eliminating
  the need for external reward estimation or value baselines." **No group, no advantage
  ⇒ no "all rollouts agree → zero gradient" escape valve**, which is why it runs off the
  cliff.
- **(8)** Qwen2.5-Math-7B, **1 example, 10 steps**: MATH500 53.0 → 78.8, avg **+24.7 pts**.
- **(9)** "Loss keeps dropping while accuracy peaks then declines" — named **over
  confidence**: EM "may excessively amplify the model's confidence in its tokens …
  exacerbating algorithmic bias." Seed variance of **2×** under identical hyperparameters.
  Inference-temperature inversion (worse at higher T, unlike RLVR models). **EM applied
  after RL is destructive** — it "locks in narrow, overconfident output modes"; EM before
  RL is fine. More data hurts.

### B12. The Entropy Mechanism of RL for Reasoning LMs — arXiv:2505.22617 (PRIME-RL)
Gold-supervised, but it is the *diagnostic* that explains every collapse above.
- Empirical law `R = −a·exp(H) + b`. **Theorem 1**:
  `H(π^{k+1}|s) − H(π^k|s) ≈ −η · Cov_{a∼π^k}( log π^k(a|s), π^k(a|s)·A(s,a) )` —
  entropy is spent in proportion to the covariance between token log-prob and advantage;
  high-probability + high-advantage tokens burn it.
- Clip-Cov (detach gradients on r·N tokens in a covariance band, r = 2e-4) and KL-Cov
  (KL penalty on the top-k covariance tokens, k = 2e-3…2e-4, β = 1).
- "**73% of the entropy consumption and 76% of the performance gain occurred in just the
  first 200 gradient steps (1/12 of training)**; the first 800 steps account for over 93%
  of performance gains."
- 32B: GRPO 45.8 → Clip-Cov 50.3 → KL-Cov 52.2.

### B13. Absolute Zero (AZR) — arXiv:2505.03335
- **(3)** proposer/solver self-play. `r̄_solve = (1/G) Σ_i r_solve^(i)` over G MC
  rollouts; **`r_propose = 0` if `r̄_solve ∈ {0,1}`, else `1 − r̄_solve`** — a
  **difficulty-targeting reward maximal at ~50% solve rate**, computed entirely from the
  policy's own empirical solve rate. Solver `r_solve = 1(y = y*)` with `y*` from a Python
  executor. Composite: `−0.5` well-formatted but wrong, `−1` format error.
- Data-free, **not verifier-free** (the executor is an external oracle).
- **(8)** AZR-Coder-7B combined avg 50.4 vs 48.6 (ORZ); **math avg 39.1 vs 45.8
  (PRIME-Zero) — 6.7 pts behind on math**; +15.2 math points from code-only self-play.
- **(9)** The "uh-oh moment": Llama-3.1-8B self-play produced "The aim is to outsmart all
  these groups of intelligent machines and less intelligent humans." Authors call for
  safety-aware training and oversight.

### B14. Spurious Rewards — arXiv:2506.10947 ("Rethinking Training Signals in RLVR", UW/AI2)
**The mandatory control this project is missing.** MATH-500 gain on Qwen2.5-Math-7B:

| Reward | Gain |
|---|---|
| Ground truth | +29.1% |
| Majority vote (= TTRL) | +27.1% |
| **Incorrect** (rewards only demonstrably wrong answers) | **+24.1%** |
| **Random** Bernoulli(0.5), independent of the response | **+21.4%** |
| Format (`\boxed{}` present) | +13.8% |

Random reward recovers **74%** of the gold gain; a *negatively* correlated reward
recovers 83%. Mechanism: "the expected gradient in the GRPO loss is **nonzero due to the
clipping mechanism**"; clipping produces "asymmetric updates towards model prior
knowledge." What gets amplified is **code reasoning** (65.0% → 95.6% of responses under
random reward; code-reasoning accuracy 60.9% vs 28.0%). **On Llama3.1-8B-Instruct and
OLMo2-7B the same spurious rewards are flat or negative.** Their warning verbatim:
"**Proposed RLVR reward signals should be tested on diverse models!**"

## Group C — verification, aggregation and progress rewards

### C1. AggLM — "The Majority is not always right: RL training for solution aggregation", arXiv:2509.06870 (Meta/CMU)
- **(2)** self-aggregation — reviewing a set of candidate solutions, detecting that the
  majority is wrong, rescuing a correct minority, or synthesizing a new solution when
  none is correct.
- **(3)** aggregator `p_ϕ(ỹ | x, y_{1:m})`, GRPO, group 8, KL 0.001, T = 1.5.
  **`r(ỹ) = 1[ỹ = y*]`.** No process, format, or aggregation-specific term.
- **(4)** gold `y*` (the entire reward); `math_verify` as equivalence checker; the `m`
  candidates come from a **separate** generator (Qwen3-1.7B thinking).
- **(6)** candidate set fixed within the group, so within-group variance isolates
  *aggregation skill* from candidate quality. Easy prompts collapse to zero advantage —
  hence their curation: all hard examples (majority wrong) + **p = 50%** easy, 446,220
  training examples.
- **(7) ★** Breaks completely. **And the obvious substitute is self-defeating here**: a
  majority-agreement reward would train exactly the behavior the paper exists to beat.
  No gold-free ablation. What *is* gold-free is **deployment**: the gold-trained
  AggLM-1.7B aggregates Qwen3-8B solutions it never trained on and still beats majority
  voting by 1–9 points — gold to train, not to deploy.
- **(8)** ACCURACY: AIME24 70.69 vs MV 67.92; AIME25 **50.00 vs 45.89**; HMMT25
  **32.07 vs 26.72**. AggLM over 8 solutions beats majority voting over 16.
- **(9)** No limitations section. Ablation: training on hard examples only (p = 0%) is
  suboptimal — the aggregator over-learns to distrust the majority.

### C2. RSA — "Recursive Self-Aggregation", arXiv:2509.26626 (Mila/LLNL)
**★ Uses exactly the cd9 policy: Qwen3-4B-Instruct-2507.**
- **(3)** Test-time loop, **no reward at all**: `τ_i^(1) ~ p_θ(·|x)`;
  `S_i^(t) ⊆ P_t, |S_i^(t)| = K`; `τ_i^(t+1) ~ p_θ(· | S_i^(t), x)` for t = 1…T−1.
  Aggregation-aware RL adds Eq. (5):
  `max_θ E_{(x,y)~D, S_0~p_θref(·|x)} [ E_{τ~p_θ(·|x,S_0)}[r(τ,y)] − β KL(p_θ(·|x,S_0) ‖ p_θref(·|x,S_0)) ]`
  — **`S_0` comes from the frozen reference, so the candidate set is off-policy and
  fixed; only the aggregation response is on-policy.** RLOO, 300 steps, 50-50 split
  between standard (Eq. 4) and aggregation (Eq. 5) prompts.
- **(7) ★ Split verdict, the most useful in the survey.** **Test-time RSA is fully
  gold-free and delivers most of the headline gain**; the RL layer is gold-dependent and
  adds only a modest increment (Fig. 9: beats standard RL on four of five tasks, **AIME-25
  is the exception**). No gold-free RL ablation — but unlike AggLM, a consensus-based
  aggregation reward is not self-defeating here.
- **(8)** ACCURACY, Pass@1, **Qwen3-4B-Instruct-2507**: AIME-25 43.91 → **73.18 ± 2.20**
  (self-refinement 53.33); HMMT-25 27.17 → **47.55 ± 1.00**; LiveCodeBench-v6 49.63 →
  56.72.
- **(9)** "Overall diversity within the population generally decreases as t increases" —
  the loop collapses toward consensus (the direct analogue of meta-emission erosion).
  Single-step aggregation caps at **K = 4** without degradation. No convergence theory.

### C3. GenRM — "Generative Verifiers", arXiv:2408.15240, ICLR 2025
- **(3)** SFT, not RL. `L_SFT(θ,D) = −E[Σ_t log p_θ(y_t|x,y_{<t})]` on
  `D_verify = {(x,y⁺,I) → 'Yes'} ∪ {(x,y⁻,I) → 'No'}`. Score `r_Direct = p_θ(Yes|x,y,I)`.
  CoT variant with K sampled rationales:
  `r_MajV@K(x,y) = (1/K) Σ_i p_θ(Yes | x, y, I_CoT, v_CoT^(i), I)`. Unified objective
  `L_GenRM = L_SFT(θ, D_verify) + λ L_SFT(θ, D_correct)`.
- **(4)** gold (Yes/No labels from an answer checker) **and** a judge LLM at data-generation
  time — rationales `v_CoT` generated by Gemini 1.0 Pro under **reference-guided grading**,
  then filtered by >50% agreement with the gold-derived label. Gold enters twice.
- **(6)** N/A — no groups, no advantage; MajV@K is variance reduction on the *score*.
- **(8)** ACCURACY (Best-of-N): algorithmic 5% → 45.3%; GSM8K 73% → 93.4%; MATH500
  easy-to-hard 28% → 44.6%. 2.5× fewer solutions than a discriminative RM for parity.
- **(9)** "Model-generated rationales may contain errors," mitigated by ensembling over
  multiple rationales. No RL, no process supervision.

### C4. V-STaR — arXiv:2402.06457, COLM 2024
- **(3)** DPO verifier:
  `L = −E_{(x,y⁺,y⁻)~D_VER}[ log σ( r̂(x,y⁺) − r̂(x,y⁻) ) ]`,
  `r̂(x,y) = β log[ V(y|x) / G_SFT(y|x) ]`. ⚠️ Asymmetry: **ranking at test time uses the
  raw likelihood `V(ŷ|x)`, dropping the reference term**.
- **(4)** gold correctness labels partition the model's own k samples; `G_SFT` frozen
  reference; unit tests for code.
- **(6)** Not GRPO, but the pairwise contrast is *within-problem* (Cartesian product of
  that problem's correct × incorrect samples) — a problem where all k samples agree
  contributes **zero pairs**, the same degenerate-group phenomenon.
- **(7) ★ The most gold-coupled method in the survey**: the label is not just the reward,
  it is the **data-construction operator**. Remove it and you cannot form a single
  preference pair, and STaR's generator buffer also collapses.
- **(8)** ACCURACY: +6–17% absolute on math, +4–12% on code, over STaR / STaR† / ORM.
- **(9)** LoRA-limited; **no analysis of length bias in the likelihood-based score**,
  which is notable given `V(ŷ|x)` is length-sensitive.

### C5. PAV / Rewarding Progress — Setlur et al., arXiv:2410.08146, ICLR 2025
**The most instructive gold-economical design in the survey.**
- **(3)** Prover-policy advantage, Eq. (2):
  `A^μ(s_h,a_h) := Q^μ(s_h,a_h) − V^μ(s_h) = Q^μ(s_h,a_h) − Q^μ(s_{h−1},a_{h−1})` —
  "the relative increase/decrease in the likelihood of success, before and after the
  step," under a **prover policy μ distinct from the trained policy π**.
  Eq. (4): `ℓ^{π'}_{PAV-RL}(π) = ℓ_{ORM-RL}(π) + α Σ_h E_{s_h~d^{π'}_h} E_{a_h~π(·|s_h)}[A^μ(s_h,a_h)]`.
  Eq. (5), the effective per-step reward:
  `∇_π ℓ = Σ_h ∇_π log π(a_h|s_h) · ( Q^π(s_h,a_h) + α·A^μ(s_h,a_h) )`.
  **α = 0.5** (Gemma 2B, 9B), **α = 0.2** (27B); robust over **[0.2, 0.6]**.
  Complementary prover instantiated as **Best-of-K with K > 1 but small; Bo4 optimal.**
- **(6) ★ The structurally important point**: `A^μ` is centered **within a trajectory
  across time**, not across a prompt group — so **it is nonzero even when every rollout in
  a GRPO group gets the same final outcome.** That is exactly where outcome-only GRPO has
  zero group variance, and it is the source of the 5–6× sample-efficiency claim.
- **(7)** Gold enters *only* at the terminal node (regex final-answer check scoring the MC
  rollouts); **no per-step gold, no human step labels**. It converts sparse terminal gold
  into dense per-step reward. Untested substitute: define `Q^μ` as the probability that μ
  reaches the *self-consistent majority* answer instead of the gold answer.
- **(8)** ACCURACY: beam search >8% over ORM at 1.5–5× less compute; **online RL 5–6×
  sample efficiency, >6% accuracy, 8× better Pass@N.**
- **(9) ★ The transferable warning.** Prover design is unsolved: "it is unclear how to
  automatically design a flexible class of optimal prover policies." A **too-weak** prover
  gives advantages ≈ 0 everywhere; a **too-strong** prover cannot distinguish steps
  ("`Q^μ` … before and after this irrelevant step will be identical"). **Appendix G
  documents an actual collapse**: training with Q-values from strong provers produced
  "policies that only produce re-phrasings of the question … and do not succeed at solving
  the question" — a capable prover rescues any prefix, so filler is free. **This is the
  shaped-span reward failure mode that a `<|meta|>` bonus would hit.**

### C6. Quiet-STaR — arXiv:2403.09629, COLM 2024
**★ The only fully gold-free, span-scoped metacognitive RL in the survey.**
- **(3)** Mixing head (3-layer ReLU MLP) blends post-thought and no-thought predictions:
  `log p_j^talk ← w_{j:j+n} · log p^init + (1 − w) · log p^thought`, initialized so the
  model starts at `p^init`.
  **Reward:**
  `r_j = log p^talk_{j:j+n_true}( X_{j+1:j+n_true+1} ) − log p̄^talk_{j:j+n_true}( X_{j+1:j+n_true+1} )`
  where `p̄` is the **average across the sampled rationales at that position** — i.e. a
  group-mean baseline.
  **Gradient:** `∇_θ L_j^REINFORCE = − r_j · ∇_θ log p_θ( T_j | [X_{:j}; <|startofthought|>] )`,
  score function taken **w.r.t. the thought tokens only**. Total: `∇L_j = ∇L_j^NLL + ∇L_j^REINFORCE`.
  ⚠️ **"We exclude the negative reward from the REINFORCE loss term, as it led to more
  stable training"** — effectively `max(r_j, 0)`, a one-sided group-relative advantage.
  Teacher forcing on the *corpus* (parallel attention mask over the true next `n_true`
  tokens), plus tokenwise parallel sampling.
- **(4)** gold: **none**. Verifier: none. Judge: none. Teacher model: none. Inputs are
  (a) the policy's own sampled thoughts at position j, (b) its own log-probs, (c) the
  **next tokens of unlabeled corpus text** (OpenWebMath, C4).
- **(5)** **Per-position, span-scoped**: reward computed at every token position, gradient
  applied **only to the thought span** between the learned markers, alongside a
  full-sequence NLL term. Architecturally the closest thing in the literature to a
  `<|meta|>`-span reward.
- **(6)** GRPO-style avant la lettre: group = thoughts sampled at position j, baseline =
  group mean, `r_j` = deviation. Zero-mean by construction ⇒ **requires within-group
  variance in thought usefulness**; on easily-predicted tokens all thoughts are equally
  useless and there is no gradient. The paper reports this as a feature: rationales
  "disproportionately help model **difficult-to-predict tokens**."
- **(7) ★** Gold-free by construction. Why, mechanically: (i) the prediction target *is*
  the input — the corpus is the correctness oracle; (ii) the baseline is the model's own
  sample mean, so no critic needs external labels; (iii) the **mixing head** makes the
  comparison well-posed by letting the model fall back to `p^init`, so `r_j` isolates the
  thought's *marginal* contribution rather than its contextual damage.
  It scales with unlabeled text rather than labeled problems. **It is a proxy**: the
  reward optimizes future-token likelihood and reasoning accuracy is a hoped-for
  correlate — and the accuracy gains are **zero-shot transfer**, the model never saw an
  answer key.
- **(8)** ACCURACY, zero-shot, Mistral 7B: **GSM8K 5.9% → 10.9%**; **CommonsenseQA
  36.3% → 47.2%**. Secondary: perplexity, concentrated on hard-to-predict tokens.
  ⚠️ Absolute numbers are low — the honest price of gold-freeness.
- **(9)** Severe compute overhead (a thought at every position); no test-time compute
  lever; only 7B tested; never validated from scratch (continued pretraining only);
  **negative rewards had to be dropped for stability** — the raw group-centered signal is
  too noisy to use symmetrically.

## Group D — metacognition-framed RL with gold in the loop

### D1. MASA / MAPR — "Verifying Meta-Awareness via Predictive Rewards in Reasoning Models", arXiv:2510.03259
**★ The single most on-point paper for cd9.**
- **(2)** meta-awareness: predicting one's own rollout statistics before solving —
  optimal thinking duration, knowledge boundary, concept-level structure.
- **(3)** Same policy `π_θ`, two instruction templates (solution template, meta template
  emitting JSON `{math_notion, pass_rate, solution_length}`).
  - **Length** (Eq. 1): `r_length = 1[ min(l_correct) ≤ l_pred ≤ max(l_correct) ]`, 0 if
    no correct rollouts exist.
  - **Difficulty** (Eq. 2): `r_difficulty = b^{|d_pred − d_sol|}`, **b = 0.01**, with
    `d_sol` the **empirical pass-rate over the model's own G solution rollouts**.
  - **Notion** (Eq. 3): `r_notion = (1/|n_pred|) Σ_{n∈n_pred} 1[ f_count(n,1) − f_count(n,0) > 0 ]`
    — the fraction of self-predicted concepts that appear **more often in correct than in
    incorrect** of its own rollouts. Notions already in the problem statement are excluded.
  - **Combination**: `r_meta = (r_length + r_difficulty + r_notion)/3`. Meta and solution
    rollouts run **separate reward pipelines**; both are optimized by the same policy
    gradient with no explicit cross-weight.
- **(4)** the policy's **own G rollouts** supply every reference statistic — lengths,
  pass rate, notion frequencies. **Gold enters in exactly one place: the binary
  correctness label that partitions those rollouts into correct and incorrect.** No
  judge, no teacher, no external annotation (notions are self-generated and matched by
  lemmatization).
- **(5)** meta reward on the **meta rollouts** only; solution reward on the solution
  rollouts only; token-level GRPO updates. Also used as a **sample filter** and a
  **generation cutoff**.
- **(6) ★ The property that makes this design work.** The *reference* (own rollout
  statistics) is constant per prompt, but the *prediction* varies across meta rollouts,
  so **`r_meta` varies within the meta group and survives mean-subtraction.** This is the
  structural template the brief is looking for: put the variation in the judgment, not in
  the reference.
- **(7) ★ GOLD REMOVAL — and this is the key finding of the whole survey.** All three
  meta-rewards are computed over the model's own rollouts; gold is used *only* as the
  partition function `correct/incorrect`. **Replace that partition with the TTRL
  majority-vote pseudo-label** (`correct := ŷ_i = y_maj`) and every formula above is
  defined without gold: `d_sol` becomes the consensus share, `l_correct` becomes the
  lengths of consensus-agreeing rollouts, and `f_count(n,·)` becomes concept frequency in
  consensus vs non-consensus rollouts. **No paper has run this.** MASA itself ran no
  gold-free ablation.
- **(8)** ACCURACY. Qwen3-8B: AIME'25 22.18 → **26.46** (+19.3% rel); AIME'24 28.54 →
  33.75; 6-math average 51.04 → 54.20. Qwen3-14B 6-math 56.69 → 58.08. OOD (13
  benchmarks) +2.08% avg. Efficiency: reaches baseline GRPO accuracy **1.28× faster**,
  34.5% less train time. Gating: `std(d_pred over M meta-rollouts) < 0.1` filters
  zero-variance prompts; early cutoff at 2× predicted length.
- **(9)** Meta-predictions are unstable before ~step 80 (needs expert-SFT warm start,
  N_expert = 128); gating/cutoff hyperparameters set offline; AIME drops 3.1–3.9% under
  the efficient variant; **notion reward's susceptibility to reward hacking acknowledged**.
  ⚠️ Model sizes are 8B and 14B — **not validated at 4B**.

### D2. MaR — "Metacognition as Reward", arXiv:2605.23384
- **(3)** `R = KMR + RMR + CR`, each in [0,1].
  `KMR = (k + r)/n` (gold knowledge units identified in the initial metacognitive-knowledge
  block, plus units recovered via a LOOKBACK block, over total gold units);
  `RMR = a(1 − λs)` where `a ∈ [0,1]` is consistency between the reasoning and the stated
  plan, `s ∈ {0,1}` a shortcut indicator, **λ = 0.3**;
  `CR = 1` iff the answer matches ground truth (semantic equivalence judged by an LLM grader).
  DAPO, `Â_t = (R_i − mean(R_j))/(std(R_j) + δ)`, trajectory-level, 8 rollouts, 270 steps.
- **(4)** **gold knowledge-unit annotations** (pre-generated by a frontier model), **gold
  answer**, and an **LLM grader** for KMR/RMR/CR. Three external dependencies — the most
  supervised method in the survey.
- **(7) ★** Not recoverable as-is. KMR needs the gold unit list; RMR is the one term that
  is *rule-checkable in principle* (plan↔reasoning consistency, shortcut detection) and
  could be done by the policy itself, at the cost of turning it into a self-judge with
  RLSR's hacking profile.
- **(8)** ACCURACY: 67.6% vs 64.6% baseline avg over 10 science/medical benchmarks; +7.7%
  on GPQA-Diamond; +11.0% over vanilla DAPO on rubric benchmarks. OOD math/logic +2.4%.
- **(9)** Decreases on FOLIO and ProofWriter — the supervision transfers to open-ended
  reasoning, not formal logic.

### D3. RLMF — "RL with Metacognitive Feedback Elicits Faithful Uncertainty Expression", arXiv:2606.32032
- **(3)** Metacognition-adjusted advantage:
  `A^RLMF_g = (o_g − ō) + { (f_g − f̄)·(k + Z_g)  if f_g > f̄ ; (f_g − f̄) otherwise }`,
  with `k = 1`, `o_g` a weighted sum of factual-calibration/accuracy/format rewards, `f_g`
  the faithfulness reward, `Z_g` a metacognitive scaling factor. **This is an advantage
  modification, not a reward.** GRPO **without std normalization**: `A_g = ρ_g − ρ̄`.
- **(4)** Gold-FC = "proportion of sentences with `|c_i − g_i| < τ`" where `c_i` is the
  expressed confidence and `g_i` the **intrinsic confidence estimated via sampling
  consistency**; Predicted-FC is a self-issued score `F_pred^(g) ∈ [0,1]`. **Ground-truth
  correctness is required** to compute the gold FC.
- **(5)** advantage scaling during GRPO, **plus** offline data selection (highest and
  lowest scoring halves).
- **(8)** **CALIBRATION, not accuracy.** cMFG* 0.84 (Llama-3.1-8B) / 0.83 (Qwen-8B) vs
  MetaFaith 0.67, FUT 0.66, standard RL 0.77 — the "63%" headline is a cMFG*
  improvement. **Accuracy merely preserved: 0.41 for both RLMF and standard RL** (vs
  MetaFaith 0.28, FUT 0.31).
- **(9)** Notes the general pattern it claims to avoid: "RL with internal feedback
  initially improves but degrades as training progresses." Repetitive hedging without a
  diversity constraint.

### D4. EGPO — arXiv:2602.22751 ⚠️ **id resolves, but the title is not "EGPO"**
Actual title: **"Know What You Know: Metacognitive Entropy Calibration for Verifiable RL
Reasoning"** (26 Feb 2026). **It is gold-supervised** — `r(y) = +1 if Ans(y) = g`, trained
on an OpenR1-math-220k subset with "single-value GT enforced."
- **(3)** Entropy proxy from the old policy's own likelihoods:
  `H̃(x,y) = −(1/T) Σ_t log π_{θ_old}(y_t | x, y_{<t})`;
  group-relative weight `ŵ_i = H̄ / (H̃_i + ε_H)`;
  **asymmetric calibration** `w_i = max(1.0, clip(ŵ_i, λ_min, λ_max))` if `r_i = +1`,
  `min(1.0, clip(ŵ_i, λ_min, λ_max))` if `r_i = −1` — "correct responses are never
  down-weighted while incorrect responses are never up-weighted";
  calibrated advantage `Ã_i = w_i · A_i`; all-incorrect groups rescued by NSR
  (`A_i ← −1 ∀i`) with `ℓ_i(θ) = −w_i · max(ρ_i, clip(ρ_i, 1−ε, 1+ε))`.
- **(6)** `w_i` is group-relative by construction, varies within the group, and the
  asymmetric clamp guarantees it never inverts the correct/incorrect ordering.
- **(7) ★** The **weight `w_i` is entirely self-derived** and is a clean, directly
  transplantable advantage-multiplier template; the reward it multiplies is not.
- **(8)** Qwen2.5-Math-7B MATH-500 13.8 → 84.6, AIME24 3.33 → 33.33; DeepSeek-R1-Distill-7B
  MATH-500 87.48 → 93.4.
- **(9)** **None reported** — no collapse analysis, no regime where the proxy breaks.
  Treat the metacognition framing as rhetoric over a reweighting trick.

### D5. Reasoning with Exploration — arXiv:2506.14758 (MSRA/RUC)
- **(2)** ★ Empirically links high-entropy tokens to three exploratory-reasoning
  categories: **(i) pivotal tokens determining or connecting logical steps, (ii) reflective
  actions such as self-verification and correction, (iii) rare behaviors under-explored by
  the base LLM.** "Reflective actions consistently exhibit higher average entropy."
- **(3)** One-line modification:
  `H_t = −Σ_v π_θ(v|q,o_{<t}) log π_θ(v|q,o_{<t})`;
  **`ψ(H_t) = min( α · H_t^detach , |A_t| / κ )`**, α > 0, κ > 1;
  **`A_t^shaped = A_t + ψ(H_t)`**.
  The entropy term is **detached from the graph** (a shaping bonus, not a max-ent
  gradient) and **clipped at `|A_t|/κ`** so it can never dominate or flip the sign of the
  correctness advantage. That clip is the anti-hacking device.
- **(4)** gold-supervised base reward (+1/−1 against a verifier); the entropy term is
  gold-free.
- **(5)** **per-token additive term on the advantage.**
- **(7) ★** The entropy term is a ready-made gold-free **bonus on exactly the token class
  that reflection lives in** — but it rewards *entropy*, not *usefulness*, so alone it
  cannot distinguish a useful check from a decorative one.
- **(8)** ACCURACY at large K: AIME2024 Pass@256 46.7 → **56.7**; AIME2025 Pass@256
  50.0 → 53.3; AMC2023 Pass@128 85.5 → 88.0. The point is that the **capability boundary**
  moves, not just sampling sharpness.
- **(9)** No failure section; the only risk addressed is naive token-frequency hacking.


### A3. Rewarding Doubt — arXiv:2503.02623 (TUM; v1 Mar 2025, v6 Feb 2026)
- **(3)** Pure **logarithmic** scoring rule, **no correctness term**:
  `R(a, p̂, j) = log(p̂) if j(a)=1 ; log(1−p̂) if j(a)=0`, clipped with **ε = 0.001** to
  `log(max(p̂,ε))` / `log(min(1−p̂, 1−ε))`, then **normalized to [−1,1]**. Out-of-format
  responses get a fixed **−3**. QAMPARI multi-answer multiplies the reward by **5**.
  Note this is exactly the log score RLCR's Corollary 1 rejects; it is unbounded below
  before clipping, which is why ε exists.
- **(4)** gold + a rule-based judge `j(·)` (exact match; **word-overlap F1 > 0.5** on
  TriviaQA/QAMPARI) + the model's own `p̂` (integer 0–10 → [0,1]).
- **(5) ★ A designated span, and only that span.** The MDP is `s_t = (q, a, c_{1:t−1})`
  with the action space being the **next confidence token**. The answer `a` is generated
  first and then "treated as fixed inputs alongside the question" — **only the confidence
  generation is optimized.** PPO with a value baseline, Llama-3-8B-Instruct + LoRA.
- **(6)** N/A — PPO, no group. ⚠️ If ported to GRPO: with the answer frozen, `j` is
  constant per (q,a), so a group of confidence-only rollouts would have reward varying
  **only through `p̂`** — fine for the scoring rule, **zero accuracy signal by
  construction**.
- **(7)** `j(a)` is the only external input; delete it and both branches vanish. No
  gold-free ablation. The paper proposes (does not implement) LLM-as-a-judge / RM / NLG
  metrics as substitutes. ⚠️ **Structural caveat**: the log score is unbounded, so a
  noisy pseudo-label is far more destructive here than in RLCR's bounded Brier — one
  mislabeled confident-correct answer costs `log(ε) ≈ −6.9` before normalization.
- **(8) CALIBRATION ONLY, by construction** (accuracy is provably unchanged — the answer
  is frozen). TriviaQA, Llama-3-8B-Instruct: ECE 0.3459 → **0.0226**, AUROC 0.5858 →
  **0.8592**, accuracy 0.6310 → 0.6309. Qwen-2.5-3B AUROC 0.5981 → **0.9065**. Beats a
  trained probe on AUROC everywhere, and the probe costs accuracy (0.6231 vs 0.6497).
- **(9)** ε-clipping "prevents differentiation within [0,ε] and [1−ε,1]" — saturation
  handled by fiat. ⚠️ **Their own Goodhart warning about the metric**: on CommonsenseQA
  ECE stays flat while AUROC rises, and they conclude "a model consistently assigning
  moderate confidence values could appear well-calibrated under ECE, yet fail to offer
  meaningful distinctions." **Do not gate an experiment on ECE alone.** Format failures
  needed an explicit −3 penalty, implying degenerate format was observed.

### A4. SaySelf — arXiv:2405.20974, EMNLP 2024 Main
- **(2)** confidence expression **+ error localization in natural language** —
  "self-reflective rationales that clearly identify gaps in their parametric knowledge."
  The closest of the calibration family to an externalized metacognitive span with
  semantic content.
- **(3)** `R = 1 − 2·(1(response) − confidence)²` — an affine rescale of Brier onto
  [−1,1]. PPO.
  ⚠️ **There is no `+ 1_{y≡y*}` term.** `R = 1` for a correct answer at c=1 **and**
  `R = 1` for a wrong answer at c=0. By RLCR's Theorem 1 this is calibration-proper but
  **correctness-indifferent**: nothing penalizes producing an answer you know is wrong
  and declaring c=0. **This is the hedging degeneracy RLCR's first term exists to close**,
  and SaySelf is protected only by SFT initialization, not by the objective.
- **(4)** RL stage: gold via a containment heuristic. **SFT stage additionally uses**: 100
  sampled chains at T=1.2, clustering with the **Instructor embedding model** at
  similarity 0.9, target **`c = round(S_c/N × 10)` where `S_c` is the size of the cluster
  containing the correct response**, and **GPT-4** to write the first-person rationale.
- **(7) ★ Two separable dependencies, and one has a free substitute.** The RL reward
  needs gold. But the SFT confidence target `c = round(S_c/N · 10)` needs gold only to
  identify *which cluster is correct* — **take the largest cluster instead of the correct
  cluster and you have TTRL**, at zero extra cost, since the K-rollout clustering is
  already being computed. The rationale needs GPT-4 but not gold. **No gold-free ablation
  was run.** ⚠️ Note the paper's own motivation for adding RL: SFT alone "tends to produce
  homogeneous confidence levels" — so the **gold-dependent RL stage is doing the
  sharpening the self-supervised SFT stage cannot.**
- **(8) CALIBRATION ONLY.** Mistral-7B ECE: HotpotQA 0.3558, TruthfulQA 0.3368,
  StrategyQA 0.3907 (vs R-Tuning 0.4141 / 0.4111 / 0.4477). ⚠️ These absolute ECEs
  (~0.33–0.39) are an order of magnitude worse than RLCR's in-domain 0.03 or Rewarding
  Doubt's 0.023 — different setups, **do not present as comparable**.
- **(9)** SFT produced **inverted** confidence (lower for correct, higher for incorrect) —
  the stated reason RL was added. "Unfaithful CoT reasoning could produce unfaithful
  rationales" — the rationale may be post-hoc. Plus the correctness-indifference above.

### A5. SEED-GRPO — arXiv:2505.12346 (ZJU, 18 May 2025)
- **(2)** difficulty-aware allocation. ⚠️ There is **no metacognitive output** — the model
  never expresses anything; the metacognition lives entirely in the optimizer.
- **(3) Advantage modification, not a reward** (the paper says so explicitly).
  Eq. 6: `Â_i = A_i · f( α · SE(q) / SE_max(q) )`, `A_i = r_i − r̄` (Dr.GRPO-style, **no
  std division**), `SE_max = log G`, α = 0.02, ε = 0.2.
  Entropy estimator Eq. 5: `SE(q) ≈ −(1/K) Σ_k log p(C_k|q)`,
  `p(C_k|q) = Σ_{o_i ∈ C_k} π_{θ_old}(o_i|q)`.
  ⚠️ **The paper never defines `f`** — it says only "linear, exponential, or focal styles."
  From the reference implementation (`seed_grpo.py` L817–821):
  `advantages *= 1/(1 + alpha*semantic_entropies)` with `alpha = 0.0417/log(num_samples)`.
  So the actual operator is **`Â_i = A_i · 1/(1 + α·SE(q))`** — a *reciprocal*, not
  linear. **At G=8 the multiplier is bounded in [0.960, 1.000] — a ≤4% effect.**
- **(4)** gold (`r_i ∈ {0,1}` by rule-based verification) + the policy's own G rollouts +
  `π_{θ_old}` sequence probabilities. Clustering is **final-answer string identity only**
  for math (an NLI path exists in the repo but is "not feasible" for varying-length math
  answers).
- **(5)** advantage multiplier, broadcast over all tokens of every rollout in the group.
- **(6) ★ The clean illustration of the multiplicative-placement rule.** `SE(q)` is
  **constant across all G rollouts of a prompt**. Because it is applied *multiplicatively
  after* `A_i = r_i − r̄`, it is not erased by mean-subtraction — **but it contributes
  zero within-group discrimination.** It cannot change which rollout is preferred, only
  how much this prompt counts relative to other prompts in the batch. **Had it been added
  as a reward term it would have been exactly annihilated by `r̄`.**
- **(7) ★ Asymmetric, and this is the interesting part.** Deleting gold kills `r_i`, hence
  `A_i`, hence everything. **But `SE(q)` itself already needs no gold** — it is pure
  clustering of the policy's own rollouts. The modulation half is self-supervised; the
  base half is not. The natural composition — reuse the *same clusters* to produce a
  TTRL majority-vote `r_i` — is **free**, since the clustering is already computed.
  Not done, not mentioned.
- **(8) ACCURACY (no calibration metric is reported at all).** Qwen2.5-Math-7B avg over
  AIME24/AMC/MATH/Minerva/Olympiad: Dr.GRPO 51.4 → SEED-GRPO **56.6** (G=16: 58.2).
  ⚠️ **Cite with the caveat**: a +5.2-pt claim backed by a ≤4% advantage rescale is not
  mechanistically plausible on its face.
- **(9)** Answer-only clustering carries no reasoning-step semantics; G=10 is
  non-monotonic vs G=8 and G=16; **no collapse or gradient-magnitude analysis is given.**

### A6. RLPR — arXiv:2506.18254 (Tsinghua/NUS/OpenBMB, 23 Jun 2025)
- **(2)** self-verification **without a verifier** — "the LLM's intrinsic probability of
  generating a correct free-form answer directly indicates its own evaluation."
- **(3)** Eq. 2, probability reward: `r = f_seq({ p_i | o'_i ∈ y* })` with
  **`f_seq = (1/|y*|) Σ`** — the *mean* of the reference-answer token probabilities in a
  sequence `o'` where the model's answer has been **replaced by the reference answer**.
  The paper explicitly rejects the product form: "(0.01, 0.7, 0.9) and (0.05, 0.7, 0.9)
  yield vastly different scores under the product."
  Debiasing, Eqs. 3–4: **`r̂ = clip(0, 1, r − r′)`** where `r′` is the same score computed
  by decoding the reference answer **without intermediate reasoning `z`**. So the reward
  is *the improvement in reference-answer probability attributable to the model's own
  reasoning* — **a counterfactual over the reasoning span**, which is unusually close to
  what a meta-span reward wants. Curriculum filter: drop prompts with
  **`std(r̂) < β`**, β by EMA; empirically 0.5 (Qwen) / 0.9 (Llama) / 1.0 (Gemma).
- **(4)** the reference answer `y*` **as text to score, never as a verdict**; the policy's
  own token probabilities in one extra forward pass; own rollouts. **No verifier, no
  judge, no reward model, no post-processing.**
- **(6) ★ A subtle annihilation the paper does not flag.** `r̂` varies within the group
  (each rollout has a different `z`). But `r′` depends only on `(Q, y*)` — **it is
  constant per prompt and therefore exactly annihilated by GRPO's mean-subtraction.**
  What it *does* change is the `clip(0,1)` boundary, i.e. which rollouts get pinned to 0,
  and that does survive centering.
- **(7)** `y*` is load-bearing but **far more weakly coupled than any other gold-using
  row** — it is a string to take a likelihood of, not a verdict, so it tolerates free-form
  answers, partial credit, and synonyms. The gold-free variant is obvious: **score the
  likelihood of the majority-vote answer instead of `y*`**; the machinery is unchanged.
  **Not ablated.** ★ Reward-quality number worth quoting: **PR ROC-AUC 0.93 (math) / 0.81
  (general) vs a rule-based verifier's 0.95 / 0.61** — the self-signal is 2 pts worse on
  math and **20 pts better** on general domains. Even Qwen2.5-**0.5B** as the scorer beats
  a specifically-trained General-Verifier.
- **(8) ACCURACY.** Qwen2.5-7B-Base on 77k non-math prompts: TheoremQA 47.3 → **55.4**,
  Minerva 49.4 → **56.5**, MMLU-Pro 54.5 → 56.0, GPQA 34.2 → 37.6. Ablation: using
  sequence likelihood instead of the token-prob mean costs **−21.9 / −22.3** — the
  mean-vs-product choice is the largest design lever in the paper.
- **(9)** "Reward quality relies on the model's inherent probability calibration; smaller
  models show marginal gains on mathematical tasks." **Prompt-template sensitivity** is an
  explicit failure axis verifier-based methods do not have. "Probabilities of 1e−4 versus
  1e−5 can lead to a tenfold difference in reward."

## Group E — multi-turn self-correction and reflection RL

### E1. SCoRe — arXiv:2409.12917 (Google DeepMind)
- **(3)** Stage I, KL-constrained init (Eq. 3):
  `max_θ E[ r̂(y₂,y*) − β₂ D_KL(π_θ(·|x₁) ‖ π_ref(·|x₁)) ]` — maximize turn-2 reward while
  a **strong KL anchors turn 1** to the base model.
  Stage II, multi-turn RL with shaping (Eq. 4):
  `max_θ E[ Σ_{i=1,2} r̂(y_i,y*) − β₁ D_KL(π_θ(·|x_i) ‖ π_ref(·|x_i)) ]`, with the
  progress bonus folded into turn 2:
  **`b̂(y₂|y₁,y*) = α·( r̂(y₂,y*) − r̂(y₁,y*) )`**.
  **α = 10, β₁ = 0.01, β₂ = 0.1 (MATH) / 0.25 (MBPP).** Since `r̂ ∈ {0,1}`, the bonus takes
  only **{+10, 0, −10}**: it pays only for a correctness *flip* and charges heavily for
  c→i.
- **(4)** gold (or unit tests) at training time only; a frozen `π_ref` for both KLs. No
  judge, no probe, no group statistics.
- **(5)** per-turn scalar on the whole response of each turn; the bonus on turn 2 only.
- **(6)** Not GRPO — on-policy REINFORCE-style, no group normalization. The bonus varies
  across sampled trajectories for the same prompt.
- **(7) ★** Total collapse. `r̂` appears three times, and — critically — **inside the
  bonus as a difference of two gold-graded scalars**. Without gold only the two KL terms
  remain, which merely pull toward `π_ref`. **No gold-free ablation.** A sibling-majority
  substitute is structurally possible (the bonus needs only a per-attempt scalar) but
  **would corrupt exactly the flips SCoRe wants**: on a problem the model mostly gets
  wrong, majority says "wrong" for a correct attempt-2, deleting the reward.
- **(8) ACCURACY.** Gemini 1.5 Flash, MATH: acc@t1 52.6 → 60.0, acc@t2 41.4 → **64.4**,
  Δ(t1,t2) **−11.2 → +4.4**. HumanEval: Δ +3.0 → **+12.2**. Ablations: **w/o reward
  shaping** Δ +2.6 (bonus worth ~1.8 pts); **w/o Stage I** Δ +2.2; **w/o multi-turn**
  Δ = **−2.4** (self-correction goes negative).
- **(9)** **Behavior collapse** is the named enemy — models "learn to produce the best
  first response followed by making any minor edits." **Minimal-edit degeneracy** shown
  via edit-distance histograms. **Distribution shift between turns** is the stated reason
  SFT on self-correction traces fails. Δ_{i→c} 5.8% / Δ_{c→i} 1.4% for SCoRe vs 5.4% /
  3.6% for Pair-SFT — **the win is mostly in not breaking correct answers.**

### E2. Reflect, Retry, Reward — arXiv:2505.24726 (Writer, 30 May 2025)
**★ The published architecture closest to a `<|meta|>`-span reward.**
- **(3)** ⚠️ **There is no reward equation in the paper.** It is a gating rule plus an
  advantage mask, stated in prose: "If it succeeds however, we use GRPO to reward only the
  tokens that were generated in the self-reflection. This is possible by **setting the
  advantage terms for all other generated tokens to zero**. … we do not reward the correct
  answer, we only reward the self-reflection." So:
  ```
  reward fires  ⟺  [attempt-1 FAILS] ∧ [attempt-2 SUCCEEDS]
  Â_t = Â_GRPO   for t ∈ self-reflection span
  Â_t = 0        for all other t
  ```
  KL 0.001, lr 5e-7, effective batch 256 **failures**, ≤1,750 steps.
- **(4)** a **binary task validator only**. APIGen: exact match against the gold call
  (gold-dependent). **Countdown: "Does the equation evaluate to the target answer?"
  (gold-free).**
- **(5)** designated span = the self-reflection tokens, as an advantage mask; plus a
  **sample filter** — only attempt-1 failures enter the batch at all.
- **(6)** GRPO; varies within group — the G sampled reflections for the same failed
  problem differ in whether they rescue attempt 2. ★ If all G reflections fail (too hard)
  or all succeed (too easy), the group advantage is zero: **the method self-curricularizes
  to the model's reflection frontier.**
- **(7) ★ The strongest gold-free statement in the self-correction literature**, verbatim
  from §3: "it is sometimes possible to define a task-dependent validator that meets this
  criteria **without ground-truth labels**, such as … mathematical equations (Does the
  equation evaluate to the target answer?), or code (Does the generated code execute?)."
  **One of their two headline tasks (Countdown, +16.0% avg) is already fully gold-free** —
  though it is not framed as an ablation. Could sibling majority substitute? Structurally
  yes (the gate needs only `¬correct(y₁) ∧ correct(y₂)`), **but majority voting is
  systematically wrong precisely where attempt-1 fails.** Not tested.
- **(8) ACCURACY.** APIGen Qwen-2-1.5B 34.8 → **52.9** (2nd attempt); Countdown
  Qwen-2.5-1.5B 10.2 → **45.0**. Averages **+9.0% function calling, +16.0% Countdown**.
  Trained Qwen-2-7B beats vanilla Qwen-2-**72B**. Catastrophic-forgetting check: <1%
  degradation on MMLU-Pro/GSM8K/HellaSwag/MATH.
- **(9)** No hacking or collapse reported. **Capability floor**: Qwen2/2.5-0.5B and
  Llama3.2-1B were dropped entirely, and **Llama3.2-3B "was unable to learn to
  self-correct on the function calling task."** ★ **Length shrinks, it does not blow up** —
  trained reflections are "much shorter, clearer, and more generalisable" vs vanilla
  "long, confusing, and redundant." The authors flag this as contradicting the
  CoT-verbosity intuition and leave it open.

### E3. Credit Assignment with Resets — arXiv:2605.25507 (SRPO = Self-Reset PO)
**★ The published analogue of the cd9 "인용 자리 반사실" (cited-position counterfactual).**
- **(3) Advantage/sampling modification, not a reward.** Reward is unchanged binary RLVR.
  Algorithm: sample iid rollouts until the first incorrect one (the **seed**); discard it;
  then **RRPO** picks `h* ~ Unif{1..H_s}` (random reset) while **SRPO** sets `h*` = the
  **self-localized index of the seed's first erroneous thought**; form `x* = (x₀, ỹ_{1:h*−1})`
  and sample G suffix rollouts from it. Each of the two groups is normalized
  independently. Shared-prefix loss:
  `L_SP(θ) = −(1/G) Σ_i (1/T_i) Σ_t Â_i log π_θ(y_{i,t} | x*, y_{i,<t})`,
  with the **prefix tokens masked**. No PPO clipping, no KL. Default 1×4 under an
  8-rollouts-per-prompt budget.
- **(4) ★** The localizer is **the policy itself**: "the same policy π_θ that generated
  the seed is prompted to analyze its own reasoning trace … and returns the index of the
  first incorrect thought. **SRPO requires no external step-level feedback.**" Gold is
  needed for `r_H` and to identify which rollout is the failed seed. Claude Opus 4.5
  appears **only in the post-hoc audit**, never in training.
- **(5)** per-token advantage with **prefix masking** — credit flows only to suffix tokens.
- **(6) ★** Two groups, each normalized within itself, so the shared-prefix group's zero
  point is **the difficulty of the reset state, not of the prompt**. Measured consequence:
  shared-prefix rollouts deliver **~2.5× more per-token gradient signal** than base
  rollouts (9.27e−5 vs 3.73e−5), higher at 10 of 11 training steps.
- **(7) ★★ The row that answers "does a self-supervised position label carry signal?"**
  Decomposition: reset-point selection is **already gold-free**; seed identification and
  `r_H` need the verifier. And the paper **quantifies the gold-free part**: auditing
  against a Claude Opus 4.5 oracle on ≈17-step chains, (a) SRPO concentrates resets in the
  early-middle while RRPO is ~uniform, "indicating that SRPO is actively localizing rather
  than resetting blindly"; (b) **roughly half** of SRPO's localizations sit at or before
  the oracle's failure step; (c) correction rate decays monotonically with deviation from
  the oracle; (d) **clean prefixes correct nearly 2× as often as erroneous ones — 28.7%
  vs 16.3% Pass@4.** That is a directly usable effect size for a self-chosen position
  label, and it is well above the cd9 G1 gate's observed .511/.551/.569.
- **(8) ACCURACY.** Qwen2.5-14B-Instruct and OLMo-3-7B-Instruct, 400 NuminaMath problems,
  3 seeds, compute-matched. **SRPO best on 7/10 tasks (Qwen) and 6/10 (OLMo); RRPO ≈ GRPO.**
  MATH-Lvl5 52.1 → 55.2; StrategyQA 69.5 → 74.9; CSQA 66.1 → 80.6; physics 26.9 → 45.5.
  LiveCodeBench: matching pass rates **2–3× faster**. ⚠️ **SCoRe collapses on OLMo-3-7B**
  (hmmt 0.0, below the untrained base).
- **(9)** **Localization quality is the active bottleneck** — half the resets overshoot
  into the erroneous region. **Reset-group cold start**: the shared-prefix group starts
  with lower pass rates and is near-degenerate until self-correction improves. Requires
  verifiable rewards; "extending reset-based frameworks to non-verifiable settings …
  remains an open direction." ★ **No distribution shift by construction** — thought
  boundaries are self-determined during generation.

### E4. SRPO (self-reflective) — arXiv:2608.23493 (24 Aug 2026)
- **(3) A reward *substitution*** — the outcome reward never enters the gradient.
  Stage 1: `p = Reflect_{π_θ}(x, τ, o)` (2–5 bullets, ~90 tokens);
  `x̃ = [p ; x]` (**prepend** — "reset with memory"); `π_T(·|x) := π_θ(· | [p;x])` — the
  **self as teacher, same weights**.
  Stage 2, per-token reverse-KL reward:
  **`r_t = sg[ log π_T(a_t|s_t) − log π_{θ_old}(a_t|s_t) ]`**, teacher-forced on the
  *student's own* on-policy tokens; `E` under `π_{θ_old}` recovers `−KL(π_{θ_old} ‖ π_T)`.
  `r̄ = (1/|V|) Σ_{t∈V} r_t`; `A_t = r_t − r̄`; clipped surrogate, ε ≈ 0.1–0.2.
  **There is no outcome term in Stage 2 at all.**
- **(4)** the terminal outcome `o` **only to condition the reflection**; the policy's own
  logits twice; its own rollout. No external teacher (that is the ablation baseline), no
  judge in the loop.
- **(6) ⚠️ "Group-relative" in name only.** `r̄` is a mean **over token positions within
  one trajectory**, not over a prompt group. There is no prompt group. Ablation
  "single-sample advantage (no group)": 73.3 → 71.1 on AIME'24.
- **(7) ★ The most gold-light architecture in the survey**: 100% of the token-level
  training signal is a log-ratio between two forward passes of the same weights. Gold
  appears only in Stage 1, to tell the reflection what happened; removing it degrades the
  *content* of `p` but breaks no equation. Ablations: `w/o reflection` 73.3 → **65.8**;
  `w/ outcome-only feedback` → **67.2** (i.e. the *diagnostic content* is worth +6.1 over
  a bare outcome signal); **`w/ external teacher reflection` → 71.5 — self-reflection
  beats an external teacher's reflection.** Forward KL instead of reverse → 69.4; append
  instead of prepend → 68.5; no state reset (Reflexion-style) → 66.3.
- **(8) ACCURACY.** Qwen3-8B from a shared SFT init: AIME'24 **73.3±1.4** at **0.26× GRPO
  train FLOPs** vs GRPO 68.0, SCoRe 70.2, OPD-72B-teacher 72.5. WebShop 64.7, ALFWorld
  76.8, SWE-Bench-Lite 31.2, avg steps 10.2 (shortest of all methods). Scaling: **1.5B
  +7.8, 8B +5.3, 32B +3.8 over GRPO — gains shrink with scale**, i.e. this family favours
  small models.
- **(9) ★ A reflection-quality failure taxonomy worth copying as a diagnostic**: generic
  advice **42%**, incorrect diagnosis **35%**, beyond capability **23%**. Quality→gain
  correlation **r = 0.72** (score-5 reflections give 34% improvement, score-1–2 give ~5%).
  Human eval: 68–74% Effective, 18–22% Redundant, **8–10% Detrimental**. Verbose
  reflections (>10 points) cost 3.3 pts. Context-accumulation degeneracy costs 7.0 pts.
  ★ **Explicitly tested and ruled out: self-teacher quality collapse** — reflection
  helpfulness across 500 iterations is flat (3.72 → 3.76 → 3.79).

### E5. RLVMR — arXiv:2507.22844 (Tencent, 30 Jul 2025)
**★ The one paper in this survey that actually ran the gold-removal ablation, and the
answer was negative.**
- **(3)** Tags `<planning> <explore> <reflection> <monitor>`.
  `R(τ) = r_s` on success, 0 otherwise. Dense per-step meta rewards:
  `r_planning` **iff the trajectory ultimately succeeds**; `r_explore` iff the action
  targets a **new** object/location; `r_reflection` iff a `<reflection>` step is **followed
  by a corrective action after a sequence of failures**; `r_t^format = −λ_format` on
  malformed output. GRPO-MR:
  `A_k^traj = (R(τ_k) − μ_R)/σ_R`; `A_{t,tag}^MR = (r_{t,tag}^MR − μ_tag)/σ_tag`;
  **`A_t = α·A_k^traj + (1−α)·A_{t,tag}^MR`**, with **α = 0.5, λ_format = 0.1,
  λ_KL = 0.01**. ⚠️ `r_s` and the three `r_MR` magnitudes are **never given numerically
  [unverified]**. Requires a cold start (SFT on 200 GPT-4-annotated trajectories).
- **(4)** rule-based tag/action checks + environment state + the environment success
  signal + `π_ref`. GPT-4 is cold-start only, never in the RL loop.
- **(6) ⚠️ Not prompt-group centering — batch centering.** The reference class for a
  reflection step is **all other reflection steps in the batch**, across prompts.
- **(7) ★★ THE ABLATION (Table 3, Qwen-1.5B, L2 unseen):**
  | Variant | ALFWorld L2 | ScienceWorld L2 |
  |---|---|---|
  | RLVMR (full) | **56.3** | **26.5** |
  | **w/o outcome reward** | **12.5** | **7.8** |
  | w/o meta-reasoning reward | 45.3 | 20.3 |
  | w/o cold start | 40.6 | 18.8 |
  Rule-checkable meta tags **alone**, with the outcome signal deleted, collapse from 56.3
  to **12.5** — far below even the SFT cold start. Their reading, verbatim: "the
  meta-reasoning rewards are locally effective … but **without the final outcome signal,
  the agent cannot learn which explorations ultimately lead to a successful trajectory**"
  and "**Outcome-based rewards remain indispensable.**" Removing the *meta* rewards costs
  only 11.0 points. **This is the single hardest published number against a purely
  rule-checkable metacognitive reward.**
- **(8) ACCURACY.** Qwen-7B ALFWorld L0/L1/L2: **91.4/91.8/83.6** vs GiGPO 89.5/90.2/67.2
  (**+16.4 on L2**). Qwen-1.5B ALFWorld L1 87.9 beats GPT-4o's 66.0.
- **(9)** GRPO-7B hits a **31.2% repetitive-action rate** on L2 and 14.8% invalid actions;
  RLVMR halves both. ★ **Tag-spamming is pre-empted structurally, not observed**: the
  −0.1 format penalty plus the *conditional* reward (a `<reflection>` pays only if a
  corrective action follows) means emitting tags without behavior earns nothing.
  Policy collapse without the cold start (−15.7 pts).

## Group F — metacognitive CONTROL: abstention, allocation, early exit

### F1. TruthRL — arXiv:2509.25760 (Meta/UVA, ICML 2026)
- **(3)** Binary baseline `r = {+1 correct, −1 otherwise}` vs **ternary
  `r = {+1 correct, 0 uncertain, −1 incorrect}`** — in the usual notation, abstain = 0 and
  **c = 1** for a wrong answer.
- **(4)** gold **plus an LLM judge** (Llama-3.3-70B) doing three-way
  correct/uncertain/incorrect classification — a rule matcher cannot produce the
  "uncertain" class.
- **(6) ★ The argument is precisely a group-centering argument**: under binary reward an
  abstaining rollout and a hallucinating rollout receive **identical** advantage, so
  centering cannot separate them. Ternary breaks the tie.
- **(7)** Fatal. ⚠️ **Partial natural experiment inside the paper**: replacing the LLM
  judge with a rule-based verifier (still gold-using, just coarser) **collapses the model
  into abstaining on the vast majority of queries, truthfulness −3.6.**
- **(8)** CRAG hallucination **43.5% → 19.4%**, truthfulness 5.3 → 37.2. ⚠️ The headline
  metric is *truthfulness*, not accuracy.
- **(9)** Over-abstention collapse under a coarse verifier; on hard questions TruthRL
  itself is at **84.5% uncertainty** — accuracy and truthfulness diverge by design.

### F2. Rewarding Intellectual Humility — arXiv:2601.20126 (27 Jan 2026)
Ternary `(+1, r_abs, −1)` with **`r_abs` swept; recommended ≈ −0.25 to 0.3.** The useful
result is the *shape*: too low → no abstention learned; too high → abstention collapse.
**The safe band is narrow and sits near zero.** Over-abstention is worse in **smaller
models**; "larger models showed greater robustness to abstention incentives."

### F3. Alignment for Honesty — arXiv:2312.07000, NeurIPS 2024 (**SFT, not RL**)
Included for its definitions, which every RL abstention paper implicitly optimizes:
`S_prudence = (N⑧+N⑨)/(N⑤+N⑥+N⑧+N⑨)`, `S_over-cons = N⑦/(N①+N④+N⑦)`,
`S_honesty = ½(S_prudence + (1 − S_over-cons))`.
★ **Known/unknown label**: sample **m = 10** responses; the question is *known* if the
fraction correct ≥ **τ = 0.1**. **That is exactly the group pass rate GRPO already
computes** — and its gold-free substitute is the *agreement* rate among the m samples.
Nobody in this batch makes that swap.

### F4. L1 / LCPO — arXiv:2503.04697 (CMU)
**The cleanest demonstration that a rule-checkable reward controls behavior and cannot
report whether the behavior helped.**
- **(3)** L1-Exact: `r = 1(y = y_gold) − α·|n_gold − n_y|`, **α = 0.0003**.
  L1-Max: `r = 1(y = y_gold) · clip(α·(n_gold − n_y) + δ, 0, 1)`, **δ = 0.5**.
- **(7) ★** `α·|n_gold − n_y|` is a pure tokenizer count — **no gold, no verifier,
  nothing.** Delete the indicator and L1-Exact becomes a *perfect* length controller with
  **zero pressure toward correctness**. In L1-Max the length term is a *multiplier* on the
  indicator, so deleting gold makes the optimal policy "emit `n_gold` tokens of anything."
- **(8)** **100–150% relative / 20–25% absolute** over S1 budget-forcing at 512/1024
  tokens; L1-Max 1.5B surpasses GPT-4o at equal token length.
- **(9)** Does not generalize to requested lengths longer than trained; on OOD MMLU
  length-adherence error **>40%**. Pure SFT for length control failed entirely.

### F5. AdaptThink — arXiv:2505.13417, EMNLP 2025 Main
- **(3)** Constraint `max E 1(y₁ = </think>) s.t. E R(x,y) ≥ E_{y'∼π_θref} R(x,y')`;
  penalized form with `δ = 1/λ`:
  **`A(x,y) = 1(y₁=</think>)·δ + R(x,y) − R̄_ref(x)`**, where
  `R̄_ref(x) = (1/K) Σ_{i=1}^K R(x, y'_i)`, `y'_i ∼ π_θref`. **K = 16, δ = 0.05, ε = 0.2.**
  Importance sampler forces `t = 1` to be 50/50 `</think>` vs a long-think opener.
- **(6) ★ Not GRPO group-centering** — the baseline is a **pre-sampled, frozen, per-prompt
  constant `R̄_ref(x)`**, making the advantage an absolute comparison against the
  *initial* policy. That is what lets "don't get worse than you started" be enforced
  per prompt, and it is a design option worth noting for any meta arm.
- **(7)** Delete `R` and `R̄_ref` and `A = δ·1(nothink)` — **a constant positive bonus for
  not thinking with nothing opposing it → immediate total collapse to always-NoThinking.**
  Self-supervised substitute: replace `R` by majority-vote agreement across the K rollouts
  and `R̄_ref` by the init model's agreement rate — the equations are unchanged in form.
  ⚠️ Caveat: agreement is inflated exactly on problems the model is *consistently* wrong
  about, which is where Thinking should win.
- **(8)** 1.5B: **−53.0% length, +2.4% accuracy**; 7B: −40.1% / +2.4%. ★ The δ sweep is
  the useful artifact: δ=0 → +5.5 acc / −32.8% len; δ=0.05 → +2.4 / −53.0%;
  **δ=0.1 → −0.5 acc / −64.5% len** (accuracy goes negative).
- **(9)** **Mode collapse** — at init `π_θold(y₁=</think>|x) ≈ 0`, so NoThinking is never
  sampled and never learned (cold-start impossibility) — hence the forced 50/50 sampler.
  **Implicit thinking**: the model learns to reason at length *inside* NoThinking mode,
  defeating the measurement.

### F6. Thinkless — arXiv:2505.13379 (NUS)
- **(3)** `r(a, y*, c) = { 1.0 if c=<short> and correct; 1.0 − γ if c=<think> and correct;
  −1.0 if incorrect }`, γ ∈ (0,1). **DeGRPO** decouples the control-token loss from the
  response loss with weight **α = 1/1000** on the control token; advantage is
  `Â = r − mean(r)` — **mean-centering only, deliberately no std normalization.**
- **(9) ★ A named, diagnosed collapse whose mechanism transfers directly to any
  single-token metacognitive decision.** Vanilla GRPO collapses within ~120 steps because
  the control-token gradient is normalized by *total response length* and
  `T_think ≫ T_short`, so the `<think>` token's update is systematically suppressed.
  DeGRPO's α = 1/1000 exists solely to fix this.
- **(8)** Long-form usage cut 50–90%; AIME **100% think** (correctly refuses to shortcut
  the hard set), GSM8K 13.31% think.

### F7. AdaCoT — arXiv:2505.11896 (ByteDance Seed)
- **(3)** `R(x,r) = R_base(x,r) − α₁·P_miss(x,r) − α₂·P_over(x,r) − γ·P_fmt(r)`; α₁, α₂
  are the Pareto knobs tracing the accuracy/cost frontier.
- **(4) ★** The "should CoT trigger" label comes from **principle-guided assessment by a
  separate 15B auxiliary model**, and `R_base` from a reward model — **no gold answer
  anywhere.** ⚠️ But this is *relocated* supervision, not self-supervision: "gold-free"
  ≠ "label-free". Delete the labeler too and the decision boundary has no anchor.
- **(6)** **PPO with a critic — no group centering at all.** The only such row here.
- **(8)** Production traffic: CoT trigger rate down to 3.18% (mobile), −69.1% tokens.
  Benchmark avg **62.8% at 53.3% CoT rate vs 65.0% at 100% CoT** — an explicit ~2.2-pt
  accuracy *sacrifice* for half the compute. Trigger ~100% on AIME/MATH, **<1% on SimpleQA**.
- **(9)** **Decision boundary collapse** — "the model might revert to homogeneous
  behavior, either always or never triggering CoT." Fix: **Selective Loss Masking**,
  `L_SLM = Σ_{k ≠ k_decision} ℓ_k` — mask the loss on the single pivotal decision token.

### F8. SelfBudgeter — arXiv:2505.11274 (PKU + ByteDance)
**★ The sharpest evidence that a self-prediction reward needs an outcome anchor.**
- **(3)** `R(C,F,ℓ,b,b_max) = { r_f if F=0 ; PB(b,b_max) + PreB(·) if F=1 }`;
  `PB = 0 if b ≤ b_max else r_b`;
  `PreB(s_min, s_max, ℓ, b, α, b_best) = { s_min if |ℓ−b|/b > α ; s_min + (s_max−s_min)·0.5·(1+cos(·)) else }`.
  **r_f = −1, r_b = −0.4; correct band [0.5, 1]; wrong band [−0.5, 0].**
  The model **first outputs a token budget `b`** — its own difficulty estimate,
  externalized — then solves within it.
- **(5) ★ Structure worth copying: correctness selects the reward *band*; the budget term
  positions within the band.**
- **(6)** GRPO, 5 rollouts/prompt; `b` is generated per-rollout so **the predicted budget
  itself varies within the group**, which is what makes budget prediction learnable.
- **(7) ★★** `PreB` compares `ℓ` to **`b`, which the model itself emitted** — entirely
  self-referential, rule-checkable, **no gold, no verifier, not even an external target.**
  A gold-free SelfBudgeter trains perfectly well and is **trivially hackable, which the
  paper observed happening**: "models inflate predicted budgets mid-training to exploit
  maximum reward band." **Correctness is the only thing that makes a self-predicted
  budget mean anything**; without it, "predict what you'll do, then do it" is satisfied by
  any self-fulfilling prophecy.
- **(8)** GSM8K 1.5B 73.09%/2865 tok → **84.10%/1232** (+11 pts, 43% compression);
  MATH500 74.93 → 78.47 at 44% compression; AIME2025 22.22 → 21.11 at **70% compression**.
- **(9)** budget-inflation hacking; over-compression eliminates 3.94% of genuine reasoning
  steps; cold-start sensitivity; adherence degrades on hard problems.

### F9. AdaCtrl — arXiv:2505.18822 (TMLR 2026)
**★ The single most transferable gold-free idea in the control family.**
- **(3)** The model emits an explicit **`[Easy]` / `[Hard]` self-assessment tag**.
  `r(y_i) = r_o(y_i) + α·r_x(y_i) + β·r_l(y_i)`, **α = β = 0.5**.
  `r_o = ±1` by correctness.
  **Difficulty-estimation calibration**:
  `r_x(y_i) = { 1.0 if I(t_i, t̂_i)=1 ; 0.0 if I(t_i, t̂_i)=0 ; −1.0 if no tag }`, where
  `t_i` is the model's **self-generated tag** and **`t̂_i` is derived from accuracy across
  rollouts with threshold δ**.
  Difficulty-aware length: `r_l(y_i) = 1 − [1 − cos((l_ij/L_i)π)]/2` **only when
  `t_i = [Easy]`** — length pressure applies only when the model has claimed the problem
  is easy.
- **(6) ★** `t̂_i` is **constant per prompt** (a thresholded group pass rate) while `t_i`
  **varies per rollout**, so `r_x` varies within the group. The paper notes this enables
  "per-problem difficulty recalibration as model capability evolves" — **the difficulty
  label is policy-dependent and moves during training.**
- **(7) ★★** `r_l` is a pure rule check. `r_o` dies without gold. **But `r_x` only needs
  `t̂_i`, and `t̂_i` is a thresholded group statistic** — swap group *pass rate* for group
  *agreement rate* and you get a **fully gold-free self-assessment calibration reward**:
  *"was your `[Easy]`/`[Hard]` claim consistent with how much your own rollouts agreed?"*
  A monitoring reward scored entirely from the policy's own rollouts. **Nobody does this;
  AdaCtrl is one substitution away from it.**
- **(8)** vs R1-SFT-RL: GSM8K **+2.05% acc, −91.04% length**; MATH500 **+7.20%, −62.05%**;
  AIME2025 +1.67%, −12.14%; AIME2024 −0.00%, −10.06%. ★ **Compute is cut hardest exactly
  where it was least needed** — the allocation result one wants.
- **(9)** ⚠️ **Not analyzed at all** — no study of misclassified difficulty. The paper
  rewards calibrated self-assessment and never audits what happens when it is wrong.

### F10. GRPO-LEAD — arXiv:2504.09696
- **(3)** `R_accuracy(o|q) = { exp(−α·z) if correct ; −1 if incorrect }`,
  `z = (|o| − μ)/(σ + ε)`.
  Group correctness ratio `ρ_q`; **logistic difficulty reweighting**
  `w(ρ_q) = A + (B − A)/(1 + exp[k(ρ_q − ρ₀)])`, **A = 0.4, B = 1.5, ρ₀ = 0.75, k = 10**,
  applied as **`w(ρ_q)` to correct responses and `w(1 − ρ_q)` to incorrect ones.**
- **(6) ★** `ρ_q` is a per-prompt constant, **but the asymmetric application makes the
  effective multiplier vary within the group** — it tilts the correct/incorrect balance
  rather than uniformly rescaling the prompt. This is the second published trick (after
  SEED-GRPO's multiplicative placement) for keeping a per-prompt scalar alive under
  centering, and it is the stronger of the two.
- **(7)** `exp(−α·z)` is rule-checkable. `ρ_q` → **group agreement rate**, same logistic,
  no gold, **zero structural change**, because `w(·)` only ever consumed a scalar in [0,1].
- **(8)** 14B AIME24 pass@1 0.641 → 0.650, cons@32 0.833 → **0.867**; AIME25 pass@1
  0.505 → **0.539**. Modest accuracy gain *with* shortening.
- **(9)** Does not generalize past math — **coding tasks show length *increases***.

### F11. RISE / Trust-But-Verify — arXiv:2505.13445 (NeurIPS 2025)
- **(3)** `r_o(y,y*) = { 1 if boxed and matched ; −0.5 if boxed unmatched ; −1 if unboxed }`.
  ⚠️ **There is no separate verification reward formula** — "the original reward r from
  the generation phase is **reused as the ground-truth score** for the verification task,"
  and verification is correct when the model's self-assigned score matches `r_o`.
- **(6)** PPO with GAE, no group centering.
- **(7) ★** Structurally fatal, and worth calling out as a category: **RISE's
  "self-verification" is distilled from the gold verifier — the verification target
  literally *is* `r_o`.** Remove gold and there is no verification label at all. This is
  the cleanest example of a metacognition reward that is supervised monitoring in
  disguise. A gold-free version must change what is being taught from "predict the
  verifier" to "predict your own reliability."
- **(8) ★ The finding is the gap**: RISE-3B 33.5% vs Zero-RL-3B 32.5% (**+1.0 accuracy**);
  RISE-7B **+1.2**. But RISE-1.5B self-verification **74.5% vs 26.8% (+47.7 pp)**.
  **Learning to verify well did not translate into solving much better.**

### F12. DEER — arXiv:2504.15895 (**training-free, no reward**)
Monitors reasoning-transition markers ("Wait"), induces a trial answer, scores confidence
μ from logits, **exits when μ > λ**. **−19.1% to −80.1% CoT length with +0.3% to +5.0%
accuracy** across 11 LRMs / 10 benchmarks. ★ Relevance: a **fully gold-free,
inference-time controller built purely on the model's own confidence**, strong enough to
control compute *and* improve accuracy with no verifier anywhere. It is the existence
proof the RL papers lack — and it cannot learn (λ is hand-tuned).

### F13. The Hallucination Tax of Reinforcement Finetuning — arXiv:2505.13988 (ACL Findings EMNLP 2025) [abstract-level]
Standard RFT **reduces refusal rates by >80%** — ordinary verifiable-reward RL actively
destroys abstention. Mixing in **10% synthetic unanswerable math (SUM)** restores it
(Qwen2.5-7B 0.01 → 0.73). This is the empirical reason F1/F2 exist.

## Group G — on-policy self-distillation and span/counterfactual credit

Every paper here defines its span signal as a **log-ratio between two forward passes of
the same policy** — one with privileged context, one without. The privileged-context
question splits the batch cleanly:

| Method | Privileged context = gold? | Gold in the *meta* signal? |
|---|---|---|
| SD-Zero 2604.12002 | **No** — own response + a binary-reward *string* | No |
| AMR-SD 2605.18529 | **No** — self-generated hint/critique + peer rollout | No |
| TAPO 2606.18844 | **No** — own correct sibling rollout | No |
| IBPO 2605.16302 | **No** — sibling rollouts | No |
| AntiSD 2605.11609 | **Partly** — verified rollout, **falls back to gold solution** | Yes on fallback |
| RLRT 2605.10781 | **No** — own correct rollout (explicitly contrasted with RLSD's gold) | No |

### G1. Self-Distillation Zero — arXiv:2604.12002 (Princeton PLI)
- **(3)** Phase 1: `L_SRT = L_revision + L_generation`, standard NLL.
  Phase 2, on-policy self-distillation:
  **`L_SD(θ) = E_{(x,a)~D} E_{y~π_θ(·|x)} Σ_t D_KL( π_θ(·|x, y_{<t}) ‖ π_{θ_SRT}(·|x, y, P_r, y_{<t}) )`**
  ★ **The privileged context is a literal two-string switch on the binary reward:**
  ```
  P_r = "Let me rephrase the above solution."                        if r(y_init,a) = 1
        "Wait, this response is not correct, let me start over."     if r(y_init,a) = 0
  ```
  **No reference solution is placed in the teacher's context.** Uniform per-token weight;
  teacher and student score the *same* string so alignment is trivial; the asymmetry is
  purely in the conditioning. ⚠️ The equation puts the student first (`D_KL(student ‖
  teacher)`, reverse/mode-seeking) while the prose calls it forward KL — naming conflict.
- **(6) ★** N/A in the best way: SD-Zero **replaces** GRPO. A per-token KL is never
  group-centered, so **nothing cancels within a prompt group.** That is a genuine
  structural advantage over any group-centered reward.
- **(7)** Gold appears only as `r ∈ {0,1}`. A self-supervised substitute is one step away —
  replace `r` with own-majority agreement and keep the two-string switch. **Not run.**
  ⚠️ Related evidence in their Table 8: the *SDFT baseline* "degrades substantially" when
  given only final-answer labels instead of full gold solutions — **gold-solution-
  conditioned teachers are the fragile ones**; reward-string conditioning is robust.
- **(8) ★ Headline model is Qwen3-4B-Instruct**: avg over 8 benchmarks **49.8 → 60.3**
  (GRPO 53.1, RFT 54.3, SDFT 51.2). AIME24 59.6→68.3; AIME25 45.8→60.0; HMMT25 26.7→45.4.
- **(9)** **Thinking models hurt** — the reviser cannot separate productive exploration
  from genuine error in long chains. ★ Fig. 4: **on incorrect responses the KL mass
  concentrates on a small token fraction; on correct ones it is flat — the localization
  signal only exists on wrong rollouts.**

### G2. AMR-SD — arXiv:2605.18529
**The paper architecturally closest to a verbal `<|meta|>` span, and the one whose failure
mode is the most relevant warning.**
- **(3) Advantage modification.**
  `I_t^CIG = log( π_{θ_sg}(a_t|s_t, c_i) / π_θ(a_t|s_t) )`; `Î_t = clip(I_t, −κ, κ)`, **κ = 5**;
  `Δ_t = 1{A_i ≥ 0}·λ_eff·max(0, Î_t − τ) + 1{A_i < 0}·γ_eff·max(0, −Î_t − τ)`;
  **`Â_{i,t} = A_i · (1 + Δ_t)`**; `{λ_eff, γ_eff} = {λ,γ}·max(0, 1 − t_global/T_decay)`,
  **λ = 0.2, γ = 0.1 (deliberately asymmetric 2:1), T_decay = 50.**
  ★ **The reflection bottleneck**, verbatim: "Instead of exposing the teacher to raw
  ground-truth solutions during token rescoring, we enforce a Meta-Reflection phase: the
  model generates an encouraging `<hint>` for successful trajectories or a targeted
  `<critique>` for failed trajectories, using verifier-approved peer rollouts when
  available." `c_i` is that self-written span. **Neither branch receives ground truth.**
- **(6)** `Δ_t` varies within group, but it **multiplies `A_i`**, so an all-correct or
  all-wrong group still yields zero gradient. **A modulator, not a source of within-group
  variance.**
- **(8)** Qwen3-8B math avg **62.7 vs GRPO 60.3 vs RLSD 57.2**. Ablation: removing
  Meta-Reflection costs **−3.8**.
- **(9) ★★ The warning for a 4B non-thinking policy**, verbatim: "Effectiveness of the CIG
  mechanism is fundamentally contingent on the quality of the self-generated
  meta-reflections. When the base model lacks sufficient introspective capability, the
  produced hints and critiques tend to be vague or factually miscalibrated, propagating
  corrupted supervision signals." **On Qwen2.5-7B-Instruct and on Qwen3-8B with
  `enable_thinking=False`, reflection quality was insufficient and it failed to beat
  GRPO.** A 4B non-thinking policy sits below both.

### G3. TAPO — arXiv:2606.18844
- **(3)** ZPD gate `|P| ≥ n_pos AND |N| ≥ n_neg` (**n_pos=2, n_neg=4, m_max=4**);
  **decoupled advantages** — `G_orig` and `G_ref` normalized in **separate groups**
  (explicit anti-contamination). `L_TAPO = L_GRPO(G_orig) + λ·L_ref(G_ref)`, **λ = 1.0**;
  OOD-token suppression `s_t = log p_θ(y_t|·) + H[p_θ(·)]`,
  `w_t = clamp(exp(s_t), 0.01, 10.0)`, applied per token in `L_ref`. **No tokens are
  masked** — the error prefix is trained on, just down-weighted.
  The reference is **`y⁺ ∈ P`, a correct rollout from the same group**, not gold.
- **(8) ★ The decorative-vs-causal measurement worth copying.** TAPO separately reports
  **Direct Solution Rate** (first pass) and **Effective Reflection Rate** (does the
  correction work): DSR GRPO 34.0/22.1/28.0 → TAPO 47.5/38.0/50.4 (**+13.5/+15.9/+22.3**);
  ERR GRPO 52.0/51.5/33.8 → TAPO 63.4/56.3/36.8 (**+11.4/+4.8/+3.0**).
  **First-pass gains dominate reflection gains** — training on reflection improved the
  *non-reflective* pass more than the reflective one.
- **(9)** Without cold start it loses to GRPO on two of three benchmarks. Construction
  fires only inside the ZPD — dead on easy and on too-hard problems. **Parsing success
  rate only 80–90%.**

### G4. IBPO — arXiv:2605.16302
- **(3)** ⚠️ **The counterfactual baseline is NOT leave-one-out, NOT prefix ablation, NOT
  step resampling.** It pairs the target trajectory with K−1 sibling references, preferring
  correct siblings. `φ_i = 0` if `τ_i` correct, else `s_i ∈ [0,1]`;
  **`R'_i = R(τ_i) + λ·φ_i`** then `Â'_i = (R'_i − mean)/std`.
  Concrete instantiation: `Δ(x; y, y_ref) = ρ·1[r(x,y)=0 ∧ r(x,ŷ)=1]`, **ρ = 0.5**,
  where `ŷ` is the revision produced from `x̃ = (x; y, y_ref)`. **λ ≈ 0.6.**
  Mask variant: `m_t = 1[t ∉ U]`, `U` = tokens unchanged under **Levenshtein alignment**.
- **(6) ★** Shaping happens **before** centering, so `φ_i` varies within the group by
  construction — it is nonzero only on incorrect trajectories. **The one paper whose
  design explicitly turns on within-group variation.**
- **(8)** Qwen3-32B AIME25 **85.3±1.2** vs GSPO 77.1, GSPO+SCoRe 78.3. ⚠️ **K=1 collapses
  to 78.6** — essentially all the gain requires ≥2 counterfactual siblings.
  ⚠️ **No model ≤4B; no 4B-scale evidence at all.**
- **(9)** "If the counterfactual trajectories contain systematic errors, the comparison
  signal may weaken" — correlated sibling errors kill it. Variance reduction is proven
  only under their Condition E.2, "not an unconditional guarantee."

### G5. AntiSD — arXiv:2605.11609
**★ The diagnosis that explains why gold-conditioned self-distillation would destroy a
`<|meta|>` span.**
- **(3)** `u_t = log π_θ(y_t | x, c, y_{<t}) − log π_θ(y_t | x, y_{<t}) = PMI(y_t ; c | x, y_{<t})`;
  **`A_t^AntiSD = −φ(u_t)`**, `φ(u) = ½[softplus(u) − log 2]`;
  `A_{i,t} = A_i^seq + λ·δ_t`, `δ_t = −φ(u_t)`, `λ = g·λ_max`.
  Entropy gate (Schmitt trigger): `g ← 1 if g=0 and H ≥ H_warm ; 0 if g=1 and H < τ_down`,
  **τ_down = 0.93·H_warm**, `H_warm` = median teacher entropy over W=5 warmup steps.
  Boundedness: `φ(u) ≥ −½log 2 ≈ −0.347` as `u → −∞`, so **the deliberation side is capped
  while the positive side is unbounded**.
- **(2) ★ The diagnosis**: privileged context inflates teacher confidence on tokens already
  implied by the solution and **deflates it on "Wait", "Let", "Maybe"** — exactly the
  deliberation tokens that drive search. **Ordinary self-distillation actively penalizes
  the token class a metacognitive span is made of.**
- **(4)** ⚠️ Privileged context is "a verified solution sampled from the rollout group when
  at least one rollout is correct, **else from the dataset**" — the one paper here that is
  **not** gold-free.
- **(6) ★** `δ_t` is **added after** `A_i^seq`, so it survives group centering and is
  nonzero even when the whole group is correct or the whole group is wrong. **The cleanest
  "does not cancel in GRPO" construction in the survey.** Measured: **additive 62.8 avg vs
  multiplicative 56.5 (−6.3, and the speedup halves).**
- **(7) ★★ The no-teacher ablation is the direct threat to any own-rollouts-only design**,
  verbatim: "Without external information from the privileged context, the per-token term
  **degenerates into a function of the student's own probability, producing a
  positive-feedback signal that reinforces whatever the policy already emits**" — **all
  three models collapse within ~70 steps.** The asymmetry must come from genuinely
  external information; the model's own prior logits do not qualify. A sibling rollout or
  a self-written critique does (see G2, G6).
- **(8) ★ Includes Qwen3-4B-Instruct-2507 — the exact cd9 policy.** Avg over
  AIME24/25/26, HMMT25, Minerva: **GRPO 51.3, SD 45.9, AntiSD 62.8 (+11.5, and 10× fewer
  steps)**. Qwen3-8B: 57.4 / 30.6 / 65.7. ⚠️ **Standard SD underperforms GRPO on every
  single model** (catastrophically on Qwen3-8B: 30.6 vs 57.4).
- **(9)** no-teacher self-reinforcement collapse; no-gate collapse near step 90 (⚠️ **it
  ignites faster before collapsing, so early curves mislead**); reverse-KL-ascent collapse
  (49.5); τ_down = 0.93 is "not per-model sweet spot but the value that transfers."

### G6. RLRT / Rebellious Student — arXiv:2605.10781
- **(3) Pure advantage multiplier.**
  `D̂_t = log( P_S^t(y_t) / P_T^t(y_t) )` with `P_S = π_θ(·|h_t)` (no privileged context)
  and `P_T = π_θ(·|h_t, c)`; **`w_t^RLRT = (P_S^t/P_T^t)^{sign(A)}`** — note **RLSD uses
  the exact inverse** `(P_T/P_S)^{sign(A)}`.
  `A_t = A·[(1−λ) + λ·clip(w_t, 1−ε_w, 1+ε_w)]` **only when `r(y^{(k)}) = 1`**, else
  `A_t = A`. **λ = 0.5; ε_w = 1.0 base / 0.5 instruct / 0.2 thinking** (tightening with
  model maturity).
  In RLRT's own runs **`c` = a correct rollout**; the RLSD *baseline* is the one
  conditioned on the ground-truth answer.
- **(8) ★ Includes Qwen3-4B-Base and -Instruct.** Qwen3-4B-Base avg@16 AIME24/25/26:
  RLRT 22.5/18.5/19.8 vs GRPO 15.0/14.4/12.3 vs **RLSD 13.3/11.2/9.0**.
  ⚠️ **RLSD (gold-conditioned self-distillation) is *below* plain GRPO on nearly every
  cell** — same finding as AntiSD, independently.
- **(9)** ★ **The reward gate is load-bearing**: RLRT-all (no `r=1` restriction) tracks
  RLRT then diverges around step 40 — length and entropy grow unbounded, training
  collapses. "Without the gate, the reverse weight reinforces teacher-divergent tokens on
  failed rollouts, conflating valuable exploration with spurious divergence." The SDPO
  baseline collapses within 20 steps due to "**excessive suppression of hedging and
  reflective tokens (e.g. 'wait', 'hmm')**."

## Group N — negative results and mechanism papers at the cd9 operating point

### N1. Distilling Self-Consistency into Verbal Confidence — arXiv:2604.24070
**★ Essentially the cd9 experiment, already run, on Gemma-3-4B-it, pre-registered.**
- Setup: labels from the model's **own 10 rollouts at T=0.7**; `n_correct` mapped to a
  confidence target (0 → 5%, 10 → 95%); LoRA r=16 SFT with the confidence in the assistant
  turn.
- **Pre-registered result: STOP.** AUROC2 **0.554 → 0.509** (δ = −0.052, CI [−0.077,
  −0.027]); accuracy 57.2% → 49.6%; ceiling rate 97.7% → 98.4%.
- **Root cause — the lesson**: a pre-registered **modal filter** (train only on items whose
  modal answer is correct) caused **label-entropy collapse**. 84.6% of items sat at extreme
  `n_correct` (0 or 10), and the filter removed all `n_correct = 0` items, so the model saw
  almost only 95% targets and learned to emit the mode.
  Their conclusion, verbatim: **"Confidence training requires label entropy. Any
  training-set filter that removes low-confidence examples will collapse the label
  distribution and guarantee failure."**
- **Post-hoc rescue**: drop the filter, keep all 2,000 items including the 893 with
  incorrect modal answers → AUROC2 **0.554 → 0.774** (shuffled-target control 0.501);
  ceiling 97.7% → 49.8%. MMLU accuracy 54.2% → **77.4%**. ⚠️ The model learned a **binary
  discriminator**, not continuous calibration (494 items at 5%, 498 at 95%; the 95% bin was
  77.1% accurate, the 5% bin 22.3%).
- **★★ The within-problem signal diagnosis, and it maps directly onto the cd9 G1
  rejection.** At 4B the self-consistency distribution is **bimodal**: of 2,000 items,
  **963 (48.2%) at n_correct = 10, 729 (36.5%) at n_correct = 0, only 308 (15.4%)
  intermediate.** Raw 10-sample self-consistency gets **AUROC2 = 0.999**; logit entropy
  **0.701**; linear probes on hidden states **0.6–0.8**; **verbal confidence 0.554.**
  Their reading: "The internal signal exists; the verbal channel's weakness is not a
  capacity gap but a **readout failure**."

### N2. When Should a Language Model Trust Itself? — arXiv:2605.02915
Same-model self-verification is **a conditional signal, not a general one**. ARC-Challenge
self-verify vs LL-AVG AUROC: Qwen-7B **0.886 vs 0.555**; Qwen-1.5B 0.765 vs 0.557;
TinyLlama-1.1B 0.525 vs 0.484; **DeepSeek-R1-Distill-8B 0.463 vs 0.511 (harmful)**.
TruthfulQA-MC: Qwen-1.5B **0.548 vs 0.611 (harmful)**; TinyLlama **0.363 vs 0.562**.
Verdict: "Self-verification should not be adopted as a blanket confidence wrapper without
task-specific validation."

### N3. The Two-Stage Decision-Sampling Hypothesis — arXiv:2601.01580
Factorizes the policy into a **sampling policy** `π_sample` and a **decision policy**
`π_d` (STOP vs RESAMPLE):
`P(τ|Q;θ) = [∏_k π_sample(A_k,T_k|s_{k−1};θ)]·[∏_{k<T} π_d(R|s_k;θ)]·π_d(S|s_T;θ)`.
**Theorem 3.1**: surrogate (RL) rewards give both components the *identical*
trajectory-level advantage weighting → *balanced*. **Theorem 3.2**: SFT/KL penalties give
`d_k^sample ~ O(L_k)` but `d_k^decision ~ O(1)` → **length-weighting heavily constrains
`π_sample` while leaving `π_d` under-regularized and under-optimized.** That is a
first-principles account of why RL builds self-correction and SFT does not.
★ **The decorative-reflection signature, operationalized.** Qwen2.5-7B-Instruct trained
only on 4×5/5×4 multiplication, all test OOD: SFT-with-reflection 3×6 **49.0%** vs RL
**90.0%**; 3×9 **0.0%** vs **34.0%**. Decomposition: `p_{d|C}` (STOP given correct) stays
~85–95% for **both**, but **SFT's `p_{d|W}` (RESAMPLE given wrong) collapses to ~5% at 3×6
and ~0% at 3×9 while RL's holds 40–60%.** The SFT model still *emits* reflection text — it
just stops rejecting its own wrong answers. **`p_{d|C}` high + `p_{d|W}` ≈ 0 is
text-shaped metacognition with zero discriminative content.** Their phrase: reflection in
SFT models is "decorative rather than functionally causative."
⚠️ Caveat: this is observational. **No ablation suppresses or randomizes reflection tokens
and measures the accuracy delta.**

### N4. When and Why Does Unsupervised RL Succeed in Mathematical Reasoning? — arXiv:2603.16578
Clusters token-entropy trajectories into Execution / Logic / Thinking states and tracks the
convex-hull volume of the 3-D phase-space manifold. Success = "tightly enveloped";
failure = "loosely enveloped." Type I (exploration stagnation) collapses manifold volume to
**0.006**; Type II (weak base model, Llama3.1-8B) explodes it to **8.125**. Ordering of
stability: **Llama < DeepSeek-Distill < Qwen3**; DeepSeek-Distill-Llama-8B is the
transitional case where "unsupervised methods experience very brief initial improvement
before rapid collapse." Finding worth noting: **a plain length penalty consistently
outperforms entropy minimization** as a label-free objective. Tested only up to 8B.


## Group H — additional self-rewarding rows

### H1. CoVo — "Consistent Paths Lead to Truth", arXiv:2506.08745, NeurIPS 2025
★ The one label-free reward that is neither consensus nor entropy: it scores **likelihood
geometry along the trajectory**, which is the closest published thing to "is my own
solution converging?"
- **(3)** Split τ into T newline steps, `s_i = [x, t_0..t_{i−1}]`.
  `d(s_i, y) = −(1/|y|) Σ_j log π_θ(y[j] | s_i, y[:j])` (Eq. 1). With K distinct final
  answers, build `D ∈ R^{T×K}`, column 0 being the trajectory's own answer.
  **`Con(τ) = (1/T) Σ_i 1( D[i,0] = min_k D[i,k] )`** (Eq. 4) — the fraction of prefixes
  from which the trajectory's own answer is already the closest.
  **`Vol(τ) = (1/T) · max{ i : D[i,0] ≠ min_k D[i,k] }`** (Eq. 5) — how late the last
  wobble occurs.
  Trajectories are **grouped by final answer** (size G) and aggregated vectorially:
  `v_i = Con(τ_i)·[cos Vol(τ_i), sin Vol(τ_i)]`; `r_int^V = (1/G)·‖Σ_i v_i‖` (Eqs. 7–9).
  Curiosity term `r_cur = d(s_i, s_{i+1}) − ln[KL(P_{i+1}, U) + 1]` (Eq. 10).
  **`r_covo = r_int + r_cur` — plain sum, no coefficient.** REINFORCE++ backbone.
- **(4)** per-token log-probs of each candidate answer under each prefix of each
  trajectory (N·T·K teacher-forced passes) + newline segmentation + the distinct-answer
  set. **No gold in the reward**; gold only at evaluation.
- **(5)** `r_int` is a **single scalar per answer-group**, broadcast identically to every
  trajectory sharing that answer; `r_cur` is step-level.
- **(6) ★** When all N rollouts agree (K=1), every trajectory is in one group, `r_int` is
  identical, and normalization zeroes it. The paper never reports that fraction but names
  the mechanism as the motivation for `r_cur`: "model diversity may decrease as training
  proceeds… This poses difficulty for the intrinsic reward to perform meaningful
  comparisons across distinct answers." Ablation: curiosity alone is far worse than
  intrinsic alone (Llama MATH-500 46.2 vs 51.6).
- **(7)** Already gold-free, and the reported cost is ≈0: MATH-500 CoVo vs supervised
  GRPO — Llama3.2-3B **51.2 vs 51.8**, Qwen2.5-3B **68.2 vs 67.4**, Qwen2.5-7B
  **78.4 vs 78.2**. ⚠️ That parity is itself suspicious given Spurious Rewards: two of
  three backbones are Qwen and **no random-reward control is run.**
  ★ **Proposition 1 (Model Collapse)** formalizes the majority-vote failure:
  `r(x,y) = 1[y = argmax_{y'} C(y')]` ⇒ "π_θ(y*|x) converges to 1 as training proceeds…
  If y* is not the ground truth, reward hacking will happen." That is SRT's empirical
  result, proved.
- **(8)** ★ Claimed feature separation of `Con(τ)`, correct vs incorrect: MATH-500
  **0.832±0.194 vs 0.215±0.136**; MMLU 0.818 vs 0.236; GPQA 0.787 vs 0.218. **If that
  replicated, `Con(τ)` would be a near-oracle self-verifier** — and it is a *within-problem*
  discriminator computed with no gold, which is exactly what the cd9 G1 gate failed to
  find. Worth an independent replication before trusting.
- **(9)** Diversity collapse is the load-bearing assumption (`r_int` needs K>1); O(NTK)
  extra forward passes; no per-trajectory credit inside an answer-group; `Vol(τ)` is
  ill-defined when `Con = 1` (Algorithm 1 silently sets Vol ← 0, an undocumented
  convention); Qwen-heavy evidence with no spurious-reward control.

### H2. CARE — "When Self-Belief Misleads: Active Label Acquisition for RLVR", arXiv:2605.25864
★ **The paper that measures the gold-free gap directly, at 1.7B / 4B / 8B.**
- **(3)** **Corrective Advantage Gap** `s_i = ‖A_i − Ã_i‖₂` (Eq. 4) — the L2 distance
  between the GRPO advantage vector computed with **gold** labels and the one computed
  with **majority-vote pseudo-labels**. CAG is an *oracle* quantity, so a two-stage
  learned head predicts it. The advantage-mixing rule (Eq. 7):
  `A_{i,g}^mix = A*_{i,g}` if the prompt got a gold label, else `c_i^(1) · Ã_{i,g}` where
  `c_i^(1) ∈ [0,1]` is a learned probability that the pseudo-label equals gold; prompts
  predicted unreliable *and* low-CAG are **dropped from the loss entirely**.
  Budget `p = 20%` of prompts get gold.
- **(6) ★** CAG is defined *as* a distance between group-normalized advantage vectors, so
  the framework is native to the group-centering question: "when a pseudo-label is correct,
  it induces the same advantage vector as the ground-truth label and thus has zero CAG."
  Unanimous groups have CAG = 0 and cost nothing.
- **(7) ★★ The numbers.** Qwen3-Base on DAPO-17k, avg over AIME24/25, MATH500, AMC23,
  HMMT25, Olympiad:
  | | 1.7B | **4B** | 8B |
  |---|---|---|---|
  | Vanilla (no RL) | 14.89 | 14.72 | 25.39 |
  | **GT (100% gold)** | **26.69** | **41.14** | **44.59** |
  | TTRL (0% gold) | 23.18 | **32.03** | 36.52 |
  | Random 20% gold | 22.86 | 32.15 | 35.46 |
  | Oracle-CAG 20% | 22.76 | 34.95 | 39.51 |
  | **CARE 20%** | **25.67** | **37.07** | **43.59** |
  **The fully label-free gap is 3.5 / 9.1 / 8.1 points — and it is *widest at 4B*.**
  20% well-placed gold recovers most of it (residual 1.0 / 4.1 / 1.0). CARE matches GT at
  40% of annotations and surpasses it at 80%.
- **(9) ★ Their collapse characterization**, verbatim: "the accuracy of pseudo-labels
  continually decreases, forming a **dead loop that keeps reinforcing incorrect internal
  beliefs**, ultimately leading to training collapse… **Unsupervised RLVR driven by
  internal beliefs is not always reliable.**" And a finding that indicts pseudo-labels
  outright: training the baselines on **only** their 20% annotated samples and dropping all
  pseudo-labeled data *improves* them — "excessive erroneous pseudo-labeled samples may
  even hinder training." Also: **Oracle-CAG selection alone still collapses** — correcting
  the right 20% is necessary but not sufficient; CARE survives only because Stage-I also
  drops and downweights.

### H3. CME — "Label-Free Reinforcement Learning via Cross-Model Entropy", arXiv:2605.29009
★ An explicit attack on self-referential metacognition, plus the only **per-token**
label-free reward in the survey.
- **(2)** Their taxonomy, verbatim: TTRL, Evol-RL, RENT, EM-RL, Intuitor "share a common
  property: the reward derives entirely from the generator's own outputs. We refer to such
  rewards as **self-referential**. The risk is structural: when the model is systematically
  wrong in a way multiple rollouts agree on, or assigns high confidence to incorrect
  outputs, the reward reinforces the error rather than correcting it."
- **(3)** `CME_{i,t} = −log π_φ( y^aligned_{i,t} | x, y^aligned_{i,<t} )`; `r_{i,t} = −CME_{i,t}`;
  **advantages normalized across the G responses at each token position**:
  `Â_{i,t} = (r_{i,t} − μ_{r,t})/σ_{r,t}`. `π_φ` is a **frozen separate-family verifier**
  (gemma-3n-E4B-it). Cross-tokenizer alignment by character-overlap weights.
  Objective identity: `E[r] = −H(π_θ, π_φ) = −H(π_θ) − D_KL(π_θ ‖ π_φ)`.
- **(6) ★** Because CME is continuous and normalized per position, **within-group variance
  is essentially never zero — there is no degenerate-group failure mode.** The flip side:
  the signal never goes quiet, so there is no natural stopping condition.
- **(7) ★★ Their random-verifier control is an independent replication of the Spurious
  Rewards effect in an open-ended domain**, verbatim: "a random verifier does provide some
  lift over the base (**55.8% vs 50%**), but every real-weighted verifier exceeds it, and
  gains scale with verifier capability." **Roughly a quarter of CME's headline lift over
  base (55.8 → 70.0 against a 50.0 floor) is attributable to nothing.** The paper does not
  subtract this floor in its headline claims.
- **(8)** AlpacaEval 2.0 tie-adjusted win rate vs RENT (the direct self-referential
  comparison): Qwen2.5-0.5B **71.4 vs 58.0**; Llama-3.2-1B-It **94.8 vs 67.0**;
  Qwen2.5-0.5B-It **83.5 vs 64.0**. Self-as-verifier diagnostic: Qwen2.5-0.5B verifying
  *itself* gets 63.0 vs cross-family 70.0. Their reading: "A cross-family verifier scores
  against an independent distribution; a same-family verifier or self-confidence reward
  can only re-weight what the generator already finds likely."
- **(9)** Authors': "vulnerable to drift toward verifier-style outputs or mode collapse
  toward the verifier's prior; we did not observe either… but we did not test long-horizon
  training." Found on reading: an internal contradiction about whether RENT baselines were
  run; verifier named two different ways; the 55.8% floor is never subtracted; **no math
  or reasoning benchmark at all.**

### H4. Self-Rewarding Language Models — arXiv:2401.10020, ICML 2024
- **(3)** No closed-form reward. The policy scores its own N=4 candidates with an
  **additive 5-point rubric** (relevance / coverage / usefulness / clarity / expertise,
  1 point each, unweighted); take argmax and argmin as a DPO pair; **discard the pair if
  the scores tie.** Iterative DPO with the external RM replaced by the policy itself.
- **(5)** sequence-level, **offline, as a preference-pair selector** — the score's
  magnitude is discarded, only the ranking survives into the DPO loss.
- **(6)** No group centering. The analogous degeneracy is the **tie rule**: a group of 4
  with identical scores produces no training pair at all, so as the policy sharpens the
  effective dataset shrinks. Tie fraction is never reported.
- **(8)** AlpacaEval 2.0 vs GPT-4-Turbo: M₁ 9.94% → M₂ 15.38% → M₃ **20.44%**. Reward-model
  agreement with held-out human preferences: 65.1% → 78.7% → 80.4% → 81.7%.
- **(9) ⚠️ Two corrections to the common framing.** First, **the paper does not report a
  reward-hacking failure** — it explicitly leaves it open: "It would also be good to
  understand if so-called 'reward-hacking' can happen within our framework, and in what
  circumstances." What it *does* report is the footprint: **average AlpacaEval generation
  length 1092 → 1552 → 2552 over three iterations**, on a metric with a documented length
  bias. Cite it as an unaddressed length-bias confound, not a demonstrated collapse.
  Second, and most important for a math survey: **reasoning did not improve at all** —
  "there are some tasks for which this approach does not improve, such as **mathematics and
  logical reasoning**, indicating that our current training approach mainly allows the
  models to better utilize their existing knowledge." **LLM-as-own-judge self-rewarding
  produced zero reasoning gain in its founding paper.**

---

# Part 3 — Synthesis

## Q1 — Families, by the mathematical object rewarded (≤200 words)

**F1 · Agreement with a reference label.** `r = 1[ŷ ≡ y_ref]`. Needs a *label*: gold
(AggLM C1, RSA C2, SCoRe E1, EGPO D4), majority vote (TTRL B1, SRT B5, EMPO B4, OM-GRPO
B9), a cross-view vote (Co-rewarding B3), or a proof (JURY-RL B10).

**F2 · A proper scoring rule on a stated probability.** `f(c), g(c)` over a verbalized
`c`. Needs a *correctness bit plus an emitted number* (RLCR A1, Rewarding Doubt A3,
SaySelf A4, RLMF D3), and per Theorem 1 of A2 the pair must satisfy
`h(c)=f′/(c−1)=g′/c ≤ 0` and `f(a⁺) ≥ g(a⁺)` or it is hackable.

**F3 · A counterfactual improvement delta.** `Q(after) − Q(before)` — SCoRe's
`α(r̂(y₂)−r̂(y₁))`, PAV's `A^μ = Q^μ(s_h,a_h) − Q^μ(s_{h−1},a_{h−1})`, IBPO's `φ_i`, RLPR's
`r − r′`. Needs *two evaluable states and a scorer for both*.

**F4 · A likelihood shift between two conditionings of the same policy.** Quiet-STaR's
`log p^talk − log p̄^talk`, SD-Zero's per-token KL, AntiSD's PMI `u_t`, RLRT's `D̂_t`,
SRPO-2608's reverse-KL `r_t`. Needs only *privileged context the unconditioned pass
lacks*.

**F5 · An entropy or confidence level.** RENT, Intuitor, EM, SEED-GRPO, DEER. Needs
*only the policy's own distributions*.

**F6 · A rule-checkable structural property.** RLVMR tags, L1/AdaCtrl lengths,
SelfBudgeter's `|ℓ−b|`, format rewards. Needs *a parser*.

## Q2 — Best gold-free instantiation per family (≤200 words)

**F1** — already gold-free via majority vote; the closest-to-gold instance is **OM-GRPO
(B9, gap 0.3–0.65 pts)**, which estimates from the answer span but masks the gradient off
it. Cost is measured by **CARE (H2): 9.1 points at 4B**, the widest gap of any scale it
tested.

**F2 — cannot be made gold-free without changing what is measured.** The scoring rule
needs a correctness bit; substituting consensus makes the target *the group's own
agreement rate*, and the model satisfies it by collapsing `c` onto self-consistency. This
is not speculation: **N1 (2604.24070) ran exactly this on Gemma-3-4B and pre-registered
STOP** (AUROC2 0.554 → 0.509).

**F3 — cannot be made gold-free in general**, because `Q` is a success probability. The
one exception is **Reflect-Retry-Reward (E2)**, whose Countdown validator
(`equation == target`) is self-checkable — the only published gold-free improvement-delta
result.

**F4 — natively gold-free, and the only family where that is structural.**
**Quiet-STaR (C6)** is the existence proof; **AntiSD (G5), RLRT (G6), AMR-SD (G2), SD-Zero
(G1)** all condition on an *own correct rollout* or a *self-written critique*, not gold.
⚠️ **AntiSD's no-teacher ablation** shows the limit: strip the privileged context entirely
and the term "degenerates into a function of the student's own probability," collapsing all
three models in ~70 steps.

**F5 — natively gold-free**, and natively degenerate (B7, B11, B6).

**F6 — natively gold-free and constitutionally blind.** RLVMR's ablation: 56.3 → **12.5**.

## Q3 — Ranking against the four cd9 constraints, and the recommendation (≤200 words)

| | (a) verbal span | (b) own-rollout labels | (c) survives centering | (d) works at 4B / chance internals |
|---|---|---|---|---|
| **F4 likelihood shift** | ✅ span-scoped natively | ✅ | ✅ additive per-token, never centered away | ✅ **AntiSD +11.5 on Qwen3-4B-Instruct-2507; SD-Zero 49.8→60.3 at 4B** |
| **F3 counterfactual delta** | ✅ | ⚠️ needs a correctness bit | ✅ (pre-centering shaping) | ⚠️ IBPO has no model ≤32B |
| **F1 agreement** | ❌ scores the answer | ✅ | ⚠️ dies on unanimous groups | ⚠️ 9.1-pt gap at 4B |
| **F6 rule-checkable** | ✅ | ✅ | ✅ | ❌ blind (56.3→12.5) |
| **F2 scoring rule** | ✅ | ❌ | ✅ | ❌ **pre-registered failure at 4B** |
| **F5 entropy** | ❌ | ✅ | ✅ | ❌ collapses |

**Recommendation: F4 as the span signal, gated by F1 from the model's own rollouts.**
Concretely — **MASA (D1) with gold replaced by sibling majority.** MASA's three meta-rewards
already read *only* the correct/incorrect partition of the policy's own G rollouts; gold
enters solely as the partition function. Swap `rule_verify(o_i, y*)` for
`agrees_with_majority(o_i)` and `r_length`, `r_difficulty = 0.01^{|d_pred − d_sol|}`, and
`r_notion` are all defined with no gold. It satisfies (a) the meta span is the predicted
statistic, (b) labels are own-rollout, (c) **the reference is constant per prompt while the
prediction varies, so the reward survives mean-subtraction** — the exact structure the
brief needs, and (d) MASA's largest gains are at its smallest model (8B: +13.04% vs 14B:
+6.63%), with the family-mate AntiSD's headline result sitting on Qwen3-4B-Instruct-2507.
The published result that makes this look viable is **TTRL's lucky-hit asymmetry — label
accuracy 37% but reward accuracy 92%** — which says a majority partition is a far better
*partition* than it is a *label*, and MASA only ever uses it as a partition.

⚠️ **Run these three controls first, or the result is uninterpretable.**
(i) **Random Bernoulli(0.5) reward and a format reward** — Spurious Rewards (B14) gets
+21.4 and +13.8 on MATH-500 with Qwen, 74% and 47% of the gold-reward gain, from GRPO's
clipping bias alone, and the effect **does not replicate on Llama/OLMo**.
(ii) **Label-entropy audit before training** — N1's lesson: at 4B the own-rollout
distribution is bimodal (48.2% at 10/10, 36.5% at 0/10, only 15.4% intermediate), and any
filter that removes the low-consensus tail "will collapse the label distribution and
guarantee failure."
(iii) **The decorative-reflection diagnostic** — TAPO's DSR/ERR split (G3) and N3's
`p_{d|C}` vs `p_{d|W}` pair. N3's signature, `p_{d|C}` high with `p_{d|W}` → 0, is
text-shaped metacognition with zero discriminative content, and it is cheap to measure.

---

# Appendix — what could not be verified

- **RLCR's format-reward coefficient**: stated only as "weighted equally", no equation. [unverified]
- **SEED-GRPO's `f`**: undefined in the paper; recovered from `seed_grpo.py` as `1/(1+x)`.
  The multiplier range [0.960, 1.000] at G=8 makes the +5.2-pt headline mechanistically
  implausible. Cite with the caveat.
- **RLVMR's `r_s` and the three `r_MR` magnitudes**: never given numerically. [unverified]
- **SCoRe's venue**: commonly cited as ICLR 2025, not stated on the abs page. [unverified]
- **Spurious Rewards' venue**: search returns conflicting venues. [unverified]
- **F2 (arXiv:2601.20126) `r_abs` sweep table**: only the recommended range −0.25 to 0.3
  was recoverable. [partially unverified]
- **The Hallucination Tax (arXiv:2505.13988)**: abstract-level only. [unverified]
- **"Yeom et al." from the metacognition survey's RL list**: could not be resolved to a
  specific arXiv id. The nearest match surfaced is RLMF (2606.32032), a different group.

## 추가(2026-09-15 15:30) — arXiv:2512.16848 LaMer «Meta-RL Induces Exploration in Language Agents» (Jiang 외, EPFL/ETH)
- **무엇**: 한 과제를 시도 N=3 순차로 굴리는 trial 단위 meta-RL. 크로스-에피소드 리턴 G_t^(n) = g_t^(n) + Σ_{m>n} γ_traj^{m−n} g_0^(m)(γ_traj=.6) — **뒤 시도의 결과 보상이 앞 시도의 반성 토큰으로 흘러든다**. 내부 루프는 gradient 가 아니라 자기 반성 텍스트(실패 시 반성문을 다음 시도 문맥에 넣음). 보상은 환경 결과뿐, 교사·검증기 모델·내성 신호 없음, 온폴리시(GiGPO/GRPO 호환).
- **수치**(Qwen3-4B, GiGPO 대비 p@1/p@2/p@3): Sokoban 41.6/43.6/44.1→42.4/52.0/55.9 · MineSweeper 52.0/54.9/55.1→44.1/66.4/74.4 · Webshop 73.4/74.6/75.2→67.8/84.4/89.1. **p@1 은 두 환경에서 하락**(−7.9/−5.6), 이득은 p@3. ALFWorld OOD 58.1→81.0. 어블레이션(p@1): **궤적만 34.8 vs 반성문만 56.4** vs 둘 다 55.9 → 오답 궤적을 문맥에 두면 깎인다.
- **우리와의 관계**: (i) 메타 스팬을 «동시각 내부 신호»가 아니라 **그 스팬이 유발한 다음 시도의 결과**로 채점 — 그룹 중심화를 견디는 문제 안 신호(난이도 축 아님). cd8 EVCM 의 RL 판. (ii) 궤적-only 열세 = 우리 끌개 발견(사실만 .617 > 본문 제시 .513)의 RL 대응. (iii) 회의: 단일턴 수학은 재시도가 새 정보를 안 줌, p@1 하락, 비용 2배. → S3 후보 (r): 2-시도 GRPO·반성문만 문맥(f 포함, 본문 없음)·크로스-에피소드 크레딧. 관문 = 1차 p@1 ≥ GRPO ∧ 2-시도 정확도 > .617. 반성-only 문맥 자체는 이 논문 어블레이션에 이미 있으므로 새롭다고 주장하지 말 것.
- 요약 원문: scratchpad/paper_2512.16848.md (세션1 0915).
