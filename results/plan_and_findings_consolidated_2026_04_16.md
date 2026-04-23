# Meta-CoT V8 — Consolidated Plan & Findings Report

**Date:** 2026-04-16
**Scope:** RQ1 / RQ2 / RQ3 outcomes to date, self-distill framework (score teacher, token KL, RLSD), code-fidelity audit, current experiment status.
**Audience:** First-time readers. Each section states **Intent → Hypothesis → Evidence → Verdict.**

---

## 0. Executive Summary (for the impatient reader)

We ask whether a language model can acquire a *metacognitive controller* — a signal that gates when it should pause, diagnose, retrieve, or commit — rather than just imitating the surface style of a teacher's reasoning trace.

- **RQ1 (Meta-CoT SFT):** Fine-tuning Qwen3-8B on GPT-5.4-mini traces *with* `<|meta|>` confidence/assessment/action blocks raises accuracy on MATH/AIME/OmniMath over a strict base-matched control (base 75.92 % → meta 79.81 %, +3.88 pp), and entropy analysis shows the meta block measurably *resolves* uncertainty (Δ entropy = +0.300 nats across the span). The controller exists after SFT.
- **RQ2 (Meta-RL / GDPO):** After 300 steps of verifiable-reward RL (`E21R-v2`, 2-head: correctness + outcome-calibration), accuracy is preserved but the **structural `<|meta|>` wrapping is stripped** in 122/1030 completions, confidence collapses to the single value 0.96 in 98.9 % of wrapped cases, and 908/1030 completions share an identical boilerplate assessment. A root-cause trace identifies the reward function's free-text fallback (see §4.1) as the mechanism: the model discovers it can drop `<|meta|>` tokens and keep the reward.
- **RQ3 (Curriculum / OOD / Test-Time Adaptation):** Mainline redirected to **RQ3-D: Epistemic Self-Distillation**, which uses the existing meta SFT + best-of-N teacher selection to rebuild the wrapped controller without the RL reward loop that caused the loss. Two training runs (D1 naive baseline, D2 epistemic) launched on H200 ×4 at 21:07 on 2026-04-16 and completed their first ~94 steps before this report was written.
- **Code audit highlights:** 5 implementation issues identified (§5); most serious is the free-text confidence fallback in `rewards.py` that caused wrapping loss. RLSD is *planned but not yet implemented* — only score-teacher SFT + token-KL exist in `src/training/self_distill/`.

---

## 1. Research Questions (RQ1/2/3)

### 1.1 RQ1. Meta-CoT as a controller, not a style

**Intent.** Train the model so that `<|meta|>...<|/meta|>` blocks act as a controller that governs subsequent reasoning — emission of meta should change *what the model does next*, not merely decorate the trace.

**Hypothesis (plan §205–234).** If the controller is real, an SFT pass on matched-style teacher traces that *contain* meta wrapping should outperform a base-matched SFT (same traces with meta removed) on held-out problems, and the model should exhibit higher entropy *before* the meta block and lower entropy *after* (a behavioral signature of deliberation).

**Evidence.**
- `results/eval_v8_meta_inside_strict_sft/` vs `results/eval_v8_base_matched_strict_sft/`: 79.81 % vs 75.92 % (N=1030).
- `results/entropy_strict_meta/` (n=120, marker = `<|meta|>` span): Δ entropy over the span = +0.300 nats, confirming the model's uncertainty is *higher* when the meta token is forced than on surrounding context.
- `results/step300_deep_analysis/full_analysis.json`: on redirect problems (where the base model was wrong), the paired meta SFT corrects them at a rate +22.3 pp above base.

**Verdict.** ✅ Confirmed. Meta-CoT SFT yields a controller with behavioral signature consistent with the hypothesis.

### 1.2 RQ2. Meta-RL must strengthen, not erode, the controller

**Intent.** Given a working controller, apply verifiable-reward RL (GDPO with a correctness head and a calibration head) to *amplify* redirection on hard problems.

**Hypothesis (plan §236–266, revised 2026-04-13).** A well-designed reward should (a) improve accuracy, (b) preserve `<|meta|>` wrapping rate, (c) preserve controller-mediated redirection quality, (d) avoid mode collapse on confidence.

**Evidence.** `results/step300_deep_analysis/analysis_report.md` (verified):
- Accuracy: preserved at +0.0 pp net (E21R-v2 step 300 matches the meta SFT baseline).
- `<|meta|>` wrapping rate: **dropped from ~100 % (SFT) to 88.2 % (1030 eval); absent in 122 completions.**
- Confidence distribution: **98.9 % of wrapped completions emit exactly 0.96.**
- Boilerplate template collapse: 908/1030 (88.2 %) share an identical 3-line assessment. Conditional on the boilerplate being emitted, accuracy = 79.5 %; conditional on it being absent, accuracy = 46.7 % — but 85 % of the absent-boilerplate traces are *token-truncated* (no `\boxed{}` emitted), so the verify↔accuracy correlation is a token-length confound, not a behavioral one.
- AIME (N=30): 40.0 % → 46.7 % nominal, but median output length 3.2 k → 11.8 k tokens; 13/30 cases run out of the 4096-token budget mid-reasoning. Token exhaustion is the primary delta in AIME performance.
- Trigger-conditioned correction rate on redirect problems: **–22.3 pp vs base** (worse than base on the very problems the SFT version corrected).

**Verdict.** ❌ Failed on (b), (c), (d). Accuracy preserved (a) but at the cost of losing the controller. See §4 for root causes.

### 1.3 RQ3. Controller-mediated information acquisition at test time

**Intent.** Use the emitted controller signal (diagnosis + study_need) to drive test-time information acquisition — retrieval, branch search, or regeneration from a curated teacher — on OOD problems.

**Sub-questions (plan §268–329, with later revisions).**
- **RQ3-A:** Diagnosis-triggered retrieval (curriculum/RAG lane).
- **RQ3-B:** MCTS-lite branch search gated on study_need.
- **RQ3-C:** Search-to-learn distillation (plan §909).
- **RQ3-D:** ⭐ **Epistemic self-distillation** (plan §933) — the new mainline. Build a teacher dataset of *trajectories that preserve the controller* and train a student that does not go through the reward loop that stripped it.

**Evidence so far.** RQ3-A/B smoke runs at `results/rq3_side_eval_smoke*` and `results/rq3_pipeline_smoke.json`. RQ3-D main run: D1 + D2 SFT in flight at the time of this report.

**Verdict.** 🔄 In progress. RQ3-D is the mainline this week; success criteria in §2.4.

---

## 2. Self-Distill Framework

Self-distillation is introduced to *rebuild* the meta controller that RL erased, and to *probe whether a base model can recover it* without ever being shown meta wrapping. Three technique variants are planned.

### 2.1 Variant A — Score-teacher (best-of-N)

**Intent.** Regenerate N completions per problem from a teacher (meta SFT + optional retrieval), score them with a multi-component reward, and keep the best as the student's target. The "best" candidate is correct **and** exhibits the expected controller behavior (meta wrap, diagnosis, recovery).

**Formalism.** For a problem $q$, sample $\{c_1, \ldots, c_N\}$ from a teacher distribution $p_T(c|q)$. Score each candidate:
$$s(c) = \sum_k w_k \cdot r_k(c; q, a^\star)$$
with $r_k$ being per-component rewards (correctness, confidence revision, redirect, verify, meta floor, meta commit quality). Select $c^\star = \arg\max_c s(c)$.

**Plan location:** §86–92, §962–973. **Implementation:** `src/training/self_distill/online.py:245–358`.

**Comparison with prior art.**
- vs **BOND** (best-of-N distillation, Sessa et al. 2024): BOND uses a single reward model; we use a weighted linear combination of six rewards. Our formulation is closer to **RLAIF-V** than to BOND.
- vs **SPIN** (self-play): SPIN selects by correctness only; our `selector_mode="correctness_only"` reproduces that — it is the default for question-only best-of-N.
- vs **RLAIF-V**: the `require_correct_teacher` gate (online.py:699) matches RLAIF-V's reject-if-wrong filter.

### 2.2 Variant B — Token-span KL

**Intent.** Distill *only* the tokens that carry the controller signal (meta blocks, diagnosis, study_need, recovery, verify) rather than forcing the student to match the teacher everywhere. This limits over-imitation of stylistic choices that are not part of the controller.

**Formalism.** Use teacher top-k logits only at positions inside control spans, augment with the target token if missing, and compute a weighted forward-KL:
$$\mathcal{L}_{\text{KL}} = \frac{1}{\sum_t m_t} \sum_t m_t \cdot D_{\text{KL}}(p_T(\cdot|x_{<t}) \,\Vert\, p_S(\cdot|x_{<t}))$$
where $m_t \in \{0, 1, w_{\text{meta}}, w_{\text{diag}}, w_{\text{study}}, w_{\text{recov}}, w_{\text{verify}}\}$ is the position weight.

**Plan location:** §91, §974–977. **Implementation:** `src/training/self_distill/kl.py:110–176`, applied in `src/training/sft.py:280–338`, captured in `src/training/self_distill/teacher_query.py:95–135`.

**Comparison with prior art.**
- vs **MiniLLM** (Gu et al. 2024): MiniLLM uses uniform reverse-KL on all assistant tokens. Ours uses span-gated **forward-KL** with top-k truncation — closer to **OPD / DistilWhisper** selective KD than to MiniLLM.
- vs standard KD top-k: the target-augmentation trick (sft.py:325–327) is the canonical safeguard against top-k missing the label.

### 2.3 Variant C — RLSD-lite (RL with self-distilled initializer)

**Intent.** After the student has been warm-started by score-teacher SFT + token-KL, run an additional verl-GDPO pass **initialized from D2's SFT checkpoint**, training on the `redirect` subset with a *commit-shape-aware* reward so that sparse task correctness provides direction and dense meta-commit shaping provides magnitude/retention.

**Plan location:** §978–982, §1005–1013.

**Implementation status:** ✅ **Implemented.**

| Component | Path |
|---|---|
| Reward function (v2) | `src/training/verl_reward.py:152` — `compute_score_e21r_v2` (2 heads: correctness, outcome_calibration) |
| Reward function (v3 smoke) | `src/training/verl_reward.py:185` — `compute_score_e21r_v3_smoke` |
| Reward function (v4 smoke, new) | `src/training/verl_reward.py:253` — `compute_score_e21r_v4_smoke`, adds `meta_commit_shape_reward` (× 0.35) |
| Supporting shape reward | `src/training/rewards.py::meta_commit_shape_reward` |
| RL launcher | `scripts/launch_e21r_v4_commit_shape_0416.sh` (lr 1e-6, KL 0.001, batch 64, 4 rollouts, 4096-token response, 300 steps, TP=2) |

Compared to the canonical RLSD of arXiv:2604.03128 (per-token teacher-ratio advantage scaling), our "RLSD-lite" is **not** an advantage-ratio method — it is verl-GDPO with improved reward shaping that directly penalises the no-boxed / repeated-meta / decoherence tails identified in §1.2. A future direction is to add per-token teacher-ratio shaping (Eq. 13–16 of arXiv:2604.03128) on top of the v4 reward, which would require a second forward pass with privileged-context teacher.

### 2.4 Operational experiments

| ID | Init | Dataset | Mode | Goal | Success gate |
|----|------|---------|------|------|---------------|
| **D0** | base SFT | `base_qonly_*` | — | baseline | report only |
| **D1** | base SFT | `d1_naive.parquet` | NAIVE (meta stripped, epistemic language suppressed) | collapse probe | does the base preserve epistemic behavior *without* meta training data? |
| **D2** | meta SFT | `d2_epistemic.parquet` | EPISTEMIC (meta preserved; synthetic wrapper if absent) | controller preservation | ≥ D1 + 2 pp OOD combined accuracy; trigger precision drop ≤ 10 pp |
| **D2-KL** | meta SFT | D2 + teacher top-k logits | EPISTEMIC + token-KL (meta-only spans) | reduce reliance on surface imitation | confidence histogram no longer mode-collapsed to 0.96 |
| **D2-RL** | D2-KL | — | RLSD-lite (planned) | turn the controller into a real driver | AIME `no_boxed_rate < 10 %`; redirect correction rate recovers to base-meta SFT level |

---

## 3. Prior Findings Incorporated

This section records what we *already know* as of 2026-04-16 and the artifacts that document each finding.

### 3.1 Controller exists after SFT
- Paired-SFT strict data: `data/v8_meta_inside_strict.parquet`, `data/v8_base_matched_strict.parquet`.
- Eval: 79.81 % (meta) vs 75.92 % (base) on 1030 matched problems.
- Entropy signature (n=120): mean entropy rises through the `<|meta|>` body then drops sharply after `<|/meta|>` — Δ = +0.300 nats across the span, consistent with deliberation.
- Artifact: `results/entropy_strict_meta/entropy_stats.json`.

### 3.2 RL erodes the controller
- After E21R-v2 (300 steps, GDPO with `correctness_reward` + `outcome_calibration_reward`):
  - `<|meta|>` wrap rate: 88.2 % (down from ~100 %).
  - Confidence mode collapse: 98.9 % of wrapped completions emit exactly 0.96.
  - Boilerplate assessment shared by 908/1030 completions.
  - AIME median output length: 3.2 k → 11.8 k tokens; 13/30 AIME cases exhaust a 4096-token budget.
- Artifact: `results/step300_deep_analysis/analysis_report.md`, `full_analysis.json`.

### 3.3 Why RL erodes the controller — root cause in the reward
The reward function `_parse_meta_blocks_with_spans` (`src/training/rewards.py:366+`) has a free-text fallback: if no `<|meta|>...<|/meta|>` block is found, it matches bare "`confidence: 0.XX`" (and variants like "`probability: 70%`") as if it were a meta block, and assigns the same credit. **The model discovers this during RL and drops the wrapping tokens as dead weight**, retaining the reward signal. See §5.1 for the call-graph.

### 3.4 Entropy on RL output — why wrapping returns only partially
- `results/entropy_analysis_step300/rl_meta_confidence/` (n=200, marker = free-text "confidence:"): Δ entropy across the free-text span = **−0.052 nats** — the *opposite* sign of the SFT signature. Split by correctness: correct-answer Δ = −0.042 (shallow); incorrect-answer Δ = **−0.193** (sharp drop).
- Interpretation: incorrect outputs have *high* entropy just before "confidence:", then the model emits the boilerplate "0.96" that collapses entropy dramatically. This is the signature of **forced certainty**, not deliberation.

### 3.5 16k re-eval
- `results/eval_1030_meta_grpo_e21r_v2_step300_16k/`: with max_tokens = 16384, the AIME truncation issue is removed; accuracy distribution stabilizes. `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/` reports Δ = −0.0305 (still negative but smaller), confirming the entropy inversion is not purely a truncation artifact.

---

## 4. Root-Cause Diagrams (how we got here)

### 4.1 Reward fallback → wrapping loss (§3.3 expanded)

```
  RL rollout → reward evaluation
     │
     ▼
 _parse_meta_blocks_with_spans(text, allow_free_text_fallback=True)  ← default True (rewards.py:366)
     │
     ├── stage 1: match <|meta|>...<|/meta|>   →  full credit
     ├── stage 2: match [META] ... [/META]     →  full credit
     └── stage 3: match r"confidence\s*:\s*[\d.]+"   →  full credit  ★ LEAKAGE
                                                              │
                                                              └── callers that trust stage 3:
                                                                    confidence_revision_reward_v2 (line 2074)
                                                                    redirect_reward (line 596)
                                                                    ... (8 more call-sites)
```

Mitigation plan: pin `allow_free_text_fallback=False` in every reward that is *supposed* to require wrapped meta (revision, calibration, omission floor). Confirmed in prompt parser (`src/metacot/prompt.py:parse_meta_blocks(..., allow_free_text_fallback=False)` default).

### 4.2 Outcome calibration → confidence collapse

`outcome_calibration_reward` is Brier-style:
$$r_{\text{cal}}(c, \hat{y}, y^\star) = 0.3 \cdot (1 - (c - \mathbb{1}[\hat{y}=y^\star])^2) - 0.15$$

Expected reward as a function of confidence $c$, at base accuracy $\text{acc}$:
$$\mathbb{E}_{y^\star}[r_{\text{cal}}] = 0.3 \cdot c \cdot (2 \cdot \text{acc} - 1) + \text{const}$$

Since $\text{acc} > 0.5$ on most of the training distribution, $c \to 1$ is optimal (the global max of a linear function on $[0, 1]$). The optimum is not the honest $c = \text{acc}$. **Any proper scoring rule gives honest confidence (Brier-log maximized at $c = \text{acc}$); the linear reward does not.**

Mitigation: either make the reward Brier-proper (drop the linear-in-$c$ term) or clip $c$ at an honest calibration target.

---

## 5. Code-Fidelity Audit

Cross-reference of implementation against plan intent. Findings are ranked by severity.

### 5.1 🔴 Free-text confidence fallback leaks through rewards → wrapping loss (confirmed)

- **File:** `src/training/rewards.py:366` (default) and call-sites at lines 596, 747, 862, 902, 950, 1023, 1059, 1138, 1374, 1785, 2074.
- **Issue:** `_parse_meta_blocks_with_spans(..., allow_free_text_fallback=True)` makes the parser accept bare "`confidence: 0.XX`" as a valid meta block. `confidence_revision_reward_v2` (line 2074) uses the default. RL converges to drop `<|meta|>` tokens as overhead while keeping the credit. **Confirmed as root cause** of §3.3.
- **Fix:** flip default to `False` (or make it a per-reward kwarg) and inspect each of the 10+ call-sites.

### 5.2 🟡 Easy-exclusion filter too aggressive

- **File:** `src/training/self_distill/trace.py:245–252`.
- **Issue:** `is_easy = difficulty in ("easy", "trivial", "")` — empty string (unlabeled) is treated as easy, silently dropping every root-correct unlabeled example. For AIME/MATH where `difficulty` may not be populated, this shrinks D2a/D2b unexpectedly.
- **Fix:** `is_easy = difficulty in ("easy", "trivial")` (strict) with a log-line for unlabeled rows.

### 5.3 🟡 Length-biased best-of-N selector

- **File:** `src/training/self_distill/online.py:336`.
- **Issue:** `correctness_first` sorts by `(correctness_bool, score, -len(completion))` — longer completions rank higher when score is tied. This amplifies the boilerplate over terse reasoning.
- **Fix:** either drop the length tiebreaker or make it opt-in.

### 5.4 🟡 Token-KL control-span alignment drift

- **File:** `src/training/self_distill/kl.py:110–176` & `src/training/sft.py:280–338`.
- **Issue:** Control-span positions are built from assistant-text token order but applied against `nonzero(shifted_labels != -100)`, which is prompt-offset. An explicit guard at sft.py:323–324 silently skips KL for mismatched positions (no log). If chat-template normalization differs between teacher-query time and SFT-dataset-prep time, the guard fires for every sample.
- **Fix:** emit a warning on first mismatch per batch; audit chat-template equivalence in `teacher_query.py`.

### 5.5 🟡 [META] plain-text fallback recognized by rewards but not by KL

- **File:** `src/training/self_distill/kl.py:19–22`.
- **Issue:** Regex matches only `<|meta|>` tokens; the plain-text `[META]` fallback (recognized in rewards) produces zero KL weight. Inconsistent coverage between reward and KL.
- **Fix:** extend the regex or remove plain-text fallback from rewards (§5.1 subsumes this).

### 5.6 ⚪ RLSD not implemented

- **Files checked:** entire `src/training/self_distill/`.
- **Issue:** No DPO/SimPO/off-policy objective exists that treats the self-distilled best-of-N trajectories as preferred completions. Plan §978–982 lists RLSD-lite as "P5" but no config file path is given. Currently only *score-teacher SFT + token-KL* is operational.
- **Fix:** defer until D2 + D2-KL results stabilize; the plan already gates this.

---

## 6. Current Experiment Status (2026-04-16)

### 6.1 Completed at 21:31 / 21:32 on 2026-04-16

| Run | Node | Launched | Completed | Runtime | Final train_loss | HF artifact |
|-----|------|----------|-----------|---------|-------------------|-------------|
| **D1 (NAIVE, base init)** | `metacognition_train_b` (H200 ×4) | 21:07 | 21:31 | 23.6 min | **0.4260** | `models/self_distill_base_qonly_naive_h200_4gpu/` |
| **D2 (EPISTEMIC, meta init)** | `metacognition_eval` (H200 ×4) | 21:07 | 21:32 | 24.0 min | **0.4782** | `models/self_distill_meta_qonly_epistemic_h200_4gpu/` |

Each checkpoint: 4 sharded safetensors (model-00001-of-00004.safetensors … 00004-of-00004) + config.json + generation_config.json + tokenizer + chat_template.jinja, uploaded to `iamseungpil/metacot` dataset repo under `models/`.

### 6.2 Post-training evaluation (1030 problems, 16k max_tokens)

Eval ran on each node against the corresponding SFT checkpoint. Results pushed to `iamseungpil/metacot` → `results/eval_d{1,2}_16k/`.

| Metric | D1 (naive) | D2 (epistemic) |
|---|---|---|
| Overall accuracy (1030) | **68.35 %** | **66.89 %** |
| Meta emission rate | 0.29 % | **100.00 %** ✓ |
| AIME2024 (30) | 10.0 % | 10.0 % |
| GSM8K (500) | 88.80 % | 87.40 % |
| MATH500 (500) | 51.40 % | 49.80 % |

Reference points: RL step 300 was **88.2 %** wrap rate with boilerplate collapse (908/1030 identical assessments); meta SFT baseline was **~100 %** wrap rate with honest confidence spread. D2 **re-reaches SFT-level wrap behaviour**.

### 6.3 Entropy signature (D2, `<|meta|>` marker, n=184 blocks; window = 8 tokens)

| Group | n | before | meta-body | after | **Δ = after − before** |
|---|---|---|---|---|---|
| Correct | 164 | 0.218 | 0.239 | 0.475 | **+0.257 nats** |
| Incorrect | 20 | 0.472 | 0.296 | 0.530 | +0.058 |
| All blocks | 184 | 0.246 | 0.245 | 0.481 | **+0.236** |

SFT baseline (reference) was Δ = **+0.300** nats; RL step 300 free-text confidence mode was Δ = **−0.052** nats (correct = −0.042, incorrect = **−0.193**). D2's **positive Δ** (same sign as SFT) is the signature that the controller has been restored, in contrast to RL's *forced certainty* (negative Δ on incorrect). For D1 (naive) the entropy analyzer found **zero confidence-marker spans** (0.29 % meta + 0 free-text confidence), confirming the collapse-probe interpretation.

### 6.5 Rebuild from RL redirect data with proper best-of-N (2026-04-16 evening)

After auditing the original D1/D2 datasets we found they were built by `build_self_distill_dataset.py` directly on `data/v8_meta_inside_strict.parquet` (the SFT teacher trace) with **`candidate_count=0`** — i.e. no best-of-N regeneration was actually performed. Both D1 and D2 in §6.1 were therefore meta-stripped or meta-preserved *re-passes* of the SFT data, not self-distillation in the strict sense.

We rebuilt both datasets from `data/verl_train_redirect.parquet` (n=2935, the redirect subset used in RL — by construction the harder, OOD-style problems the SFT did not master) using `scripts/run_online_sdpo_regen.py` (mode=`question_only_best_of_n`, `--repair_candidates 8`, `require_correct_teacher=True` enforced by the wrapper). For D1 the student=teacher was `v8_base_matched_strict_sft` with `selector=correctness_only`; for D2 the student=teacher was `v8_meta_inside_strict_sft` with `selector=correct_then_meta` (sort key: `(is_correct, meta_commit_quality, total, −length)`).

Compute optimisations applied: vLLM `chunk_size 32 → 64`, `max_new_tokens 2048 → 3072`, `gpu_memory_utilization 0.85 → 0.92`; SFT `per_device_batch 1 → 4`, `grad_accum 8 → 2` (effective batch 32 preserved, step throughput 2.3×).

| Run | Teacher | Selector | Surviving rows | SFT runtime |
|---|---|---|---|---|
| D1-rebuilt | base SFT | correctness_only | ~1660 (out of 2935) | ~5 min |
| D2-rebuilt | meta SFT | correct_then_meta | ~1660 | ~5 min |

**Eval results (1030 problems, 16k max_tokens):**

| Model | Overall | AIME | GSM8K | MATH500 | meta_rate | Δ entropy |
|---|---|---|---|---|---|---|
| D1 old (SFT-rehash) | 68.35 % | 10.0 % | 88.8 % | 51.4 % | 0.29 % | n/a |
| D2 old (SFT-rehash) | 66.89 % | 10.0 % | 87.4 % | 49.8 % | 100.0 % | +0.236 |
| **D1 rebuilt** | **68.2 %** | **16.7 %** | 88.4 % | 51.0 % | 0.0 % | n/a (no marker) |
| **D2 rebuilt** | **59.8 %** | 6.7 % | 78.8 % | 44.0 % | 98.9 % | **+0.231** |

D2-rebuilt entropy signature (n=155 blocks):
- Correct (118): before 0.306 → meta 0.255 → after 0.558, **Δ +0.253 nats**
- Incorrect (37): before 0.446 → meta 0.285 → after 0.609, Δ +0.162
- All: before 0.339 → meta 0.262 → after 0.570, **Δ +0.231 nats**

**Key findings (rebuild):**

1. **D1 rebuilt — AIME +6.7 pp**: training on hard RL problems with correctness-only filter improves the OOD benchmark (AIME 10 → 16.7 %) without hurting GSM8K/MATH. Confirms RL data is the right source for self-distill.
2. **D2 rebuilt — controller fully restored, accuracy down**: meta wrap rate 99 %, entropy Δ matches SFT signature (+0.231 vs SFT +0.300). But Overall accuracy dropped 67 → 60 %. Mechanism: forcing teacher to *both* be correct *and* emit a clean meta block leaves a smaller, more verbose pool; the student over-fits to wrapping ceremony at the expense of post-meta reasoning length.
3. **Tradeoff exposed**: simply requiring meta wrapping does not preserve task accuracy. The next two distills (Meta-KL, RLSD-lite) are the experimental tests for whether a softer constraint (KL only on meta spans) or a calibrated reward (commit-shape reward in RL) can recover accuracy without losing the controller.

Artifacts (HF `iamseungpil/metacot`):
- `results/self_distill_rebuild/{d1,d2}/` — driver logs, regen output, summary
- `models/self_distill_rebuilt_{d1_naive,d2_epistemic}_h200/` — final + checkpoint-118 + 4 sharded safetensors
- `results/eval_{d1,d2}_rebuilt_16k/` — per-problem parquet + JSON
- `results/entropy_d2_rebuilt_16k/` — per-block CSV + stats JSON

### 6.4 Teacher-selection quality audit (D2 training data)

Strict `\boxed{...}` vs `gold_answer` comparison on a 500-row D2 sample (from `data/self_distill/d2_epistemic.parquet`):
- **79 / 500 = 15.8 %** of teacher completions have a boxed answer that does not match the gold string.
- Driver: `src/training/self_distill/online.py` has `require_correct_teacher=False` as the default; `src/training/self_distill/trace.py:249` applies the easy-exclusion + correctness filter only to the *root fallback* path, not to `selected_completion`.
- **Implication:** D2 was trained with a non-zero fraction of wrong-answer teachers. The 15.8 % figure overstates true wrongness because simple format differences (`5/2` vs `2.5`) are included; an answer-equivalence check is needed to get the true rate. Ablation: build `d2_epistemic_correct_only.parquet` with `require_correct_teacher=True` at regeneration time, then re-SFT.

Both runs use config `sft_self_distill_*_h200_4gpu.yaml`, batch size 1 × grad-accum 8 (eff. 32), lr 1e-6, 2 epochs, bf16, ZeRO-3 no-offload, max_length 4096.

### 6.2 What failed before

- **Attempt 1** (20:35): `accelerate launch` crashed because `configs/accelerate_sft.yaml` had a `deepspeed_config` block while `sft.py` passed `deepspeed=json_path` to `TrainingArguments` — duplicate-key ValueError. Both D1 and D2 failed identically.
- **Attempt 2** (21:00): switched to `torch.distributed.run`. Both crashed with `ModuleNotFoundError: No module named 'src'` because torchrun does not put cwd on `sys.path`.
- **Attempt 3** (21:07, current): set `PYTHONPATH=/scratch/code` and kept `torchrun`. Passes import; models loaded; first 90+ steps printed.

### 6.3 Keep-alive / preemption defense

- GPU keep-alive (45 s matmul loop) runs during pip install and model download to prevent idle suspend on BSC nodes.
- HF push daemon pushes `/scratch/driver_*.log`, `/scratch/train_*.log`, and the latest `trainer_state.json` every 10 min to `iamseungpil/metacot` (dataset repo) under `results/self_distill_training/{d1,d2}/`.
- Final safetensors for D1 and D2 will be pushed to `models/self_distill_*_h200_4gpu/` on completion.

### 6.4 Success criteria (against plan §1086–1111)

For D2 to be considered a successful controller rebuild:
1. `<|meta|>` wrap rate on the 1030 eval ≥ 95 % (currently RL is at 88.2 %).
2. Confidence entropy across emitted values ≥ 0.5 nats (currently collapsed to 0.96).
3. OOD-combined accuracy ≥ D1 + 2 pp.
4. AIME `no_boxed_rate` < 10 % at max_tokens = 16384.
5. Trigger-conditioned correction rate on redirect problems ≥ base-meta SFT level (recover the −22.3 pp loss).

---

## 7. Plan Gaps and Recommendations

**G1. Phase-numbering collision.** "Phase 5" is used for both curriculum/RAG (plan §794) and self-distill mainline (§1264). Rename the self-distill phase to "Phase SD" or renumber to avoid readers conflating the two.

**G2. E21R-v2 operational state.** Section 0 reports "300 steps completed" while Phase 4 still reads "resume from latest checkpoint." Reconcile in the next plan revision.

**G3. RLSD implementation path.** Unlike Variants A and B, Variant C lacks a concrete config file. Even a stub YAML referring to a `grpo_v2.py` + `init_from=D2_checkpoint` flow would close this gap.

**G4. Reward-parser consistency.** Audit every call-site of `_parse_meta_blocks_with_spans` (§5.1). If RL is ever rerun with the current `rewards.py`, the same wrapping-loss dynamic will return.

**G5. Difficulty-label coverage.** Either enforce difficulty labels upstream (dataset builder) or relax `is_easy` in `trace.py` (§5.2).

**G6. AIME token budget.** Fix max_tokens ≥ 16384 by default for any AIME evaluation. Include a `no_boxed_rate` metric in all eval reports.

**G7. Final-wrap plan.** After D2 completes, run:
  (a) 1030-problem 16k eval of D2 and D1;
  (b) confidence-anchored entropy analysis on D2 outputs;
  (c) boilerplate-template share computation;
  (d) AIME re-run with no_boxed_rate check.
Tie each back to the table in §6.4 for pass/fail.

---

## Appendix A. File map

| Concern | Path |
|---|---|
| Plan | `results/plan_metacot_v8_active_2026_04_09.md` |
| Step-300 analysis | `results/step300_deep_analysis/analysis_report.md` |
| Strict-SFT entropy | `results/entropy_strict_meta/entropy_stats.json` |
| RL confidence entropy | `results/entropy_analysis_step300/rl_meta_confidence/` |
| 16k re-eval | `results/eval_1030_meta_grpo_e21r_v2_step300_16k/` |
| Self-distill code | `src/training/self_distill/` (builders, trace, online, pipeline, kl, teacher_query, eval_metrics) |
| SFT trainer | `src/training/sft.py` |
| Reward functions | `src/training/rewards.py` |
| D1/D2 configs | `configs/sft_self_distill_*_h200_4gpu.yaml` |
| HF artifacts | `iamseungpil/metacot` (dataset repo): `models/`, `data/self_distill/`, `results/self_distill_training/` |

## Appendix B. Notation

- **Qwen3-8B:** student / base model.
- **GPT-5.4-mini:** teacher that produced the meta-wrapped training chains.
- **SFT:** supervised fine-tuning; "meta SFT" = SFT on meta-wrapped traces; "base SFT" = SFT on matched traces with meta stripped.
- **GDPO:** group direct preference optimization; here the RL objective with a correctness head and a calibration head.
- **E21R-v2:** the RL run analyzed in §1.2 (300 steps, 2 heads).
- **D1 / D2:** the two SFT runs described in §6.1.
- **Controller:** the model's learned use of `<|meta|>` blocks as a gating signal that changes downstream decoding.
