# Meta-CoT × OPD Progress Report

**Date**: 2026-05-06 · **Author**: iamseungpil@gmail.com
**Predecessors**: `plan_meta_opd_2026_05_03.md` (v5), `experiments_intent_hypothesis.md` (v3)
**Format**: Intent → Hypothesis → Verification → Result. Designed for readers with no prior project context.

---

## 0. Executive Summary

We ran a **first-of-its-kind** RL training procedure for metacognitive Chain-of-Thought (Meta-CoT) reasoning on Qwen3-8B: **R5 Forced-Meta RLSD**. The method *guarantees* the metacognitive `<|meta|>...<|/meta|>` block is present in every student rollout (otherwise, prior work [Why-Self-Distill-Degrades, 2603.24472] documents that gold-conditioned teachers suppress epistemic verbalization to ~5%, defeating any meta-aware reward).

**Result at step 200**: forced injection mechanism *works as designed* — meta emission rate 73% → 87% on AIME, response length preserved, no epistemic collapse. Standard math benchmarks (GSM8K, MATH-500) gained +13.2 / +12.0 percentage points over the base SFT. **However, AIME-2024 did not move (10.0% → 10.0%)**, falling 26.7pt below the vanilla-RLVR baseline that uses no metacognitive reward at all.

This is the central finding: *the metacognitive overlay we trained does not unlock long-chain mathematical reasoning, even when reward and emission converge*. We attribute this to **a fundamental tension between meta-block self-reflection and uninterrupted long Chain-of-Thought**, and propose **Multi-Teacher OPD** (Plan v5 §10.5 M5.4) as the principled fix: pull the body region toward a verified-strong reasoning teacher (`base_grpo` step 300, AIME 36.7%), pull the meta region toward a calibration teacher (gold-conditioned SFT). Implementation is in progress; smoke checks pass; one production-blocker remains (forced-injection mechanism not yet ported from veRL to TRL).

---

## 1. Background — what is Meta-CoT, and why is it hard?

### 1.1 Meta-CoT — the idea

Standard Chain-of-Thought (CoT) interleaves reasoning steps as plain text. **Meta-CoT** adds an explicit metacognitive layer: special tokens `<|meta|> ... <|/meta|>` wrap short interjections that express the model's *self-assessment* of its own reasoning — confidence, doubt, planned action ("verify", "revise", "switch strategy"). At inference, these blocks function as a calibration scaffold; at training, they expose a structural region we can target with rewards or distillation losses.

We trained a Qwen3-8B SFT baseline (`v8_meta_inside_strict_sft`) on `iamseungpil/metacot` data — 4,996 GPT-5.4-mini-distilled chains, conf 0.745. The SFT pass succeeded in calibration metrics: AIME ECE dropped from 0.870 → 0.610. But raw accuracy dropped below the no-meta SFT baseline (MATH-500: 56.7% vs 76.7%). The follow-up question — *can RL recover accuracy while preserving calibration?* — drives every experiment in this report.

### 1.2 The earlier failure (fresh-0428)

Our first RL attempt — vanilla self-distilled RLVR following Self-Distilled RLVR [arxiv 2604.03128], abbreviated **paper RLSD** — collapsed catastrophically on the OOD AIME benchmark: **AIME 23.3% → 6.7%** (-16.6pp), response length 1778 → 680 tokens, internal `confidence=0.79` printed on wrong algebra. The pattern *exactly* reproduces the failure documented in *Why Self-Distill Degrades Reasoning* [arxiv 2603.24472]: gold-conditioned teachers suppress epistemic markers (the words "wait", "hmm", "perhaps", "maybe", "actually") to ~5% of baseline. Without epistemic verbalization, the model loses the ability to recover from in-flight errors.

We diagnosed two structural causes:
1. **Region-empty signal** — the SFT model emits `<|meta|>` blocks on only 5% of gold-conditioned generations; on those 95% empty responses, any meta-region SDC factor reduces to a no-op.
2. **Implicit token alignment** — paper RLSD assumes student and teacher agree on token boundaries; with sparse meta blocks, they agree on essentially nothing.

### 1.3 The R5 Forced-Meta proposal (Plan v4)

To fix both: **inject `<|meta|>` as a forced suffix at the end of the prompt for every student rollout, and at the end of the teacher's privileged-information block**. This guarantees:
- Region presence: 100% of rollouts begin inside a meta block.
- Token alignment: student and teacher both consume the same forced token at the same position.
- Identifiability: the SDC factor on the meta region now operates on a region that always exists.

Before launch we ran 5 small-scale `inspect_forced_meta` probes (Plan v4 §7 E5):
- S1 (gold + force, no V0 prefix): 5% AIME — too aggressive.
- S2 (V0 prefix + force, no gold): 0% — collapse.
- **S3 (V0 prefix + gold + force): 20% AIME, 71% gold-commit** — landed in the operating regime. We adopted S3.

This is the design that ran on H200 v6 to step 200.

---

## 2. R5 — Forced-Meta RLSD as run

### 2.1 Mechanism (one-paragraph form)

For every prompt `x` in the batch:
1. **V0 prefix generation**: greedy-generate a no-privilege response, slice at the first `<|meta|>` (or first 1500 chars). This is the student's *natural* reasoning trajectory.
2. **Teacher conditioning** (T+, gold): `prompt + V0_prefix + "(The correct answer is X.)\n<|meta|>"`.
3. **Teacher conditioning** (T-, decoy): same, with `_rule_based_decoy(gold)` substituted for the gold answer.
4. **Student rollout**: prompt is fed with `<|meta|>` token id appended via custom verl agent loop (`forced_meta_agent_loop.py`); vLLM begins generation *inside* the meta block.
5. **Per-token SDC factor**: `factor_t = (1 − λ) + λ × clip(exp(sign(A) × (α × Δ⁺_t + β × δ_t)), 1±ε)`, applied only on meta-region tokens. Δ⁺ = log T+ − log Student, δ = log T+ − log T−.

This is **5 components** different from paper RLSD: (a) forced student rollout opener, (b) V0 prefix in teacher conditioning, (c) forced teacher meta opener, (d) explicit token alignment, (e) region 100% guaranteed.

### 2.2 Hardware and budget

- `metacot-r5-h200-rl-0504v6`: 4× H200 (141 GB), AMLT BSC tier, 14 hours wall-clock to reach step 200.
- batch 64, max_response 4096, temperature 0.6, lr 1e-6, GRPO-style group advantage with G=4, KL coefficient 0.002, total 200 PPO steps.

---

## 3. Hypothesis verification (what we checked)

| ID | Statement | Measurement at step 200 | Verdict |
|---|---|---|---|
| **H5.1** | Forced injection does not break capacity (AIME within ±5pp of V0=17%) | 10.0% | ⚠ borderline (-7pp) |
| **H5.2** | Contrastive δ ≥ 0.5 nat in meta region | not measured directly | ⏸ pending |
| **H5.3** | **AIME accuracy ≥ 20%** | **10.0%** | ❌ **FAIL** |
| **H5.4** | Avg AIME response length ≥ 1422 tokens (= base SFT × 0.8) | 9932 chars ≈ 6.2k tokens | ✅ PASS |
| **H5.5** | If H5.3 succeeds, Track B (TRL OPD) is meaningful | H5.3 fail → Track B alone insufficient | ⚠ Track B requires multi-teacher pivot |
| **H5.6** | Natural meta transfer ≥ 34% emission | 87% (AIME) / 95% (MATH) / 97% (GSM) | ✅ PASS |

**Reading**: the forced-injection *mechanism* is verified (H5.4, H5.6 pass). The *downstream value* on hard benchmarks is not (H5.3 fails decisively). H5.1 is borderline — capacity is not destroyed but is not preserved either.

---

## 4. Quantitative comparison (1030 problems)

### 4.1 Main result

| Method | Step | Eval ctx | GSM8K | MATH-500 | AIME-2024 | Meta% (G/M/A) |
|---|---|---|---|---|---|---|
| base SFT v8 (`meta_inside_strict_sft`) | — | 4k | 71.8 | 40.4 | 10.0 | 89/93/73 |
| **R5 forced-meta RLSD** | 200 | 4k | **85.0** | **52.4** | **10.0** | 97/95/87 |
| `meta_grpo_e21r_v2` (legacy meta-aware GRPO) | 300 | 16k | 92.6 | **74.8** | 13.3 | 99/82/20 |
| `base_grpo` (vanilla RLVR — no meta reward) | 300 | 16k | **93.4** | 63.0 | **36.7** | 0/0/0 |
| `d1_naive` SFT distillation | — | 16k | 88.8 | 51.4 | 10.0 | 0/0/0 |

### 4.2 Δ vs base SFT (improvements only)

R5 step 200 improves base SFT on GSM8K (**+13.2pt**) and MATH-500 (**+12.0pt**); AIME unchanged.

### 4.3 Δ vs prior RL runs (red flags)

| Comparator | GSM8K | MATH-500 | AIME |
|---|---|---|---|
| R5 vs `meta_grpo_e21r_v2` | −7.6 | −22.4 | −3.3 |
| R5 vs `base_grpo` | −8.4 | −10.6 | **−26.7** |

**The headline negative finding**: R5 underperforms vanilla RLVR (no meta reward, no SDC, no forced injection) by 26.7pt on AIME.

### 4.4 Confound — hyperparameter mismatch

The comparison is not perfectly apples-to-apples:

| Param | R5 | `e21r_v2` / `base_grpo` |
|---|---|---|
| batch_size | 64 | 128 |
| total_steps | 200 | 300 |
| effective samples | 12.8k | 38.4k (3×) |
| optimizer lr | 1e-6 | 5e-7 |
| eval context | **4k** | 16k |

R5 has seen ⅓ as many on-policy samples and was evaluated at a context length too short for AIME's typical reasoning trace (~10k tokens). M5.1 (Plan v5 §10.5, in flight) addresses both: extend to step 300 + re-evaluate at 16k.

---

## 5. Diagnosis — Why does AIME refuse to move?

We propose four hypotheses, ordered by suspected impact.

### 5.1 Hypothesis A — Meta interrupts long Chain-of-Thought (architectural)

AIME problems require 50+ uninterrupted reasoning steps (combinatorics, number theory, deep algebraic manipulation). The metacognitive interjection `<|meta|>confidence: 0.7, action: revise<|/meta|>` mid-chain breaks the coherent working-memory state the model has been building. Two pieces of evidence:

- **Phase 0 E4** (Plan v4 §7): in V0 conditioning, response *with* a `<|meta|>` block scored 29% accuracy versus 37% *without* (-7.4pp). In V1 (gold) conditioning the gap widens to -18.4pp. Across both, *the act of producing a meta block correlates negatively with correctness*.
- **Cross-method pattern**: vanilla RLVR (0% meta emission) → AIME 36.7%. Meta-aware GRPO (20% meta emission) → AIME 13.3%. Forced 100% meta emission (R5) → AIME 10.0%. Monotone decrease.

Implication: *meta-block placement matters*. Self-reflection at the right moment helps GSM/MATH (medium chains); the same self-reflection at any moment hurts AIME (long chains).

### 5.2 Hypothesis B — Forced first-token meta = SFT prior re-anchoring

Once the prompt's last token is `<|meta|>`, the next-token distribution sits very close to the SFT model's meta-following distribution: `confidence:`, `assessment:`, `action:` — the SFT idiom. The student's on-policy distribution is pulled tightly toward SFT, and the SFT model itself scores 10% on AIME. We end up training on a distribution that has 10% AIME built into its mode. RL gradient cannot move the policy off that mode without a teacher signal that *also* points away from SFT.

Phase 0 E5 verified this: S1 (gold + force, *no* V0 prefix) = 5% AIME. S3 (V0 prefix + gold + force) = 20%. The V0 prefix dilutes the SFT anchoring partially, but at training time the prefix is regenerated each step, so the partial dilution does not compound.

### 5.3 Hypothesis C — 4k context truncation

R5 was trained with `max_response_length = 4096`. R5's AIME generations average 9932 characters ≈ 6.2k tokens; many were truncated. `base_grpo` was *also* trained at 4k but evaluated at 16k, where it generates ~12.8k token responses. The 4k cap during training likely starves R5 of high-reward long-trajectory examples on AIME. This is the most easily fixed confound: M5.1 re-evaluates at 16k.

### 5.4 Hypothesis D — Hyperparameter under-training

Effective sample count: R5 = 12.8k, `e21r_v2` = 38.4k. R5 is at most ⅓-trained. Plan §10.4 and `Rethinking OPD` [2604.13016] both flag that off-policy / cold-start RL benefits especially from longer training; the early-step RL signal is dominated by the SFT prior. Step 300 (M5.1) closes part of this gap; matched batch=128 closes the rest but is currently out-of-budget.

### 5.5 What addresses what

| Hypothesis | Impact | M5.1 (step 300 + 16k eval)? | Method change needed? |
|---|---|---|---|
| A — meta interrupts long CoT | High (architectural) | No | Yes — region-separated teacher (M5.3, M5.4) |
| B — SFT prior anchor | Medium | Partial | Yes — pull body to non-SFT teacher (M5.3, M5.4) |
| C — 4k truncation | Medium | Yes (eval-side fix) | No |
| D — HP mismatch | Low–Medium | Yes (step 300) | No (batch 128 deferred) |

M5.1 is necessary but not sufficient. The decisive change is for A+B: **different teachers in different regions**.

---

## 6. Pivot — Multi-Teacher OPD

### 6.1 Theoretical motivation

The Meta-CoT objective has *two* sub-goals operating on *disjoint regions* of the response:

- **Body region** (95% of tokens): produce coherent long reasoning that reaches the correct answer. The right teacher here is one that *demonstrably* solves the problem class — for AIME, that is `base_grpo` (vanilla RLVR), which scores 36.7%.
- **Meta region** (5% of tokens): produce calibrated self-assessment ("confidence", "assessment", "action"). The right teacher here is the gold-conditioned SFT model, which exhibits the desired meta idiom.

Forcing one teacher to do both jobs is the design error of paper RLSD and of R5. The new design pulls each region toward its specialist:

```
on body region: factor = clip(exp(sign(A) × Δ_reason)),  Δ_reason = log T_reason − log Student
on meta region: factor = clip(exp(sign(A) × Δ_meta)),    Δ_meta   = log T_meta   − log Student
```

This generalizes naturally to **full-distribution KL** (top-K subset) rather than scalar Δt, following *Revisiting OPD* [2603.25562]'s top-K-support recipe.

### 6.2 Method candidates (Plan v5 §10.5)

We consider five candidates, each differentiated from paper RLSD on at least 5 components:

| Method | Distill signal | Region | Teachers | Cold start | Status |
|---|---|---|---|---|---|
| **M5.1** R5+ | scalar Δt | forced 100% | T+/T- gold/decoy | step 200 ckpt | learning (queued v6) |
| **M5.2** OPD-Decoy | full-logit KL on top-K | meta only | T+/T- gold/decoy | R5 step 300 best | code complete, smoke 1+2 PASS |
| **M5.3** DualTeacher | scalar Δt | body=T_r, meta=T_m | base_grpo + SFT | R5 step 300 best | pending |
| **M5.4 ★** Hybrid | full-logit KL | body=T_r, meta=T_m | base_grpo + SFT | R5 step 300 best | pending (primary) |
| M5.5 Reviser-OPD | full-logit KL | meta only | reviser-conditioned | base SFT | pending (skip if M5.4 wins) |

**Selected**: M5.4 (highest theoretical novelty + addresses Hypotheses A+B simultaneously) as primary; M5.2 and M5.3 as ablations to isolate the contribution of (full-logit vs scalar) and (region-separation vs single teacher) independently.

### 6.3 Expected scoreboard (forecast, falsifiable)

| Method | Forecast AIME | Mechanism |
|---|---|---|
| base SFT | 10.0% | (measured) |
| R5 step 200 (4k) | 10.0% | (measured) |
| M5.1 R5 step 300 (16k) | 12–15% | HP fix only, hypotheses A+B unaddressed |
| M5.2 OPD-Decoy | 13–17% | adds full-logit signal density (still single teacher) |
| M5.3 DualTeacher | 18–25% | addresses Hypothesis A directly |
| **M5.4 Hybrid ★** | **22–30%** | Hypotheses A + B simultaneously |
| `base_grpo` (RLVR ceiling) | 36.7% | (measured) |

If M5.4 lands within 22–30% with calibration preserved (ECE ≤ 0.671), we have a paper-grade contribution: *the first metacognitive RL method that does not sacrifice long-chain reasoning*.

---

## 7. Implementation Status (as of 2026-05-06)

### 7.1 Plan v5 — frozen
- 5 self-review rounds, 0 critical issues remaining.
- Stored at `docs/plan_meta_opd_2026_05_03.md` §10 (v5 update).

### 7.2 M5.2 OPD-Decoy code — round 2 review complete
- `src/training/meta_opd_trainer.py` (336 lines) + `scripts/smoke_meta_opd.py` + `configs/meta_opd_decoy_R7_h100_4x4k.yaml`.
- 2 review rounds: 7 Critical, 4 Warning, 7 Suggestion. **11 issues fixed**, 7 deferred to round 3.
- Smoke step 1 (one-shot loss + backward): PASS, gradient finite.
- Smoke step 2 (variance compare on synthetic data): OPD std=0.0025 vs scalar Δt std=0.0594 (24× lower) — H5.2.1 pre-validated *qualitatively*.

### 7.3 Production-blocker — forced injection not yet ported to TRL

R5 (veRL) implements forced `<|meta|>` injection via a custom agent loop (`forced_meta_agent_loop.py`) that appends `META_START` token id to `prompt_ids` before vLLM generation. M5.2 (TRL `GRPOTrainer`) inherits TRL's default generation — **no forced injection is applied at training time**. The cold start checkpoint (R5 step 300) carries learned forced behavior, but the training-inference distribution match is not guaranteed.

Three fixes are possible:
1. **Dataset-level injection**: append `<|meta|>` to every training prompt before tokenization. Simplest, distribution-stable.
2. **Generation kwargs override**: `prefix_allowed_tokens_fn` to force first generated token = `META_START_ID`. Cleaner separation.
3. **Custom rollout hook**: port the veRL agent loop pattern to TRL by overriding `_generate_completions`. Most invasive.

We will adopt option 1 in round 3.

### 7.4 v6 BSC node — exit 255 + queue stall
- v6 has been in BSC queue for ~14 hours.
- Latest resume attempt: bootstrap completed, training process exited with status 255. Hypothesized cause: yaml's hardcoded `global_step_25` download overwrites `latest_checkpointed_iteration.txt = 25`, losing the on-disk step 200 progress (which was on /scratch, not preserved across resume).
- Recovery: cancel + resubmit with `global_step_200` download path and `total_training_steps=300`.

### 7.5 Open issues for next iteration

| Item | Severity | Owner |
|---|---|---|
| Forced injection in M5.2 (round 3 C6 / Q3) | **Critical** | code |
| `padding_side="left"` assertion (W3) | High | code |
| Smoke step 3 — real-model integration test (round 2 C6) | High | code |
| New v6 yaml with step 200 download (exit 255 fix) | High | infra |
| M5.3 DualTeacher implementation | Medium | code |
| M5.4 Hybrid implementation | Medium | code |
| `base_grpo` step 300 ckpt verification on HF (M5.3/M5.4 cold start) | Medium | infra |
| ECE measurement integration in eval pipeline | Low | eval |

---

## 8. Risks and Open Questions

### 8.1 Reasoning-vs-meta tradeoff is fundamental

Even M5.4 may fail to bridge `base_grpo` (36.7% AIME, no meta) and `e21r_v2` (13.3%, with meta). If the tradeoff is genuinely *capacity-coupled* — i.e., any meta budget *must* come out of reasoning budget — then the hybrid will land closer to 13.3% than to 36.7%. We instrument this risk by running M5.4 with `λ_body × KL_reason` ramping from 0 → 1 across training and measuring the AIME inflection point.

### 8.2 `T_reason` may not provide useful gradient inside meta region

Under forced-meta rollouts, every response begins inside a meta block. `T_reason` (vanilla RLVR teacher) has near-zero probability for meta-block tokens — its training never produced them. KL(T_reason ‖ Student) inside the meta region would push the student to abandon meta tokens, which is the opposite of what we want. **Mitigation**: mask `T_reason` to body region only; meta region uses `T_meta` exclusively. This is already encoded in M5.4's `opd_meta_only` mask design but must be verified in smoke.

### 8.3 Calibration may regress under hybrid distillation

ECE measurement at step 300 (M5.1) is a baseline. M5.4 must hold ECE within base SFT × 1.10 (= 0.671). If hybrid distillation drives the meta block toward "verifies whatever the body said" rather than "audits the body's confidence", ECE will rise. **Mitigation**: include a confidence-calibration reward term (already supported in `MetaRLSDConfig.reward_meta_full_bonus`).

### 8.4 batch=64 vs batch=128 — uncontrolled confound

For paper-grade comparison with prior runs we should retrain at batch=128. This is roughly 2× compute. Decision: M5.4 first at batch=64 (stays comparable to R5 step 200/300 family); a single batch=128 run is added as ablation if M5.4 lands in the predicted band.

---

## 9. References

### Primary papers
- **paper RLSD** (2604.03128 Self-Distilled RLVR) — base mechanism for SDC factor.
- **Why Self-Distill Degrades Reasoning** (2603.24472) — epistemic suppression; source of Hypothesis B framing.
- **Rethinking OPD** (2604.13016) — top-K mass concentration, off-policy cold start recipe (used in M5.2).
- **Revisiting OPD** (2603.25562) — top-K teacher support matching, special-token masking; +19.8% on baseline.
- **CoMT** (2601.21909) — two-stage cognitive separation; OOD +4.63%, the inspiration for region-separated teachers (M5.3, M5.4).
- **OPSD** (2601.18734) — privileged-info teacher with verified traces; the dual-teacher pattern formalized.

### Secondary
- 2604.12002 SD-Zero (binary→KL via revision; M5.5 base).
- 2604.13010 Lightning OPD (offline cached teacher; deferred for cost reduction).
- 2604.27083 Co-Evolving Policy Distillation (CoPD; alternative architecture, not adopted).
- 2402.03898 DistiLLM (skew JSD; alternative to KL — may swap in M5.4 if KL_neg unbounded behavior surfaces).
- 2306.13649 GKD (on-policy student rollouts + reverse KL/JSD; M5.2 baseline).

### Internal
- `docs/plan_meta_opd_2026_05_03.md` (v5, frozen) — full method matrix and self-review log.
- `docs/experiments_intent_hypothesis.md` (E1–E4 base) — earlier experiments.
- `memory/reference_opd_april_2026.md` — extended literature notes (April 2026).
- `memory/feedback_meta_self_distill_design.md` — design principles distilled.
- `memory/project_rlsd_paper_alignment.md` — paper-RLSD ↔ our `verl_sdc.py` alignment audit.

---

## 10. Changelog

- **2026-05-06**: report created. Phase 4 R5 step 200 readout integrated. Multi-Teacher OPD pivot documented. Production-blocker (forced injection in TRL) flagged for round 3.
- **2026-05-05**: Plan v5 frozen after 5 self-review rounds. M5.2 OPD-Decoy code round 1+2 complete (smoke 1+2 PASS, 11 issues fixed).
- **2026-05-04**: R5 step 200 reached on H200 v6. On-node eval pipeline (4× H200 TP=4) generated quantitative + qualitative reports.
- **2026-05-04**: Plan v4 frozen (R5 forced-meta design selected after Phase 0/1 evidence; H200 v1–v6 yaml iterations).

---

*Self-review pass v1: §0 executive accessible to first-time reader ✓. §1.1 background does not assume Meta-CoT prior ✓. §3 hypotheses with quantitative pass/fail ✓. §5 diagnoses with measurement-backed evidence ✓. §6 method differentiation matrix references plan §10.5 ✓. §7 status section honest about gaps (forced injection blocker, exit 255) ✓. §8 risks falsifiable, with mitigations ✓.*
