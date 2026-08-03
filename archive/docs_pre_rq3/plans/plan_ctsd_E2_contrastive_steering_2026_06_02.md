> ⚠️ **ARCHIVED 2026-08-03 — superseded plan.** The PT-RLSD / CTSD / meta-ROD plan chain
> terminates at `plan_pt_rlsd_v517_2026_05_09_FINAL.md` ("plan iteration 종료") and the whole
> generation is superseded by `docs/redesign/`, `docs/CONSTITUTION.md` and
> `docs/EXPERIMENT_PLAN.md`. Kept as a historical record, not as current guidance.

# PLAN — E.2 Contrastive Meta-Content Steering (LOCKED pre-registration, 2026-06-02)

> **Origin.** B.4 (teacher SCORING) failed (AUC 0.49); C.1 (whole-template leverage) showed
> fixed-position generic injection is net-neutral-to-harmful and difficulty dominates. User
> pivot: stop swapping whole templates; instead **STEER the model's OWN self-emitted meta
> CONTENT along the contrastive (gold−decoy) direction** and measure whether that improves the
> answer. Position stays the META-STUDENT's own choice (self-trigger); the teacher only shapes
> CONTENT. Steering ≠ scoring, so this is genuinely new information beyond B.4.

## Intent (the thing every check is graded against)
The self-teacher must improve meta CONTENT so the response moves toward correct, specifically:
- **(a) Wrong-direction → REDIRECT**: when pre-meta reasoning explores a wrong path, the steered
  meta should flag it and switch approach (explore/abandon-and-retry).
- **(b) Overconfident → VERIFY**: when pre-meta reasoning is confidently wrong, the steered meta
  should slow down and check.
Success = steered meta is MORE corrective than the model's own meta AND the post-meta reasoning
turns toward the correct answer — confirmed BOTH quantitatively (Δacc) AND by direct qualitative
ultrathink reading of (problem, pre-meta reasoning, meta, post-meta reasoning) per case.

## Models (BOTH, compared)
- **E20a SFT** `/home/v-seungplee/sft_e20a_local` (meta-emitting, A.6 winner, base==teacher).
- **v8_strict SFT** `/home/v-seungplee/sft_v8_strict_local` (RL cold-start, strict meta format).
- `<|meta|>`=151669, `<|/meta|>`=151670. Report each model's self-emit rate.

## Problem selection (criteria, pre-registered)
- Benchmarks gsm8k / math500 / aime2024 (difficulty = dominant axis; never pool-cancel).
- **Headroom stratum**: baseline no_meta acc ∈ (0,1) exclusive (model sometimes right → room + capability).
- **Capability split (pass@k)**: record per-problem pass@k. `pass@k=0` = capability wall (a meta
  CANNOT help; analyzed SEPARATELY, never in the primary). Primary = `pass@k>0` (model CAN solve,
  may fumble → meta has a chance). This is the "is it just too hard?" control.
- **Failure-mode tags (for qualitative focus, not gates)**: overconfident-wrong (wrong + low pre-meta
  entropy), wrong-direction (wrong sub-goal). Used to pick cases for qualitative ultrathink.

## Steering mechanism (META-ONLY, the core)
At generation, a custom logits-processor tracks meta-span state via the meta token ids and applies
the contrastive shift ONLY inside a meta span:
```
inside meta span (after 151669, before 151670):
    logit_steered = logit(·|ctx) + α · [ logit(·|ctx+gold_reveal) − logit(·|ctx+decoy_reveal) ]
outside (reasoning body): α = 0  (untouched)
```
- gold/decoy via `_decoy_utils._rule_based_decoy` (decoy: ≠gold string+numeric, valid-form, not equiv — VERIFIED).
- Position = the student's own `<|meta|>` emission (NOT forced). Optional secondary: force-open one
  meta at argmax-entropy if a rollout self-emits none (clearly flagged).
- α sweep {0.5, 1.0, 2.0} to see under/over-steering.

## Arms (per problem, paired Δacc vs `no_meta`)
- `no_meta` (meta span suppressed / regenerated without meta)
- `self_meta` (model's own meta, unsteered)
- `steered@α` (contrastive-steered meta content, meta-only)
- **leakage control** = restrict the Δacc(steered) gain to steered metas that do NOT contain the
  gold answer (regex + numeric check on the generated meta). The gain MUST survive this restriction.

## Locked hypotheses + verifiable criteria
- **H1 — content improvement (stance shift).** Steered meta is more corrective than self_meta:
  stance(self_meta) vs stance(steered) classified into {verify, redirect/explore, commit, generic};
  among cases tagged overconfident-wrong the steered stance shifts toward VERIFY, among
  wrong-direction toward REDIRECT. Criterion: corrective-shift rate ≥ 0.5 on tagged cases AND
  qualitative confirmation on ≥ 8 hand-read cases per model.
- **H2 — steering → correctness (primary go/kill).** `Δacc = acc(steered@α*) − acc(self_meta) ≥ +0.05`,
  paired p<0.05, realized_MDE ≤ 0.05, on `pass@k>0` problems, reported per benchmark AND per model;
  α* = best α on a held-out half (cross-fit, no winner's curse).
- **H3 — leakage guard.** H2 gain MUST survive restriction to answer-free steered metas. If the gain
  vanishes when answer-containing metas are removed → leakage artifact → FAIL (not PASS).
- **POWER HARD-GATE**: realized_MDE > 0.05 → that cell INCONCLUSIVE (never FAIL). gradeable_rate ≥ 0.5.
- **PASS**: H2 holds (≥+0.05, p<0.05, MDE≤thr) on pooled OR gsm8k, surviving H3, with H1 corrective
  shift → contrastive steering improves meta content toward correct → proceed to RL reinforcement
  (self-distill with the contrastive direction as advantage shaping, [[rlsd-vs-sdpo-reference]] fix).
- **FAIL**: H2 powered null/negative OR gain is pure leakage (H3) → contrastive direction is empty/
  leaky for steering → pivot to outcome/entropy-counterfactual credit.
- **INCONCLUSIVE**: under-MDE → scale k/N. Never read as substantive null.
- Stat: `probe_utils.paired_perm_test` (sign-flip, 5000); report mean Δ, sd, n, MDE per cell.

## Turn-granular logging (REQUIRED — for qualitative ultrathink + audit)
Per (model, problem, arm) write a JSONL record:
```
{ model, problem_id, benchmark, gold, decoy, baseline_acc, pass_at_k, failure_tag,
  reasoning: { pre_meta, post_meta },               # CoT before/after the meta span
  meta: { self_text, steered_text, alpha },          # the meta content (both arms)
  action: { arm, steered: bool, masked: bool, meta_pos_frac, steer_alpha },
  skill:  { stance_self, stance_steered, stance_shift },   # verify/redirect/commit/generic
  grid:   { per_arm_correct_over_k, no_meta_correct, self_correct, steered_correct },
  meta_contains_answer: bool }
```
Saved under `reports/e2_steering_<model>_<tag>.jsonl` (+ summary JSON). This is the artifact I read
for the qualitative direction-of-reasoning analysis.

## Implementation & staging (Karpathy minimal-change, clean code)
- New `experiments/probes/e2_contrastive_steering.py`, **import-only** reuse: a3 (templates,
  `raw_entropy`, `first_boxed_token_idx`, `find_meta_spans`), b2 (`make_decoy`,
  `extract_first_meta_block`, `find_answer_token_mask`), b4 (pool/phase pattern), _decoy_utils,
  common/grading (robust_grade/is_gradeable), common/vllm_gen, common/probe_utils. DO NOT modify
  a3/a3b/a6/probe_utils/env/grading/vllm_gen/rewards.
- **Steering needs token-level logit control** → HF generation with a custom LogitsProcessor (vLLM
  lacks per-step contrastive logit injection). Phase-separated: P0 vLLM baseline+grade+headroom →
  P1 HF steered/self/no_meta generation (the 2 extra context forwards for gold/decoy are the
  contrastive signal) → P2 grade + stats + JSONL. vLLM & HF never co-resident (b1 OOM lesson).
- **STAGING (start tiny — "does the code run")**: smoke = 1 model (E20a), 2 problems, k=2,
  max_new≈256, α=1.0 only, assert JSONL written + logits-processor toggles on meta ids + decoy
  valid. THEN scale: both models, per-benchmark headroom, k=16, α sweep, max_new=16384.

## Decision logic
- PASS → RL reinforcement (contrastive advantage shaping, corrected direction).
- FAIL(leakage) → drop gold-reveal contrast; outcome/entropy-counterfactual.
- FAIL(null) → steering direction empty; capability-limited hard problems → meta for tractable band only.
- Qualitative ultrathink (mine) is a CO-EQUAL gate: numbers without a coherent reasoning-direction
  story are treated as INCONCLUSIVE, not PASS.
