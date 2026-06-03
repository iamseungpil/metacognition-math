# PLAN — E.3 Contrastive-Direction A/B for Meta-Content Steering (LOCKED pre-registration, 2026-06-03)

> **Origin.** E.2 ([[e2-steering-probe]]) built v8_strict self-meta harvest + meta-only contrastive
> steering. Small tests showed the answer-direction (gold−decoy) is NOISY and sign-unstable
> (−0.167 @v8smoke vs +0.10 @paired n=10) and tends to inject HINDSIGHT OVER-CONFIDENCE, while a
> stance-direction (cautious−confident) edged it (+0.20 paired). But everything was n≤10 (noise-
> dominated) and used only VERBALIZED confidence + accuracy — no objective uncertainty. E.3 settles
> WHICH contrastive teacher direction best improves the answer, decomposing answer-vs-stance and
> oracle-vs-deployable, with OBJECTIVE uncertainty metrics, at power, in two pre-registered phases.

## Intent (every check grades against this)
A contrastive teacher must improve meta CONTENT so the response moves toward correct — by REDIRECTing
wrong-direction reasoning and VERIFYing overconfident reasoning — WITHOUT injecting hindsight over-
confidence or leaking the answer. E.3 finds the best steering DIRECTION (the future RL advantage-
shaping signal) cheaply at inference before committing to RL.

## Steering mechanism (FIXED for E.3)
LOGIT-level contrastive decoding, META-CONTENT-ONLY: at each self-emitted meta token,
`scores += α·(logit|ctx_A − logit|ctx_B)`, where ctx_A/ctx_B are the two reveal contexts of the
SAME frozen v8_strict, advanced in KV-lockstep, applied ONLY between 151669 and 151670 (verified
meta-only; the body/continuation is never steered). NOT activation/hidden-layer steering (logit vs
activation is a deferred future axis). Position = the model's OWN self-emitted <|meta|> (self-
trigger; E20a self-emits 0%, v8_strict ~9-32% → v8_strict is the model). Extraction via
b2.extract_first_meta_block (open-only safe).

## Arms — grounding factorial (6 arms)
`grounding` factor = is gold revealed on BOTH reveal sides (so the answer CANCELS in the contrast)?

| arm | direction (ctx_A − ctx_B) | isolates | oracle? |
|---|---|---|---|
| `self` | (no steer) | baseline | — |
| `gold_decoy` | (gold∧confident) − (decoy∧confident) | ANSWER axis (reference) | yes |
| `cautious` | (cautious) − (confident) | STANCE axis, ungrounded | no (deployable) |
| `gold_stance` | (gold∧cautious) − (gold∧confident) | STANCE axis, gold-grounded (answer cancels) | yes-but-answer-free signal |
| `conf_down` | (conf:0.15) − (conf:0.95) | CONFIDENCE-number axis, ungrounded | no (deployable) |
| `gold_conf_down` | (gold∧conf:0.15) − (gold∧conf:0.95) | CONFIDENCE-number axis, gold-grounded | yes |

- α sign: stance/conf arms steer TOWARD cautious / low-confidence (the inversion hypothesis); α>0.
- COHERENCE CAVEAT (pre-registered): `gold_conf_down` is internally CONTRADICTORY ("know the answer X yet
  claim confidence 0.15") → its reveal-stream direction may be noisy. Included as a coherence test: if
  it is noise, that itself shows grounding does not transfer to the confidence-number axis. `gold_stance`
  is coherent ("know X, still verify carefully"). Drop `gold_conf_down` in Phase 2 if Phase 1 shows noise.
- α sweep deferred: Phase 1 fixes a single moderate α (e.g. 0.6) chosen by a quick pre-check (avoid the
  α=1.0 off-distribution destabilization seen in E.2); the DIRECTION comparison is the Phase-1 target.

## What the decomposition answers
1. **ANSWER vs STANCE**: `gold_decoy` vs `gold_stance` — teach the answer-direction or the verification stance?
2. **GROUNDING main effect (= need the oracle?)**: grounded vs ungrounded per axis (`gold_stance` vs
   `cautious`; `gold_conf_down` vs `conf_down`). grounded≈ungrounded → deployable (no oracle).
3. **STANCE vs CONFIDENCE-number**: `cautious` vs `conf_down` — full stance or pure confidence axis.
4. **gold_conf_down coherence**: does contradictory grounding kill the signal.

## Problem selection
- Benchmarks gsm8k / math500 / aime2024 — DIVERSE difficulty & type, STRATIFIED (C.1: difficulty dominates).
- v8_strict baseline on a LARGE pool; KEEP problems where v8_strict SELF-EMITS a meta (extract_first_meta_block).
- Headroom (no_meta acc ∈ (0,1)) + pass@k capability split (pass@k=0 = capability wall, analyzed separately —
  controls "is it just too hard?"). Self-trigger position p_self from the baseline.

## Metrics (OBJECTIVE uncertainty is core — the E.2 gap)
Per (arm × problem), over k continuations from the steered-meta point:
- **Accuracy** — paired Δacc vs `self` (the outcome).
- **Objective uncertainty**: (a) answer-token ENTROPY via a3/a2 `raw_entropy`; (b) SELF-CONSISTENCY =
  fraction of k continuations agreeing on the modal final answer (empirical confidence).
- **Calibration**: |verbalized_confidence − empirical_accuracy| (ECE-like); does the direction REDUCE it.
- **Verbalized confidence**: parsed "confidence: X" (subjective — reported but NOT the criterion).
- **Qualitative (my ultrathink)**: read problem / pre-meta / self-meta vs steered-meta / post-meta across
  arms on sampled cases — does the steer produce concrete alternative-method verification / redirect, and
  does the reasoning turn toward correct (CO-EQUAL gate; numbers without a coherent story = INCONCLUSIVE).

## Two phases (DISCOVERY → CONFIRMATION; winner's-curse-safe)
- **Phase 1 (DISCOVERY/compare, moderate n~40-50, all 6 arms, stratified)**: rank directions by paired Δacc
  + the 4 decomposition contrasts + objective metrics + qualitative. Output = best 1-2 directions. NO PASS claim.
- **Phase 2 (CONFIRMATION/powered, n~100-120, the Phase-1 winner(s) only, FRESH problems)**: pre-registered
  test of ACTUAL improvement. Separate problem set → no winner's curse.

## Verifiable criteria (Phase 2)
- **POWER HARD-GATE**: realized_MDE = 1.96·sd·√(2/n) > 0.05 → INCONCLUSIVE (never FAIL). gradeable_rate ≥ 0.5.
- **PASS** (direction improves): paired `Δacc(winner − self) ≥ +0.05`, p<0.05 (`paired_perm_test`), MDE≤0.05,
  on pass@k>0 problems, pooled AND per-benchmark; AND the gain SURVIVES the leakage guard (answer-free steered
  metas only) AND is accompanied by a coherent objective-uncertainty story (entropy↓ where correct / calibration↑)
  AND qualitative confirmation. A deployable (ungrounded) winner passing → no-oracle teacher.
- **FAIL**: winner powered null/negative, or gain is pure leakage / pure verbalized-confidence inflation
  (no objective backing) → that direction is not a teacher → reconsider (activation-steering axis, or RL-only).
- **INCONCLUSIVE**: under-MDE → scale k/N. Never a substantive null.

## Performance (gating dependency — MUST precede powered runs)
Current steered arm runs the WHOLE continuation on slow HF (~10-20× vLLM); 6 arms × stratified N is infeasible.
FIX: HF only for the META span (where steering applies) → hand off the prefix+steered-meta to vLLM for the
(long, unsteered) continuation. Reduces steered-gen to ≈vLLM speed + small HF meta overhead. Build + verify
this BEFORE Phase 1 (else use a reduced Phase-1 n as a fallback).

## Decision logic
- Phase 1 ranks → Phase 2 confirms. PASS (deployable winner) → that no-oracle direction becomes the steering/
  RL teacher. PASS (oracle-only winner, e.g. gold_stance) → RL-teacher uses it (oracle available in training).
- The winning DIRECTION = the RL advantage-shaping contrast ([[rlsd-vs-sdpo-reference]] fix: replace the
  gold-likelihood teacher with this validated direction; sign from RLVR correctness).
- INCONCLUSIVE never read as substantive null.

## Staging
Smoke (1-2 self-meta problems, all 6 arms, tiny) green → Phase 1 (n~40-50) → Phase 2 (n~100-120). Karpathy
minimal-change; extend e2_contrastive_steering.py with the 6 contrast modes + objective metrics + the
HF-meta→vLLM-continuation handoff; import-only from a3/a2/b2/_decoy_utils/common; iterative-code-review.
