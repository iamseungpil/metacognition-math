# s3b — Meta-Format Priming (fix forming, keep PMI) design & experiment plan

**Date**: 2026-06-15  **Author**: ctsd-phase-c autonomous loop  **Branch**: ctsd-phase-c
**Status**: APPROVED direction (user 2026-06-15): **both** form-fixing levers — format SFT-priming
(primary) + constrained decoding (best-effort assist) — with **PMI likelihood reward kept**. Pending
user review of this spec, then writing-plans → ultracode implementation → node launch.

Builds on: s3 (data-regime + composition, run `dcpo_v4_s3`). s3 succeeded on data/length/correctness
but **plain-ified** (meta abandoned). This spec fixes only the meta **forming**, preserving everything
s3 got right.

---

## 1. North-star (unchanged)

Strengthen useful metacognition to raise reasoning accuracy. s3 showed the data regime and composition
work, but the meta channel collapsed by *form*, not by reward design. s3b targets **forming only**.

## 2. Problem — meta forming fails, so PMI is starved and the policy plain-ifies

s3 evidence:
- `meta_emit_rate` 0.60 (step1, ~SFT) → **0.15** (step42): RL abandoned meta.
- `wellformed_rate` 0.22 (step1) → 0.06: form was weak **from SFT onward**.
- `pmi_member_rate` ~0.13: PMI (the usefulness signal) reaches almost no meta.
- `correctness` rose to 0.58: the model learned to **just answer without meta** (plain-ification).

Root cause (diagnosed across this session): the meta delimiters `<|meta|>` / `<|/meta|>` are **added
tokens with zero pretraining prior**, exposed only ~6.5k times in SFT. The **closing** tag is hit
hardest because (a) its position is variable (meta block length CV 0.69) and (b) it competes with the
strong native `</think>`. So in free generation the model can't reliably *pair* (dup_open / swapped /
discard, not "no close"). Weak form → low PMI coverage → meta earns little reward → strong R_corr pulls
the policy to drop meta. **Form is a generation-ability problem, not a reward-tuning one** (RL form
reward failed in s1/s2/s2b). Fix it with SFT/decoding, not RL.

## 3. Design — fix forming, keep PMI (two levers)

### 3.1 Format SFT-priming (primary, we fully control it)
A short SFT pass on top of the existing `v8_meta_inside_strict` SFT, with three changes:
- **(a) Simplify the meta block to a short, fixed template** (`confidence: / assessment: / action:`,
  capped ~80 tokens) so "when to close" is predictable (kills the CV-0.69 variability). Rebuild the
  SFT meta traces to this template.
- **(b) Initialize the meta-token embeddings from think tokens** — copy the `<think>`/`</think>`
  embedding rows into `<|meta|>`/`<|/meta|>` before this SFT, transplanting the native pairing prior
  into the added tokens.
- **(c) Short format-priming SFT** on the simplified-template data so the model re-learns clean
  open→close by imitation (low LR, few epochs, mixed with general data to avoid forgetting).

### 3.2 Constrained decoding (best-effort assist)
A state-machine **custom logits_processor**: once `<|meta|>` is emitted, bias toward closing
(`<|/meta|>`) within N tokens, forbidding a second open. Injected into the vLLM rollout
`sampling_params` (same site where we already inject `logit_bias`). **Best-effort / gated**: verl 0.7.1
does NOT expose guided decoding and our "free reasoning + bounded meta block" case needs a *partial*
constraint, so this carries Ray-serialization + verl-integration risk. Ship it **behind a flag**; if it
doesn't integrate cleanly, s3b still stands on §3.1 alone. (Verified: vLLM supports logits_processors;
verl passes `SamplingParams(**sampling_params)` so injection is possible but non-trivial.)

### 3.3 PMI likelihood reward — KEPT, and it recovers
PMI is the only usefulness signal (SFT/decoding handle *form*, PMI handles *which meta helps*). With
form fixed by §3.1/§3.2, `pmi_member_rate` recovers from 0.13 toward ~1.0, so PMI finally grades most
meta and selectively strengthens useful metacognition. Do NOT replace RL with SFT — SFT can't optimize
"meta caused the answer". Structure = `existing SFT → format-priming SFT (§3.1) → RL (PMI + composition,
+ optional §3.2 decoding)`. This is the same "prime behavior → RL for usefulness" pattern as the
redirect-priming spec; the two priming jobs can later merge.

### 3.4 tier-1 auto-correction widened (minor)
Extend the existing replace-malformed-then-route (`dcpo_format_replace`) to recover a slice of
`discard` (e.g. keep the first valid pair) so PMI coverage gets an extra lift even when generation slips.

### 3.5 Bundled bug fix — signal.alarm grading
`rewards.py::_check_correctness` passes `timeout_seconds=None`, which the node's math_verify turns into
`signal.alarm(None)` → TypeError flood → string-match fallback (symbolic golds mis-scored, log flood).
Fix here (pass a positive timeout or guard signal availability) since it has been silently degrading
correctness since s2b. Small, isolated, with a regression test.

## 4. What we DON'T change

Data (`verl_train_meta_mix` — signal alive), composition (anchor / emit-route / len_cost / trunc — length
controlled), PMI likelihood reward, R_corr/R_cal. Everything s3 got right is preserved.

## 5. Hypotheses (falsifiable)

- **HF1 (form holds).** Post-priming `wellformed_rate` starts > 0.5 and stays > 0.4 through RL (vs s3's
  0.22→0.06). *Falsified if* it decays below 0.3 again.
- **HF2 (PMI coverage recovers).** `pmi_member_rate` > 0.7 (vs s3 0.13). *Falsified if* it stays low
  despite wellformed rising.
- **HF3 (no plain-ification).** `meta_emit_rate` stays ≥ 0.5 through RL (vs s3 → 0.15). *Falsified if*
  emit collapses < 0.3 again.
- **HF4 (usefulness, the verdict).** Held-out per-benchmark self_consistency ≥ Meta SFT
  (gsm8k 0.885 / math 0.518 / aime 0.167), difficulty-stratified, **with meta present** (emit > 0.5).
  *Falsified if* below Meta SFT or meta is boilerplate.

## 6. Metrics

Priming: rollout `wellformed_rate` / `meta_emit_rate` right after format-SFT (before RL). RL: the s3
dashboard (wellformed, meta_emit, pmi_member, acc_with/without, eff_ratio_meta, correctness, length) +
`dcpo/discard_rate` (should drop). Final: held-out eval vs Meta SFT, difficulty-stratified.

## 7. Decision tree

- HF1 fails (form still decays) → §3.2 decoding becomes mandatory (not best-effort); escalate.
- HF2 ok, HF3 fails (emit still drops) → raise w_emit / meta_floor (RL holds the channel).
- HF4 fails (held-out ≤ Meta SFT) → meta present but not useful; revisit PMI sign-gate / add redirect priming.

## 8. Operational

- **Order**: format-SFT-priming (short, 1 GPU ok) → push primed ckpt → RL stage (H100 4-GPU, init = primed
  ckpt, config = s3 config + decoding flag). wandb `metacot-dcpo-v4`, run `dcpo_v4_s3b`.
- New code release asset; pull_resume durability; save best ckpt separately.
- Built via ultracode (karpathy surgical + clean code + TDD) after writing-plans; node launch after I
  verify the primed ckpt's rollout wellformed_rate jumped (the priming actually worked) before spending
  the RL node.
- Follow-up: merge with redirect Harvest/Prime; difficulty-stratified eval; revisit constrained decoding
  if verl later exposes guided decoding natively.
