# PLAN — E.1 Entropy-Triggered Injection Leverage (LOCKED pre-registration, 2026-06-02)

> **Origin.** C.1 ([[c1-sweep-result]]) showed fixed-position forced injection is pooled
> net-neutral-to-harmful: GOOD_META NEVER beats raw no_inject at any fixed fraction
> (0.25/0.5/0.75), the inject-per-se cost is significant EARLY (neutral −0.069 @0.25,
> p=0.003), and the only robustly-signed axis is problem DIFFICULTY (gsm8k +0.084 vs
> aime −0.358 @0.25). The user's decision: **pivot the premise** — stop testing fixed
> positions; test whether injecting at the model's OWN uncertainty point (argmax token
> entropy) can beat raw. This is the decisive go/kill for the entire inference-time
> injection premise before any teacher work.

## Intent link
North star = a TEACHER guiding meta POSITION + CONTENT so Meta-CoT > Base SFT. C.1 demoted
POSITION (fixed fractions don't help) and demoted CONTENT (GOOD never beats raw). The last
live form of "good POSITION" is **timing = the model's own max-uncertainty token** (entropy
trigger, the canonical `meta_inject.py` mechanism). E.1 asks: **does ANY injection policy —
here, uncertainty-targeted — beat no_inject net-positive?** If yes → the teacher is still
worth building (it would guide CONTENT at entropy positions). If no (even at the model's own
uncertainty) → the inference-time forced-injection premise is dead → Phase D (training-time
meta shaping, no inference inject).

## Locked hypotheses (pre-registered before any measurement)
- **H-ENT-MAIN (primary go/kill).** Inject `GOOD_META` at the argmax-token-entropy body
  position of the no_inject trajectory: companion `Δacc = acc(entropy_GOOD) − acc(no_inject)`
  **≥ +0.05**, paired p<0.05, on the headroom stratum — tested BOTH pooled AND per-benchmark
  (difficulty must not pool-cancel as it did in C.1).
- **H-ENT-INJECT (inject-per-se at the uncertainty point).** `entropy_neutral` (MARKER_ONLY
  at argmax-entropy) companion vs no_inject: does targeting uncertainty REMOVE the early-inject
  penalty C.1 found at fixed 0.25 (−0.069)? Descriptive isolation of timing vs content.
- **H-ENT-STANCE (secondary, the C.1 stance×position signal).** `entropy_stance` = inject
  verify-stance (GOOD_META) if argmax frac < 0.5 else commit-stance (BAD_META). Tests whether
  position-matched stance beats fixed GOOD. Reported, no kill claim.

## Verifiable criteria (PASS / FAIL / INCONCLUSIVE)
- **POWER HARD-GATE** (in-script, MDE = 1.96·sd·√(2/n)): per hypothesis & per stratum, if
  realized_MDE > 0.05 → that cell INCONCLUSIVE (never FAIL). KILL guarded behind MDE≤0.05.
- **GRADEABLE GUARD**: gradeable_rate ≥ 0.5 per stratum else that stratum INCONCLUSIVE.
- **PASS** (premise survives): entropy_GOOD companion ≥ +0.05 AND p<0.05 AND MDE≤0.05 on the
  POOLED stratum OR on the gsm8k stratum → uncertainty-targeted injection HAS leverage →
  proceed to Step-2 teacher (guides CONTENT at entropy positions; corrected outcome-conditioned
  direction per [[rlsd-vs-sdpo-reference]]).
- **FAIL** (premise dead): entropy_GOOD companion powered (MDE≤0.05) AND null/negative
  (≤ +0.02, p≥0.05) on BOTH pooled AND gsm8k → even the model's own uncertainty point yields no
  injection benefit → forced inference-time injection is dead → **Phase D** (training-time meta
  shaping; drop inference inject). Aligns with C.1's neutral-arm harm + [[b4-result-decision]].
- **INCONCLUSIVE**: any decisive stratum under-MDE → re-run with more k/N (never a substantive null).
- Stat test: `probe_utils.paired_perm_test` (sign-flip, 5000); report mean Δ, sd, n, realized MDE.

## Leakage & confound controls
- **Clean paired DV** (C.1 pattern): per-problem paired Δacc vs a TRUE no_inject raw-prefix
  baseline; each arm differs from no_inject ONLY in the injected segment at the SAME entropy
  position → difficulty cancels within the pair. NEVER label a meta by its own continuation.
- **Headroom stratum**: keep problems with no_inject acc ∈ (0,1) exclusive (per the SAME
  baseline used to locate the entropy position).
- **Templates verified** (a3:69-83): GOOD_META = confidence 0.3 / "slow down, recompute,
  verify"; BAD_META = confidence 0.95 / "no need to re-check, commit"; both answer-free,
  length-similar, stance-only (CONFIRMED 2026-06-02, the C.1 "template not found" flag was a
  wrong-directory false alarm).
- **Entropy position** = argmax of per-token entropy over the pre-`\boxed` body span of the
  no_inject trajectory (b1 `raw_entropy` + `candidate_positions["argmax"]`); fall back to a
  valid body candidate if argmax lands in the boxed/answer span.
- **Difficulty stratification**: report pooled AND per-benchmark (gsm8k / aime2024 / math500)
  so the C.1 Simpson cancellation cannot recur silently.

## Implementation & staging (Karpathy minimal-change)
- New `experiments/probes/e1_entropy_trigger.py`, **import-only** reuse: a3 (GOOD_META/
  BAD_META/MARKER_ONLY/`raw_entropy`/`first_boxed_token_idx`/`find_meta_spans`), b1
  (`candidate_positions`/`stratified_hard_pool`/`load_tokenizer`/phase-sep pattern/`grade`),
  b4 (representative_pool), common/grading (robust_grade/is_gradeable), common/vllm_gen,
  common/probe_utils (paired_perm_test). DO NOT modify a3/a3b/a6/probe_utils/env/grading/vllm_gen.
- **Phase-separated** (vLLM & HF NEVER co-resident — b1 OOM lesson):
  P0 vLLM: baseline gen (k) + grade + per-benchmark headroom stratum.
  P1 HF: `raw_entropy` on each kept baseline trajectory → argmax-entropy body position (free HF).
  P2 vLLM: inject {neutral, GOOD_META, BAD_META/stance} at the entropy position + gen (k) + grade.
  P3 stats: per-arm × per-stratum paired Δacc + perm p + MDE + verdict JSON.
- **Arms**: no_inject (reused baseline), entropy_neutral, entropy_GOOD, entropy_stance.
- **STAGING**: smoke (smoke=1, n≈4, k=2, max_new≈256) green first; then full
  k=16, ~stratified headroom (gsm8k+aime+math500), max_new=16384, max_model_len=20480,
  single local A100, vLLM-only except P1.
- **Cost**: ~ (kept × 3 inject arms × k) continuations + baseline + one HF entropy pass.
  Comparable to C.1 P2 (~15-20h) since entropy adds one HF pass, not extra generation arms beyond 3.

## Decision logic
- **PASS** → uncertainty-targeted injection has leverage → Step-2 teacher (content @ entropy pos).
- **FAIL** → inference-time injection premise dead → Phase D (training-time meta shaping).
- **C1a-style nuance**: if entropy_GOOD passes on gsm8k but fails on aime → difficulty-gated
  teacher (inject only where the model is both uncertain AND the problem is tractable).
- **INCONCLUSIVE never read as a substantive null.**
