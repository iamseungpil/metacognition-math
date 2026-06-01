# PLAN — C.1 Leverage Probe (LOCKED pre-registration, 2026-06-01)

> **Origin.** B.4 (teacher-steering) was a decisive negative, but its interpretation
> exposed that we never established **leverage** — does varying meta POSITION/CONTENT
> actually move the outcome? — before testing teacher *identifiability* (and B.4's DISC
> label was difficulty-confounded). C.1 fixes the ordering: measure leverage FIRST,
> with a clean per-problem paired **Δacc-vs-no_inject** DV. Design hardened by a 31-agent
> adversarial workflow (REVISE→PASS; 20 surviving gaps folded in). NO training.

## Intent link
North star = a TEACHER that guides meta POSITION and CONTENT (→ Meta-CoT > Base SFT).
C.1 is the **precondition**: a teacher can only matter if WHERE and WHAT you inject move
outcomes. C.1a = position leverage; C.1b-DIRECTION = **answer-FREE** content leverage
(exactly what a deployable contrastive teacher supplies). It bounds the max benefit of
perfect position+content selection, licensing (or killing) the teacher-identifiability re-run.

## Locked hypotheses (pre-registered before any measurement)

- **H-C1a-POSITION (primary, single class).** Content fixed at a3 `GOOD_META` (answer-free
  verify/recompute), inject at **body-frac 0.75** of the pre-`\boxed` span (a3
  `first_boxed_token_idx` cap; a plausible late/pre-boxed point. PROVENANCE CORRECTION
  2026-06-01: B.4 gave only a BETWEEN-problem OBSERVATIONAL corr(inject_frac, effect)≈0.39
  — one argmax position per problem, NOT a causal position sweep — so 0.75 is UNVALIDATED
  as a causal "safe" position. Implication: PASS@0.75 is self-sufficient (leverage exists),
  but FAIL@0.75 is AMBIGUOUS (no-leverage vs wrong-position) and CANNOT license a global
  KILL without the {0.25,0.5,0.9} position sweep):
  mean paired `Δacc = acc(inject@0.75) − acc(no_inject)` **≥ +0.05**, `paired_perm_test`
  p<0.05, on the headroom stratum. Other 5 classes {0.25, 0.5, 0.9, just-before-boxed,
  right-after-first-candidate-answer} are EXPLORATORY (curve + Holm-adjusted p; no PASS claim).
- **H-C1b-DIRECTION (primary / LINE-DECISION gate, answer-free, within-problem).** Position
  pinned at the C.1a class (re-estimated on a held-out baseline draw): `acc(productive=GOOD_META)
  − acc(misleading=BAD_META)` **≥ +0.04**, paired p<0.05; companion `acc(productive) −
  acc(no_inject)` **≥ +0.05**, paired p<0.05. Both must hold for a full PASS.
- **H-C1b-ANSWER (secondary / EXPLORATORY ceiling on answer-INFORMATION; NEVER a kill gate).**
  Licensed contrast = `acc(answer-aware-GOLD) − acc(answer-aware-DECOY)` (both generated in the
  IDENTICAL gold/decoy-reveal generation prompt, same decode, then leakage-stripped), run in
  TWO masking conditions (unmasked upper bound / masked conservative). answer-aware vs
  answer-blind is descriptive only (generation-context confounded).

## Verifiable criteria (PASS / FAIL / INCONCLUSIVE)

- **POWER HARD-GATE** (in-script, mde = 1.96·sd·√(2/n) per b4): for every hypothesis,
  if `realized_MDE > threshold` → that hypothesis is **INCONCLUSIVE** (never FAIL/KILL).
  The KILL branch is physically guarded behind `realized_MDE ≤ threshold`.
- **GRADEABLE GUARD**: `gradeable_rate ≥ 0.5` else whole run INCONCLUSIVE; aggregate each Δ
  only over problems gradeable in BOTH paired arms.
- **H-C1a PASS**: primary-class Δ ≥ +0.05 AND p<0.05 AND power_ok AND realized_MDE≤0.05.
- **H-C1b-DIRECTION PASS** (line-decision): Δ(prod−mislead) ≥ +0.04 AND p<0.05 AND companion
  prod−no_inject ≥ +0.05 AND p<0.05 AND power_ok. FAIL only if both null/negative WITH power_ok.
- **H-C1b-ANSWER**: report gold−decoy Δ + p in both masking conditions + residual leak rate +
  drop rate; CONTAMINATED/INCONCLUSIVE if leakage-guard drop rate > 30%. Never gates the line.
- Stat test: `probe_utils.paired_perm_test` (sign-flip, 5000); report mean Δ, sd, n, realized
  MDE, Holm-adjusted p for the 6-class position family.

## Leakage & confound controls
- **Clean paired DV**: per-problem paired Δacc vs a TRUE no_inject raw-prefix baseline
  (`[prompt+base[:p]]`, no marker); each arm differs from no_inject ONLY in the injected
  segment → problem difficulty cancels. NEVER label a meta by its continuation outcome (B.4 fix).
- **Headroom stratum**: keep problems with fresh k-continuation no_inject acc ∈ (0,1) exclusive
  (drops ~52% ceiling + ~9% floor that diluted B.4).
- **Fixed-position isolation**: all answer-free arms pinned at the same C.1a position.
- **Triple leakage guard** on answer-aware metas: (1) numeric/symbolic-variant regex from gold,
  (2) math_verify equivalence over number/expr spans, (3) held-out LLM judge on a sample;
  drop-rate ceiling 30% → else CONTAMINATED. Replaces literal-substring-only masking.
- **Generation-condition control**: oracle contrast is GOLD-reveal vs DECOY-reveal (b4 `make_decoy`)
  in IDENTICAL format/decode; answer-aware vs answer-blind is descriptive only.
- **Oracle generator fix**: `build_teacher_input` is a SCORING harness — NOT used to generate.
  A new gold/decoy-hint continuation generator emits the meta, stripped of the hint at inject.
- **Multiplicity**: one primary position class (0.75) at α=.05; others exploratory + Holm.
  C.1b position re-estimated on a held-out half (not selected on its own eval data).

## Decision logic
- **DIRECTION PASS** → answer-free content leverage exists → PROCEED: re-run teacher
  identifiability = **B.4 REDONE with a clean Δacc label** (label a meta by its OWN paired
  Δacc-vs-no_inject, not by continuation outcome); if C.1a also PASS, teacher must guide BOTH
  position and content.
- **DIRECTION FAIL @0.75** (power_ok, MDE≤thr) → `FAIL_AT_0.75`, NOT a global KILL. Because 0.75 is
  unvalidated (see provenance correction), a single-position null is ambiguous → run the {0.25,0.5,0.9}
  **position sweep FIRST**; only if content shows no leverage at ANY position → KILL → **Phase D**
  (training-time meta-shaping or a different DV).
- **C1a PASS but DIRECTION FAIL** → position-only teacher; defer content steering.
- **Oracle (H-C1b-ANSWER) NEVER triggers KILL.** **INCONCLUSIVE never read as a substantive null.**

## Implementation & staging
- New `experiments/probes/c1_leverage.py` (~420 LOC), import-only reuse of a3/a6/b3/b4/common,
  phase-separated (vLLM & HF never co-resident): P0 vLLM headroom baselines → P1 CPU position
  classes → P2 vLLM C.1a sweep → split position lock → P3 vLLM oracle gen + triple guard →
  P4 vLLM C.1b arms → stats/report JSON.
- **Cost**: full design ≈ 250–300k continuations @16k ≈ 18–30 GPU-h on 4×A100 (≈1–1.5 day node),
  or multi-day on the single local A100.
- **STAGING (cost gate)**: run **C.1-core first** = the DIRECTION line-decision gate
  {no_inject, neutral, GOOD_META, BAD_META} at the single position 0.75, headroom stratum,
  k=32 — the decisive precondition test, locally feasible in hours. Position sweep (C.1a 6-class)
  + oracle (C.1b-ANSWER) follow only if core passes / venue allows.

---

## STEP 1 EXECUTION SCOPE (locked 2026-06-01, the leverage gate "good meta → correct?")

This is the FIRST of the user's two-step validation (Step 1 = leverage + define "good"
causally; Step 2 = does the corrected teacher guide toward it). Step 1 is **purely causal
Δacc — NO teacher, NO HF forward, vLLM-only** (cheaper than B.4). The oracle (H-C1b-ANSWER)
is DEFERRED to a later step; Step 1 ships only the answer-free leverage gate.

- **Probe**: new `experiments/probes/c1_leverage.py`, import-only reuse of a3 (GOOD_META/
  BAD_META/MARKER_ONLY/first_boxed_token_idx), b4 (representative_pool + phase pattern),
  common/grading (robust_grade/is_gradeable), common/vllm_gen, common/probe_utils
  (paired_perm_test). NO a6/teacher.
- **Stratum**: two-sided HEADROOM — keep problems whose fresh k-continuation no_inject
  robust_grade acc ∈ (0,1) exclusive (drops ceiling+floor that diluted B.4).
- **Position**: primary class = body-frac **0.75** of the pre-`\boxed` span; exploratory =
  {0.25, 0.5, 0.9} (Holm-adjusted, no PASS claim). Computed from token offsets via
  `first_boxed_token_idx` (no entropy/HF needed).
- **Content arms** (each injected after the identical fixed-position prefix; differ ONLY in
  the injected segment): `no_inject` (raw), `neutral` (MARKER_ONLY), `GOOD_META`, `BAD_META`.
- **DV**: per-problem paired `Δacc = acc(arm) − acc(no_inject)`, acc = fraction of k
  continuations robust_grade-correct; aggregate over problems gradeable in BOTH paired arms.
- **Verifiable EXIT**:
  - H-C1b-DIRECTION (line-decision): `Δacc(GOOD_META) − Δacc(BAD_META) ≥ +0.04`, paired p<0.05,
    AND companion `Δacc(GOOD_META) − no_inject ≥ +0.05`, paired p<0.05, at position 0.75.
  - H-C1a-POSITION: primary-class GOOD_META `Δacc ≥ +0.05`, paired p<0.05; full position curve reported.
  - POWER HARD-GATE: realized MDE = 1.96·sd·√(2/n); if realized_MDE > threshold → INCONCLUSIVE
    (never FAIL). gradeable_rate ≥ 0.5 else INCONCLUSIVE.
  - **PASS** (leverage exists) → proceed to Step 2 (corrected outcome-conditioned teacher vs
    per-meta causal Δacc). **FAIL_AT_0.75** (power_ok) → run the {0.25,0.5,0.9} position sweep to
    disambiguate (no-leverage vs wrong-position) BEFORE any KILL → Phase D.
- **Run params**: k=24, N target ~200 headroom (pool ~2.5×), max_new=16384, max_model_len=20480.
  vLLM-only, single local A100, phase = {P0 baselines → P1 CPU positions → P2 arms → stats}.
