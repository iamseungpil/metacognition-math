# PLAN — CTSD Phase B probes: can inference *guide* position & content? (v1, 2026-06-01)

> **Scope.** Three inference-only probes (local A100, no training) that answer,
> *before* committing any RL compute, the user's two questions:
> 1. **Position** — can we *find* (via a gold-free metric, incl. teacher-based)
>    a body position where force-injecting `<|meta|>` causally helps? (B.1)
> 2. **Content** — on the E20a substrate, does the model's *own* injected meta
>    carry good/bad variance a contrastive teacher can grab? (B.2)
> 3. **Causal gate** — does marker-only inject causally help on E20a? (B.3)
>
> **Relation to prior probes (what is NEW vs done).**
> - A.2 measured entropy→*rollout* wrongness (AUC 0.749). It did NOT measure
>   *per-position* causal benefit. **B.1 does.**
> - A.3b compared 3 fixed position *rules* (argmax/onset/random) on v8 and found
>   argmax neutral / onset harmful. It did NOT (a) map the full position→benefit
>   curve, (b) correct for winner's curse, (c) test *teacher-based* position
>   metrics, (d) run on E20a. **B.1 adds all four.**
> - A.6 showed a teacher discriminates *template* good/bad content (E20a AUC 0.95).
>   It did NOT test E20a's *self-generated* meta. **B.2 does.**
> - A.3 ran the causal gate on **v8 only** (`--model SFT_V8_STRICT`). **B.3 = the
>   same probe on E20a** — the substrate the user wants to train.
>
> **Karpathy minimal-change mandate.** B.3 adds ZERO new code (run `a3_inject_causal.py`
> with `--model <E20a>`). B.1/B.2 are new files that *import* the proven machinery
> (`a3b.body_candidates`/`pick_position`, `a3.gen_batch`/grading, `probe_utils`
> teacher+entropy+AUC+perm, `a6.score_meta_logp`/`build_prompt_with_meta`). a3/a3b/a6
> are NOT modified — their published verdicts stay intact.

---

## Substrate & data readiness (measured 2026-06-01)

Hard-wrong headroom (where inject *could* help), from R10v2 evals on HF:

| substrate | aime wrong | math500 wrong | gsm8k wrong | natural meta |
|---|---:|---:|---:|---:|
| v8_strict | 26/30 | 209/500 | 50/500 | 87–99% (decorative) |
| **E20a** | 19/30 | 96/500 | 40/500 | **0–3%** |

- Correct/wrong labels: ✅ ample on both. Pre-register selection = hard-wrong
  (math500+aime) pooled, seed `STRATIFIED_SAMPLE_SEED`.
- good/bad **content** cases: constructable (A.6 template gold/decoy, proven AUC
  0.95) + B.2 outcome-labels E20a-self meta.
- good/bad **position** cases: **NOT pre-labeled** — B.1 *constructs* them by
  causal measurement. Risk: A.3b suggests good positions may be rare on v8 →
  B.1 Stage-1 is exactly the test of whether they exist at all.

---

## B.1 — Position-benefit metric discovery (NEW)

**Intent.** Determine (i) whether body positions exist where marker-only inject
causally improves accuracy *beyond selection-bias chance* (position headroom), and
(ii) whether a deployable gold-free metric ranks positions by that benefit —
including teacher-based metrics (does T+ want a meta here? do T+/T− diverge here?).
Run on v8 *and* E20a to test the clean-substrate hypothesis.

**Hypotheses (pre-registered).**
- **H-B1a (headroom):** mean over problems of [Δacc(best-REAL position, held-out k2)
  − Δacc(best-RANDOM position, held-out k2)] ≥ **+5pp**, paired p<0.05, on ≥1 substrate.
  (Held-out + real-vs-random kills winner's curse.)
- **H-B1b (findability):** ∃ gold-free metric with AUC(metric → 1[Δacc_p>0]) ≥ **0.65**
  AND AUC(metric) − AUC(position-index baseline) ≥ **+0.05**.
- **H-B1c (substrate):** headroom and/or findability is stronger on E20a than v8.

**Verification.**
- Sample: N=60 hard-wrong (40 math500 + 20 aime), fixed seed; per substrate.
- Per problem: 1 baseline rollout (seeded) → per-token raw entropy (a2 def).
  Candidate positions P = {entropy p50, p75, p90, argmax, onset} ∪ {R=4 random body
  positions}, each ≥ MIN_TOK=50, before first `\boxed`, outside existing meta spans.
- Per candidate p: from shared prefix `[prompt + resp[:p]]`, generate k=8
  continuations in 2 arms — **marker** (`\n<|meta|>\n` appended, model fills+closes)
  and **noinject** (continue raw). Δacc_p = acc(marker) − acc(noinject). (a3b design.)
- **Winner's-curse control (cross-fit):** split k into k1=4 (select argmax-Δacc
  position) + k2=4 (estimate that position's Δacc, held-out). Compare best-REAL vs
  best-of-R-RANDOM under identical k1-selection / k2-estimation.
- Gold-free metrics per p (all computable at inference, no gold answer):
  ① raw entropy H_p ② onset slope (H_p − H_{p-w}, w=8) ③ **pause-propensity**:
  P(`<|meta|>`=151669 | prefix→p) under the v8 meta-emitting SFT (NOT E20a — E20a
  emits ~0% meta so its P(meta)≈0 everywhere is degenerate) ④ teacher T+/T− next-token
  KL at p (gold-aware vs decoy-aware teacher contexts, a6 `build_teacher_input`) ⑤
  student−T+ logp gap. Baseline metric = normalized position index (a useful metric
  MUST beat this).
- Metric eval: pool all (problem,p) with measured Δacc_p → AUC(metric vs 1[Δacc_p>0])
  + Spearman(metric, Δacc_p). (Metrics ③④⑤ come from ONE teacher/ref forward per
  problem over the full prefix — per-position next-token dists in a single pass, not
  per-position forwards. ④ needs a per-problem decoy answer: reuse a6 decoy gen.)
- **Multiplicity control:** split problems 50/50 into discovery/confirmation (fixed
  seed). A metric counts as passing H-B1b only if AUC≥0.65 on the *confirmation* half
  (discovery half picks the candidate metric). Guards against 5-metric × 2-substrate
  false positives.
- **Power guard:** baseline `boxed_rate` < 0.5 → INCONCLUSIVE (raise `--max_new`).
- **Gates:** Stage-1 = H-B1a; Stage-2 = H-B1b (confirmation-half). Stage-1 FAIL on
  *both* substrates → position-guiding is dead → STOP B-line, go Phase D (experiential SFT).
- **Staging & cost:** smoke (n=2) → v8 (N=40) → E20a (N=40). Heaviest probe:
  N·|P|·2·k ≈ 40·9·2·8 ≈ 5.8k continuations/substrate + teacher forwards (~A100 2–4h
  each). N reducible to 40; raise k if INCONCLUSIVE on power.

## B.2 — E20a self-generated content variance (extends A.6)

**Intent.** On E20a base, marker-inject (argmax; B.1-best as sensitivity), let E20a
generate its OWN meta + continuation; test whether the contrastive teacher score
(T+ − T−, E20a teacher = A.6 winner) on that self-meta carries variance tracking
continuation correctness — the target v8 lacked (A.1).

**Hypotheses (pre-registered).**
- **H-B2a (variance exists):** correct-leading and wrong-leading E20a-self metas each
  number ≥ Nmin=15 (else INCONCLUSIVE — "no variance" is itself the A.1 echo on E20a).
- **H-B2b (discriminable):** AUC(contrastive_score → continuation correct) ≥ **0.65**,
  perm p<0.05.

**Verification.**
- Sample: N=60 E20a hard-wrong; marker-inject at argmax → k=8 (meta+continuation) →
  grade outcome (math_verify).
- contrastive_score = mean_logp_{T+}(meta_tokens) − mean_logp_{T-}(meta_tokens),
  answer-token-masked (a6 `find_answer_token_mask`).
- AUC + perm p (probe_utils). AUC≥0.65 → contrastive β justified; <0.65 with
  variance present → drop β (marker + correctness only).

## B.3 — E20a causal gate (= A.3 on E20a, marker decision)

**Intent.** Decisive offline causal gate for E20a training: does marker-only inject
at the deployable position causally help accuracy on E20a?

**Hypothesis (pre-registered).** acc(marker b) − acc(no-inject a) ≥ **+3pp**, paired
p<0.05, power-guarded. (Decision on **b** marker, not **c** content — training is marker-only.)

**Verification.** `python experiments/probes/a3_inject_causal.py --model <E20a> --select
mixed --n 48 --k 6` — **argmax position (a3 default), ZERO code change**. Only if B.1
surfaces a materially better rule do we add a small `--inject-rule {argmax,onset,...}`
flag to a3 (one surgical arg, default argmax) and re-run. PASS (b−a≥+3pp, p<0.05) →
Phase C on E20a. FAIL + power-OK → inject non-causal on E20a → STOP → Phase D.

---

## Decision integration (autoresearch loop)

```
B.1 (position) ─┬─ Stage-1 FAIL (both substrates) ─────────────▶ STOP → Phase D (SFT)
                └─ PASS → best position+metric
                      ├─▶ B.3 causal gate at best position
                      │      PASS ─▶ TRAIN (Phase C on E20a)
                      │      FAIL ─▶ STOP → Phase D
                      └─▶ B.2 content variance
                             AUC≥0.65 ─▶ keep contrastive β
                             AUC<0.65 ─▶ marker + correctness only
```

- **TRAIN gate** (Phase C on E20a): B.3 PASS **AND** B.1 Stage-1 PASS. B.2 sets β on/off.
- autoresearch "modify→verify→keep/discard→repeat" iterates over: B.1 metric choice,
  B.3 inject position, B.2 teacher/clip. No decisive FAIL → no auto-launch of training.
- Pre-registered thresholds locked in THIS commit before any measurement.
