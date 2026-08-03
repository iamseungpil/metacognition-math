# PLAN — B.4 Teacher-Steering Causal Gate (pre-registered, 2026-06-01)

> **Why this replaces the B.1/B.2/B.3 framing.** An adversarial alignment audit
> (34-agent workflow, 2026-06-01) found the B-suite mis-aligned with the stated
> intent — *"find a TEACHER that guides meta POSITION and CONTENT"* — on all four
> lenses (intent-fidelity, headroom-validity, teacher-search, statistical power):
> - **No probe tests the teacher GUIDING meta.** b1/b2 are read-only scorers/rankers;
>   b3 injects FIXED hand-written GOOD/BAD templates with no teacher in the loop.
> - **The dependent variable is inject-RESCUE of wrong answers**, but the valid 16k
>   run showed only **24% of the hard pool is genuinely wrong** (E20a solves 68/90).
>   So the DV is blind on the 76% already-correct items, and the +5pp/+3pp gates
>   exceed a plausible ceiling on the shrunken target.
> - **No tie to the north-star** (trained Meta-CoT ≥ Base SFT).
>
> B.4 is the minimal redesign that puts the **teacher in an active steering role**
> and uses a DV with signal across the **whole** distribution. Reuses b2 (self-meta
> generation), a6 (contrastive teacher scoring), b3 (paired causal gate machinery).
> No change to a3/a6/b1/b2/b3 — new file only.

---

## Intent → testable question

**Intent.** Find a teacher that GUIDES meta CONTENT (position held fixed at the
deployable argmax-entropy rule — position optimization is a separable later probe).

**Question.** At a fixed position, does a teacher that *selects which meta to inject*
guide the student better than no guidance — specifically, better than a **random**
selection from the **same** self-generated meta pool (which removes "a marker helps"
and "any meta helps" confounds, isolating the teacher's *selection* contribution)?

## Pre-registered hypotheses (locked before any measurement)

- **H-B4-GUIDES (primary).** Over a representative sample, the teacher-top-1 meta
  beats a random-1 meta (same pool, same position) on per-problem continuation
  accuracy: `mean_p[acc(teacher_top) − acc(random)] ≥ +0.04`, paired perm p<0.05.
  *Null:* teacher selection is no better than random.
- **H-B4-NOHARM (gating constraint).** On the already-CORRECT stratum, re-injecting
  the teacher's pick does not break correct answers:
  `acc(teacher_top) − acc(no_inject) ≥ −0.02` on correct items, one-sided paired
  perm p<0.05. Required EVEN IF GUIDES passes (an inject that rescues wrong answers
  but harms correct ones is a net loss vs the north-star).
- **H-B4-DISC (precondition).** The teacher's contrastive score discriminates
  correct-leading from wrong-leading metas in the pooled set:
  `AUC(contrastive_score → continuation_correct) ≥ 0.60` (mann_whitney_auc).
  If AUC<0.60 → **INCONCLUSIVE** (this teacher cannot tell good from bad meta, so
  selection is meaningless — not a refutation of the concept).

## Verifiable PASS / FAIL / INCONCLUSIVE

- **INCONCLUSIVE** if `gradeable_rate < 0.5` (16k power floor) OR `AUC_disc < 0.60`
  OR realized power for +0.04 < 80% (report MDE at realized n_gated, k).
- **PASS** iff GUIDES (≥+0.04, p<0.05) AND NO-HARM (≥−0.02, p<0.05) AND power_ok.
- **FAIL** (power_ok, AUC≥0.60, but GUIDES fails) → the contrastive teacher does NOT
  steer content → STOP the contrastive-content line → Phase D (experiential SFT).

## Design & verification

- **Sample.** Representative stratified `N=120` (≈40 each aime / math500 / gsm8k),
  sampled IGNORING correctness, then each tagged correct/wrong by FRESH 16k robust_grade
  (`common/grading.robust_grade`). Expect ≈24% wrong / 76% correct → both strata present.
- **Position.** argmax body entropy (a3/b2 `body_argmax_entropy_pos`), before first
  `\boxed`, ≥MIN_TOK — held fixed (this probe varies CONTENT, not position).
- **Self-meta pool.** At p*, marker-inject and generate `M=12` continuations (E20a
  fills its own meta+tail); extract the meta block from each (b2 `extract_first_meta_block`).
  Pool = the closed meta blocks (problems with <2 distinct metas → drop, logged).
- **Teacher scoring.** contrastive_score(meta) = `mean_logp_T+(meta) − mean_logp_T−(meta)`,
  answer-token-masked, T+ gold-aware / T− decoy-aware (a6 `build_prompt_with_meta` /
  `score_meta_logp` / `find_answer_token_mask`; E20a teacher = A.6 winner, base==teacher).
- **Arms (re-inject at p*, then generate k=8, robust_grade).**
  - `teacher_top` = re-inject argmax-contrastive meta.
  - `random` = re-inject a uniformly random meta from the SAME pool (seeded).
  - `no_inject` = raw continuation from prefix (baseline for NO-HARM).
- **DV.** per-problem acc = mean robust_grade over k=8. GUIDES = paired (teacher_top −
  random) over all N; NO-HARM = paired (teacher_top − no_inject) over the correct stratum.
  DISC = pooled AUC over all (problem, meta) using each meta's own continuation outcome.
- **Stats.** `probe_utils.paired_perm_test`, `mann_whitney_auc`; report MDE.
- **Compute.** phase-separated (vLLM gen → free → HF teacher), per the b1 OOM fix.
  ≈120·12 self-meta + 120·3·8 arm ≈ 4.3k×16k continuations + 120 teacher passes ≈ 4–6h A100.
- **Substrate.** E20a primary (substrate to train; 0% natural meta → injected metas
  fully attributable; E20a is the A.6 teacher so base==teacher = one model load).
  v8 sensitivity arm only if E20a passes.

## Staging
smoke (N=4, M=4, k=2, max_new small) green → full E20a N=120 16k → verdict → loop.
