# DCPO v4 — 2-Stage RL + Dense Likelihood R_meta (2026-06-11)

User feedback (3 items) + RLT paper (arXiv 2506.08388) + adversarial design review (agent a98ebcf4).
Replaces CF *regeneration* with frozen-reference *likelihood delta*; format taught in a separate short stage.

## 0. Why (one paragraph)
v3's causal R_meta (c_with − c_without via CF regeneration) is sparse (~95% zeros), slow (full 2nd
generation), and fragile (leak guard ungrades 50–75% of CFs). v3l/v3m showed format penalties drive
emission collapse when value learning and format correction share one run. Fix: (Stage 1) teach format
alone, fast; (Stage 2) dense R_meta = how much the meta raises the frozen SFT base's likelihood of the
model's OWN post-meta continuation, sign-gated by final correctness.

## 1. Stage 1 — format-only RL (~50 steps)
- Rewards: R_corr (w1.0, ANSWER) + **uncentered post-centering emission floor** for wellformed metas
  (reuse v3m `dcpo_meta_floor` mechanism — length-neutral, trusted-classes only). NO CF, NO likelihood.
- R_meta/R_cal weights 0 in this stage. Format head active (w 0.1).
- Exit: wellformed ≥ 70–80% AND Δ-distribution non-degenerate on a held-out scoring batch
  (content check — review M4: form-only exit invites hollow-format overfit).

## 2. Stage 2 — dense likelihood R_meta
### Signal
- prefix = prompt + response up to first `<|meta|>`; meta = tag-inclusive block;
  C = model's OWN tokens after `<|/meta|>` → `</think>` + answer (native position).
- Δ_t = logP_ref(C_t | prefix+meta+C_<t) − logP_ref(C_t | prefix+C_<t), frozen SFT base via the
  ref-policy worker custom-batch path (verl_sdc.py `_build_teacher_logprob_batch` precedent), **T=1.0**
  (review M1; rollout-temp scoring compresses PMI by 1/T).
- Aggregation: decided by offline probe among {sum-clipped, top-k mean, avg, max}; max−min REJECTED
  (direction-blind).
- Sign-gate (review M3): R_meta_row = (+1 if correct else −1) × clip(agg(Δ), 0, c) — correct rollouts
  can only get ≥0, wrong only ≤0.
- Centering (review I2): group-mean-subtract over **meta-emitting non-gated rows only** (member mask);
  no-meta rows contribute nothing.

### Routing (review I4)
- Dense Δ head routes to **META_CONTENT \ CONF** (new mask expression). CONF stays Brier-only —
  dense PMI on the conf token rewards "stating a confidence the continuation echoes", fighting Brier.
- R_corr / R_cal / format head unchanged. Region masks unchanged.

### Emission stability (review I1 — floor STAYS)
- Centered Δ is mean-zero ⇒ no standing emission pull; length cost erodes emission ⇒ collapse redux.
- KEEP the v3m floor in Stage 2 (small, e.g. 0.05–0.1). Floor removal only as a LATE Stage-2 action
  gated on measured emission stability. "Should I emit" = floor; "was this meta good" = centered Δ.
- Mild length cost: small coefficient, added to the R_corr scalar (response-length budget), weak per
  RLT warnings; introduced with a warmup alongside w_meta (review M4).

### Anti-hack guards
1. Frozen scorer (never the live policy).
2. **Build a real meta↔continuation overlap detector** (review C2 — the "v2 boilerplate detector"
   does NOT exist; only the field-label regex does). N-gram/LCS overlap above threshold ⇒ Δ invalid.
3. Meta containing the literal boxed answer string ⇒ Δ invalid.
4. clip(agg(Δ), 0, c) + member-mask centering.

### Splice alignment contract (review C3 — bug magnet)
- The without-arm (prefix+C) is a sequence the model never produced. Re-tokenize BOTH arms
  independently; locate C by decode-and-rematch (never assume token-index correspondence; BPE can merge
  across the deleted boundary). Subtract only over the C-span that decodes byte-identically in both
  arms; ASSERT identity before trusting Δ. Mirror of the v3b silent-bug class.

## 3. Offline probe FIRST (kill-or-go gate, local A100)
On existing eval generations (~8k with meta):
- Compute Δ under all aggregations; compare with old causal c_with−c_without on the graded subset.
- **Placebo control (review C1)**: inject a contentless coherent meta ("Let me continue.") — real meta Δ
  must beat placebo Δ. KILL criterion.
- **Continuation-shuffle control**: C from a different rollout of same problem; Δ should drop to ~0.
- **Selection-bias split (review I5)**: report separately on (a) graded subset (leak-guard-passed = least
  entangled metas) and (b) leaked subset (most entangled — the population that dominates training).
  (b) is load-bearing; correct/wrong separation must hold there.
- Probe also fixes clip threshold c from the T=1 Δ distribution.

## 4. What gets removed/kept
- REMOVED: CF regeneration wrap (gate sdc_counterfactual=false), signature suppression, leak guard
  (all CF-generation-only machinery goes dormant, not deleted).
- KEPT: region masks/routing, Dr.GRPO per-head centering, format 3-tier parser, floor, R_cal (weight
  re-tunable), 5-way sync rule, rollout table observability.

## 5. Review traceability
C1 placebo/shuffle controls → §3; C2 overlap detector → §2 guards; C3 alignment contract → §2;
I1 floor stays → §2; I2 member-mask centering → §2; I4 CONF carve-out → §2; I5 probe split → §3;
M1 T=1.0 → §2; M2 floor-mechanism reuse for stage-1 bonus → §1; M3 clip-at-0 sign gate → §2;
M4 w_meta warmup + content exit-check → §1/§2.

---

## Amendment A1 (2026-06-11, post cross-shuffle probe): placebo-corrected Δ′

Findings (report `docs/reports/2026-06-11-dcpo-v3-v4-study.md`, H5–H7): raw Δ
passes placebo-t (17.9) and AUC (0.78) but is **86% generic text-presence**
(placebo mean 0.114 vs real 0.133); cross-problem shuffle retains 0.52. The
content increment (+0.019 right / −0.046 wrong vs placebo) is the trainable
signal.

Spec changes:
1. **Stage-2 reward** = sign-gated `agg(Δ_real) − agg(Δ_placebo)` per row
   (aggregate-level subtraction = probe `verdict_corrected` semantics). Knob
   `dcpo_pmi_placebo_correct: true`; third scored arm (prefix + PLACEBO_META +
   continuation) reusing the real without-arm logprobs (without-span equality
   enforced at splice; divergence → row fails closed, member 0 — no raw
   fallback inside a centering group). Ref cost ×1.5. PLACEBO_META SSOT moved
   to `src/training/dcpo_pmi.py`.
2. **Stage-1 exit gate** now grades the **corrected** battery on the gs50
   checkpoint (probe `--shuffle-mode cross_problem`, read `verdict_corrected`)
   and re-freezes `dcpo_pmi_clip_gate` from `recommendation_corrected` (the
   corrected Δ′ distribution is much narrower: mean 0.019 vs 0.133).
3. **Stage-2 watch item**: the v3m floor only *delays* collapse (8–12 steps);
   if emission decays in the first ~60 stage-2 steps while Δ′ flows, extend
   the w_meta warmup (M4) instead of raising the floor.
4. New observability: `dcpo/pmi_placebo_fail_rate`.

Gate: ship Δ′ only if the per-row corrected probe (E-corr, running) returns
`verdict_corrected.overall == PASS`; corrected-AUC failure → pre-registered
stop + re-decision (raw Δ + group-centering argument vs stage-1-only).

### A1 addendum (2026-06-12): E-corr verdict = PASS; signed shuffle criterion

The per-row corrected probe initially printed FAIL, traced to two grading
artifacts, both fixed (commit pending):
1. **Direction-blind collapse ratio**: |shuffle|/|real| graded mean's
   wrong-content SIGN FLIP (−0.0456 vs +0.0188, ratio 2.43) as "did not
   collapse". The criterion's intent is "wrong content must not EARN" →
   replaced with the SIGNED criterion (real>0 and shuffle < 0.25·real;
   negative retention passes). Same direction-blindness we rejected in
   max−min (spec §2) — reproduced in our own metric, now test-locked.
2. **AUC-shopping method choice**: the corrected verdict picked topk_mean
   (AUC_ent 0.764) whose wrong-content retention is +0.50 — wrong content
   still earns half; gameable. Verdict now grades the TRAINING method (mean).

Final E-corr (method=mean, guard-filtered n=7877): mean_gt0 t=17.9 PASS /
signed retention −2.43 PASS / corrected AUC_ent 0.714 PASS → **PASS**.
Payout simulation (sign-gated, clip 0.1085): honest content E[R]=+0.0106 vs
noise content +0.0015 (7× margin); two-sided clip would widen the margin
(+0.0076 vs −0.0337) at the cost of punishing honest-negative rows — keep
the one-sided M3 gate, note two-sided as a fallback if stage-2 shows mashing.
clip_gate re-frozen 0.342 → 0.1085 (corrected p95).
