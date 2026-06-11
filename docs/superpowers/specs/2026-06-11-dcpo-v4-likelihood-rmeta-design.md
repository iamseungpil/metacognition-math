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
