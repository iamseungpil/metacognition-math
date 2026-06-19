# Confidence-Conditioned Redirect/Verify Teacher-Distill Design

**Date:** 2026-06-19
**Status:** converged design, ready to implement (GPU-free smoke only this pass)
**Predecessors:** `2026-06-18-redirect-priming-counterfactual-design.md` (Stage-A CF harvest),
`2026-06-14-redirect-priming-from-failed-rollouts-design.md`, PG0 pilot (memory `pg0-raw-onpolicy-harvest-infeasible`).

---

## §0 North-star + success criteria

### North-star
Useful metacognition raises math accuracy. The model must **SELF-EMIT a confidence
score** and **DECIDE redirect-vs-verify-vs-nothing from it**:
- **low confidence / on a wrong path → REDIRECT** (switch to a genuinely different method);
- **high confidence but checkable → VERIFY** (independent confirmation);
- **not useful → emit nothing** (no decorative meta).

Calibration (emitted confidence matching correctness) is a **sub-signal** of that
decision, not the goal. The goal is decision → accuracy.

### Why this design (evidence that forces it)
The PG0 pilot (`scripts/pg0_yield_pilot.py`, memory `pg0-raw-onpolicy-harvest-infeasible`)
established three facts that kill the obvious cheaper paths and define this one:
1. **On-policy harvest from the cold-start SFT is infeasible** — accepted-redirect
   yield ≈ 141 ≪ 1500 target (`pg0_verdict` → STOP). We cannot mine enough causal
   redirect demos by re-rolling the student alone.
2. **Forced redirect helps EASY, hurts HARD** — `[pg0-raw]` `mean_gap(R-Nc)=+0.082`
   overall but `gap_by_attribute` shows `difficulty=hard` is negative (≈ −0.034).
   Redirect must be **anchored on easy/medium only**.
3. **The existing v8 SFT redirect data is DECORATIVE** — only 758/4264 rows have a
   real switch though 2199 are labeled redirect, 1340 of them hard, with **no causal
   filter**. Training on it yields hollow meta → RL **forming-collapse**
   (v1 acc 0.651 < baseline 0.786).

The fix is a **teacher-distill** pipeline that builds *causally-filtered*,
*on-distribution*, *student-confidence-conditioned* demonstrations.

### Success criteria
- **S1 — calibrated emitted confidence.** The confidence the model emits in its
  `<|meta|>` block tracks STUDENT correctness: bins of emitted confidence are
  monotone in measured pass-rate; ECE on the held-out eval drops vs the
  decorative-v8 SFT. (Confidence label = student self-consistency; §1.)
- **S2 — causally appropriate decision.** On held-out problems, REDIRECT demos
  recover low-conf-wrong cases (R flips wrong→right while the no-redirect control
  stays wrong) and VERIFY demos **confirm** high-conf-right cases without
  regressing them (no-harm confirm gate; a verify "demo" that flips a wrong prefix
  is a redirect, not a verify, and is scored as such — §4). Measured by the same
  counterfactual gate used to build the data (§4) replayed on held-out.
- **S3 — meta-ON accuracy survives, no forming-collapse, no HOLLOW meta.** Meta-ON
  math accuracy on the 1030-eval (`scripts/eval_vllm_1030.py`) ≥ **baseline 0.786**
  and ≥ **v1 0.651**, and the metacognition **survives RL**. Crucially, S3 is NOT
  satisfied by raw meta-emit rate (a high emit rate with no causal benefit is
  exactly the v8 decorative failure mode and would game a bare "emit rate did not
  collapse" check). The RL-survival gate is **utility-conditioned**: on held-out
  problems where the model fires a meta block, the accuracy *attributable to the
  firing* must be positive — measured by the §4-style counterfactual replay
  (meta-ON vs the same prefix with the meta block ablated, lower-CI > 0 in
  aggregate), not by how often meta fires. We additionally report meta-emit rate
  only to confirm it neither collapses to ~0 (under-forming, the v1 mode) nor
  saturates toward firing-on-everything (over-forming, the v8 decorative mode);
  both extremes fail S3 regardless of headline accuracy. S3 is the gate; S1/S2 are
  the mechanism.

---

## §1 The confidence signal — student self-consistency

**Decision: confidence label = STUDENT self-consistency, never teacher confidence.**

For each candidate problem, roll out the cold-start student
(`v8_meta_inside_strict_sft`, the SFT init for all RL per `CLAUDE.md`) N times
(N = 8, temp 0.8, mirroring the PG0 rollout config in
`scripts/pg0_yield_pilot.py:217-224`). Define

```
conf_target(problem) = pass_rate = mean_k [ _check_correctness(rollout_k, gold) ]
```

graded by `src.training.rewards._check_correctness` (answer-blind grading).
`agreement` (modal-answer share) is an alternative we record but the headline
target is `pass_rate`.

**Why not teacher confidence.** The teacher (TRAPI GPT-5.4) is a *more capable*
model; its confidence reflects *its* competence, not the student's. Distilling
teacher confidence would teach the student to claim confidence it cannot back up
— exactly the overconfidence we are fighting. The student's own pass-rate is the
only quantity that makes "emit confidence ≈ p" a *truthful, calibrated* statement
about *this* model's correctness.

**Why gold is used only for correctness, never to measure confidence.** Gold
enters solely through `_check_correctness` to grade rollouts and to run the causal
filter (§4). It never enters the confidence target's *value* beyond the binary
correct/incorrect tally that defines pass-rate — there is no gold-derived
"difficulty" or "the answer is X so confidence should be Y" leak. Confidence is a
property of the *student's rollout distribution*, computed before any teacher call.

---

## §2 Anchoring (two source bands: wrong-prefix for redirect, right-prefix for verify; easy/medium only)

**Decision: anchor each demo on a REAL student rollout; the band depends on the
ACTION; exclude hard for both.**

The two actions need *opposite* confidence regimes, and §1 makes confidence =
student pass-rate. A single low-pass-rate band would force every demo (verify
included) to carry a low measured confidence, which is incoherent with VERIFY
("high conf but checkable") and would make the teacher state a high confidence the
measured value contradicts — re-introducing the exact overconfidence we fight and
breaking S1's truthfulness. So we split the source band by action:

- **REDIRECT source — low-conf WRONG-prefix band.** Keep problems in the PG0
  frozen redirect band (`BAND_LO=0.125, BAND_HI=0.5` in
  `scripts/pg0_yield_pilot.py:62`) that have ≥1 wrong rollout. The **wrong rollout
  prefix** is the anchor, spliced at `splice_index(n_tokens, frac)` with
  `frac ~ U[SPLICE_LO, SPLICE_HI]` = U[0.30, 0.70]
  (`scripts/harvest_redirect_cf.py:49-53`). `conf_target` here is in [0.125, 0.5]
  → the teacher truthfully states a *low* confidence and redirects.
- **VERIFY source — high-conf band, anchor on a RIGHT-but-checkable prefix.** Keep
  problems with `pass_rate ≥ VERIFY_BAND_LO = 0.625` (mostly-right; high measured
  confidence) and anchor on a **correct** rollout prefix spliced the same way.
  `conf_target` is high (≥ 0.625) → the teacher truthfully states a *high*
  confidence and names an independent check. This is the only regime in which
  VERIFY's "high conf" claim is calibrated to the student, and it is the only
  regime in which the §4 verify *no-harm-confirm* gate (prefix-was-right) is even
  reachable. The "always anchor on a wrong prefix" rule of earlier drafts applied
  to REDIRECT only; applying it to verify would make every verify demo a disguised
  redirect (see §4) and leave the confirm gate dead.

In both bands the teacher continuation starts from text the student *actually
produced*.

**Why on real student rollouts (both bands).** Demos must be on-distribution. A
teacher-only solution would be OOD relative to where the student actually is; the
student cannot imitate a recovery (or a check) from a prefix it would never write.
Anchoring on its own prefix makes the meta+continuation a *reachable* edit of its
own trace.

**Why easy/medium only (exclude hard), both actions.** PG0 fact #2: forced
redirect *hurts* hard problems (`gap_by_attribute` `difficulty=hard` negative). On
hard problems the student is wrong because of a *capability* gap, not a
*path-choice* gap, so a redirect cannot recover it and the demo would teach
"redirect when stuck" as a reflex that fires uselessly. We filter on
`split_tags.difficulty ∈ {easy, medium}` (the `tags` dict already threaded through
`_load_pool` → `scripts/pg0_yield_pilot.py:140-151`) for **redirect AND verify** —
hard contributes only the untouched v8 base/verify corpus at mix time (§5), never
a freshly-distilled demo of either action.

---

## §3 Teacher conditional generation

**Decision: teacher generates CONDITIONED on [problem + student prefix (wrong for
redirect / correct for verify) + measured confidence], and the action is fixed by
the §2 source band the confidence came from.**

The teacher (TRAPI GPT-5.4, `gpt-5.4_2026-03-05`, via
`src.metacot.generator.get_trapi_client` + `generate_single_chain` retry/backoff
machinery, concurrency from `scripts/gen_metacot_v2.py` `ThreadPoolExecutor`)
receives:
- the problem,
- the student's spliced anchor prefix — a **wrong** prefix in the redirect band, a
  **correct** prefix in the verify band (§2),
- the **measured student confidence** `conf_target` (§1).

It must produce a `<|meta|>` block (format from `src.metacot.prompt_behavior` /
`prompt_control_v4.py`: `<|meta|> … confidence: 0.xx … <|/meta|>`,
`META_START/META_END = <|meta|>/<|/meta|>`) that:
1. **states confidence ≈ `conf_target`** (truthful to the measured student value —
   this is what makes S1 trainable);
2. **selects the action from that confidence — and the action is already pinned by
   the §2 source band, so the steer cannot drift**:
   - **REDIRECT source band** (low `conf_target` ∈ [0.125, 0.5], wrong prefix) →
     **REDIRECT**: name a concrete trigger (contradiction / failed substitution /
     unit mismatch — reuse the redirect scenario rules in
     `prompt_behavior.BEHAVIOR_SYSTEM_PROMPT` and
     `prompt_control_v4.build_control_v4_prompt` redirect branch, which requires a
     low-confidence intervention + a *real* strategy switch), then switch to a
     genuinely different method. The stated confidence MUST be low (≈ `conf_target`,
     < 0.5) — coherent with the measured value.
   - **VERIFY source band** (high `conf_target` ≥ 0.625, right-but-checkable
     prefix) → **VERIFY**: state high confidence ≈ `conf_target` and name an
     *independent* check (`prompt_control_v4` verify branch /
     `BEHAVIOR_SYSTEM_PROMPT` verify scenario), perform it, and finalize. The check
     must be genuinely independent (a *different* derivation / back-substitution /
     bound), not a re-statement of the same steps (the §4 verify gate enforces this
     causally).
   - otherwise → **NONE** (no meta block; plain correct continuation).
3. then writes the **correct continuation** to the gold answer.

Because the action is fixed by the band (low-conf-wrong → redirect, high-conf-right
→ verify), `select_action` is a function of the band/`conf_target`, NOT of a
free teacher choice; `prefix_correct` is therefore *derived from which band the
anchor came from* (wrong-band ⇒ False, right-band ⇒ True) rather than re-judged,
removing the dead path where an always-wrong prefix would make verify unreachable.
The teacher is *steered* by the measured confidence rather than free to invent its
own, so the emitted-confidence label and the chosen action are coherent by
construction. Action selection is recorded as a label (`action ∈
{redirect, verify, none}`) on every candidate for the §4 filter and the §5 mix.

---

## §4 Causal filter (gold for correctness/causality only)

**Decision: keep a demo only if its action is CAUSALLY load-bearing**, reusing the
already-built + tested counterfactual logic in `scripts/harvest_redirect_cf.py`.

For each teacher candidate, regenerate ANSWER-BLIND continuations (k = 8, temp 0.9,
gold hidden) from the candidate's own anchor prefix (wrong prefix for redirect,
correct prefix for verify) in arms, grade each with `_check_correctness`, then
apply the gate:

**Redirect demos** — `accept_redirect(r_grades, nprime_grades, nc_grades,
bprime_grades, margin=ACCEPT_MARGIN=0.5)`
(`scripts/harvest_redirect_cf.py:78-91`). Arms:
- `R` = redirect block then continue,
- `N'` = null-meta (confidence restatement, no switch) then continue,
- `Nc` = plain continuation (no meta),
- `B'` = plain-prose backtracking with `<|switch|>` masked (credits redirect
  *content*, not a free second attempt).

Acceptance is on the **one-sided 95% lower-CI bound** (`lower_ci_diff`) of
(R − best control) ≥ margin, with `MIN_K=4` guarding small-k noise. This enforces
"redirect flips wrong→right AND the no-redirect control stays wrong." Diagnostics
`raw_yield_stats` / `gap_by_attribute` localize where it helps (re-used as-is).

**Verify demos** — analogous gate, *new* helper `accept_verify` (§6, M-cf):
arm `V` = verify block then continue, control `Nv` = plain continue. Verify demos
are sourced from the **right-prefix high-conf band** (§2), so `prefix_correct=True`
is the *only* legitimate case and the gate is a **no-harm confirm**:

- **(a) confirm / no-harm (the only accepted verify case here):** the prefix was
  right (high `conf_target`) and the verify arm's lower-CI rate is **≥** the plain
  control AND stays high (≥ `conf_target − slack`). I.e. inserting the independent
  check did **not** break a correct answer and did not waste budget into a wrong
  one. This is what "VERIFY" means under the north-star: confirm a high-confidence
  answer, do not regress it.
- **(b) flip-from-wrong is explicitly NOT a verify demo.** A check that flips a
  wrong prefix right is causally a *redirect* (it changed the path), and admitting
  it here would (i) relabel redirect content as verify, gaming S2, and (ii) bring a
  wrong (low-conf) prefix into the verify corpus, breaking the high-conf
  calibration of §1–2. Such a candidate, if it ever arises, is routed back through
  `accept_redirect`, never accepted as verify.

This reuses `lower_ci_diff` and the `arm_rate`/`MIN_K` machinery; the "stays high"
clause is the only verify-specific addition (it stops a vacuous gate that would
pass a verify block which silently turns confident-right into wrong).

**Gold usage.** Gold enters *only* `_check_correctness` (arm grading) — it filters
*acceptance*, it is never shown to the model and never sets the confidence value
(§1). No leak.

---

## §5 Loss masking + parquet assembly + mix ratio

**Loss mask.** Each accepted demo has shape
`[prompt][wrong_prefix]<|meta|>…<|/meta|>[correct continuation]`. We MUST NOT teach
the model to produce the wrong prefix. Reuse
`src.training.segment_loss_mask.redirect_train_spans(prompt_len, prefix_len,
total_len)` → `build_segment_loss_mask(...)` to train **only** the meta block +
correct continuation; `[prompt]+[wrong_prefix]` is masked to `-100`. The module's
fail-closed contract (negative lengths → mask everything) is preserved.
**Precedence:** `teacher_kl.enabled=false` whenever this mask is used
(per the module docstring — the teacher_kl path keys off a single `prompt_len`
boundary and would re-introduce loss on the masked prefix).

**Parquet assembly.** Mirror the v8 SFT parquet format via
`scripts/build_v8_strict_paired_data.py` conventions (`META_BLOCK_RE`,
`load_messages`, `THINK_RE`, `CONF_RE`, the messages/`split_tags` schema). Each row
carries `split_tags = {difficulty, scenario∈{redirect,verify,none}, trigger,
action, conf_target}` and the precomputed `prompt_len`/`prefix_len` so the trainer
can build the segment mask without re-tokenizing the cut.

**Mix ratio.** Mix the causally-filtered redirect/verify demos with the v8
base + verify corpus so the model does not over-fire meta. Target band **15–30%**
distilled demos (the PG0-registered SFT mix band), exact ratio a swept unknown
(§9). Hard problems contribute **only the untouched v8 base/verify corpus**, and
**no freshly-distilled demo of either action** (neither redirect nor verify) is
built on a hard problem — §2 already excludes hard from both source bands, so the
mixer's hard-row invariant is `action == none` (base) only. (The legacy v8 verify
rows on hard are pre-existing decorative-risk corpus, not products of this
pipeline; they are mixed unchanged for capability coverage, not counted toward the
15–30% distilled band.)

---

## §6 The 4 code modules + interfaces + reuse

All four are **pure where possible + unit-tested**; GPU/TRAPI wiring lives in a
`main()` marked `# pragma: no cover`.

1. **`src/distill/confidence_label.py`** — student self-consistency confidence.
   - `conf_target(grades: list[int]) -> float` (= `arm_rate`);
     `in_redirect_band(pass_rate) -> bool` (PG0 low band [0.125, 0.5]);
     `in_verify_band(pass_rate) -> bool` (high band ≥ `VERIFY_BAND_LO=0.625`);
     `select_anchors(rollouts, gold, tags) -> list[Anchor]` keeping easy/medium and
     emitting BOTH a redirect anchor (in-redirect-band, on a **wrong** rollout
     prefix, `prefix_correct=False`) and a verify anchor (in-verify-band, on a
     **correct** rollout prefix, `prefix_correct=True`); the `Anchor` carries
     `action`, `prefix_correct`, and `conf_target` so the band fully determines the
     action downstream (no free re-judgement).
   - Reuses: `rewards._check_correctness`, `harvest_redirect_cf.arm_rate`,
     PG0 band constants, `pg0_yield_pilot._load_pool` schema (tags coercion).

2. **`src/distill/teacher_conditional.py`** — confidence-conditioned teacher gen.
   - `build_conditional_prompt(problem, prefix, conf_target, action) -> str`
     (the `prefix` is the wrong prefix for redirect, the correct prefix for verify);
     `select_action(prefix_correct, conf_target) -> {redirect,verify,none}` —
     band-derived: `prefix_correct=False` (redirect band) → redirect,
     `prefix_correct=True` (verify band) → verify, never a free choice;
     `parse_teacher_output(text) -> {meta, conf_emitted, action, continuation}` plus
     an assertion that `conf_emitted` is on the same side of 0.5 as `conf_target`
     (reject demos where the teacher's stated confidence contradicts the measured
     student value — guards S1 truthfulness).
   - Reuses: `prompt_behavior.BEHAVIOR_SYSTEM_PROMPT`,
     `prompt_control_v4.build_control_v4_prompt` (+ `META_START/META_END`,
     `CONF_RE`), `generator.get_trapi_client`/`generate_single_chain`,
     `gen_metacot_v2` concurrency. `main()` = TRAPI wiring.

3. **`src/distill/causal_filter.py`** — accept/reject by counterfactual.
   - `accept_verify(v_grades, nv_grades, conf_target, margin, min_k, slack) -> bool`
     (new, §4) — CONFIRM/NO-HARM only: requires `prefix_correct=True` (asserted),
     `lower_ci_diff(v, nv) >= 0` AND `arm_rate(v) >= conf_target - slack`; a verify
     candidate built on a wrong prefix is REJECTED here (routed to
     `accept_redirect`), so flip-from-wrong can never be relabeled verify.
     Thin re-export of `accept_redirect`; `filter_candidates(cands) -> kept`
     dispatching on `action`.
   - Reuses: `harvest_redirect_cf.accept_redirect`, `lower_ci_diff`, `arm_rate`,
     `MIN_K`, `ACCEPT_MARGIN`, `_check_correctness`. `main()` = vLLM arm regen.

4. **`src/distill/assemble_sft.py`** — mask + parquet + mix.
   - `build_row(anchor, teacher_out, prompt_len, prefix_len) -> dict`;
     `loss_mask_for_row(row) -> list[int]`; `mix(distill_rows, base_rows, ratio)
     -> DataFrame` enforcing the 15–30% band and hard→base-only invariant.
   - Reuses: `segment_loss_mask.redirect_train_spans`/`build_segment_loss_mask`,
     `build_v8_strict_paired_data` (`META_BLOCK_RE`, `load_messages`, schema).
     `main()` = parquet I/O.

---

## §7 GPU-free smoke strategy

All tests run on CPU with `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python
-m pytest`. **No real network / GPU.**
- **Mock the TRAPI teacher.** Patch `get_trapi_client` to a fake client whose
  `responses.create` returns canned `<|meta|>… confidence: 0.x …<|/meta|>` +
  continuation strings (well-formed, malformed, no-meta) — exercises
  `parse_teacher_output`, `select_action`, retry/backoff.
- **Mock the vLLM rollout.** Feed pre-baked grade lists / arm texts directly into
  the pure functions (`conf_target`, `select_anchors`, `accept_redirect`,
  `accept_verify`, `filter_candidates`) — these are already grade-list-in,
  bool-out (the PG0/CF pattern), so no LLM is needed for the decision logic.
- **Assembly smoke** runs on a tiny synthetic in-memory frame: assert the mask
  zeros the prefix, the mix ratio lands in [0.15, 0.30], no hard row carries a
  freshly-distilled redirect OR verify action (hard distilled rows = 0), every
  verify row has `prefix_correct=True` and a high `conf_target` while every redirect
  row has `prefix_correct=False` and a low `conf_target`, and a verify candidate
  built on a wrong prefix is rejected by `accept_verify` (so flip-from-wrong is
  never relabeled verify).
- Every `main()` stays `# pragma: no cover`; the pure logic is fully covered.
  **Tests MUST pass.**

---

## §8 Experiment + intent metrics (→ Metrics phase)

1. **Build:** N=8 student rollouts on the easy/medium RL-train pool → `conf_target`
   → split into the low-conf redirect band (wrong-prefix anchors) and the high-conf
   verify band (right-prefix anchors) (§1–2) → conditional teacher gen (§3) →
   causal filter (redirect flip gate + verify no-harm-confirm gate, §4) →
   assemble + mix 15–30% (§5).
2. **SFT:** from `v8_meta_inside_strict_sft`, segment-masked, `teacher_kl=off`.
3. **Eval (forward ref to Metrics phase):** `scripts/eval_vllm_1030.py` 1030-set
   (GSM8K 500 + MATH 500 + AIME 30), max_tokens=4096.
   - **S1:** binned emitted-confidence vs pass-rate monotonicity + ECE vs v8.
   - **S2:** held-out counterfactual replay (R−B' redirect gap > 0 on low-conf-wrong;
     verify no-harm-confirm on high-conf-right — V lower-CI ≥ plain control AND stays
     high) via `raw_yield_stats`/`gap_by_attribute`.
   - **S3:** meta-ON acc ≥ 0.786 and ≥ 0.651; then **SFT→RL** and check the
     UTILITY-conditioned gate — counterfactual meta-block-ablation lower-CI > 0 on
     fired problems (accuracy is *caused* by the firings) — with meta-emit rate
     reported only to confirm it sits between the under-forming (~0) and
     over-forming (fires-on-everything) extremes, never used as the pass criterion.

---

## §9 Honest risks

- **Teacher capability gap.** GPT-5.4 may recover prefixes the student never could,
  producing demos that are correct but unreachable → wasted/OOD even after the
  easy/medium + on-prefix anchoring. Mitigation: anchor strictly on student
  prefixes (§2); the §4 causal gate still requires the *student-distribution* arm
  regen to flip, not the teacher's own continuation.
- **Confidently-wrong bucket rarity.** The most valuable redirect case
  (high-emitted-conf but wrong path) may be rare in the band; if too few survive
  the filter we fall back toward the verify-heavy / low-conf-redirect majority and
  S2 weakens for the confidently-wrong sub-claim. We record bucket counts; a thin
  bucket is reported, not papered over.
- **Mix-ratio unknown.** 15–30% is the registered band but the *optimal* ratio is
  unmeasured; too much distilled data risks over-firing meta (the v8 decorative
  failure mode), too little fails to form the behavior (v1 collapse). Ratio is a
  swept hyperparameter in the experiment, not a fixed constant.
