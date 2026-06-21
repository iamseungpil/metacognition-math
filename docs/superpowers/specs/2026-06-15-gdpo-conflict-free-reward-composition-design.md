# Conflict-Free GDPO Reward Composition — design & experiment plan

**Date**: 2026-06-15  **Author**: ctsd-phase-c autonomous loop  **Branch**: ctsd-phase-c
**Status**: APPROVED (user 2026-06-15: scope **A, PCGrad excluded**; ③ **Anchor-on-R_corr+EMA**;
② **R_emit first-token routing**; ① length containment with **medium** "open-meta-then-truncation"
penalty). Pending user review of this spec before writing-plans.

This spec defines the **stage-C reward composition** used by the redirect pipeline
([2026-06-14-redirect-priming-from-failed-rollouts-design.md](2026-06-14-redirect-priming-from-failed-rollouts-design.md));
it replaces that doc's §8 length-containment sketch with a concrete composition design.

---

## 1. North-star (unchanged)

Reinforce **useful** metacognition to raise **accuracy**; metacognition is a means, calibration a
sub-goal. This spec is narrowly about **how multiple reward heads are composed so they do not fight
each other** under GDPO (region-routed Dr.GRPO advantage). It is a precondition for stage-C RL: we
want to give correctness / PMI-likelihood / format / emission / length / redirect signals *together,
strongly*, without one dominating or one structurally killing another (the s2b failure).

---

## 2. Problem — disjoint routing exists, yet s2b's heads fought

The current GDPO compose (`src/training/dcpo_region.py::compose_dcpo_region_advantage`) already
centers each head independently (Dr.GRPO, mean-subtract, no /std) and routes each onto a **disjoint**
token region (ANSWER / META_CONTENT / CONF / FORMAT). So same-token additive conflict cannot happen
by construction. Yet s2b collapsed (2026-06-14 postmortem: length inflation → format/emission
collapse → PMI-coverage ratchet). Reading the compose code, disjoint routing leaves **three
conflicts unaddressed**:

1. **① Length collapses the mask *structure*, not the reward axis (the real killer).** When length
   inflates and truncates at `max_response_length`, `classify_dcpo_format` labels rows
   truncation/discard and `build_dcpo_region_masks` empties META_CONTENT / gates R_meta. Length does
   not fight R_format at a token — it *deletes the region R_format and PMI would route onto*. This is
   a structural conflict upstream of composition; fixing the composition cannot reach it.
2. **② R_emit is the only non-disjoint head.** Lines 1156-1158: R_emit is added as
   `advantages + w_emit * A_emit * rm` — broadcast per-token over the **entire** response mask, so it
   sits on ANSWER and META tokens alongside the other heads. Emission pressure thus mixes directly
   with correctness and quality (s2 boilerplate; s2b silence both originate here).
3. **③ Centering subtracts mean but not std, so the weak head (PMI) is buried.** Dr.GRPO omits /std
   on purpose (difficulty-bias avoidance). Consequence: per-head advantage scales differ wildly —
   R_corr is ±1-scale, PMI Δ'~0.05. Even with w_meta 0.8 the quality signal (~0.04) is dwarfed by
   floor/emit on the same meta tokens. Disjoint routing does not equalize magnitude.

The approved fix (scope A) addresses all three: **structure (①) + composition (②③)**, excluding
gradient surgery (region disjointness already removes token-level direction conflict; PCGrad's
per-head backward is too invasive for the benefit — YAGNI).

---

## 3. Design

### 3.1 Anchor-on-R_corr scale normalization (③)

`A_corr` stays exactly as today — Dr.GRPO centered, **not** normalized (preserves the main signal
and its difficulty-bias-avoidance property). The **auxiliary heads** (meta, cal, format, emit, and
the optional SCoRe bonus) are, after their own Dr.GRPO centering, **rescaled to the R_corr scale**
before applying `w_head`:

```
A_head_norm = A_head * (ema_mean_abs_corr / max(ema_mean_abs_head, floor))
contribution = w_head * A_head_norm * region_mask
```

where `ema_mean_abs_*` is an EMA-tracked running `mean(|centered advantage|)` per head. Effect:
`w_meta=0.8` now means "0.8× the correctness signal's strength," interpretable and stable as PMI's
raw scale drifts. The `floor` on the denominator prevents noise amplification when a head's spread is
near zero (PMI can be sparse) — robust normalization, not naive division.

**Warmup.** Normalization is OFF for the first N steps (EMA not yet stable); during warmup the heads
use raw `w_head` (current behavior). After warmup, anchor scaling engages.

### 3.2 R_emit first-token routing (②)

Replace the global `* rm` broadcast with routing onto the **response's first token(s)** — the point
where the emit/abstain decision is made. Both silent and emitting rows have a first token, so the
centered R_emit (silence = negative) still penalizes silence, while ANSWER and META tokens are no
longer touched (disjoint restored). Boilerplate is NOT prevented here — it is prevented by §3.1
keeping PMI above emit in magnitude, so "open meta → small +; content graded strongly by PMI →
only useful meta survives."

### 3.3 Length containment for structure preservation (①)

- `dcpo_len_cost` **0.06–0.10** (from 0.03), warmed up with / ahead of w_meta (already scaled by
  `dcpo_w_meta_scale`; the base was just too small). `max_response_length` stays **4096** (redirect
  needs the room — shrinking it kills redirect on hard problems).
- **Meta-block token-length cap** (`dcpo_meta_len_cap`): floor and PMI must not pay for sheer meta
  length. Caps the meta span that earns floor/PMI so "write a longer meta to farm reward" is removed.
- **Medium "open-meta-then-truncation" penalty** (`dcpo_trunc_open_penalty`, user: medium). Today
  truncation is treated as a blameless length problem (R_format 0, R_meta gated) — in s2b this became
  the abstention escape (open a meta, run out of budget, never close, pay nothing vs −0.2 for a
  malformed close). We add a **medium** penalty applied **only** to rows that *opened* a meta and
  then truncated before closing — NOT to legitimately-hard problems whose answer simply ran long
  without a meta. This shuts the escape without punishing genuine length needs.

### 3.4 redirect / SCoRe (relation to the pipeline)

redirect is graded as a kind of meta by the **PMI head** (no separate reward axis). The optional
**SCoRe wrong→right bonus** (a rollout that transitions incorrect→correct within one response) is a
small **auxiliary head** — and because it is anchor-normalized (§3.1), it composes with the others
without fighting them.

### 3.5 EMA state

A handful of scalars (per-head running `mean|advantage|`) updated each step. Tiny, no checkpoint
schema change beyond optional persistence; if not persisted, re-warms on resume (a few steps).

### 3.6 Data flow

rollout → region masks (unchanged) → head scalars (unchanged) → **[new] EMA update + anchor
rescale of auxiliaries** → **[new] R_emit first-token routing** → weighted sum → (unchanged floor /
gate / discard-exclusion). Length knobs act in the existing populator block (len_cost on R_corr
scalar; cap/trunc-penalty at mask/format classification).

---

## 4. Interfaces (Karpathy minimal-change; default-off = byte-identical)

`compose_dcpo_region_advantage` gains, all defaulting to the current behavior:

| Knob | Default (= current) | New behavior |
|---|---|---|
| `dcpo_anchor_norm` | `false` | enable §3.1 anchor scaling of auxiliary heads |
| `dcpo_anchor_ema` | `0.9` | EMA decay for `mean\|advantage\|` tracking |
| `dcpo_anchor_warmup_steps` | `0` | steps before anchor scaling engages |
| `dcpo_emit_route` | `"global"` | `"first_token"` → §3.2 routing |
| `dcpo_emit_first_n` | `1` | how many leading tokens carry the emit signal |
| `dcpo_meta_len_cap` | `0` (off) | cap meta tokens eligible for floor/PMI |
| `dcpo_trunc_open_penalty` | `0.0` (off) | medium penalty for open-meta-then-truncation |
| `dcpo_len_cost` | (existing) | value raised to 0.06–0.10 in the stage-C config |

Every new knob off ⇒ the compose output is **byte-identical** to today (regression-locked). Each
piece is independently testable: anchor scaling is a pure tensor transform; emit routing is a mask
swap; length knobs live in the populator/classifier.

---

## 5. Hypotheses (falsifiable)

- **HC1 (anchor keeps PMI alive).** With anchor on, the logged effective scale ratio
  `eff_scale_meta / eff_scale_corr` tracks `w_meta` within ±20% across training (PMI no longer
  buried). *Falsified if* PMI effective scale decays toward 0 despite anchor on.
- **HC2 (emit routing is clean).** First-token routing leaves ANSWER-token advantage unchanged
  (unit test) and emission does NOT collapse (no s2b silence ratchet); general accuracy unregressed.
  *Falsified if* emission collapses or answer learning regresses.
- **HC3 (length contained, structure preserved).** `response_length/mean` < 1500, `clip_ratio` <
  0.20, and `wellformed_rate` does NOT ratchet (stays > 0.4 past warmup) — the s2b collapse does not
  recur. *Falsified if* length inflates or wellformed collapses again.
- **HC4 (heads don't fight).** With correctness/PMI/format/emit/length all on at meaningful weights,
  `acc_with ≥ acc_without` holds through training and no single head's diagnostic dominates/cancels
  another. *Falsified if* a boilerplate (s2) or silence (s2b) mode reappears.

## 6. Metrics

New: `dcpo/eff_scale_{corr,meta,cal,format,emit}` (post-anchor effective scales),
`dcpo/emit_route_answer_leak` (should be 0), `dcpo/trunc_open_rate` (open-meta-then-truncation rate).
Existing watch (from s2b): `wellformed_rate`, `pmi_member_rate`, `meta_emit_rate`,
`response_length/mean` (<1500), `clip_ratio` (<0.20), `acc_with` vs `acc_without`,
`gdpo/correctness/mean`.

## 7. Tests (Keep/Discard criteria)

1. **Anchor scaling** — two synthetic heads with 10× different raw scales compose at the
   weight-implied ratio after warmup; degenerate (zero-spread) head hits the denominator floor, no
   NaN/inf.
2. **Emit first-token** — silent row receives negative emit advantage on token 0; ANSWER/META token
   advantages are bit-identical to the no-emit baseline.
3. **Length** — `len_cost` subtracts the right amount from the R_corr scalar; `meta_len_cap` zeroes
   floor/PMI beyond the cap; `trunc_open_penalty` fires only on opened-then-truncated rows, not on
   meta-less long answers.
4. **Regression (Karpathy lock)** — all new knobs at defaults ⇒ compose output byte-identical to the
   pre-change function across the existing dcpo_region test suite.

## 8. Decision tree

- HC1 fails (PMI still buried) → denominator floor too high or EMA too slow; lower floor / raise EMA
  responsiveness.
- HC2 fails (emission collapses) → first-token signal too weak; widen `dcpo_emit_first_n` or raise
  `w_emit` (anchor keeps it from dominating).
- HC3 fails (length still inflates) → raise `dcpo_len_cost` toward 0.10 and/or tighten
  `dcpo_meta_len_cap` / `dcpo_trunc_open_penalty`.
- HC4 fails (a fight reappears) → inspect which head's effective scale spiked; the anchor logs make
  the culprit visible (this is the whole point of §3.1's observability).

## 9. Operational

- **Implementation order**: this composition is built + unit-smoked FIRST (it is reused by stage-C),
  then folded into the redirect pipeline's stage-C config. Built via ultracode (karpathy-guidelines
  surgical change + autoresearch Verify/Keep-Discard on the unit tests) after writing-plans.
- **Config**: stage-C config sets `dcpo_anchor_norm=true`, `dcpo_emit_route=first_token`,
  `dcpo_len_cost=0.06–0.10`, `dcpo_meta_len_cap` + `dcpo_trunc_open_penalty` (medium), on top of the
  PMI recipe.
- **wandb**: `metacot-dcpo-v4` project; new effective-scale metrics on the dashboard.
