# Redirect-Priming v2 (Counterfactual) — design & experiment plan

**Date**: 2026-06-18  **Author**: ctsd-phase-c autonomous loop  **Branch**: ctsd-phase-c
**Status**: REV-5 (ultracode rounds: 25→30→18→18 confirmed; criticals 4→3→3→2, converging).
**REV-5 keystone (round-4 single-most-important)**: the causal estimand is **R − B′** =
"redirect *content* given a matched second attempt", NOT "meta-on vs frozen continuation". This
neutralizes the two round-4 criticals at once: the second-attempt/self-consistency confound (C-1)
and the off-policy degenerate-arm-B artifact (C-2).
**Two keystones**: (1) redirect emitted as a discrete `<|switch|>` token; (2) because banning a
token does NOT ban the *behavior* (the model routes around it in plain prose — review C1), the
causal measure also uses a **redirect-BEHAVIOR detector**, and feasibility is decided by cheap
**empirical PRE-GATES** before any GPU training. If a pre-gate fails, the experiment is invalid
and we STOP — better than spending GPU-hours on an unmeasurable claim.

---

## 0. HARD PRE-GATES (cheap smokes; ALL must pass before harvest/RL; any fail → STOP)

- **PG0 — yield pilot (cheapest first, round-5 I-7).** On a tiny harvest pilot, run the
  expected-yield calc (emission × in-band frac × accept-prob × pool) against a **pre-registered
  numeric target accepted-redirect count** for a 15–30% SFT mix, under a hard GPU-hour ceiling.
  Low projected yield → STOP/redesign in minutes, before the big harvest.
- **PG1 — redirect *behavior* separability, TWO-SIDED (load-bearing; review C1 + round-5 I-4).**
  On ≥20 labeled traces under the TRUE `-inf` `<|switch|>` mask: (a) redirect **BEHAVIOR rate**
  (via §5 detector, **measured recall ≥ ~0.8 reported as a number** — pre-registered numeric
  false-negative bound) drops far below arm-A; (b) verify/confidence meta keeps emitting at arm-A
  rate; **(c) arm-B per-arm degeneracy (repetition / length-floor / %-no-answer) stays ≈ arm-R**
  (else hard-masking just breaks fluency, not behavior — INSUFFICIENT). Fail any → **STOP** (invalid
  before any GPU-hour).
- **PG2 — CF leak-guard viability.** Primed-SFT checkpoint: v3m leak-guard ungrade < 50% AND
  ≥10 graded nonzero-R_meta rows/batch. (v3l died at ~3/4 ungrade.)
- **PG3 — verification primitives (pure-python; scipy/statsmodels absent).** Unit tests pass:
  saved>broke-but-not-significant → REJECT; placebo-tie → INSUFFICIENT; parse-fail row → DROP;
  hard-mask `emitted_switch_without == 0`.
- **PG4 — SFT-init no-degradation.** `v8_redirect_primed_sft` meta-off 1030 ≥ v8-init meta-off.
- **PG5 — micro-RL collapse probe (review I2, I7).** 20–30 steps of Stage-2 (CF ON) on a tiny
  batch with I1 (BAND-emission, round-4 I-8) and C2 (acc_without) tripwires LIVE. **The probe MUST
  reach FULL `len_cost` weight inside its window** (set probe warmup≈0 OR run ≥ warmup+30) — else
  it runs entirely inside the warmup ramp where the documented collapse engine (full len_cost) is
  ~0 and PG5 passes trivially while the real run collapses at step~60. Report band-vs-offband
  emission. Answers the central collapse bet for ~1% of a full run.
- **PG6 — RL-headroom (review I5).** Measure `acc_with − acc_without` on `v8_redirect_primed_sft`
  (NO RL) on the frozen band. If the primed gap already ≥ the target final gap, RL adds nothing →
  STOP/redesign (harder band) before any Stage-C spend.

---

## 1. North-star + success (review C3, I1, I5)

Reinforce **useful** metacognition (redirect) to raise **accuracy**. Success requires ALL of
(tightened per round-4 I-3 so the gates cannot be passed at the meta-dead floor via a
suppressed-arm artifact):
- **(S1) absolute floor with margin**: final primed+RL **meta-ON** overall 1030 ≥ **v1 0.651 + δ**
  (pre-registered margin) AND **`acc_with` RISES in absolute terms over primed-SFT `acc_with`**.
  Mid-RL S1 tripwire (round-4 M-4): STOP if meta-ON overall < 0.651−δ early, or truncation > base.
- **(S2) net-positive meta, won the RIGHT way**: `acc_with_overall > acc_without_overall` on the
  final model AND the gap **improves over primed-SFT pre-RL gap**, AND **`acc_without` is
  NON-DECREASING vs primed-SFT** — **REJECT if > 50% of gap improvement comes from `acc_without`
  falling** (gap-gaming). Report the Δacc_with / Δacc_without decomposition.
- **(S3) causal redirect**: significant **R − B′** recovery on the failing distribution (§5).
Mechanism-first = a GATE, not a terminus. Beating e4 0.786 is later.

## 2. Problem (unchanged): A = meta net-harmful & pruned (acc_with 0.71<0.81; PMI≈0; verify_exec
−0.66; wf collapse). B = regime < baseline on hard, meta-independent (v1 0.651 vs e4 0.786) —
out of scope, isolated by pre+post-RL meta-off guards.

## 3. Reward (NEW code on the dormant V4 cf branch; review C2, I6, I7, M4)

**★REV-6 round-5 C-1 — the −inf hard mask + behavior detector apply to the TRAINING `c_without`
producer, not just eval.** The whole reward rests on `c_without` (`verl_sdc.py:590,:3358`); today
it is produced with the soft `-100.0` the model routes around (the exact leak that made PMI≈0.04)
plus `signature_suppression_ids` that perturb answer content. **Fix: Stage-C `c_without` must use
the NEW `-inf` `<|switch|>` LogitsProcessor (§7) and be graded by the §5 behavior detector +
degeneracy health gate (round-5 I-2), with a RUNTIME tripwire: halt if the `c_without`-arm
redirect-BEHAVIOR rate exceeds PG1's numeric bound during training.** Same rigor in harvest arms
(round-5 I-1). I1 collapse guard adds a behavior-rate live tripwire, not token-count only
(round-5 I-3); pre-register concrete continuous rmeta_pos/neg thresholds (today >0.5/<−0.5).

Counterfactual `acc_with−acc_without` is the only correctness-aligned option but historically
sparse/slow/fragile; PMI is correctness-agnostic/length-inflating. Priming addresses sparsity
ONLY (leak-rate = PG2). Wiring (all NEW; live cf branch is a no-op, R_meta∈{−1,0,+1}, k=1):
- **k≥16 suppressed CF draws aggregated to continuous** `c_without ∈ [0,1]` (mean); R_meta
  continuous → **redefine rmeta_pos/neg thresholds + slope tests for the continuous regime**
  (review I7; today thresholds are >0.5/<−0.5). Producer+consumer+logging change.
- Positive R_meta only on **frozen hard-band `<|switch|>` rows**; **negative term**
  `R_meta −= λ·1[emit switch AND c_without correct]`, **λ small & subordinate to recovery**;
  pre-register it must NOT reduce emission-induced recovery (M4).
- **CF compute budget (I6)**: k≥16 ≈ large gen multiplier — **fix padding to active rows only**
  (today pads to full B, `verl_sdc.py:3322`); state per-step gen multiplier + wall-clock gate.
- Subset-score `f` s.t. `f·emit·B ≥ 10`; PMI weight 0 (diagnostic).
- `len_cost` warmup RAMP (full step-0 = documented collapse engine). Stage-1 format-hold (no CF,
  ~50 steps) → Stage-2 CF.
- **LIVE RL tripwires (counts w/ binomial CIs, every 10 steps)**:
  - **I1 PRIMARY collapse guard** = absolute `count(<|switch|> rows)/B` below floor (< ⅓ of
    post-prime step-0 rate) over any 10-step window → halt. `rmeta_pos_rate` SECONDARY, always
    logged with denominator n.
  - **C2 gap-gaming guard** = halt if `acc_without` (frozen hard band, binomial CI) FALLS while
    `rmeta_pos` rises (model degrading the suppressed arm instead of learning recovery). Also
    step<30 halt if `rmeta_pos>0` but `acc_with−acc_without ≤ 0`.

## 4. Approach — 3 stages

### Stage A — Harvest (`scripts/harvest_redirect_cf.py` NEW; gen base `run_online_sdpo_regen.py`)
Gates: (1) FROZEN source pass-rate band [0.125,0.5] measured once; (2) failed pool;
(3) splice 30–70%, drop prefix<50tok/prefix-solves; (4) well-formed = single meta block with
`<|switch|>` + conf<0.5; (5) **prefix-forced 3-arm, k≥16 (review I3 yield)**: R=`<|switch|>` tail,
N′=null-meta tail, Nc=compute-matched non-meta tail; accept by a **single pre-registered statistic**
(lower-CI-bound on R−max(N′,Nc) effect ≥ margin) — not the over-constrained AND-of-three that gave
~11–48% accept; fresh held-out re-confirm; **expected-yield calc (emission×band×accept×pool) with
STOP gate if projected accepted < needed for 15–30% mix**; (6) ≤2/problem; (7) **record source
ids — eval disjoint (I9)**.

### Stage B — Prime (`build_meta_template_sft.py` ext → `v8_redirect_primed_sft`; review M5)
`<|switch|>` is a **full vocab/embedding/data chain** (NEW sub-tasks, all before PG4): add token
to tokenizer + assign id; `resize_token_embeddings` + verify lm_head tie; init embedding from
`decision/switch` think-tokens; regenerate primed SFT parquet with the token; confirm RL configs +
frozen-ref + PMI base share the resized vocab.
- **Segment loss-mask (I7, M5)**: masking lives in `tokenize_row` via a single `prompt_len`
  boundary (`sft.py:103-106`) and a separate teacher_kl span path (`sft.py:116-139`). Extend with
  **token-id-index segment boundaries (chat-template-robust, not char offsets)**: mask
  `[prompt]+[wrong prefix]`, train on `[meta…<|switch|>…]+[correct continuation]`; **state
  teacher_kl precedence** (define whether teacher_kl is on for priming SFT).
- Mixing 15–30% redirect + rest normal. Light SFT (low LR, 1–2 ep).

### Stage C — RL (`verl_sdc` DCPO v4, `sdc_counterfactual=true`): reward §3; gate Stage-2 entry on
PG2 + PG5.

## 5. Verification (review C1, C2, I2, I4, M1, M3)

**Load-bearing = R − B′ on the FAILING distribution** (held-out, disjoint from harvest, frozen
source-band stratified). The estimand is redirect *content given a matched second attempt*:
1. **Attempt-matched arms (round-4 C-1/C-2)**. **Arm R** = `<|switch|>` redirect allowed.
   **Arm B′** = prefix-forced continuation that MAY restart/backtrack in **plain prose**, with ONLY
   the `<|switch|>` id masked via a **TRUE −inf `logits_processor`** (round-4 I-2: live code uses
   soft −100 which a primed model overrides; reuse the `MetaCloseLogitsProcessor` wiring path).
   B′ gives the same second-attempt affordance, so **R − B′ isolates redirect CONTENT**, not the
   mere opportunity to resample. Report the effect as R − B′ across a **logit-bias delta sweep
   (−2,−5,−20,−inf)**: a true causal effect is monotone & saturating; a decoding-break artifact
   grows with bias → pre-register non-monotone/blow-up ⇒ **INSUFFICIENT** (round-4 C-2).
2. **Per-arm degeneracy health gate (C-2)**. Before a row counts: repetition rate, output-length
   floor, %-no-final-answer; require arm-B′ degeneracy ≈ arm-R (else the gap is an off-policy
   artifact, not redirect). `_check_correctness` garbled→False must be FLAGGED, not scored as a
   "meta saved it" win.
3. **Behavior detector with MEASURED recall (round-4 M-3)**. Detect redirect behavior in both arms
   via an **LLM-judge (primary) + regex pre-filter**; PG1 hand-labels ~30 known redirects to
   **report detector recall as a number**; recall < ~0.8 ⇒ behavior measure invalid ⇒ INSUFFICIENT.
   PG3 asserts zero `<|switch|>` TOKEN IDs in `response_ids` (not text).
2. **Placebo P** = ban a matched **non-redirect meta** token (N′ block) + a compute-matched
   non-meta control; require `acc_A−acc_B` exceed `acc_A−acc_P` and the non-meta control (M-8).
3. **Asserts (I4)**: `emitted_switch_without==0` AND `emitted_verify/conf_without ≈ arm-A rate`
   (record `emitted_switch_*` separately from `emitted_meta_*`).
4. **Truncation control (M3)**: per-arm truncation rate; Δ valid only if truncation_A≈truncation_B,
   else stratify on "answer within budget" or give arm B a length handicap = mean meta-block length.
5. **Parse gate + pure-python McNemar exact binomial** (`min(b,c)~Binom(b+c,½)`); pre-registered
   min discordant pairs (else "underpowered, not negative"); success = significant.
6. **No-regression floor = SAME-REGIME** `v8_redirect_primed_sft` meta-off, frozen band (M6).
7. Plus S1/S2 gates (§1).

**H1a survival**: I1 absolute `<|switch|>`-row guard PRIMARY; rmeta_pos SECONDARY w/ denominator.
**Guards**: PG4 pre-RL + **post-RL meta-off ≥ pre-RL** (B-into-A contamination check).

## 6. Risks: every prior risk now has a PRE-GATE (§0) or LIVE tripwire (§3). Biggest residual =
PG1 (is redirect even separable as behavior?) — resolved empirically before GPU spend.

## 7. Code: reuse vs NEW (honest)
- **Reuse**: `prompt_behavior.py`; `run_online_sdpo_regen.py`; `_check_correctness`;
  `validate_v8_strict_data.py`; `cf_prefix_agent.py` prefix-forcing primitive (RL-bound, k=1);
  v3m leak-guard.
- **NEW (the bulk)**: a **configurable hard-ban `LogitsProcessor`** that sets `<|switch|>` logit to
  `-inf` (round-5 I-6: `MetaCloseLogitsProcessor` is hardcoded to meta-open/close ids, NOT a
  configurable banner, and vLLM `logit_bias` is added/clamped ≠ −inf — verify vLLM honors it),
  wired into BOTH the eval AND the Stage-C `c_without` producer; k=4–8 (not 16; wider CIs, stated
  per-step wall-clock budget) continuous `c_without` + `cf_correct: float→list[float]`;
  `<|switch|>` vocab/embed/data chain; `tokenize_row` segment-mask + teacher_kl
  precedence; `harvest_redirect_cf.py` (3-arm prefix-forced, k≥16, CI-stat, holdout, yield-calc);
  the §3 reward wiring in the dormant cf branch (k≥16 continuous, negative term, band gate, subset
  floor, padding fix, tripwires) + `cf_correct: float→list[float]` + V4+cf integration test;
  redirect-behavior detector; eval rewrite (switch-ban, prefix-force, placebo, behavior-detector,
  leak+parse+truncation gates, pure-python McNemar) + verification unit tests (PG3); configs.

## 8. Review convergence (rounds wjluhaolr/wb36l276p/wydbphtdv)
REV-4 resolves the round-3 criticals: C1 → behavior detector + PG1 (behavior-rate, post-RL audit,
INSUFFICIENT path); C2 → live acc_without gap-gaming tripwire; C3 → S1 hard floor ≥ v1 0.651.
Importants I1–I7 + minors M1–M5 folded into §3/§4/§5. **The residual unknowns are now empirical
PRE-GATES (§0) — by design they are answered in the smoke phase (PG1–PG5), not by more spec text.
PG1 in particular can declare the experiment invalid before any GPU spend, which is the correct
fail-fast.** Remaining wording-level items deferred to writing-plans.
