# RL Data Regime Redesign — design & experiment plan

**Date**: 2026-06-15  **Author**: ctsd-phase-c autonomous loop  **Branch**: ctsd-phase-c
**Status**: APPROVED direction (user 2026-06-15): broaden to general metacognition at medium
difficulty; static mixed-build + dynamic sampling + rollout n↑ all three; n=8; data-change-first —
run the full data+composition loop (s3) before further changes. Pending user review of this spec.

Companion to the composition spec
([2026-06-15-gdpo-conflict-free-reward-composition-design.md](2026-06-15-gdpo-conflict-free-reward-composition-design.md))
and the redirect-priming spec
([2026-06-14-redirect-priming-from-failed-rollouts-design.md](2026-06-14-redirect-priming-from-failed-rollouts-design.md)).
The composition fixes the *reward-side* collapse (length ratchet); this spec fixes the *data-side*
collapse (GRPO signal loss). s3 = data redesign + composition together; redirect priming is layered on
**after** s3 validates.

---

## 1. North-star (unchanged)

Strengthen confidence self-introspection + metacognition (check assumptions when stuck, verify before
answering, switch approach when uncertain) to **raise reasoning accuracy**. Calibration is a sub-goal.
The reward stack (R_corr / PMI R_meta / R_cal / R_format / R_emit / floor + composition) is unchanged;
only the **training data** changes.

## 2. Problem — the data caused two of the three collapse engines

The s2b collapse had three compounding causes. The composition spec fixes engine (1). This spec fixes
(2) and (3), which the composition cannot touch:

1. **Length ratchet** (reward-side) — fixed by the composition spec.
2. **GRPO signal loss** — the current RL subset is easy-0% / medium-62% / hard-38%, redirect-scenario
   100%, MATH 80% + omni-math 20% (omni-math entirely hard/olympiad), run at **rollout n=4** (verl_e4
   base default). On hard problems a small group is often all-wrong → group reward variance 0 →
   advantage 0 → **no gradient to strengthen the meta channel** (GRPO variance ∝ p(1−p); p→0 kills it).
   DAPO/Online-Difficulty-Filtering: middle difficulty (~20–80% solve rate) maximizes signal.

   **Key finding (data already exists):** the source corpus `data/v8_meta_inside_think.parquet` (6,329
   rows) ALREADY contains the mix we want — difficulty easy 2051 / medium 2750 / hard 1528; scenario
   redirect 3261 / verify 3068; source gsm8k 2051 / omni-math 820 / hendrycks MATH (multiple subjects).
   The collapse-prone `verl_train_redirect` was produced by `build_v8_redirect_subset` filtering this
   down to `scenario==redirect ∧ difficulty∈{medium,hard}`. So broadening is a **filter widening**, not
   a new data-generation job.
3. **Grading fragility** — ~26% of omni-math gold answers are prose/non-numeric, so rule-based grading
   scores correct answers as 0 → another advantage-zero source.

Targeting redirect (a hard-problem behavior) by selecting only hard problems was self-defeating: it
killed the very signal that would reinforce redirect. The self-distillation-degrade paper (zip,
arXiv:2603.24472) predicts exactly this — narrow task coverage + uncertainty suppression degrades hard/
OOD reasoning — and selects analysis data at solve-rate 0.125–0.5. Gandhi 2025 mixes 3/4-number
Countdown 50:50 and primes behavior. DAPO uses dynamic sampling + integer-gold + clip-higher.

## 3. Design — broaden to general metacognition at middle difficulty

### 3.1 Source mix (restore easy, multi-source)
The corpus already mixes gsm8k (easy–medium) + hendrycks MATH (medium–hard) + omni-math (hard); the old
subset dropped gsm8k/easy entirely. Widening the filter restores them — no new generation. Keep the
corpus's natural source proportions (gsm8k ~32% / MATH ~55% / omni-math ~13% of the full 6,329), which
is already medium-skewed; optionally cap omni-math hard share. No fixed target ratio is forced.

### 3.2 Scenario mix (general metacognition, not redirect-only)
Replace redirect-100% with **redirect + verify** (the only two `scenario` values; redirect 3261 /
verify 3068). **Confidence is NOT a separate scenario** — it is carried inside both via the
`has_conf_drop` / `has_overconfidence` fields, so a redirect+verify mix already exercises confidence
self-introspection. Keep the corpus's natural ~50/50 redirect/verify split (tunable). Redirect also
gets later priming support (redirect-priming spec).

### 3.3 Difficulty: coarse static + precise dynamic
Static build keeps a coarse difficulty label mix (easy slice + medium-majority + hard slice) — NO
expensive per-problem solve-rate pre-measurement. Precise signal is secured by **dynamic sampling at
train time** (§3.4). Rationale: dynamic sampling makes solve-rate filtering redundant and adapts to the
live policy.

### 3.4 Dynamic sampling (the single highest-impact fix)
Drop prompts whose group is all-correct or all-wrong (advantage 0) from the training batch and
oversample until the batch is filled with mixed (0 < #correct < G) prompts (DAPO). Verify whether verl
exposes this (`algorithm.filter_groups` / dynamic-sampling config); if present, enable it; if not,
implement the DAPO filter+oversample in the verl_sdc rollout/advantage path. (Resolved in the plan.)

### 3.5 Rollout n: 4 → 8
Larger group ⇒ mixed (partial-correct) groups occur far more often ⇒ dynamic sampling discards less and
real gradient is frequent. n=8 matches DAPO/Gandhi/RLT group sizes.

### 3.6 Grading robustness
At build time, filter omni-math rows whose gold is prose/non-numeric (or integerize per DAPO) so
rule-based grading never scores a correct answer 0. Keep only `\boxed`-extractable / numeric gold; log
how many rows are dropped.

### 3.7 Hyperparameters
clip-higher (ε_low 0.2 / ε_high 0.28) for entropy preservation; len_cost 0.08 (composition spec);
weak KL kept (SFT-lineage init — full KL removal is for base-model RL, risky here); conservative lr;
max_response_length stays 4096. composition knobs (anchor/emit-route/meta_len_cap/trunc_open_penalty)
all ON.

## 4. Interfaces (Karpathy minimal-change)

| Need | Reuse / modify | New |
|---|---|---|
| Source + scenario + difficulty mix | `src/training/verl_gdpo_data.py::build_v8_redirect_subset` (currently filters scenario==redirect ∧ difficulty∈{medium,hard}) | generalize to a parameterized `build_v8_meta_subset(scenarios, difficulties, source_mix, grading_filter)` → new parquet `verl_train_meta_mix.parquet` |
| Grading filter | `_check_correctness` / gold parsing in rewards.py (reuse to test extractability) | build-time prose-gold drop |
| Dynamic sampling | verl `algorithm.filter_groups` if present, else DAPO filter+oversample in verl_sdc | enable or implement |
| rollout n | config `actor_rollout_ref.rollout.n` | set 8 |
| clip-higher | verl clip config | set 0.2 / 0.28 |
| composition | s3 config inherits s2c knobs | — |

Each unit is independently testable: the data builder (assert distribution of the output parquet), the
grading filter (assert dropped rows are non-extractable), dynamic sampling (assert all-0/all-1 groups
contribute no gradient / are refilled).

## 5. Hypotheses (falsifiable)

- **HD1 (signal alive).** With mixed data + dynamic sampling + n=8, the fraction of training prompts
  with a usable (mixed) group is high and `gdpo/correctness/mean` is stably positive — NOT the s2b
  near-zero/volatile correctness. *Falsified if* most groups are still degenerate.
- **HD2 (no collapse).** Combined with composition, `wellformed_rate` stays > 0.4 and `meta_emit_rate`
  stable through training; length < 1500 / clip < 0.20. *Falsified if* the s2b collapse recurs.
- **HD3 (meta useful, broadened).** `acc_with ≥ acc_without` holds, and held-out per-benchmark
  self_consistency ≥ boilerplate baseline (gsm8k 0.906 / math 0.542 / aime 0.133) and ideally ≥ Meta
  SFT (0.885 / 0.518 / 0.167), with **difficulty-stratified** gains on hard sets. *Falsified if* meta
  is neutral/harmful held-out.
- **HD4 (redirect not lost by broadening).** Redirect still appears in rollouts at a usable rate
  (priming layered later if too sparse). *Falsified if* redirect vanishes entirely.

## 6. Metrics

Build-time: output parquet distribution (source / scenario / difficulty / dropped-gold counts).
Train-time (wandb metacot-dcpo-v4): `gdpo/correctness/mean`, dynamic-sampling discard rate / mixed-group
fraction, plus the composition dashboard (eff_ratio_meta, wellformed_rate, pmi_member_rate,
meta_emit_rate, acc_with vs acc_without, response_length, clip_ratio). Final: held-out per-benchmark
self_consistency vs boilerplate + Meta SFT, difficulty-stratified.

## 7. Decision tree

- HD1 fails (groups still degenerate) → difficulty too hard; raise easy/medium share or lower hard.
- HD2 fails (collapse recurs) → composition insufficient; revisit len_cost / anchor (composition spec).
- HD3 fails (meta neutral held-out) → meta presence learned but not usefulness; add redirect priming /
  revisit PMI sign-gate.
- HD4 fails (redirect gone) → start the redirect-priming follow-up earlier.

## 8. Operational

- **Order (user-approved): data-change-first.** Build the mixed data, run s3 (data + composition +
  dynamic + n8) as the full loop, eval; only after it works do we layer redirect priming or further
  ablations (composition-only isolation is intentionally skipped per user).
- s2b + s2c cancelled (old data). s3 = new run on `triobj_dcpo_v4_stage3` (or reuse s2c config + new
  data parquet + n8 + dynamic + clip-higher), wandb `metacot-dcpo-v4` / run `dcpo_v4_s3`,
  `trainer.project_name=metacot-dcpo-v4`, save best checkpoint separately.
- Node: H100 4-GPU basicvc; new code release asset; pull_resume durability.
- Follow-ups (separate plans, after s3): redirect Harvest/Prime (2026-06-14 spec); difficulty-
  stratified eval.
