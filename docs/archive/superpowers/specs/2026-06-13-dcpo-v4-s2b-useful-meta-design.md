# DCPO v4 Stage-2b — Useful Metacognition (anti-boilerplate) design & experiment plan

**Date**: 2026-06-13 **Author**: ctsd-phase-c autonomous loop **Branch**: ctsd-phase-c
**Status**: APPROVED direction (user 2026-06-13), pending launch after this doc + wandb setup.

---

## 1. North-star (unchanged)

Reinforce **useful** metacognition to raise **accuracy**. Metacognition is a *means*, not an
end; calibration is a sub-goal. The win condition is: a model that emits `<|meta|>` blocks and,
**because of them**, is at least as accurate as the Base SFT — ideally more, especially on hard
problems. Held-out accuracy vs Base SFT is the final verdict; everything during training is a
proxy.

Baselines (v8_strict series, 1560-problem eval, results/metacot_v8_experiment_report.md):
- **Meta SFT (our init, v8_meta_inside_strict)**: 75.4% overall.
- **Base SFT**: 75.5% (GSM8K 88.5 / MATH500 51.6 / AIME 26.7).
- Meta format is already **accuracy-neutral at SFT** (75.4 ≈ 75.5). So stage-2 RL must *improve
  on* 75.4% via useful meta — a subtle bar, not a gap-closing one.

---

## 2. What the first stage-2 run did (the boilerplate failure)

`triobj_dcpo_v4_stage2_h100_4x4k` ran from SFT, 95 steps, then was cancelled. The training
proxies (emission plateau ~0.8, correctness positive) looked healthy, but inspecting the actual
rollouts + the dcpo/ diagnostic suite (already logged to wandb) revealed the model learned
**content-free boilerplate meta**, not useful metacognition. Hard numbers at gs90:

| metric | value | reading |
|---|---|---|
| `dcpo/meta_emit_rate` | 0.81 | emits meta 81% of the time |
| `dcpo/wellformed_rate` | **0.047** | only 4.7% are cleanly opened+closed |
| `dcpo/discard_rate` | 0.457 | 46% are malformed garbage |
| `dcpo/replaced_rate` | 0.324 | 32% auto-repaired by tier-1 replacement |
| `dcpo/pmi_member_rate` | **0.371** | quality signal reaches only 37% of rows |
| `dcpo/rmeta_mean_meta_rows` | +0.030 | quality signal is small |
| `dcpo/acc_with` vs `acc_without` | **0.715 vs 0.807** | meta makes the continuation *worse* |

Qualitatively, the meta is the **same fixed phrase on every problem** ("the algebra is clean; a
numerical spot-check would confirm / verify the boundary conditions / substituting the boundary
values…") regardless of whether the answer is 18, 856, or 7√5. It is decorative, often malformed,
and slightly harmful.

## 3. Root cause — the quality signal is *starved*, not broken

Three compounding layers, all evidenced above:

1. **Discrimination works.** placebo-corrected PMI gives generic boilerplate ~0 (placebo_fail 0,
   guard_hit 0), grades direction by correctness, probe-validated (t=17.9). The signal that
   *arrives* is correct.
2. **Magnitude is weak.** raw Δ' ~0.05, ×w_meta 0.5 ≈ 0.025/token, dwarfed by the emission head
   (w_emit 0.4 ≈ 0.06/token on the same meta tokens). Emission ~2–3× the quality signal.
3. **Coverage is gated by format (the decisive one).** PMI requires splice-alignable meta.
   `pmi_member_rate 0.371` ≈ `wellformed 0.047 + replaced 0.324`. The **46% discard rows get NO
   PMI at all** — only the unconditional emission reward. So on the malformed majority there is no
   quality grading; the emission head alone shapes them → boilerplate. Format brokenness
   *starves* the quality signal of coverage.

The emission head (w_emit 0.4) — added to escape the s1b silence-collapse — is now the engine of
boilerplate: it pays for *any* emission, including malformed/decorative meta that PMI can't reach.

## 4. The fix (s2b) — config-only rebalance, staging intent preserved by schedule

The original two-stage design (format first, then content) had the right instinct: format must be
established before PMI can grade content. We collapsed it to a single run because s1b RL kept
collapsing/length-farming (the 7-launch saga). s2b keeps the **single stable from-SFT run** but
re-creates "format-first → content" via weights + schedule:

| knob | s2 (old) | s2b (new) | why |
|---|---|---|---|
| `dcpo_w_format` | 0.1 | **0.35** | establish clean format strongly; wellformed (+1) also props emission, so the format head subsumes much of the emission role |
| `dcpo_w_emit` | 0.4 | **0.15** | stop boilerplate-farming (the engine); emission held by format head + floor |
| `dcpo_w_meta` | 0.5 | **0.8** | strengthen the quality signal magnitude (the real lever; clip 0.1085 unchanged — the Δ' distribution std 0.06 never saturates it) |
| `dcpo_w_meta_warmup_steps` | 50 | **80** | format establishes first; PMI ramps in only after meta is alignable → coverage rises before content is graded |
| `dcpo_len_cost` | 0.02 | **0.03** | tighter length bound (w_emit-driven inflation risk) |
| `dcpo_format_neg` | 0.2 | 0.2 | unchanged |
| `dcpo_meta_floor` | 0.05 | 0.05 | unchanged (collapse insurance) |
| `dcpo_pmi_clip_gate` | 0.1085 | 0.1085 | unchanged (distribution doesn't saturate it) |

No code change — every diagnostic metric below is already logged. SFT init, 300 steps, new
parallel node. The cancelled run's gs90 is evaluated in parallel as the boilerplate baseline.

## 5. Hypotheses (falsifiable, with thresholds)

- **H1 (format establishes).** With w_format 0.35, `dcpo/wellformed_rate` rises from 0.047 toward
  **>0.5** within ~80 steps. *Falsified if* it stays <0.2 — then format is an SFT artifact RL
  can't fix and SFT data needs work.
- **H2 (coverage rises).** As format establishes, `dcpo/pmi_member_rate` rises from 0.37 toward
  **>0.7** — the quality signal reaches most meta. *Falsified if* member_rate stays ~0.37 despite
  wellformed rising.
- **H3 (useful meta selected).** `dcpo/acc_with` rises to **≥ dcpo/acc_without** (currently 0.71 <
  0.81 = harmful). This is the cleanest in-training north-star proxy: meta should help, not hurt.
  `dcpo/rmeta_mean_meta_rows` rises above +0.03. Meta text becomes problem-specific (manual rollout
  check). *Falsified if* acc_with stays below acc_without.
- **H4 (accuracy, the verdict).** Held-out eval of the s2b final/late checkpoint ≥ **75.4%** (Meta
  SFT) with emission >0.5 and non-boilerplate meta. *Falsified if* below 75.4% or meta is still
  boilerplate.

## 6. Metrics — what to LOG (all exist) and what to WATCH

All under `dcpo/` already logged to wandb every step. The s2b dashboard, grouped by hypothesis:

- **Format health (H1):** `dcpo/wellformed_rate` ↑ (>0.5), `dcpo/discard_rate` ↓ (<0.2),
  `dcpo/meta_emit_rate` (ok to drift down to a natural level), `dcpo/replaced_rate`,
  `dcpo/meta_unclosed_rate`.
- **Quality coverage (H2):** `dcpo/pmi_member_rate` ↑ (>0.7), `dcpo/pmi_aligned_rate`,
  `dcpo/pmi_guard_hit_rate` (~0), `dcpo/pmi_placebo_fail_rate` (~0).
- **Quality signal (H3):** `dcpo/rmeta_mean_meta_rows` ↑, `dcpo/rmeta_pos_rate` /
  `dcpo/rmeta_neg_rate`, and the headline pair **`dcpo/acc_with` vs `dcpo/acc_without`** (want
  with ≥ without).
- **Stability / cost:** `gdpo/correctness/mean` (positive), `response_length/mean` (<1500),
  `response_length/clip_ratio`, `actor/kl_loss`.
- **Final verdict (H4):** held-out eval accuracy + ECE vs SFT 75.4% / Base SFT 75.5%.

Primary success metric to watch is **acc_with vs acc_without** — it directly measures "does the
meta help the answer," which is the north-star in miniature.

## 7. Decision tree

- All four hypotheses trend right → ride to completion, held-out eval confirms ≥75.4% → SUCCESS.
- H1 fails (wellformed stays low) → format is an SFT-data problem; pause RL, fix SFT meta diversity.
- H1/H2 ok but H3 fails (acc_with stays < acc_without) → PMI captures likelihood-lift but not
  *correctness*-lift; revisit the sign-gate / consider correctness-conditioned PMI.
- Held-out eval < 75.4% despite healthy proxies → meta still net-neutral/harmful; rethink the meta
  protocol definition (what *is* a good meta habit) before more RL.

## 8. wandb setup (clean, per-experiment)

The old runs are hard to read (shared project `skilldiscovery2`, interleaved run ids, stale
config keys from resumes). Fix:
- **New dedicated project**: `metacot-dcpo-v4`.
- **One run id per experiment**, explicit `WANDB_NAME`: `dcpo_v4_s2b` (this run),
  `dcpo_v4_s2_boilerplate_gs90_eval` (the baseline eval), future arms `dcpo_v4_s2c_*` etc.
- `WANDB_RUN_GROUP=dcpo_v4_useful_meta` so the family groups together but each arm is its own run.

## 9. Operational

- **Node**: new parallel H100 4-GPU (basic VC), job `triobj-dcpo-v4-s2b`.
- **Baseline eval (parallel, separate node)**: gs90 of the cancelled boilerplate run → held-out
  1030/1560 acc+ECE vs SFT/Base-SFT. Anchors the s2b comparison.
- **Idempotency/durability**: per-stage HF push + pull_resume (same machinery as s2).
- **Config**: `configs/triobj_dcpo_v4_stage2b_h100_4x4k.yaml`; node yaml
  `h100std_triobj_dcpo_v4_s2b.yaml`.
