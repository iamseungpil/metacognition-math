# Metacognition-Math — Project Constitution

> A living governance document. It fixes (1) **what we are building and why**, (2) **how we
> diagnose problems from evidence**, and (3) **the metric dashboard + failure signatures** that
> every run is read against. When a decision or diagnosis is made, it MUST be justifiable against
> this document. When reality contradicts this document, update the document (with evidence) —
> do not quietly work around it.
>
> Last major revision: 2026-07-10 (pmi_shift starvation diagnosis).

---

## Part I — Intent (North-Star)

**We are developing a metacognition RL method that, from a self-distillation standpoint,
reinforces a *specific useful habit* so that math accuracy goes up.**

> 현행 실험명: **RQ3 매치드 래더** (`h100std_rq3_b0/b2/b3.yaml`, init =
> `models/b0_gold_sft` · `models/b23_rv_unmasked_sft`).

Precise claims:

1. **Meta is a means, not an end.** The goal is **accuracy**. Metacognitive behavior (emitting a
   `<|meta|>` block that checks an assumption, verifies before answering, or redirects a wrong
   lean) is valuable **only when it causally raises the probability of a correct answer**.
   Calibration (matching stated confidence to accuracy) is at most a *signal* of a useful habit,
   never the objective.

2. **Self-distillation framing.** The model supplies its own training signal: we reward the
   *belief-movement a meta block causes in the model's own head* (PMI-shift), not an external
   teacher's label. The reward reinforces the model's own good metacognitive moves.

3. **The two research questions (matched-arm ladder).**
   - **RQ1 = B2 − B0**: does a meta-SFT init help vs a no-meta (meta-stripped) SFT init, under
     otherwise identical GRPO?
   - **RQ2 = B3 − B2**: does adding the PMI-shift meta reward help vs plain GRPO on the same
     meta-SFT init?
   - The arms differ **only** along their stated axis. Any other difference is a bug.

4. **Success bar.** Meta must **outperform matched base on held-out math accuracy** — not tie,
   not "calibrate better while scoring lower". The final judge is the 1030-problem held-out eval
   (GSM8K 500 + MATH 500 + AIME 30), not the in-training val.

---

## Part II — Principles of Diagnosis

1. **Evidence over narrative.** Every diagnosis cites a metric with a number and, where possible,
   a `file:line`. "It feels unstable" is not a diagnosis; `actor/entropy 2.08 rising while
   acc flat` is.

2. **Verify-causes before fixing.** Do not change config on a hypothesis. Confirm the mechanism
   fires (or fails) in the trace first. A fix that is not preceded by a confirmed cause is a guess.

3. **Distinguish the lever from the symptom.** Length growth, high entropy, low val can all be
   *downstream* of one upstream cause. Trace to the most-upstream confirmed cause before acting.
   (History: len_cost, anchor_norm, and w_meta were all debated as the cause; the confirmed
   upstream cause was **weak meta emission from an under-trained base SFT** — see Part V.)

4. **Gold is for correctness/causality only — never to measure confidence.** Use `math_verify`
   (parse+verify), never the deprecated `check_correctness`.

5. **No over-claiming.** A single early data point (e.g. gs25) is not a trend. A package effect
   is not a single-head effect. State sample size and stage.

6. **Matched isolation is sacred.** If a change to one arm cannot be applied identically to its
   comparison arm, it confounds the RQ — either apply it to both or do not apply it.

7. **A null/absent signal is a FAIL, not a pass.** An inert reward (fires on 3% of rollouts with
   zero magnitude) is a broken experiment, even if nothing crashes.

---

## Part III — The Mechanism (what reward reaches which tokens)

DCPO region-split advantage (`src/training/dcpo_region.py`, compose ≈ line 1480):

```
advantages = w_corr * A_corr * ans      # correctness advantage  -> ANSWER span ONLY
           + w_meta * A_meta * meta_c    # meta reward (pmi_shift) -> META_CONTENT span ONLY
           + w_cal  * A_cal  * conf      # calibration            -> confidence span
```

Consequences that MUST be kept in mind when reasoning:

- **Correctness does NOT reach meta tokens.** `A_corr` is masked by `ans` (answer span). In a
  region-split arm (B3), the meta tokens are trained **only** by `A_meta` (pmi_shift). In a
  VANILLA_GRPO arm (B2), by contrast, the correctness advantage is applied to the whole response
  including meta tokens. This asymmetry is the core of RQ2 and the usual source of confusion.
- **`A_meta = pmi_shift`** = asymmetric sign-reversal of the gold−decoy log-odds across the meta
  block (open vs close position). `+save` on decoy→gold, `−derail` on gold→decoy. See
  `src/training/dcpo_pmi_shift.py`.
- **`anchor_norm` (`dcpo_anchor_norm: true`)** rescales `A_meta` to the correctness advantage
  magnitude so a weak PMI signal "is not buried" (`A_meta *= corr_s / meta_s`). This is a
  **double-edged** tool: it boosts a *real* PMI signal (good, as in the old instruct run) but it
  **amplifies noise to full scale when the PMI signal is inert** (bad, as in the base run). Its
  benefit is conditional on PMI actually firing.
- **Warmup.** `dcpo_w_meta_warmup_steps` linearly ramps `w_meta` (and, coupled to the same scale,
  `dcpo_len_cost`) from 0→1. Early steps therefore have weak meta pressure and weak length
  containment.

---

## Part IV — The Diagnostic Dashboard

Read every meta-RL run against these. Ranges are calibrated on the **old instruct pmishift run
that won T1 (healthy)** vs the **base B3 run (broken)** — see Appendix A.

### Emission & structure (is there material for the meta reward?)
| Metric | Healthy | Broken (base B3) | Reads |
|---|---|---|---|
| `dcpo/meta_emit_rate` | **0.98–0.99** | **0.40–0.52** | fraction of rollouts emitting a meta block. Low ⇒ SFT under-installed the habit. |
| `dcpo/discard_rate` | 0.12–0.22 | **0.45** | malformed / no-valid-meta rollouts. High ⇒ meta poorly formed. |
| `dcpo/meta_unclosed_rate` | low | 0.09 | opened `<|meta|>` never closed (truncation/drift). |

### PMI-shift engagement (is the meta reward actually firing?)
| Metric | Healthy | Broken | Reads |
|---|---|---|---|
| `dcpo/pmishift_attempted_rate` | **0.52–0.66** | **0.03** | fraction scored by pmi_shift. **<0.1 ⇒ inert reward.** |
| `dcpo/pmishift_n_save` | **8–11** | **0** | decoy→gold reversals per step (the reward's core event). 0 ⇒ dead. |
| `dcpo/pmishift_n_derail` | 13–15 | ~0 | gold→decoy reversals. |
| `dcpo/pmishift_rmeta_mean_scored` | **+1.0–+1.2** | ~−0.2 | mean scored meta reward. Near-0 ⇒ no signal. |

### Does meta help accuracy? (the north-star check)
| Metric | Healthy | Broken | Reads |
|---|---|---|---|
| `dcpo/acc_with` | **0.87** | 0.65 | accuracy of meta-emitting rollouts. |
| `dcpo/acc_without` | (n/a, ~all emit) | **0.72** | accuracy of no-meta rollouts. |
| **acc_with − acc_without** | **> 0** | **< 0** (−0.07) | **negative ⇒ meta currently hurts.** (Caveat: could be problem-difficulty selection; treat as necessary-not-sufficient.) |

### Stability (recipe health)
| Metric | Healthy | Watch | Reads |
|---|---|---|---|
| `actor/entropy` | 0.2–0.3 (converged) or 1.0–1.8 (mid-train, no collapse) | **<0.05 = collapse; >2.0 rising with flat acc = noise injection** | v1 base collapsed to 0.01 (entropy). B3 = 2.08 rising (anchor amplifying inert PMI). |
| `response_length/clip_ratio` | <0.2 | **>0.5 = degeneration** | |
| `response_length/mean` | stable | monotone growth toward cap ⇒ eventual truncation | uniform across arms is benign for the *comparison*; a single arm diverging is not. |

### Validation — read it correctly
- In-training val = **`verl_val_meta_mix.parquet` = 594 problems** (gsm8k 198 + MATH subsets + omni-math), greedy (temp 0, n=1), every 25 steps. **NOT** the 1030 held-out battery.
- `val-core/<ds>/reward/mean@1` is on the **+1/−1 reward scale**, so **accuracy = (reward + 1) / 2**.
  `reward 0.588` = **79% accuracy**, not 59%. Do not report the reward as accuracy.
- The final judge is the separate **1030 held-out eval** at gs300 (`eval_hf.py`), on base-vs-meta matched arms.

---

## Part V — Known Failure Modes & Signatures

1. **PMI-shift starvation (base, 2026-07-10).** Low `meta_emit_rate` (0.40) → low
   `pmishift_attempted_rate` (0.03) → `n_save 0`. The meta reward has no material to score, so it
   is inert; `anchor_norm` then amplifies the near-zero signal into noise on the meta tokens
   (entropy 2.08). **Root cause: the base meta-SFT under-installed the meta habit** — the corpus
   is 100% meta yet the SFT model emits meta only ~52% at RL start (gs1) and RL erodes it to 40%
   (because acc_with < acc_without). Contrast: instruct+functional-SFT emitted 99% and PMI fired.
   **Fix locus is UPSTREAM (SFT), not the reward:** deeper/stronger base meta-SFT (more epochs
   and/or higher LR so emission ≈ 99%) + an RL-side meta-emission floor to resist erosion. Only
   then do pmi_shift / anchor_norm / w_meta become meaningful levers.

2. **Base entropy collapse (v1, 2026-07-08).** All arms collapsed gs50–100 (entropy→0.01,
   clip→0.9, length 717→3582). Cause: instruct-recipe hyperparams (temp 0.6 + top_k 20 +
   GDPO std-normalization + 4096 cap) on the base substrate. Fix (v2): temp 1.0, top_k −1,
   `norm_adv_by_std=false` (Dr.GRPO), cap 8192. Confirmed by instruct twins completing 300 steps
   on the byte-identical recipe → substrate causation.

3. **Reward mis-scaling misread (recurring).** Treating `+1/−1` reward as `0–1` accuracy. Always
   convert `acc = (r+1)/2`.

4. **Package-effect over-claim.** Attributing a win to a single head when several were on
   (the T1 win was the full triobj package, not pmi_shift alone). Isolation requires all other
   heads matched/off.

5. **anchor_norm amplifying nothing.** `dcpo_anchor_norm: true` is only beneficial when PMI fires;
   with an inert PMI it injects noise. Gate its use on `pmishift_attempted_rate > ~0.3`.

---

## Part VI — Preconditions / Gates for a Valid Meta-RL Run

A meta-RL run is only *interpretable* if, in the first ~25 steps:
- `dcpo/meta_emit_rate ≥ 0.8` (else the meta reward is starved — fix the SFT first).
- `dcpo/pmishift_attempted_rate ≥ 0.3` and `n_save > 0` (else the reward is inert).
- `actor/entropy` not collapsing (`> 0.1`) and clip `< 0.2`.
- Matched arms differ only along their declared axis (audit the launcher CLI overrides).

If any gate fails, **do not spend GPU to gs300** — fix upstream and relaunch.

---

## Part VII — Verification Protocol

- **Matched-arm isolation:** enumerate every launcher override; the diff across arms must be
  exactly {SFT-init, reward-source}. Same config path, same len_cost, same recipe.
- **PMI-shift is a proxy — stratify to defend it:** the signal test must show
  shift(real_meta) ≫ shift(corrupted_meta) on the own≠gold subset (placebo), AND split own≠gold
  into gold-is-default vs not (safe-default control). Passing bare own≠gold is not enough
  (see `src/training/dcpo_pmi_shift.py` docstring).
- **Resume safety:** HF push every 90s (`--keep 2`) + `pull_resume_ckpt` + RGS completeness gate
  (abort rather than restart if HF has a step but local pull failed) + `resume_mode=auto`
  (optimizer state restored) + `WANDB_RESUME=allow` (one run). Verified across r2v1→v4 cycles.
- **Final judgment:** 1030 held-out eval, matched base vs meta, bootstrap CI, single-seed results
  flagged as provisional.

---

## Appendix A — Reference trace: old (healthy) vs base B3 (broken)

| Metric | OLD instruct pmishift (T1 win, gs181–299) | base B3 (gs37) |
|---|---|---|
| meta_emit_rate | 0.98–0.99 | 0.40 |
| pmishift_attempted_rate | 0.52–0.66 | 0.03 |
| pmishift_n_save | 8–11 | 0 |
| pmishift_n_derail | 13–15 | ~0 |
| pmishift_rmeta_mean_scored | +1.0–+1.2 | −0.2 |
| acc_with | 0.87 | 0.65 |
| discard_rate | 0.12–0.22 | 0.45 |
| SFT init | instruct + functional-redirect SFT | Qwen3-8B-Base + 1-epoch meta-SFT |
| meta emission at RL start | ~0.99 | ~0.52 (from a 100%-meta corpus) |

**The mechanism did not change; the substrate + SFT-installed meta habit did.**
