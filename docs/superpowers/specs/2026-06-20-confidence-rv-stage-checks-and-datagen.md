# Confidence-Conditioned Redirect/Verify — Per-Stage Checks, Data-Gen, and RL-Reward Decision

**Date:** 2026-06-20
**Status:** stage-by-stage intent-check plan (every stage's check defined BEFORE the stage runs)
**Companion design:** `2026-06-19-confidence-redirect-verify-distill-design.md` (the *what*/*why*).
**Companion metrics contract:** `2026-06-19-confidence-redirect-verify-experiment-metrics.md` (the B/C/D + PG gates; this doc REUSES those numeric gates, it does not re-derive them).
**Predecessors / evidence:** PG0 pilot (memory `pg0-raw-onpolicy-harvest-infeasible`: redirect helps easy +0.082, hurts hard −0.034, on-policy harvest yield ≈ 141 ≪ 1500 → teacher distill); v2 RL emission collapse (wellformed 0.615 → 0.010); v8 decorative redirect (758/4264 real switch).

This doc is the **per-stage CHECK/MONITOR plan** the user asked for: for each of the three stages (0 data-gen, 1 warm-up SFT, 2 RL) it pins, *before the stage is run*, (i) the artifact produced + its HF upload path, (ii) the exact intent-check metrics with numeric PASS/STOP thresholds (reusing the B/C/D gates from the metrics contract), (iii) the response/quality monitoring, and (iv) the explicit USER-APPROVAL checkpoint at the stage boundary. It also records the **RL-reward DECISION** and the **SAMPLE-FIRST sub-plan** (8–12 hand-anchored demos on real TRAPI, GPU-free, self-checked, user-approved, BEFORE the full GPU rollout).

---

## §0 Two design changes applied before any stage runs (user-approved 2026-06-20)

These changes are already approved and are assumed by every stage below; they are recorded here so the stage checks are unambiguous about the format being measured.

### 0.1 DROP the `<|switch|>` special token → TEXT `decision:` field inside `<|meta|>`
The `<|switch|>` special token required vocab/embedding surgery and is **unnecessary** for the distill approach: causality is measured by ablating the **whole `<|meta|>` block** (§4 / D3), not by banning a token. It is **DEAD CODE** and is removed across `src/metacot/prompt_redirect_verify.py` (`SWITCH_TOKEN` constant + every `{SWITCH_TOKEN}` reference), `scripts/build_confidence_redirect_verify_sft.py` (the `from src.metacot.prompt_redirect_verify import SWITCH_TOKEN` import and the `has_switch = _has_meta_block(...) and (SWITCH_TOKEN in ...)` redirect check), and all tests that assert on `<|switch|>`.

It is **REPLACED** with a TEXT field **`decision: redirect`** / **`decision: verify`** *inside* the `<|meta|>` block — mirroring the existing `prompt_behavior.py` `decision: switch_method` convention. The redirect "real method change" check becomes:

> the meta block contains **`decision: redirect`** AND the post-meta continuation actually uses a **different method**; the causal flip wrong→right (§4 redirect arm, lower-CI > 0) remains the **hard gate**.

So the canonical accepted block is `<|meta|> … confidence: 0.xx … decision: {redirect|verify} … <|/meta|>` followed by the (differently-methoded, for redirect) correct continuation.

### 0.2 ADD a meta-format NORMALIZE+VALIDATE filter ("inspect-and-substitute")
A real teacher run produced a perfect demo but with the close tag `</|meta|>` instead of `<|/meta|>`, which the strict checker silently **DROPPED**. Before the structural/causal filter the build driver runs, in order:

1. **`normalize_meta_format(text) -> text`** (pure, repairs repairable variants → canonical):
   - close-tag variants `</|meta|>` | `</meta>` | `<|meta/|>` → `<|/meta|>`;
   - collapse stray whitespace; ensure exactly **one** well-formed block;
   - canonicalize the `confidence:` and `decision:` line casing/spacing.
2. **`validate_meta_structure(text) -> {ok|fatal, reason}`** (pure):
   - **FATAL (drop, increment a counter):** zero or > 1 meta blocks; missing `confidence:` line; missing `decision:` line; `decision` value not in `{redirect, verify}`.
   - **repairable → repaired+kept** (the `</|meta|>` demo above is repaired and kept).

Wired into the build driver **before** `_has_meta_block` / the causal checks. Pure + unit-tested; the suite uses a MOCK teacher (no real network/GPU). The real TRAPI path is wired but only executed by the main loop later.

---

## §1 RL-reward DECISION (recorded; NOT implemented in this pass)

**Decision:** reuse the existing **DCPO triobj** reward on the **warm-up SFT checkpoint**:

```
R = R_corr (correctness) + R_meta (PMI; dcpo_w_meta back ON ~0.8) + R_cal (calibration)
```

- **DROP** the contrastive / counterfactual reward used in earlier drafts (too many rollouts to be affordable at RL scale).
- **Hypothesis:** v2 collapsed because the **SFT meta was hollow**, not because the reward was wrong. The warm-up SFT (Stage 1) makes the meta **functional** (C-gate passes), so the *same* triobj reward sustains it instead of collapsing it.
- **Decisive monitored signal:** `acc_with − acc_without` (accuracy on fired-meta problems minus non-fired) flips from **NEGATIVE** (v2: 0.71 < 0.81) to **POSITIVE** and **stays positive** while `wellformed_rate` does **not** collapse. This is reported as a co-signal alongside the D3 causal gate (D3, not `acc_with/without`, is the hard gate per the s3b lesson that `acc_with/without` is confounded).

RL implementation is deferred to the main loop; this doc only records the decision so Stage 2's checks are defined.

---

## §2 STAGE 0 — DATA-GEN (teacher distill build)

### Artifact + HF upload path
- **Artifact:** the causally-filtered SFT parquet `data/confidence_rv_sft.parquet` (rows: `messages`, `scenario∈{redirect,verify}`, `confidence_label`, `wrong_prefix`, `prefix_split_char`, `split_tags`), plus the **build log** (`summary` dict + per-row accept/reject reasons + the `format_repaired` / `format_fatal` counters from §0.2).
- **Compute:** CPU/network — TRAPI teacher (CPU) + short decode-only vLLM rollouts (student N=8 anchors + CF arms) that can piggyback an H100 slot. **No SFT training here.**
- **HF upload:** dataset repo `iamseungpil/metacot` (per `CLAUDE.md`), path `data/confidence_rv_sft.parquet`; the build log uploaded alongside as `logs/confidence_rv_sft_build.json`. Upload only AFTER the B-gate passes and the user approves (§2.4).

### 2.1 Intent-check metrics (REUSE §B B-gate + §E PG-build)
The full §B table from the metrics contract is the gate; the load-bearing rows:

| ID | Metric | PASS | STOP |
|----|--------|------|------|
| B1 | redirect demos surviving causal filter (easy/medium) | **≥ 600** | < 250 |
| B2 | verify demos surviving no-harm-confirm filter (easy/medium, prefix_correct) | **≥ 300** | < 120 |
| B3 | causal acceptance rate kept/candidates (redirect arm) | **≥ 0.25** | < 0.10 |
| B5 | demo-calibration Spearman ρ(teacher conf_emitted, conf_target) | **ρ ≥ 0.6** + 0 same-side-of-0.5 violations | ρ < 0.3 |
| B6 | hard-row purity (freshly-distilled demos on `difficulty==hard`) | **== 0** | any > 0 |
| B7 | mix ratio distilled/total | **∈ [0.15, 0.30]** | outside band |
| B8 | prefix loss-mask coverage | **== 1.0** | < 1.0 |
| B4 | confidently-wrong sub-bucket count (REPORTED, not gating) | **≥ 80** report either way | — |

**B-gate:** B1∧B2∧B3∧B5∧B6∧B7∧B8. **PG-build** pre-gate (before the FULL teacher spend): on a 200-problem dev slice with a small real teacher N (≈50), projected full-corpus B1 ≥ 600 ∧ B2 ≥ 300 ∧ B3 ≥ 0.25 (linear-scale the dev-slice accept rate); projected B1 < 250 → do NOT run the full build.

### 2.2 Response/quality monitoring (data-gen specific)
Computed on the build log, reported every build:
- **`functional_rate`** = fraction of accepted redirect demos that causally **flip wrong→right** (redirect arm lower-CI(R − best control) ≥ `ACCEPT_MARGIN=0.5`). This is the anti-decorative core; a high kept-count with low `functional_rate` is the v8 trap. *Target ≥ the B3 acceptance band; a functional_rate that decouples from kept-count flags a decorative build.*
- **`format_repaired_rate`** = `format_repaired / (format_repaired + format_fatal + well_formed)` from §0.2 (how many teacher outputs the normalize step rescued, e.g. the `</|meta|>` case). *Report value; a spike (> ~0.5) means the teacher prompt should pin the close tag, not that the data is bad.*
- **`format_fatal_rate`** = `format_fatal / total` (dropped by `validate_meta_structure`). *Report; a high fatal rate means the teacher prompt/format is broken — fix the prompt, do not lower the validator.*
- **`calibration` (demo-level)** = teacher **stated** `confidence:` vs student **measured** `conf_target`: Spearman ρ (= B5) AND the same-side-of-0.5 violation count (must be 0). This is the no-leak twin — the demo must report the *student's* confidence, not the teacher's.
- **`redirect_count` / `verify_count`** (= B1 / B2) and the `verify_catch` / `verify_confirm` split.
- **`decision_field_consistency`** = fraction of accepted rows whose `decision:` value matches the arm/band that minted them (redirect-band ⇒ `decision: redirect`, verify-band ⇒ `decision: verify`). *== 1.0 invariant (the §0.1 replacement of `<|switch|>`).*

### 2.3 SAMPLE-FIRST sub-plan (BEFORE the full GPU rollout)
A cheap dress-rehearsal that proves the teacher prompt + normalize/validate + causal-filter logic produce **real** demos before any GPU spend:

1. **Hand-anchor 8–12 sample problems** drawn from the real easy/medium RL-train pool (a spread: ~5 redirect-band low-conf-wrong, ~5 verify-band high-conf-right, ~2 boundary). GPU-free: the student "rollouts" are pre-supplied real attempts (anchors) rather than freshly decoded, so no vLLM is needed for the sample.
2. **Run the REAL TRAPI teacher** on each (this is CPU/network, no GPU) via the model fallback list `['gpt-5.4-mini_2026-03-17','gpt-5.3-chat_2026-03-03','gpt-5.4_2026-03-05']` (ENTRA bearer-token provider, NO static token; try in order on 404/503/health, retry 429/403 with backoff).
3. **Self-check the 8–12 demos** against a SAMPLE rubric (a hand-sized mirror of §2.2):
   - every demo has exactly one well-formed (or normalize-repaired) `<|meta|>` block with a `confidence:` and a `decision:` line;
   - `decision:` matches the band; the redirect continuation visibly uses a **different method**;
   - the stated `confidence:` is within `CONF_STATED_TOL=0.15` of the anchored `conf_target` (no-leak);
   - eyeball `functional_rate`: do the redirect demos actually fix the wrong path, and do the verify demos perform a **genuine independent check** (substitution / recompute / re-derive), not a restatement;
   - count `format_repaired` (e.g. any `</|meta|>` rescued) and `format_fatal`.
4. **USER APPROVAL on the samples** — present the 8–12 demos + the sample rubric tally. Only on user GO does the full GPU rollout (full pool, real student vLLM rollouts) launch. A STOP here means fix the teacher prompt / band / normalize rules and re-sample — never scale a bad teacher prompt to the full pool (PG0 "measure yield before you train" applied at demo granularity).

### 2.4 USER-APPROVAL checkpoint (stage boundary 0 → 1)
After the full build: present the **B-gate verdict** (each B row value vs threshold) + the §2.2 monitoring (`functional_rate`, `format_repaired_rate`, `format_fatal_rate`, demo calibration ρ, redirect/verify counts, B4 confidently-wrong count). **GO only on user approval AND B-gate PASS.** Any B STOP row ⇒ do NOT upload to HF, do NOT spend the SFT H100; fix band/teacher-prompt/filter and rebuild. (= `PG-sft` pre-gate.)

---

## §3 STAGE 1 — WARM-UP SFT

### Artifact + HF upload path
- **Artifact:** the SFT checkpoint `confidence_rv_sft` (init from `v8_meta_inside_strict_sft`, segment-masked via `redirect_train_spans`, `teacher_kl.enabled=false`), trained on `data/confidence_rv_sft.parquet`.
- **Compute:** H100 only (`h100std_confidence_rv_sft.yaml`).
- **HF upload:** model checkpoint pushed to HF (the H100 jobs pull/push via the `code_snapshots` + checkpoint convention in `CLAUDE.md`); record the checkpoint id for the §4 RL init and the §C eval `--model_path`.

### 3.1 Intent-check metrics (REUSE §C C-gate)
Run `scripts/eval_vllm_1030.py --model_path <ckpt> --model_name confidence_rv_sft --max_tokens 4096` on the 1030-set, then:

| ID | Metric | PASS | STOP |
|----|--------|------|------|
| C1 | meta-ON accuracy (1030) | **≥ 0.786** AND **≥ 0.651** | < 0.651 |
| C2 | ECE of emitted confidence vs correctness | **≤ v8-decorative ECE − 0.03** AND ≤ 0.20 | ≥ v8 ECE |
| C3 | confidence-bin monotonicity Spearman ρ(conf-bin, pass-rate) | **ρ ≥ 0.7** | ρ < 0.4 |
| C4 | redirect_causal_rate (held-out low-conf-wrong fired-redirect): aggregate lower-CI(R − B′) | **lower-CI > 0** AND rate **≥ 0.30** | lower-CI ≤ 0 |
| C5 | verify_noharm_rate (held-out high-conf-right fired-verify) | **≥ 0.85** AND verify-arm not below plain lower-CI | < 0.70 |
| C6 | action-appropriateness (action conditioned on situation AND emitted conf) | **≥ 0.80** | < 0.5 |
| C7 | meta-emit rate (anti-decorative, two-sided) | **∈ [0.10, 0.70]** | < 0.05 OR > 0.85 |

**C-gate:** C1∧C2∧C3∧C4∧C7, plus C5∧C6 if their fired-sample n ≥ MIN_K-scaled power (else reported underpowered, not blocking). **C4's causal lower-CI>0 is the hard gate** — a high emit rate (C7) with C4 lower-CI≤0 is the decorative trap and FAILS regardless of C1.

### 3.2 Response/quality monitoring (SFT specific)
- **emitted-confidence histogram** (10 bins) + per-bin pass-rate (the C2/C3 binning) — confirms the SFT actually emits a *spread* of calibrated confidences, not a single mode.
- **`decision:`-field distribution** over fired blocks (redirect / verify / malformed) — confirms the §0.1 text field is emitted and parseable post-SFT.
- **format health on the eval output:** `wellformed_rate` (closeable `<|meta|>…<|/meta|>`) + how often `normalize_meta_format` would have to repair the model's *own* output (should be near 0 — the model learned the canonical close tag from the repaired training data).
- **`acc_with` / `acc_without`** at SFT (descriptive baseline for the Stage-2 flip signal): record both so the RL flip negative→positive (§1) is measured against the SFT starting point.

### 3.3 USER-APPROVAL checkpoint (stage boundary 1 → 2)
Present the **C-gate verdict** (each C row value vs threshold) + §3.2 monitoring. **GO to RL only on user approval AND C-gate PASS (critically C4 lower-CI>0).** C-gate fail ⇒ do NOT submit the RL yaml; the SFT did not form causal meta and RL would only collapse it. (= `PG-rl` pre-gate.)

---

## §4 STAGE 2 — RL (SFT → RL)

### Artifact + HF upload path
- **Artifact:** the RL checkpoint (init from the Stage-1 `confidence_rv_sft` ckpt; reward = DCPO triobj `R_corr + R_meta(PMI on, w_meta~0.8) + R_cal`, §1; keep the validated emission-survival knobs `format_neg 0.2` + per-token `w_emit 0.4` so D1 does not reopen the abstention escape).
- **Compute:** H100 only (`h100std_confidence_rv_rl.yaml`, copy `h100std_triobj_dcpo_v4_s2b.yaml`). Enable `DCPO_WANDB_ROLLOUTS=1` so D1/D5 curves stream.
- **HF upload:** RL checkpoint pushed to HF on convergence (same checkpoint convention).

### 4.1 Intent-check metrics (REUSE §D D-gate)

| ID | Metric | PASS | STOP (kill the run) |
|----|--------|------|---------------------|
| D1 | meta_survival / wellformed_rate over steps | **stays ≥ 0.40**, final ≥ 0.40 | < 0.10 at any step, OR > 0.50 absolute drop from its step-10 value within 30 steps (v2 signature) |
| D2 | accuracy delta vs SFT init (1030 meta-ON) | **final acc ≥ SFT-init acc** AND **≥ 0.786** | acc < 0.786 at convergence |
| D3 | utility-conditioned causal gate (meta-ON vs same prefix meta-block ABLATED, held-out fired-meta): aggregate lower-CI | **aggregate lower-CI > 0** | lower-CI ≤ 0 at convergence |
| D4 | `acc_with` vs `acc_without` (descriptive co-signal, NOT the gate) | report both; flag if acc_with < acc_without − 0.05 | (no standalone STOP) |
| D5 | over-forming guard — meta-emit rate at convergence | **∈ [0.10, 0.70]** | → 0, OR → ~1.0 with D3 lower-CI ≤ 0 |

**D-gate (experiment "works as intended"):** D1 (no collapse) ∧ D2 (acc ≥ baseline) ∧ D3 (causal lower-CI>0). D4/D5 annotate.

### 4.2 Response/quality monitoring (RL specific) — the decisive §1 signal
- **`acc_with − acc_without` curve (the decisive monitored signal):** must flip from **NEGATIVE** (v2 baseline 0.71 < 0.81) to **POSITIVE** and **stay positive** while `wellformed_rate` (D1) does not collapse. This is the headline RL-reward-decision check; D3 (causal ablation lower-CI>0) is the hard gate that the flip is *causal* and not confounded.
- **`wellformed_rate` curve with the D1 collapse alarm** (below 0.10, or > 0.50 drop in 30 steps → kill + discard).
- **meta-emit-rate curve** (D5) joined with D3 to catch decorative over-forming (rate → ~1.0 while D3 lower-CI ≤ 0).
- **`decision:`-field stability** over RL steps (redirect/verify mix does not collapse to a single action; malformed-block rate stays low).
- **`R_meta` / `R_cal` / `R_corr` component curves** — confirm `R_meta` (PMI) stays live (does not silence to 0 as in the v3l format-penalty collapse).

### 4.3 USER-APPROVAL checkpoint (final stage boundary)
On convergence present the **D-gate verdict** + §4.2 monitoring (especially the `acc_with − acc_without` flip and D3). **D-gate PASS = experiment confirmed** (confidence self-emitted [C-gate] AND redirect/verify decision causally raising accuracy [D3] AND it survives RL [D1]). A D1 STOP-rule trip during the run kills+discards immediately (no daemon; manual relaunch on preempt). Final user approval closes the experiment.

---

## §5 Stage ordering + one-line gate summary

Strict order with a user-approval AND a numeric gate at every boundary:

**SAMPLE-FIRST (8–12 real-TRAPI demos, GPU-free) → user GO →**
**Stage 0 DATA-GEN** (B-gate: ≥600 redirect / ≥300 verify causally-filtered; functional_rate / format_repaired / demo-calibration monitored) **→ user GO + PG-sft →**
**Stage 1 WARM-UP SFT** (C-gate: self-emits calibrated confidence, ECE down, C4 redirect-causal lower-CI>0, acc ≥0.786) **→ user GO + PG-rl →**
**Stage 2 RL** (DCPO triobj `R_corr+R_meta(PMI on)+R_cal`, contrastive dropped; D-gate: no emission collapse ≥0.40, acc ≥0.786, D3 causal lower-CI>0; decisive `acc_with−acc_without` flips negative→positive without wellformed collapse) **→ user GO = experiment confirmed.**

A STOP at any pre-gate (sample self-check / B-gate / C-gate) discards the change and keeps the prior best **before** GPU spend (PG0 philosophy). No SFT H100 until §2.4 passes; no RL H100 until §3.3 passes.
