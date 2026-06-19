# Confidence-Conditioned Redirect/Verify — Experiment + Intent Metrics

**Date:** 2026-06-19
**Status:** metrics contract for autoresearch (numeric pass/stop on every signal)
**Companion design:** `2026-06-19-confidence-redirect-verify-distill-design.md` (the *what*/*why*; this doc is the *measure*/*gate*).
**Predecessors / evidence:** PG0 pilot (memory `pg0-raw-onpolicy-harvest-infeasible`); v8 decorative-redirect failure (758/4264 real switch); v1 forming-collapse (acc 0.651 < baseline 0.786); v2 emission collapse (wellformed 0.615 → 0.010).

This doc defines (i) the experiment stages, (ii) the **exact** metric autoresearch monitors at each stage with a **numeric PASS / STOP threshold and a verify-command sketch**, and (iii) the GO/STOP **pre-gates** that must clear *before* GPU spend, mirroring the PG0 "measure yield before you train" philosophy. autoresearch keeps a change only if the stage's PASS row holds and no STOP row trips.

Reference anchors used below:
- baseline (Base SFT) 1030-acc = **0.786**; v1 (decorative redirect SFT→RL) = **0.651**; v2 RL wellformed collapse **0.615 → 0.010**.
- eval harness `scripts/eval_vllm_1030.py` (GSM8K 500 + MATH 500 + AIME 30, max_tokens=4096).
- counterfactual machinery reused for S2/S3: `scripts/harvest_redirect_cf.py` (`accept_redirect`, `lower_ci_diff`, `arm_rate`, `raw_yield_stats`, `gap_by_attribute`, `MIN_K`, `ACCEPT_MARGIN=0.5`) + `src/distill/causal_filter.py` (`accept_verify`).
- confidence/anchor logic `src/distill/confidence_label.py`; teacher gen `src/distill/teacher_conditional.py`; assembly `src/distill/assemble_sft.py`.
- grading `src/training/rewards._check_correctness` (answer-blind).

A metric is reported with its **value, the threshold, and the lower bound of a one-sided 95% CI** wherever a "real effect" claim is made (reuse `lower_ci_diff`); single-point means without a CI never satisfy a PASS row that asks for a causal effect.

---

## §A Experiment stages (pipeline → what is produced → who consumes it)

| Stage | Compute | Produces | Gate doc §  |
|-------|---------|----------|-------------|
| 0. Pre-gate (yield projection) | CPU only (counts on a 200-problem dev slice, mocked teacher OFF/real teacher on small N) | projected redirect+verify yield | §E pre-gates |
| 1. Build (teacher distill) | **CPU/network** (TRAPI GPT-5.4) + short vLLM rollout for anchors/CF arms | causally-filtered SFT parquet | §B SFT-time |
| 2. SFT | **H100 only** | `confidence_rv_sft` checkpoint | §C post-SFT eval |
| 3. RL (SFT→RL) | **H100 only** | RL checkpoint | §D RL-time |

The student N=8 rollout (anchors), teacher gen, and CF arm regen are the only places a GPU touches *build*; they are short (decode-only) and can run on the SFT H100 node opportunistically, but the **teacher call itself is CPU/network** (TRAPI) and is the natural place to run while waiting for an H100 slot.

---

## §B SFT-time metrics (build quality — gate BEFORE launching SFT)

These are computed on the assembled parquet + build logs; no model training needed. They protect against the v8 *decorative* failure (volume without causality) and the §9 *thin-bucket* risk.

| ID | Metric | PASS | STOP (abandon this build) | Verify sketch |
|----|--------|------|---------------------------|---------------|
| B1 | **Redirect demos surviving causal filter** (`accept_redirect`=True, easy/medium) | **≥ 600** | < 250 → STOP (insufficient causal signal; do not SFT) | `python scripts/build_confidence_redirect_verify_sft.py --report-yield` → count rows `action=="redirect" & accepted`; cross-check vs `raw_yield_stats` |
| B2 | **Verify demos surviving no-harm-confirm filter** (`accept_verify`=True, easy/medium, `prefix_correct=True`) | **≥ 300** | < 120 → STOP | same script, `action=="verify" & accepted` |
| B3 | **Causal acceptance rate** kept/candidates (redirect arm) | **≥ 0.25** | < 0.10 → STOP (teacher producing decorative meta; band/prompt broken) | accepted_redirect / candidate_redirect from build log |
| B4 | **Confidently-wrong sub-bucket count** (emitted-conf ≥ 0.5 stated by teacher but prefix wrong → redirect) — the §9 rare-but-most-valuable case | **≥ 80** (report value either way) | (no hard STOP; thin bucket is *reported*, S2-cw claim downgraded to "underpowered") | count rows where `conf_target < 0.5` AND teacher `conf_emitted` near band AND `action=="redirect"` |
| B5 | **Demo calibration sanity** — Spearman ρ between teacher `conf_emitted` and `conf_target` across accepted rows (must track, §3 truthfulness) | **ρ ≥ 0.6** AND 0 rows fail the same-side-of-0.5 assertion | ρ < 0.3 → STOP (teacher confidence decoupled from student → S1 untrainable) | `parse_teacher_output` already asserts same-side; report ρ over accepted rows |
| B6 | **Hard-row purity** — freshly-distilled redirect/verify demos built on `difficulty==hard` | **== 0** (invariant) | any > 0 → STOP (PG0 fact #2 violated; redirect hurts hard) | assert in `assemble_sft.mix`; count hard distilled rows |
| B7 | **Mix ratio** distilled / total | **∈ [0.15, 0.30]** | outside band → STOP (re-mix, do not SFT) | `mix(...)` enforces; assert on final DataFrame |
| B8 | **Prefix loss-mask coverage** — fraction of distilled rows whose `[prompt]+[wrong_prefix]` tokens are masked to −100 | **== 1.0** | < 1.0 → STOP (would teach the wrong prefix; v8-class leak) | `loss_mask_for_row` over all rows; assert no unmasked prefix token |

**B-gate (build PASS):** B1∧B2∧B3∧B5∧B6∧B7∧B8 all PASS. B4 is reported, not gating. Failing B-gate ⇒ do **not** spend the SFT H100; fix the band/teacher prompt/filter and rebuild.

---

## §C Post-SFT eval metrics (does the model self-emit useful confidence?)

Run after Stage-2 SFT, **before** RL. Eval on the 1030-set. These map 1:1 to design S1/S2/S3-static. Compare against two stored baselines: **base 0.786** and **v8-decorative SFT** (for ECE direction).

| ID | Metric | PASS | STOP | Verify sketch |
|----|--------|------|------|---------------|
| C1 | **Meta-ON accuracy** (1030, max_tokens=4096) | **≥ 0.786** AND **≥ 0.651** | < 0.651 → STOP (regressed below v1; forming-collapse at SFT) | `eval_vllm_1030.py --model_path <ckpt> --model_name confidence_rv_sft --output_dir <out>` then acc over 1030 |
| C2 | **ECE of emitted confidence vs correctness** (S1) | **ECE ≤ v8-decorative ECE − 0.03** (strict drop) and ECE ≤ 0.20 | ECE ≥ v8 ECE → STOP (no calibration gained over decorative) | bin `<|meta|>` `confidence:` (CONF_RE) into 10 bins, ECE vs per-problem correctness on 1030 |
| C3 | **Confidence-bin monotonicity** (S1) — Spearman ρ(emitted-conf-bin, empirical pass-rate) | **ρ ≥ 0.7** | ρ < 0.4 → STOP (confidence does not track correctness) | per-bin pass-rate from C2 binning; Spearman |
| C4 | **redirect_causal_rate** (S2) — on held-out **low-conf-wrong** problems where the model fires REDIRECT, fraction where R flips wrong→right while no-redirect control stays wrong; gated on `lower_ci_diff(R, B') ≥ ACCEPT_MARGIN=0.5` aggregate | **aggregate lower-CI(R−B′) > 0** AND rate **≥ 0.30** | lower-CI ≤ 0 → STOP (redirect not causal; decorative) | replay `accept_redirect`/`raw_yield_stats` on held-out fired-redirect problems |
| C5 | **verify_noharm_rate** (S2) — on held-out **high-conf-right** problems where the model fires VERIFY, fraction confirmed (V lower-CI ≥ plain control AND stays ≥ `conf_target − slack`); flip-from-wrong NOT counted as verify | **≥ 0.85** AND aggregate verify-arm acc not below plain control lower-CI | < 0.70 → STOP (verify regresses correct answers — vacuous/harmful check) | replay `accept_verify` on held-out fired-verify problems |
| C6 | **Action-appropriateness** — of fired meta blocks, fraction whose action matches the band (redirect on low-conf-wrong, verify on high-conf-right) | **≥ 0.80** | < 0.5 → STOP (action decoupled from confidence) | join emitted action vs measured band on held-out |
| C7 | **Meta-emit rate** (anti-decorative guard, two-sided) | **∈ [0.10, 0.70]** | < 0.05 (under-form, v1) OR > 0.85 (over-form/decorative, v8) → STOP | count `<|meta|>` blocks / problems on 1030 |

**C-gate (post-SFT PASS, required to spend RL H100):** C1∧C2∧C3∧C4∧C7 PASS, and C5∧C6 PASS *if* their fired-sample n ≥ MIN_K-scaled power (else reported underpowered, not blocking). C4's causal lower-CI>0 is the hard gate — a high emit rate (C7) with C4 lower-CI≤0 is exactly the decorative trap and **fails** regardless of C1.

---

## §D RL-time metrics (does the metacognition SURVIVE RL?)

Run during/after Stage-3 SFT→RL. The dominant historical failure is **emission/forming collapse** (v2 wellformed 0.615 → 0.010). The gate is **utility-conditioned**, not emit-rate-conditioned (a high emit rate with no causal benefit is the v8 trap).

| ID | Metric | PASS | STOP (kill the RL run) | Verify sketch |
|----|--------|------|------------------------|---------------|
| D1 | **meta_survival / wellformed_rate** over RL steps (well-formed `<|meta|>…<|/meta|>` closeable blocks) | **stays ≥ 0.40** for the whole run; final ≥ 0.40 | drops **below 0.10** at any step, OR drops by **> 0.50 absolute from its own step-10 value** within 30 steps (the v2 collapse signature) → STOP | wandb `wellformed_rate` curve; alarm on the collapse rule |
| D2 | **Accuracy delta vs SFT init** (1030 meta-ON) | **final acc ≥ SFT-init acc** AND **≥ 0.786** | acc < 0.786 at convergence → STOP (RL ate accuracy; v1 mode) | `eval_vllm_1030.py` on RL ckpt vs SFT ckpt |
| D3 | **Utility-conditioned causal gate** (the real S3) — on held-out fired-meta problems, meta-ON vs **same prefix with the meta block ablated**, aggregate `lower_ci_diff(meta_on, meta_ablated) > 0` | **aggregate lower-CI > 0** | lower-CI ≤ 0 at convergence → STOP (meta not causal; decorative survived) | counterfactual meta-block ablation replay (§4-style) on RL ckpt |
| D4 | **acc_with vs acc_without** — accuracy on problems where meta fired vs where it did not (descriptive co-signal; NOT the gate, per s3b lesson that acc_with/without is confounded) | report both; flag if acc_with < acc_without − 0.05 | (no standalone STOP — D3 is the causal gate; D4 only annotates) | split 1030 by fired/not-fired, report acc each |
| D5 | **Over-forming guard** — meta-emit rate at RL convergence | **∈ [0.10, 0.70]** | → 0 (under-form) or → ~1.0 with D3 lower-CI ≤ 0 (decorative over-form) → STOP | wandb emit-rate curve + D3 join |

**D-gate (RL PASS = experiment "works as intended"):** D1 (no collapse) ∧ D2 (acc ≥ baseline) ∧ D3 (causal lower-CI>0). D4/D5 annotate. This is the full north-star: confidence self-emitted (C-gate) AND the redirect/verify decision is causally raising accuracy AND it survives RL (D3∧D1).

---

## §E GO/STOP pre-gates BEFORE GPU spend (PG0 philosophy)

Mirror PG0: *project the yield on CPU and STOP before training if it cannot clear the bar.* Each pre-gate is cheap (dev slice / mocked or tiny real teacher N).

| Gate | When | GO condition | STOP action |
|------|------|--------------|-------------|
| **PG-build** (before Stage-1 full teacher spend) | after building on a **200-problem dev slice** with a **small real teacher N (e.g. 50 problems)** | projected full-corpus B1 ≥ 600 AND B2 ≥ 300 AND B3 ≥ 0.25 (linear-scale the dev-slice accept rate to the full pool) | projected B1 < 250 → do NOT run the full teacher build; revisit band/prompt (this is the PG0 yield≈141≪1500 STOP applied to the distill pipeline) |
| **PG-sft** (before Stage-2 H100) | after full Stage-1 build | **B-gate PASS** (§B) | any B STOP row → do NOT submit SFT yaml |
| **PG-rl** (before Stage-3 H100) | after Stage-2 SFT eval | **C-gate PASS** (§C), critically C4 causal lower-CI>0 | C-gate fail → do NOT submit RL yaml; the SFT did not form causal meta, RL will only collapse it |

The ordering is strict: **PG-build → PG-sft → PG-rl**. No GPU H100 is requested until PG-sft passes; no RL H100 until PG-rl passes. autoresearch treats a STOP at any pre-gate as "discard this change, keep the prior best."

---

## §F Manual amlt launch steps (no daemon; idle-suspend reality)

There is **no babysitter daemon**; nodes idle-suspend and preempt, so each stage is a **manual `amlt run` then a manual poll/relaunch**. Steps:

1. **Stage-1 teacher build (CPU/network — TRAPI, run locally or on a CPU box; NO GPU):**
   ```
   /home/v-seungplee/miniconda3/envs/metaprobe/bin/python \
     scripts/build_confidence_redirect_verify_sft.py \
     --out data/confidence_rv_sft.parquet --report-yield
   ```
   TRAPI is CPU/network (concurrent + retry via `generator.get_trapi_client`); the short student/CF vLLM rollouts can piggyback an H100 slot if one is up, else run them on the same node before SFT. Then evaluate the §B B-gate on the produced parquet/log. **Do not proceed unless B-gate PASS.**

2. **Snapshot code → HF** (the H100 jobs pull a code tarball, per `CLAUDE.md`): build/push `code_snapshots/metacognition.tar.gz`, record the new `CODE_TAR_REVISION` into the SFT/RL yamls (mirror the `CODE_TAR_REVISION` field in `h100std_triobj_dcpo_v4_s2b.yaml`).

3. **Stage-2 SFT (H100 only):** write `h100std_confidence_rv_sft.yaml` (copy an existing H100 SFT yaml, point `--config` at a `configs/confidence_rv_sft_*.yaml` with `data=data/confidence_rv_sft.parquet`, init `v8_meta_inside_strict_sft`, `teacher_kl.enabled=false`). Launch:
   ```
   amlt run h100std_confidence_rv_sft.yaml :confidence_rv_sft -d "confidence redirect/verify SFT"
   ```
   (the remote entry is `scripts/launch_sft_remote.sh <config>`; it refuses duplicate launches).

4. **Poll Stage-2 (idle-suspend/preempt reality):** `amlt status <job>`; on preempt, **manually re-`amlt run`** (no auto-resume). When the SFT ckpt lands on HF, run the §C eval:
   ```
   /home/v-seungplee/miniconda3/envs/metaprobe/bin/python scripts/eval_vllm_1030.py \
     --model_path <sft_ckpt> --model_name confidence_rv_sft \
     --output_dir results/confidence_rv_sft_eval --max_tokens 4096 --tp_size 4
   ```
   Evaluate the C-gate. **Do not proceed to RL unless C-gate PASS (esp. C4 lower-CI>0).**

5. **Stage-3 RL (H100 only):** write `h100std_confidence_rv_rl.yaml` (copy `h100std_triobj_dcpo_v4_s2b.yaml`; init from the Stage-2 SFT ckpt; keep the validated emission-survival knobs — `format_neg 0.2` + per-token `w_emit 0.4` — so D1 does not reopen the abstention escape). Launch `amlt run h100std_confidence_rv_rl.yaml :confidence_rv_rl`. Enable wandb rollout logging (`DCPO_WANDB_ROLLOUTS=1`) so D1/D5 curves stream.

6. **Poll Stage-3 with the D1 collapse alarm:** watch wandb `wellformed_rate`; if the D1 STOP rule trips (below 0.10, or >0.50 drop in 30 steps), **kill and discard**. On preempt, manually relaunch (no daemon). At convergence run the §D D2/D3/D4 evals on the RL ckpt; D-gate PASS = experiment confirmed.

Idle-suspend/preempt notes: every stage is launch-then-manual-poll; there is no auto-resume, so a preempted SFT/RL job must be re-`amlt run` by hand, and the TRAPI build (step 1) is the only stage safe to run fully off-cluster.

---

## §G One-line summary of the gates

Build must yield **≥600 redirect / ≥300 verify causally-filtered demos** (B-gate) → SFT must **self-emit calibrated confidence (ECE down, C4 redirect-causal lower-CI>0, acc ≥0.786)** (C-gate) → RL must **not collapse emission (≥0.40, no v2-signature) and keep meta causally useful (D3 lower-CI>0, acc ≥0.786)** (D-gate). A STOP at any pre-gate (PG-build/PG-sft/PG-rl) discards the change before GPU spend.
