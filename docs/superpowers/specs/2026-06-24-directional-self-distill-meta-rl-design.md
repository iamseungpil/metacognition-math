# Directional Self-Distillation Meta-RL — Design Spec (2026-06-24)

## North-star
Model self-emits STUDENT-calibrated metacognition (redirect/verify/abstain) and it RAISES accuracy
**only when useful**. We reinforce useful metacognition via a **dense, per-token self-distillation
REWARD** (counterfactual gold-vs-decoy), routed to the meta region — capturing self-distillation's
fine-grained credit while **avoiding the epistemic-suppression** that KL-imitation self-distillation
(SDPO, arXiv 2603.24472) causes. Novelty = (1) epistemic-PRESERVING self-distillation (reward, not KL),
(2) dense per-token credit → rollout efficiency (RLT 2506.08388 argument applied to metacognition).

## Key prior results (grounding)
- PMI (always-on, emit 0.99): mediocre Δ~+0.01, hard negative. Collapse baseline.
- CF (counterfactual c_with−c_without, selective emit 0.20): **gs100 Δ+0.040, McNemar z≈3.05 SIGNIFICANT**, hard +0.052. Current best.
- decoy-DiD prereq (output-scored, token-level): **PASS within-AUC 0.685, t18** (vs sum-diluted 0.578).
- SDPO line-by-line: our gold-teacher T+ == SDPO's epistemic-suppression mechanism EXACTLY; our `conf_free` already routes around it. Reference is all input-conditioned KL; output-scored decoy-DiD exists only in ours.

## Two directions (both DCPO independent ADDITIVE heads, both on functional SFT v8_rv_functional_sft)
- **gm (gold|meta)**: `R_meta = AGG_divergent-answer-tokens[ (logp(gold_ans_tok|body+meta) − logp(gold_ans_tok|body+placebo)) − (same for decoy_ans) ]`. Scores the ANSWER under the model's own meta. Scalar → broadcast to META (or weighted by mg per-token mask). Validated (prereq PASS).
- **mg (meta|gold)**: `δ_tok = logp(meta_tok|prompt+gold_hint+body) − logp(meta_tok|prompt+decoy_hint+body)` per META token. **Routed PER-TOKEN** to each meta token (dense). gold/decoy injected ONLY at reward time (leak-free at inference).

## Structural decision (LOCKED)
- **ADDITIVE per-token (primary)**, NOT multiplicative-on-correctness. Multiplicative (RLSD) shackles meta to correctness SIGN → punishes a gold-reaching meta on a wrong rollout (backwards) + zero gradient when group correctness is flat. Additive gives the meta an INDEPENDENT signal.
- **Multiplicative (RLSD form) = S5 ABLATION only** (`Â_t = Â_corr·((1−λ)+λ·w_t)`), to empirically demonstrate the shackling harm.
- **AGG = `mean_min` (RLT, mean+α·min) over the RELEVANT token set** (gm: divergent answer-value tokens only — excludes structural `\boxed` tokens that cause dilution; mg: per-token, no agg needed or max for the scalar variant). NOT plain mean (dilution), NOT plain max for the REWARD (gameable — one token). Max is for the EVAL/detection only.
- **R3 routing**: R_correctness → ALL response tokens (whole solution); R_meta_dir → META tokens (additive). anchor_norm to keep heads balanced (R3 correctness-dominance guard).
- **over-emission penalty** (reuse CF `dcpo_w_over`): selectivity = epistemic preservation = CF's +0.040 driver.

## Decoy
`_rule_based_decoy(gold, seed, checker=_check_correctness)` — deterministic near-miss (±1/±2/sign/fraction), guaranteed ≠gold and not math-equivalent. Same decoy for gm/mg. K=1 default; K=3-averaged optional for robustness.

## Epistemic measurement (REQUIRED — this is the decision metric, not just accuracy)
Adapt SDPO `check_epistemic_tokens.py` (words: wait/hmm/perhaps/maybe/actually/alternatively/seems/might/likely/check) + our meta-emission rate. Log every 50 steps. A direction that RAISES accuracy by SUPPRESSING epistemic (emission↓, epistemic-words↓) is REJECTED even if accuracy rises.

## Files to modify (surgical, BACKWARD-COMPATIBLE — must NOT break running pmi/cf_group paths)
1. **NEW `src/training/dcpo_directional.py`**: `compute_directional_meta_reward(rows, direction, agg, per_token, decoy_seed)` — pure-ish, CPU-unit-testable core (logp scoring delegated). gm + mg.
2. **`src/training/verl_sdc.py`**: add `dcpo_rmeta_source ∈ {decoy_did_gm, decoy_did_mg}` branches NEXT TO pmi/cf_group (do not alter existing branches). Default unchanged. + epistemic-trend logging. + over-emission penalty reuse.
3. **`src/training/dcpo_region.py`**: R3 routing (R_corr on response_mask) + per-token mg routing — both OPTIONAL/flagged so existing compose is byte-identical when off.
4. **`src/eval/decoy_did_pregate.py`**: add mg-direction scoring (meta-token δ, max within-AUC) alongside existing gm.

## Experiment stages
- **S0 (signal eval, both, functional SFT, 1 GPU each)**: gm = DONE (PASS 0.685). mg = build + run (meta-token max within-AUC). Gate ≥0.60. Compare gm vs mg signal.
- **S1 (implement)**: dcpo_directional + verl_sdc wiring + R3 + epistemic logging. **TDD** (boilerplate→0, gold-favor→+, multiplicative-shackle demo, per-token routing correctness, backward-compat: existing pmi/cf unchanged).
- **S2 (smoke)**: short run — R_meta flows to META (H6-style), emit stays selective, epistemic logged, no collapse.
- **S3 (RL, gm vs mg, same functional SFT, 300)**: every 50 steps accuracy Δ (held-out) + epistemic trend.
- **S4 (decision)**: net-positive accuracy × epistemic preserved. Structure held constant (additive) → isolates DIRECTION (gm vs mg).
- **S5 (ablations, optional)**: (a) multiplicative-RLSD form (demonstrate shackling); (b) rollout-efficiency n=8 vs n=4 with dense reward (RLT efficiency claim).

## Baselines (running, keep to 300)
PMI (collapse control) · CF (Δ+0.040 significant, current best).

## S0 RESULT (2026-06-24) — DIRECTION DECIDED: gm wins, mg dropped
Signal eval on functional SFT (same 7 mixed groups): **gm within-AUC 0.685 (PASS) vs mg within-AUC 0.391 (FAIL, below chance 0.5)**. mg (meta|gold, gold-conditioned-teacher direction) does NOT discriminate useful metas — empirically confirms SDPO (gold-teacher harmful) + contrastive-teacher-confound (meta|gold = answer-coupling, weak). **Decision: use gm direction only.** mg/teacher dropped.

## FINAL 2-ARM EXPERIMENT (structure isolation, gm direction FIXED)
Both arms use the **gm contrast** (`logp(gold_ans|body+meta)−logp(gold_ans|body+placebo)) − (decoy)`, scored AFTER `<|/meta|>`, token-level mean_min over divergent answer tokens). They differ ONLY in STRUCTURE:
- **Arm DCPO (additive)**: `dcpo_rmeta_source=decoy_did_gm` — gm contrast → R_meta independent head → ADDED to META region (R3). Decoupled from correctness.
- **Arm RLSD (multiplicative)**: gm contrast → per-row weight `w = exp(sign(A_corr)·clip(gm))` → MULTIPLIES correctness advantage on META tokens: `Â_t = Â_corr·((1−λ)+λ·w)`. Shackled to correctness sign.
- Prediction: DCPO wins (multiplicative punishes good meta on wrong rollouts). Empirically test it. Both functional SFT, 2 nodes, measure accuracy Δ + epistemic.
- NOTE: this RLSD differs from old SDC — old = multiplicative + mg-direction (failed 0.391) + mean + KL-leak; this = multiplicative + gm-direction (0.685) + token-level. So it isolates STRUCTURE alone.

## Launch infra (both arms, reuse CF yaml h100std_triobj_dcpo_v4_cfgroup.yaml pattern)
resume-from-HF (pull_resume_ckpt.py, resume_mode=auto) + periodic HF push (push_ckpts_to_hf.py --interval 90) + verl log capture (>/scratch/logs/verl_main.log) + gpu_keeper + LOCAL_RANK=0 + epistemic/DCPO trend logging. Non-interactive `amlt run -y`.

## Success criteria
A direction (gm or mg) that yields held-out Δ>0 (McNemar significant) WHILE preserving epistemic verbalization (emission selective, epistemic-words not suppressed) — beating CF's +0.040 OR matching it with cleaner mechanism. If both suppress epistemic → reward-form is wrong, revert to CF + pivot.
