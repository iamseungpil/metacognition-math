# E.9 — Binned-Confidence-Injection RLVR (BCI-RLVR)

**Date:** 2026-06-05
**Branch:** ctsd-phase-c
**Status:** design approved (user), pending spec review

## 1. Motivation

The best meta-preserving RLVR run to date is **e4_baseline** (`mode=VANILLA_GRPO`,
`sdc_enabled=false`, reward = `correctness` only): **acc 0.786** on 1030@16k k=8.
But its calibration is bad in the *under*confident direction:

```
e4_baseline: acc 0.786 | mean_confidence 0.29 | ECE 0.557 | overconf_rate 0.0015
```

The model is accurate but verbalizes confidence ~0.29 — it does not trust itself.
ECE 0.557 is almost entirely this under-confidence gap. The north-star requires
Meta-CoT to beat Base SFT on accuracy **AND** improve calibration; accuracy is met,
calibration is not.

Inference steering (E.6b conf_down +5.5pp; E.7 conf_adaptive +4.3pp) proved that
conditioning the meta on a confidence target *causally* changes reasoning, but it
shifts confidence one-directionally (over→underconfident) and does not calibrate.
E.4 also showed inference effects do **not** automatically transfer to RL (the
contrastive teacher helped at inference, hurt in training: 0.714 < 0.786). So a
calibration mechanism must be (a) trained, and (b) A/B-verified against
correctness-only.

## 2. Mechanism

For each problem, verl generates `rollout.n = 4` samples (the GRPO group),
`gen_batch.repeat(n, interleave=True)` → layout `[p0s0, p0s1, p0s2, p0s3, p1s0, …]`,
so **within-group bin index = sample_index % n**.

We **force a binned confidence statement at the start of each rollout's response**:
sample with within-group index `i` gets its response **seeded** with
`<|meta|>\nconfidence: c_i\n<|/meta|>\n`, where `c_i ∈ {0.2, 0.4, 0.6, 0.8}` for
`i ∈ {0,1,2,3}`. The model then generates the reasoning + answer conditioned on that
confidence. The group is thus a **confidence sweep**: every problem is attempted at
each confidence level.

The seeded confidence tokens are placed **in the trained response region** (not the
prompt). The proper-scoring-rule reward `outcome_calibration_reward` reads the meta
confidence and scores `Brier(c_i, is_correct) = 0.3·(1 − (c_i − correct)²) − 0.15`
(+ revision credit). The rollout whose injected bin matched the outcome
(high-c + correct, or low-c + wrong) gets the highest calibration advantage; because
the seeded confidence is on-policy-scored and trained, REINFORCE raises the policy's
probability of emitting the *calibrated* confidence for that problem type.
GDPO per-reward normalization keeps `correctness` the dominant head so accuracy is
preserved.

**Why injection over a plain proper-score reward on self-emitted confidence:**
e4_baseline self-emits a narrow conf band (~0.29), giving a weak calibration
gradient. The forced sweep guarantees the full confidence range is sampled on every
problem (variance reduction), so the proper-score always has signal to select from.

**Single-phase, not two-phase.** Unlike the A.3 accuracy-injection (which regenerated
to place a meta block at the max-entropy mid-reasoning position), calibration only
needs "state confidence, then solve." So we seed the confidence at response-start in
**one** generate call — cheaper, and a simpler repack. Entropy-positioned two-phase
is noted as a future variant, not built here.

## 3. Architecture & isolation (HARD constraint)

> The new code MUST NOT mix with or break existing SDC/self-distill modes or prior
> training. Every other mode must stay byte-identical.

Isolation is achieved by three properties:

1. **Gated, additive reward mode.** New `REWARD_CONFIGS["BCI_RLVR"] =
   {funcs:[correctness_reward, outcome_calibration_reward], weights:[1.0, 0.5],
   keys:["correctness","outcome_calibration"]}`. Purely additive dict entry; no
   existing entry touched. `outcome_calibration_reward` is **imported** from the
   protected `rewards.py` (no modification).

2. **Gated rollout wrap — `fit()` is never copied or modified.** The seeding happens
   in a new `SDCRayPPOTrainer` method that **wraps**
   `self.async_rollout_manager.generate_sequences` *only* when
   `algorithm.sdc_force_inject_conf == true`. When the flag is false (all existing
   modes), the wrap is not installed → the rollout path is byte-identical. We do
   **not** override verl's `fit()` (the shared hot path).

3. **Frozen-release separation.** Live jobs (E.8, E.4) run from frozen GitHub release
   tarballs (`CODE_TAR_REVISION`), so new local code cannot affect them. Only a newly
   cut release used by the E.9 YAML carries the change.

A new flag name `sdc_force_inject_conf` is used (distinct from the legacy
`sdc_force_inject` whose hard-block guard at verl_sdc.py:1434 stays intact for the
ROD_MQ_CONTRAST_INJECT path).

## 4. Components

| unit | file | change | locally testable |
|---|---|---|---|
| binned conf segment + per-index planner | `src/training/meta_inject.py` | add `conf_inject_template(c)` + optional `confidences` arg to a new `plan_conf_prefixes`; existing funcs untouched | **yes** (pure, numpy) |
| `BCI_RLVR` reward mode | `src/training/verl_sdc.py` REWARD_CONFIGS | additive dict entry | yes (dict membership) |
| gated generate_sequences wrap + response repack | `src/training/verl_sdc.py` SDCRayPPOTrainer | new method, installed only under flag | **no — node-smoke only** |
| config | `configs/verl_e9_bci_rlvr_h200_4x4k.yaml` | copy e4 base; `mode=VANILLA_GRPO` reward heads via BCI_RLVR; `sdc_force_inject_conf=true`; bins | n/a |
| amlt job | `h200_e9_bci_rlvr.yaml` | 1-step node-smoke gate → full 300-step run → push | n/a |

### 4.1 Response repack (the node-smoke crux)
After the (single) generate call on the conf-prefixed prompts, the seeded
`<|meta|>confidence: c_i<|/meta|>` tokens live at the tail of `batch["prompts"]`.
The repack moves them into the head of `batch["responses"]` so they are trained:
`new_prompt = original_prompt`, `new_response = conf_seed ⊕ continuation`, with
`attention_mask`, `position_ids`, `response_mask` rebuilt consistently and re-padded
to the rollout width. Invariants:
- (I1) the seeded conf block is inside the scored `responses` (so
  `outcome_calibration_reward` parses it and it receives advantage/gradient).
- (I2) no double-counting: the conf tokens appear exactly once, in the response.
- (I3) shapes/padding match what `old_log_prob`/advantage expect (else PPO crashes).

## 5. Experiment

- **Control:** e4_baseline (VANILLA_GRPO, correctness-only, no inject) — already
  evaluated: 0.786 / ECE 0.557.
- **Treatment (E.9):** BCI_RLVR + binned conf seed + `outcome_calibration` head.
- Same base: v8_strict cold start, 4×H200, 300 steps, train max_response 4096,
  eval 1030@16k k=8 (matches all prior evals).
- Optional later 3rd arm (inject + correctness-only) isolates the proper-score head.

### 5.1 Pre-registered success criteria
- **Primary:** ECE < 0.35 (down from 0.557, significant) **AND**
  accuracy ≥ 0.786 − 1.5pp (correctness preserved).
- **Guards:** meta emission ≥ 95% (no collapse); mean_confidence moves 0.29 → toward
  acc (~0.78); log `train/inject_close_rate` and `train/seed_conf_hist`.

## 6. Risks & mitigations

| risk | mitigation |
|---|---|
| repack wrong (verl-specific, untestable locally) | 1-step node-smoke MUST pass before full run; assert I1–I3 on-node; smoke checks the seeded conf token receives nonzero gradient and reward parses it |
| seeding a fixed conf block hurts accuracy (A.3: full-block < marker) | keep block minimal (conf-only, no prose); accuracy guard ≥ 0.786 − 1.5pp aborts |
| forced (off-policy) conf token training bias | verl recomputes `old_log_prob` over the spliced response → standard guided-REINFORCE; importance ratio internally consistent; `use_kl_in_reward=false` in base config so no over-penalty |
| 4 bins is coarse | start n=4 for e4 comparability; note n=8 (2× compute) as a follow-up if the sweep is too coarse |
| isolation regression | unit test asserts every non-BCI mode's funcs/weights/keys unchanged; wrap installed only under the new flag |

## 7. Out of scope (YAGNI)
- Two-phase entropy-positioned injection.
- n>4 bins.
- The 3rd ablation arm (inject + correctness-only).
- GFN/KL distribution-matching calibration.
