# Matched-data Base for Clean Meta Comparison (Approach A)

**Date:** 2026-06-29  **Goal:** Remove the training-data confound so "does meta help?" is measured cleanly.

## Problem (audited)
base-vs-meta comparison is confounded: base RL = `redirect_base` (2935, **gsm8k 0%**, hard); meta RL = `meta_mix` (5344, **gsm8k 34%**); problem overlap **0%**. The accuracy gap (0.829 vs 0.786) AND the degeneration difference (base 0% vs meta 57% at 16k) may be data artifacts, not meta effects.

## Meta lineage (established)
`Qwen3 → v8_meta_inside_strict_sft (4264 matched, meta) → rv_functional SFT (1763, meta+conf/verify; init=v8_meta_inside) → DCPO RL (meta_mix 5344) → {pmi, pmi-shift, gm, cf}`

## Design: build the meta-removed twin of base
Build a base whose ONLY difference from the meta models is the absence of the meta mechanism — identical SFT problems, RL problems, and init lineage.

### §1 Data ("change only the solution")
- **base SFT-2 data** = `rv_redirect_verify_functional.parquet` (1763), parse `messages` (JSON string), STRIP `<|meta|>…<|/meta|>` blocks + conf/verify-only turns → same problems, same final answers, meta removed. Validate each stripped solution still reaches the gold answer; broken → public reference solution fallback or drop.
- **base RL data** = `verl_train_meta_mix.parquet` (5344) AS-IS — identical to meta RL.

### §2 Training (every non-meta hyperparam identical to meta)
- base SFT-1: reuse `v8_base_matched_strict_sft` (matched twin of v8_meta_inside; verify weights on HF, else re-SFT from pretrained on `v8_base_matched_strict.parquet`).
- base SFT-2: from SFT-1, SFT on §1 stripped data → `v8_base_rv_sft` (meta-removed twin of rv_functional).
- base RL: **correctness-only GRPO** on meta_mix from `v8_base_rv_sft`, `max_response_length=4096`, lr/batch/clip/kl/steps **identical to** `triobj_dcpo_v4_stage3b_h100_4x4k`, ~300 steps → `base_rv_grpo`. Tightest: same verl_sdc harness with meta reward OFF (w_meta=0, no meta routing/injection); fallback verl GRPO.

### §3 Evaluation (matched length blocks degeneration confound)
1030 (gsm8k/math500/aime) at **4k AND 16k**, math_verify graded; degeneration rate at 16k. Compare `base_rv_grpo@300` vs `pmishift gs300` (+ other 3 meta).

### §4 Success criteria
- **Primary (north-star):** clean base-vs-meta accuracy on identical data. meta≥base ⇒ meta helps (unconfounded); meta<base ⇒ genuine negative.
- **Secondary (degeneration confound):** if base-on-meta_mix degenerates ~like meta at 16k ⇒ degeneration is DATA-driven (confirmed). If not ⇒ meta mechanism contributes.

### §5 Cost / risks
~3-4 H100 jobs (base SFT-1/2 + base RL + eval). Risks: strip artifacts (validate + fallback); SFT-1 weight availability (re-SFT); **base RL hyperparams must match meta exactly** (else new confound).

## Out of scope
pmi/pmi-shift/gm/cf are NOT retrained. Capability/AIME push (harder data + long-length RL + TRAPI) = separate later phase (Approach C).

## Intent alignment
North-star = strengthen metacognition via priming-free self-distillation to RAISE accuracy. This experiment makes the "meta vs no-meta" test valid; autoresearch will watch that the meta RL signal (self-distilled contrast) is what drives any gap, not data.
