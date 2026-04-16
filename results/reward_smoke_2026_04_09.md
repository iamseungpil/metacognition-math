# Reward Smoke Log (2026-04-09)

## Scope

Files reviewed and patched:

1. `src/training/rewards.py`
2. `src/training/grpo_v2.py`
3. `src/training/verl_gdpo.py`
4. `src/training/verl_reward.py`
5. `configs/verl07_e21r.yaml`
6. `results/plan_metacot_v8_active_2026_04_09.md`

## Goal

Verify that the new confidence-centered controller (`E21R`) is internally consistent before any new RL launch:

1. healthy `redirect` examples should not score negative
2. healthy `verify` examples should not score negative
3. `E21R` must be selectable in TRL and veRL paths
4. legacy `E21` must remain reproducible

## Main fixes

### 1. Reward helper fixes

Updated `rewards.py` so that:

1. `I should check` no longer counts as uncertainty trigger
2. verify intent/execution regexes cover natural phrasing such as
   `cross-check`, `check my answer`, `Plug x=2 into ...`, `compare both sides numerically`
3. redirect diagnosis regexes cover natural failure analysis such as
   `wrong formula`, `ignored a constraint`, `conflated two cases`
4. redirect execution is no longer blocked only by `method_diff >= 0.35`

### 2. Launcher / mode wiring

Updated code so that `E21R` is actually launchable:

1. added `E21R` to `grpo_v2.py --mode` choices
2. added `E21R` to TRL GDPO gating
3. added `E21R` to veRL GDPO patch gating
4. added legacy `verl07_e21r.yaml` for the old custom-reward path

## Synthetic smoke results

### Redirect sample

Input pattern:

- low confidence
- diagnosis of weak route
- explicit next strategy
- post-meta execution of that strategy

Result after patch:

- `redirect_execution_reward_v2 = 0.6`

### Verify sample

Input pattern:

- high confidence / overcommit signal
- explicit verify intent
- actual substitution / cross-check in tail

Result after patch:

- `verify_execution_reward_v2 = 0.25`

### Natural-language checks

Also verified:

1. `I should check my answer with a quick cross-check` no longer triggers redirect penalty
2. `Let me start over and factor instead` can now count as redirect execution when followed by real solving

## 50-sample parquet smoke

Dataset:

- `results/eval_v8_E20a/eval_v8_meta_inside_E20a.parquet`
- first 50 rows

Observed means:

1. `confidence_revision = 0.1588`
2. `redirect_execution = 0.0000`
3. `verify_execution = -0.0010`

Observed counts:

1. `confidence_revision nonzero = 50 / 50`
2. `redirect_execution nonzero = 0 / 50`
3. `verify_execution nonzero = 20 / 50`

Interpretation:

1. early E20a slice behaves like verify-heavy easy math, not redirect-heavy hard math
2. reward distribution is no longer obviously broken on natural verify cases
3. redirect reward still needs validation on trigger-conditioned subset parquet, not on easy GSM-style rows

## Redirect subset parquet creation

Created:

1. `data/verl_train_redirect.parquet` — 2935 rows
2. `data/verl_val_redirect.parquet` — 326 rows
3. `data/verl_train_redirect_base.parquet` — 2935 rows
4. `data/verl_val_redirect_base.parquet` — 326 rows

Selection rule:

1. source: `v8_meta_inside_think.parquet` and `v8_base_matched_clean.parquet`
2. `scenario == redirect`
3. `difficulty in {medium, hard}`
4. row-aligned split so meta/base use the exact same question slice

Observed composition:

1. train difficulty = `medium 1813 / hard 1122`
2. val difficulty = `medium 198 / hard 128`
3. trigger = mostly `anomaly`

## 50-sample redirect-subset smoke

Dataset:

- `data/verl_train_redirect.parquet`
- first 50 rows, scored against the original V8 assistant completions

Observed means after reward cleanup:

1. `confidence_revision = 0.4359`
2. `redirect_execution = 0.2310`
3. `verify_execution = 0.0000`

Observed counts:

1. `redirect positive = 32 / 50`
2. `redirect negative = 9 / 50`

Interpretation:

1. the redirect subset now activates redirect reward as intended on most samples
2. remaining negatives come from weak SFT-style meta that declares a redirect without a strong diagnosis/execution trace
3. this is a data-quality issue at the margin, not a launch-blocking wiring bug

## Current decision

Do **not** launch new redirect RL until node/runtime allocation is re-checked.

Remaining gates:

1. freeze wrong-high-confidence baseline metric for Phase 2
2. re-check active node/runtime state before launch
3. optionally inspect the remaining redirect-negative SFT samples for data cleanup
