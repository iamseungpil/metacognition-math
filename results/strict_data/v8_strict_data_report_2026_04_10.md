# V8 Strict Data Report (2026-04-10)

## Goal

Rebuild the current paired V8 SFT corpora into a stricter subset that supports
a cleaner scientific comparison:

1. `meta` data must encode strong `verify` and `redirect` behavior.
2. `base` data must remove meta and present direct one-pass reasoning.
3. both sides must stay paired on the exact same problem slice.

## Outputs

1. `data/v8_meta_inside_strict.parquet`
2. `data/v8_base_matched_strict.parquet`
3. `results/strict_data/v8_strict_build_summary.json`
4. `results/strict_data/v8_strict_validation_summary.json`

## Strict Rules

### Common

1. exactly one user message and one assistant message
2. assistant uses a single `<think> ... </think>` envelope
3. final answer is outside think as `The answer is $\boxed{...}$`
4. paired meta/base rows must keep the same prompt and the same final boxed answer

### Meta Verify

1. exactly one non-empty meta block
2. confidence must be at least `0.65`
3. meta must contain a concrete verification action
4. redirect-style diagnosis text is not allowed

### Meta Redirect

1. one or two non-empty meta blocks
2. minimum confidence must be at most `0.45`
3. meta must contain both:
   - a diagnosis of why the current route is weak
   - an explicit switch / redirect signal
4. duplicated redirect prefixes are rejected

### Base

1. no meta tags remain
2. no route-management leakage such as:
   - `A first thought is`
   - `At first glance`
   - `I might try`
   - `I should switch`
   - `study_need:`
3. verify rows become direct solutions without the explicit check block
4. redirect rows keep only the post-switch solution path

## Result

### Build

1. input rows: `6329`
2. kept rows: `4264`
3. kept verify rows: `2065`
4. kept redirect rows: `2199`

### Validation

1. paired row count matched
2. all meta rows passed strict scenario validation
3. all base rows passed no-meta / no-route-leak validation
4. final validator status: `passed = true`

### Final Structure Stats

1. meta rows: `4264`
2. base rows: `4264`
3. meta empty-meta rows: `0`
4. base meta-leak rows: `0`
5. redirect mean minimum confidence: `0.2939`
6. verify mean confidence: `0.8113`

## Recommended Next Step

1. train fresh paired SFT models on the strict corpora
2. compare them against the previous SFT results
3. only then restart matched RL
