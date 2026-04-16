# Behavior-Uncertainty Analysis Plan

## Intent

Use saved eval artifacts to measure whether Meta-CoT changes behavior, not just accuracy.

## Hypothesis

If the controller is real, then compared to the paired base lane it should show:

1. more confidence-conditioned `verify`
2. more low-confidence or anomaly-conditioned `redirect`
3. more explicit diagnosis on hard problems
4. behavior changes that are strongest on difficult subsets such as `aime2024`

## Verification Method

Primary analysis bundle:

1. behavior counts by benchmark and difficulty
2. confidence statistics and confidence revision
3. redirect-after-low-confidence rate
4. verify-after-high-confidence-overcommit rate
5. qualitative AIME traces
6. optional entropy / information-allocation diagnostics

## Interpretation

1. More meta text without route change is weak evidence.
2. More route change without correctness recovery may indicate instability.
3. Redirect plus later correction on hard problems is the strongest behavioral evidence.

## Scope Boundary

This directory does not own node scheduling.
It consumes outputs from the mainline pipeline after they are saved.

## Output Targets

1. behavior summary table
2. confidence summary table
3. AIME qualitative casebook
4. optional entropy note
