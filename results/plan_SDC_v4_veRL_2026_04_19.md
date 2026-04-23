# SDC veRL Port — Plan v4

**Date**: 2026-04-19  
**Scope**: faithful veRL port of `sdc-shared`  
**Status**: implemented and unit-tested, real worker smoke pending

## Intent

Port improved SDC to veRL **without** changing the historical `verl_gdpo_e21r` line.

The port should preserve the original SDC intention:

1. Update **direction** comes from environment reward / GDPO.
2. Token-wise **magnitude shaping** comes from teacher policy differences (`T+`, `T-`, student old log-prob).
3. Post-meta shared structure is preserved.
4. Only differential post-meta evidence receives stronger contrastive pressure.

## Hypotheses

### H1. veRL port matches RLSD-style factorization
- scalar/group direction from GDPO reward heads
- token-wise factor from teacher log-prob differences

### H2. shared-vs-diff split is teacher-guided, not reward-only
- candidate shared structure is detected structurally (`\boxed{}` wrapper / answer framing / punctuation)
- final shared-vs-diff assignment is gated by teacher agreement threshold
- diff tokens are those outside shared structure or teacher-disagreed inside it

### H3. existing E21R baseline remains untouched
- all changes live in separate files / configs

## Verification

### Code checks
- teacher signals are attached through veRL ref-policy log-prob path
- advantage is built as `base GDPO advantage × token factor`
- meta/shared/diff masks are present in batch before advantage shaping
- no changes are made to the historical `verl_gdpo_e21r` files

### Unit checks
- `tests/test_verl_sdc.py` passes
- module import + `py_compile` pass
- patched helper test confirms teacher log-probs are attached with the expected response shape

### Smoke checks
- run short veRL smoke with `verl_sdc_e21r_shared`
- verify batch contains:
  - `sdc_teacher_pos_log_probs`
  - `sdc_teacher_neg_log_probs`
  - `sdc_meta_mask`
  - `sdc_postmeta_shared_mask`
  - `sdc_postmeta_diff_mask`

## Current Limits

1. veRL port currently implements `sdc-shared` only.
2. `sdc-uniform` and `sdc-noise` are not yet ported to veRL.
3. The current port is faithful to the RLSD-style *direction vs magnitude* split, but still uses our project-specific `T+ = prompt + gold`, `T- = prompt + decoy` construction rather than an official released RLSD codebase.
4. Real 4-GPU smoke is still required because `_compute_ref_log_prob` is exercised only through local fakes in unit tests.
