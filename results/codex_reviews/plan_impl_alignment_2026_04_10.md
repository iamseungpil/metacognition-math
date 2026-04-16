# Plan-Implementation Alignment Note (2026-04-10)

## Scope

Reviewed:
1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `src/training/rewards.py`
3. `src/training/verl_reward.py`

## Converged Findings

1. The active reward family is still scientifically usable, but two controller boundaries were too loose:
   - `verify_execution_v2` allowed `high_conf OR overcommit`, while the plan specifies `high_confidence_overcommit`.
   - `_redirect_context()` allowed verification-like tails to satisfy redirect execution.
2. `diagnosis` and `correction` are core analysis objects, but they do not need to become standalone reward heads before the next launch.
   The current five-key controller remains the right mainline unless redirect-pilot evidence shows otherwise.
3. The pre-launch smoke gate needed explicit controller-confusion cases, not just generic reward smoke.

## Actions Applied

1. Tightened `verify_execution_reward_v2()` to require `high_confidence AND overcommit`.
2. Separated verification-like tails from redirect route replacement inside `_redirect_context()`.
3. Added controller smoke tests covering:
   - high confidence without overcommit
   - high confidence with overcommit
   - redirect meta followed by verify-only tail
   - redirect meta followed by real route replacement
4. Aligned the active plan wording so Phase 3 explicitly matches the five-key veRL reward path.

## Current Interpretation

1. This is a variation-tightening pass, not a reward-family reset.
2. The main scientific thesis remains:
   - Meta-CoT learns confidence-conditioned control.
   - Meta-RL tests whether that control can be reinforced with verifiable signals.
   - Curriculum/RAG remains downstream of diagnosis quality and study-need extraction.
3. New experiment launches should wait for:
   - controller smoke pass
   - subset parquet verification
   - frozen baseline recomputation on the matched pilot subset
