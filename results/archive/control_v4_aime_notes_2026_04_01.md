# Control v4 AIME Notes (2026-04-01)

## Current AIME Pattern

Remote eval summaries from the existing 1,030-problem runs show:

| Model | AIME | Wrong Avg Confidence | Wrong Avg Meta Blocks |
|---|---:|---:|---:|
| Base SFT | 3/30 | N/A | 0.00 |
| V2 SFT | 4/30 | 0.631 | 2.58 |
| V3 SFT | 2/30 | 0.287 | 2.32 |
| E5 | 2/30 | 0.321 | 5.36 |
| E7 prev | 3/30 | 0.315 | 5.15 |
| E7 current | 1/30 | 0.263 | 4.34 |
| E8 | 2/30 | 0.238 | 5.04 |

Interpretation:

1. RL-style models are much less overconfident on AIME wrong answers than `V2`.
2. However, lower confidence alone is not enough; accuracy does not reliably improve.
3. Meta block count increases on hard failures, but many hard failures still look like low-confidence self-talk rather than diagnosis-driven control.

## Qualitative Pattern

Representative AIME failures show:

1. `V2` often states medium confidence and proceeds with a conventional solution path.
2. `E7/E8` more often interrupt themselves and lower confidence.
3. But those interruptions are still often local corrections instead of:
   - explicit failure diagnosis
   - naming the missing skill/blocker
   - decomposition into subgoals
   - real strategy replacement

## Required Loop

The next loop should explicitly teach:

1. `low confidence -> diagnose why the route is failing`
2. `diagnose -> decompose into subgoals`
3. `decompose -> choose next strategy`
4. `high confidence -> independently verify`

This is the basis for:

1. behavior-first SFT
2. diagnosis-aware GDPO
3. later curriculum / retrieval loops

## Immediate Implementation

1. `prompt_control_v4.py`
2. `gen_control_v4_trapi.py`
3. `build_control_v4_sft_variants.py`
4. `qc_control_v4_samples.py`
5. `E10` GDPO reward mode with:
   - `effective_verification_reward`
   - `effective_redirection_reward`
   - `confidence_revision_reward`
   - `diagnosis_reward`
   - `decomposition_reward`
