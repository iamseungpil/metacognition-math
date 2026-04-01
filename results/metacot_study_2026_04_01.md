# Meta-CoT Study Update

**Author**: Seungpil Lee  
**Date**: 2026-04-01  
**Project**: metacognition-math

## Fact Base

### Verified facts

| ID | Claim | Source |
|---|---|---|
| F1 | Base SFT 1,030-problem overall accuracy is 71.7%. | prior verified summary in working session |
| F2 | V2 SFT 1,030-problem overall accuracy is 72.72%. | verified eval result |
| F3 | V3 SFT 1,030-problem overall accuracy is 72.0%. | verified eval result |
| F4 | Old E7 1,030-problem overall accuracy is 69.9%. | `/scratch/e7prev_eval_results/eval_1030_grpo_v2_E7_prev.json` |
| F5 | E5 1,030-problem overall accuracy is 72.04%. | `/scratch/e5_eval_results/eval_1030_grpo_v2_E5.json` |
| F6 | Current E7 1,030-problem overall accuracy is 70.68%. | `/scratch/e7current_eval_results/eval_1030_grpo_v2_E7_current.json` |
| F7 | V2 rich eval 1,030-problem overall accuracy is 71.36%. | `/scratch/v2rich_eval_results/eval_1030_v2_sft_rich.json` |
| F8 | E8 HF upload completed. | `/scratch/metacognition/grpo_v2_E8_hf_upload.log` with `UPLOADED_E8` |
| F9 | Behavior-first pilot generation produced 770 valid chains from a requested 1,800. | `results/autoresearch_round1/gen_behavior_round1.log` |
| F10 | Pilot valid class balance is redirect 552, verify 216, straight 2. | `data/metacot_behavior_trapi_round1.parquet` generation summary |
| F11 | Local behavior data files were saved as parquet. | `data/behavior_all_sft.parquet`, `data/behavior_redirect_sft.parquet`, `data/behavior_verify_sft.parquet`, `data/metacot_behavior_trapi_round1.parquet` |
| F12 | E5 and old E7 eval bundles include `.json`, `.metadata.json`, and `.parquet`. | `/scratch/e5_eval_results`, `/scratch/e7prev_eval_results` |
| F13 | Current E7 and V2 rich currently save `.json` logs, but metadata/parquet were not uniformly present at inspection time. | `/scratch/e7current_eval_results`, `/scratch/v2rich_eval_results` |
| F14 | E8 eval is still incomplete and currently only a log file exists. | `/scratch/e8_eval_results` |

## Executive Summary

The current evidence shows that Meta-CoT can match or slightly exceed the base SFT model on overall 1,030-problem accuracy, but the intended metacognitive behaviors are still only partially learned. The strongest finished result so far is `E5` at `72.04%`, which is close to `V2 SFT` at `72.72%` and above the `71.7%` base reference. However, qualitative inspection continues to indicate that verification is often decorative and true strategy redirection remains weak.

The autoresearch direction was therefore shifted from meta-format optimization to behavior-first control. The new pilot data generation explicitly targets three behaviors: direct solve, high-confidence verification, and low-confidence redirection after contradiction. That direction is aligned with the long-term goal of OOD test-time control, but the first pilot surfaced a critical design flaw: the accepted samples collapsed toward redirect behavior and almost entirely failed to retain straight-solve examples.

## 1. Artifact Audit

### 1.1 What is currently preserved well

The project now preserves the main categories of artifacts needed for later analysis.

| Category | Current status | Notes |
|---|---|---|
| Session logs | Present | `results/session_log_2026_03_31.md`, monitor logs, follow-up logs |
| Plans and autoresearch notes | Present | `results/experiment_plan_v3.md`, `results/autoresearch_behavior_2026_04_01.md` |
| Generated training data | Present | local `.parquet` files under `data/` |
| Eval raw JSON | Present for completed runs | E5, old E7, current E7, V2 rich |
| Eval metadata/parquet | Partial | strong for E5 and old E7; uneven for newer runs |
| HF upload evidence | Present | E8 upload log confirms completion |

### 1.2 Gaps that still need cleanup

The artifact pipeline is not yet uniform. Current E7 and V2 rich were saved as JSON, but at inspection time the metadata and parquet sidecars were not consistently visible in the remote result directories. E8 evaluation is still incomplete and only a log file exists so far. This means the modified `eval_hf.py` saver is correct in code, but operationally not every older or currently running evaluation has yet produced the full bundle.

## 2. Latest Quantitative Results

### 2.1 Completed 1,030-problem runs

| Model | Overall Accuracy |
|---|---:|
| Base SFT | 71.7% |
| V2 SFT | 72.72% |
| V3 SFT | 72.0% |
| E5 | 72.04% |
| Old E7 | 69.9% |
| Current E7 | 70.68% |
| V2 rich eval | 71.36% |

### 2.2 Interpretation

These numbers support a cautious conclusion. Meta-CoT is not fundamentally collapsing accuracy anymore, because multiple meta variants now sit near or slightly above the base reference. At the same time, more meta structure has not translated cleanly into better control. `Old E7` is the clearest warning sign because it used more meta while underperforming the simpler baselines.

## 3. Behavioral Assessment

The central research question is no longer whether the model can emit meta text. That has already been achieved. The more important question is whether confidence changes actually alter behavior in useful ways.

The current answer is still mixed. Verification has been learned much more easily than redirection, but much of that verification still looks like a textual ritual rather than an independent error check. Redirection remains the weaker behavior. The model often signals uncertainty, but it does not consistently switch to a genuinely different method.

This is why the current autoresearch round is behavior-first. The objective is to move from `reported confidence` to `confidence as a control variable`, where confidence affects whether the model verifies, redirects, or continues directly.

## 4. Round-1 Behavior Pilot

### 4.1 What was attempted

The new pilot used TRAPI generation to build supervision for three scenarios:

1. `straight`
2. `verify`
3. `redirect`

The intention was to create a balanced dataset that could teach both conservative and interventionist meta behavior.

### 4.2 What actually happened

The pilot produced `770` valid chains from the requested `1,800`, but the accepted data was highly imbalanced:

| Scenario | Valid Count |
|---|---:|
| Redirect | 552 |
| Verify | 216 |
| Straight | 2 |

### 4.3 Why this matters

This pilot is still useful because it validates the behavior-first framing, but it is not suitable for a full main run. If used as-is, it would likely bias the model toward over-revision rather than calibrated control. The next iteration must therefore repair scenario balance before scaling the data to a larger run.

## 5. Current Operational Status

The project has active monitor logs, saved local pilot data, and multiple completed evaluation bundles. The main open operational issues are:

1. make eval bundle saving uniform across all active runs
2. finish or debug E8 evaluation
3. repair the remote launch path for behavior SFT and E9 GDPO

These are execution problems rather than conceptual blockers.

## 6. Proposed Next Steps

The next autoresearch iteration should keep the current high-level direction and tighten the implementation.

First, the generator and validator should be revised so that straight-solve and verify examples survive at much higher rates. Second, the remote launcher should be made reliable before the next training wave is scheduled. Only after those two issues are fixed should the project scale the behavior-first dataset toward a larger main run.

## 7. Conclusion

The project is in a better state than the early calibration-only phase because it now has stronger artifact preservation, more realistic large-scale evaluation, and a clearer behavioral target. The main unfinished work is not identifying the right research direction. It is enforcing that direction through balanced data generation and dependable execution. That is the correct next focus for autoresearch.
