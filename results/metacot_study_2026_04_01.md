# Meta-CoT Study Update

**Author**: Seungpil Lee  
**Date**: 2026-04-01  
**Project**: metacognition-math

## Fact Base

### Verified facts

| ID | Claim | Source |
|---|---|---|
| F1 | Base SFT 1,030-problem overall accuracy is 71.7%. | verified working summary and `eval_1030_base_sft.json` |
| F2 | V2 SFT 1,030-problem overall accuracy is 72.72%. | verified working summary and `eval_1030_v2_sft.json` |
| F3 | V3 SFT 1,030-problem overall accuracy is 72.0%. | `eval_1030_v3_sft.json` |
| F4 | E5 1,030-problem overall accuracy is 72.04%. | `eval_1030_grpo_v2_E5.json` |
| F5 | E7 prev 1,030-problem overall accuracy is 69.9%. | `eval_1030_grpo_v2_E7_prev.json` |
| F6 | E7 current 1,030-problem overall accuracy is 70.68%. | `eval_1030_grpo_v2_E7_current.json` |
| F7 | E8 1,030-problem overall accuracy is 67.2%. | verified working summary from remote eval |
| F8 | V2 rich 1,030-problem overall accuracy is 71.36%. | `eval_1030_v2_sft_rich.json` |
| F9 | AIME slices are Base 3/30, V2 4/30, V3 2/30, E5 2/30, E7 prev 3/30, E7 current 1/30, E8 2/30. | `results/control_v4_aime_notes_2026_04_01.md` |
| F10 | Wrong-answer average confidence on AIME is V2 0.631, V3 0.287, E5 0.321, E7 prev 0.315, E7 current 0.263, E8 0.238. | `results/control_v4_aime_notes_2026_04_01.md` |
| F11 | Wrong-answer average meta-block count on AIME is V2 2.58, V3 2.32, E5 5.36, E7 prev 5.15, E7 current 4.34, E8 5.04. | `results/control_v4_aime_notes_2026_04_01.md` |
| F12 | Behavior pilot produced 770 valid samples from a requested 1,800. | `results/autoresearch_round1/gen_behavior_round1.log` |
| F13 | Behavior pilot class balance is redirect 552, verify 216, straight 2. | pilot generation summary and preserved parquet audit |
| F14 | Completed remote full eval bundles with `.json`, `.metadata.json`, and `.parquet` are present for Base, V3, E5, and E7 prev. | remote directory inspection on `train_b` |
| F15 | Current remote eval outputs for E7 current, E8, behavior_all, behavior_redirect, and behavior_verify were observed as JSON-plus-log, without uniform parquet/metadata sidecars at inspection time. | remote directory inspection on `eval_e8` |
| F16 | Hugging Face dataset repo already contains uploaded model artifacts for Base SFT, V2 SFT, V3 SFT, E5, E7 current, E8, behavior_all_sft, behavior_redirect_sft, and behavior_verify_sft. | `HfApi.list_repo_files()` and commit history inspection |
| F17 | The Hugging Face dataset repo did not yet contain `eval`, `responses`, `results`, `study`, or `plans` paths at inspection time. | `HfApi.list_repo_files()` inspection |

## Executive Summary

The current evidence supports a narrower and more precise claim than "Meta-CoT improves reasoning." The project has already shown that it can teach models to emit meta text, and it has also shown that confidence-shaping rewards can reduce overconfidence on hard wrong answers. What has not yet been shown is the full intended control policy: when the model becomes stuck or notices contradiction, it should lower confidence and truly change method, and when the model is confident, it should still perform an independent check before committing.

This is why the present experiment family is directionally correct. The earlier V2 and V3 runs established that meta formatting and explicit confidence can be learned without destroying the base math capability. The E-series then tested whether reward shaping can make confidence better aligned with actual error. The behavior-first branch is the first branch that explicitly targets the intended actions themselves: `verify` and `redirect`. In other words, the experiment stack is now much more aligned with the research intent than before, even though the latest pilot is still too imbalanced to scale.

## 1. What the Project Is Actually Trying to Learn

The core target is not "more reflection" and not "more careful language." The target is a metacognitive control policy that changes the trajectory of reasoning.

The intended policy has two primary cases. The first case is a failure-management policy: if the model notices anomaly, contradiction, or a meaningful drop in confidence, it should not keep narrating the same route. It should diagnose why the route is weak and then redirect to another method. The second case is an overconfidence-management policy: if the model feels confident, it should not immediately finalize. It should independently verify the answer and reduce confident mistakes.

This framing matters because it explains why curriculum and retrieval are not yet the next step. Curriculum only becomes meaningful after the model can reliably recognize a weakness and act on it. Otherwise, curriculum would amplify noisy self-talk rather than useful self-diagnosis.

## 2. Quantitative State

### 2.1 1,030-problem comparison

| Model | Overall Accuracy |
|---|---:|
| Base SFT | 71.7% |
| V2 SFT | 72.72% |
| V3 SFT | 72.0% |
| E5 | 72.04% |
| E7 prev | 69.9% |
| E7 current | 70.68% |
| E8 | 67.2% |
| V2 rich | 71.36% |

These results show that the project is no longer in a simple collapse regime. Several meta variants remain near the base reference, and some exceed it slightly. However, this table alone is not evidence that the intended meta control has been learned. At best, it says that explicit meta structure can coexist with math performance.

### 2.2 AIME slice

| Model | AIME | Wrong Avg Confidence | Wrong Avg Meta Blocks |
|---|---:|---:|---:|
| Base SFT | 3/30 | N/A | 0.00 |
| V2 SFT | 4/30 | 0.631 | 2.58 |
| V3 SFT | 2/30 | 0.287 | 2.32 |
| E5 | 2/30 | 0.321 | 5.36 |
| E7 prev | 3/30 | 0.315 | 5.15 |
| E7 current | 1/30 | 0.263 | 4.34 |
| E8 | 2/30 | 0.238 | 5.04 |

This table is the clearest reason to keep the current direction but tighten the design. Reward-shaped models often become much less overconfident on hard wrong answers than V2, which means the confidence axis is not useless. But lower confidence by itself does not automatically turn into better AIME accuracy. The extra meta activity is real, yet the benefit is unstable.

## 3. Qualitative Reading of the Responses

The earlier qualitative inspection should be understood in a specific way. V2 often carries medium confidence and proceeds on a conventional path. E7 and E8 interrupt themselves more often and lower confidence more aggressively. That is a meaningful behavioral change. It is not just random perturbation.

At the same time, the changed behavior still tends to be local. The model often corrects a line, rephrases a claim, or says it should be careful, but it does not consistently do the stronger actions that the project actually needs. Those stronger actions are explicit failure diagnosis, naming the missing subskill or blocker, decomposing the problem into subgoals, and then replacing the original strategy with another one.

This is why the current interpretation is not "the idea failed." The correct interpretation is narrower: calibration and meta interruption are partially learned, but diagnosis-driven redirection is not yet stably learned. Verification is easier to teach than redirect, and redirect is still the weak link.

## 4. Why the Current Plan Is Aligned with the Intent

The current plan is aligned because each experiment family now has a distinct role instead of mixing all hypotheses together.

V2 and V3 are the representation stage. They test whether explicit meta traces and confidence language can be learned while keeping base math ability. E3 and E8 are the calibration stage. They test whether reward shaping can make confidence react more appropriately to error risk, especially on hard wrong answers. The behavior SFT branches are the control stage. They test whether the model can be directly taught the two target actions, `verify` and `redirect`, as conditional behaviors rather than as free-form narration.

This decomposition is important for research logic. If the project jumps directly from generic meta SFT to curriculum or RAG, then any later gain would be ambiguous. It would be unclear whether the model learned self-diagnosis, or whether the system merely added more external help. The present staged plan avoids that ambiguity.

## 5. What the Current Pilot Proves and What It Does Not

The behavior-first pilot proves one useful thing: the control policy can be expressed in data form, and the project now has a concrete supervised interface for the actions it cares about. The pilot does not yet prove that the data recipe is ready for a main run.

The reason is the class collapse. Out of 770 valid samples, 552 are redirect, 216 are verify, and only 2 are straight. This means the present generator-validator combination is selecting intervention-heavy examples and nearly deleting the "continue directly" case. That is unacceptable for a calibrated controller because the model would then learn to over-intervene.

So the correct next action is not to abandon the behavior-first direction. The correct next action is to repair the data so that `straight`, `verify`, and `redirect` all survive validation in a controlled ratio.

## 6. Artifact Preservation and Reporting State

Artifact preservation is partly strong and partly uneven. The strongest completed runs already have full remote eval bundles with JSON, metadata, and parquet sidecars. That is enough to support later quantitative and qualitative analysis on Base, V3, E5, and E7 prev. Newer runs are less uniform. For E7 current, E8, and the behavior SFT evaluations, the observed state at inspection time was JSON plus logs without the same sidecar consistency.

The Hugging Face dataset repo already has the main model artifacts, which is useful for model reproducibility. However, the repo still lacks a parallel archive structure for eval bundles, response backups, study reports, and plan documents. That means the model layer is better preserved than the analysis layer.

## 7. Immediate Recommendations

The next iteration should keep the current conceptual direction and repair the parts that are currently preventing a clean conclusion.

First, the control-v4 data format should encode natural meta interventions rather than rigid subfields, while still preserving extractable confidence values. Second, the next reward comparison should keep the calibration axis as its own ablation and then add behavior rewards on top, rather than replacing calibration with behavior. Third, eval saving should be made uniform so that every completed run leaves behind the same JSON, metadata, and parquet bundle.

## 8. Conclusion

The central intent in the current plan is correct. The project should not optimize for more meta text, and it should not claim success from confidence reduction alone. It should optimize for conditional control: doubt should trigger diagnosis and redirection, while confidence should trigger verification. The present experiment stack is now aligned with that intent more clearly than the earlier stages were. The main remaining gap is not conceptual alignment but implementation quality: balanced data, cleaner reward decomposition, and uniform artifact preservation.
