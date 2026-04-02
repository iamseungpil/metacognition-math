# Meta-CoT Experiment and Analysis Plan (2026-04-01)

## 1. Purpose of This Document

This document separates two questions that were getting mixed together.

1. `Experiment plan`: what should be trained and compared next
2. `Analysis plan`: how to decide whether the resulting behavior actually matches the intended metacognitive control policy

The main research target is not "more meta text" and not "a nicer calibration number." The target is an `OOD test-time control policy` in which confidence functions as an internal control variable that changes the reasoning trajectory.

## 2. Central Intent

The current plan is organized around two intended control behaviors.

### 2.1 Low-confidence failure management

When the model notices contradiction, anomaly, or a meaningful drop in confidence, it should not simply continue the same route while narrating caution. The intended behavior is:

`trigger / anomaly -> confidence drop -> brief diagnosis of why the route is weak -> redirect to another method`

### 2.2 High-confidence overconfidence management

The key point is that verification should not be triggered just because a problem is easy. Verification should appear when the model feels sufficiently confident to commit but internal evidence suggests a calibration gap, premature commitment, or confidence that is running ahead of support. If confidence is stable and well-calibrated, meta should be reduced. The intended behavior is:

`sufficient-confidence + overcommit / calibration-gap signal -> verify before final answer`

These two behaviors are the core target. Curriculum and retrieval should be treated as later-stage extensions that become meaningful only after these conditional control policies are stable.

## 3. What the Project Is Not Trying to Optimize

The current phase should explicitly avoid four false success criteria.

1. A larger number of meta blocks is not success by itself.
2. Lower wrong-answer confidence is not sufficient by itself.
3. Saying "let me try another way" is not enough unless the method actually changes.
4. Retrieval- or curriculum-assisted gains are not evidence of metacognitive control unless the model first demonstrates self-diagnosis.

The plan therefore prioritizes `conditional behavior control` over benchmark hacking.

## 4. Terms

| Term | Role | Meaning |
|---|---|---|
| `trigger` | situation signal | contradiction, failed substitution, unsupported assumption, unit mismatch, or confidence drop |
| `verify` | stabilizing action | independently check an answer or route before final commitment |
| `redirect` | control action | lower confidence after a trigger and actually switch to another strategy |
| `diagnosis` | cause interpretation | briefly explain why the current route is failing |
| `decomposition` | failure decomposition | break down why the current route is not working, what is missing, or what sub-skill is absent |

Two clarifications matter.

1. `trigger` is not a terminal target behavior. It is the condition that should activate `redirect`.
2. `verify` and `redirect` are the true target behaviors, while `diagnosis` and `decomposition` are internal structures that make redirect useful rather than decorative.
3. `meta` must remain strictly separate from CoT, calculations, and concrete solve steps.

## 5. What the Current Evidence Already Shows

The current evidence supports five claims.

1. Explicit meta format can be learned.
2. Calibration-style rewards can reduce overconfidence on hard wrong answers.
3. Lower confidence alone does not reliably improve AIME accuracy.
4. Verification is easier to learn than redirect.
5. Many current hard-problem interventions still look like local self-correction rather than diagnosis-driven control.

This is why the overall direction changed from format-centric optimization to behavior-centric control. That change is not arbitrary. It is the correct consequence of the evidence collected so far.

## 6. Why the Current Experiment Family Is Aligned with the Intent

The current experiment family is aligned because each branch now has a distinct role instead of collapsing all hypotheses into one run.

### 6.1 V2 and V3

These runs test representational feasibility.

- Can the model emit explicit meta structure?
- Can confidence be expressed?
- Can meta text coexist with the base math capability?

This stage answers whether meta traces are representable at all.

### 6.2 E3 and E8

These runs test the calibration axis.

- Can wrong high-confidence behavior be reduced?
- Do meta interventions become more common on hard examples?
- Does reported confidence become better aligned with actual error risk?

This stage answers whether confidence becomes more meaningful, not whether the final control policy is complete.

### 6.3 Behavior SFT branches

These runs test the target actions directly.

- `behavior_verify_sft`: teach high-confidence verification
- `behavior_redirect_sft`: teach redirect after anomaly or confidence drop
- `behavior_all_sft`: teach a conditional policy over straight solve, verify, and redirect

This is the first stage that directly asks whether confidence changes lead to different actions.

### 6.4 Future E10

The next RL comparison should not discard the calibration axis. It should add behavior rewards on top of it.

The clean comparison is:

1. `E3`: calibration baseline
2. `E8`: stronger calibration / confidence shaping
3. `E10`: `E8 + behavior rewards`

This decomposition is necessary if the project wants a defensible scientific claim about what each reward family is doing.

## 7. Data Design Principles for the Next Control-v5 Round

The next data generation round should be rebuilt as `control-v5`, with three simultaneous requirements: `meta purity`, `confidence-conditioned control`, and `usable failure/study signals for later RAG or curriculum`.

### 7.1 What to keep

1. Explicit meta regions should remain visible.
2. Each meta intervention should contain extractable confidence formatting.
3. Hard problems should be allowed to contain multiple meta interventions over time.

### 7.2 What to change

1. Rigid fields such as `trigger:` or `confidence_before:` and `confidence_after:` should be removed.
2. A single meta event should not be split into mechanical staged subfields.
3. Each meta block should be one natural intervention written in natural language.
4. Meta should not contain calculations, substitutions, case splits, verification CoT, or concrete solve plans.

### 7.3 What each useful intervention should contain

1. confidence self-monitoring
2. anomaly notice or calibration-gap notice
3. brief diagnosis of why the current route is weak
4. `study_need:` when a missing skill or perspective should be made explicit
5. failure decomposition when needed
6. only the control-level next action, not the actual solve steps

The policy split should also be explicit:

1. `verify`: trigger only when there is an overconfidence or premature-commit signal
2. `redirect`: trigger only when there is confidence drop, anomaly, or stuckness
3. `straight`: use little or no meta when confidence appears well calibrated

The goal is to preserve naturalness, reward extractability, and immediate downstream usefulness for retrieval.

## 8. Reward Plan

The next RL setup should decompose reward families rather than merging everything blindly.

### 8.1 Reward axes to preserve

1. `calibration_reward`
2. `confidence_revision_reward`
3. `overconfidence_penalty_reward`
4. `effective_verification_reward`
5. `effective_redirection_reward`

`calibration_reward` is not something to remove. It remains the backbone of the control setup. In particular, the project should preserve the E8-style confidence shaping and then ask whether behavior rewards add distinct value on top of that.

### 8.2 Reward axes to reinterpret in natural language terms

The following should be computed from natural-language evidence rather than rigid field presence.

1. `diagnosis_reward`
2. `decomposition_reward`

### 8.3 Additional candidate axes

1. `anomaly_notice_reward`
2. `repeated_intervention_reward`
3. `overconfidence_verify_reward`

The recommended RL ablation should therefore be:

1. `E3`: calibration baseline
2. `E5`: calibration + confidence revision family
3. `E8`: stronger calibration / overconfidence shaping
4. `E10`: `E8 + behavior rewards`

The important scientific point is not reward count. It is whether each reward axis produces a distinct behavioral change.

## 9. Experiment Plan

### Phase A. Consolidate the current comparison set

First, complete and normalize the evaluation inventory for:

1. Base SFT
2. V2 SFT
3. V3 SFT
4. E3, E5, E7 prev, E7 current, E8
5. behavior_all_sft, behavior_redirect_sft, behavior_verify_sft

This phase fixes the starting point for the next RL decision.

### Phase B. Regenerate control-v5 data

The next dataset should be built with a pilot -> critic -> improve -> main-run loop.

1. Smoke QC must show that `straight`, `verify`, and `redirect` all survive.
2. Samples must be checked qualitatively across difficulty bands.
3. Meta must remain strictly separated from CoT.
4. `verify` must be triggered by overconfidence signals rather than by difficulty.
5. `redirect` must explain why the model is failing, not just announce a new method.
6. `study_need:` must be short, parseable, and useful for retrieval.
7. Hard trajectories may contain repeated interventions, but only when a second anomaly or renewed overconfidence signal appears.

No 10k-scale main run should begin before this pilot passes.

### Phase B.5. Node allocation and execution gating

The execution policy should remain strict so that the main project and the separate analysis
project do not silently interfere with each other.

1. `metacognition_e8` is reserved for `control_v5_all_sft` and later main-project RL follow-up.
2. `metacognition_eval` is reserved for `control_v5_verify_sft` and later main-project eval work.
3. `metacognition_train_b` is reserved for `control_v5_redirect_sft` and later main-project
   training work.
4. `metacognition_run_c` is reserved for the separate
   `metacognition-behavior-uncertainty` project.

There is also an execution gate:

1. The reserved AMLT jobs may stay idle as holder jobs.
2. Actual SFT launch should not happen until `data/control_v5_10k.parquet` exists and passes QC.
3. If the generation process dies before writing the parquet, the correct action is to relaunch or
   repair generation, not to repurpose the reserved nodes.

### Phase C. SFT comparison

The starting point should be a strong existing SFT checkpoint such as `qwen3_base_sft`, not the raw base model.

Recommended comparison set:

1. `base_sft -> control_v5_all_sft`
2. `base_sft -> control_v5_verify_specialist`
3. `base_sft -> control_v5_redirect_specialist`

The primary decision criteria are:

1. accuracy retention
2. verification effectiveness
3. redirect effectiveness
4. confidence-conditioned behavior change

### Phase D. RL comparison

After the SFT comparison, select one or two strong bases and compare:

1. `E3`
2. `E5`
3. `E8`
4. `E10 = E8 + behavior rewards`

If needed, specialist branches for verify and redirect can remain for analysis, but the main scientific goal should still be a unified controller.

### Phase E. Curriculum / RAG gate

Curriculum or retrieval should start only if all of the following are true:

1. contradiction-conditioned confidence drop is real
2. confidence drop leads to actual redirect
3. high confidence leads to actual verify
4. these behaviors remain visible on the full 1,030-problem setting and hard slices

Only after that gate does a loop like "diagnose weakness -> extract `study_need` -> retrieve helpful context or examples -> retry at test time" become a valid extension of the same research program.

## 10. Analysis Plan

### A. Calibration analysis

Required metrics:

1. benchmark-level ECE
2. wrong high-confidence rate
3. correct low-confidence rate
4. wrong-answer average confidence

### B. Mid-trajectory confidence revision

Required metrics:

1. contradiction-conditioned confidence drop
2. confidence-drop -> redirect rate
3. confidence-drop -> correctness recovery rate
4. no-drop wrong-commit rate

This is the most direct test of whether confidence is functioning as a control variable.

### C. Verification analysis

Required metrics:

1. verify fraction
2. independent-check fraction
3. high-confidence-with-verify error rate
4. high-confidence-without-verify error rate
5. answer-change-after-verify rate

### D. Redirect analysis

Required metrics:

1. redirect fraction
2. strategy-switch fraction
3. redirect-conditioned recovery accuracy
4. redirect-without-real-switch fraction

This analysis must distinguish real strategy replacement from cosmetic rephrasing.

### E. Diagnosis and decomposition analysis

Required qualitative questions:

1. Does the model name why the current route is failing?
2. Does it define a useful subgoal or subproblem?
3. Is the next action consistent with that diagnosis?

This axis matters because later curriculum and retrieval are supposed to build on these weakness labels.

### F. Difficulty-conditioned compute allocation

Required metrics:

1. completion length by difficulty
2. meta-block count by difficulty
3. over-intervention rate on easy problems
4. verify or redirect rate on hard problems

## 11. Current Conclusion

The central intent inside the current plan is correct.

1. Confidence should be treated as a control signal, not just a reported scalar.
2. Low confidence should trigger diagnosis and redirect.
3. High confidence should trigger verification.
4. Curriculum should come only after these control behaviors stabilize.

The current experiment family is also broadly aligned with that intent.

1. V2 and V3 test representational feasibility.
2. E3 and E8 test the calibration axis.
3. Behavior SFT branches teach the target actions directly.
4. The future E10 stage is the correct place to combine calibration and behavior rewards.
5. Curriculum and retrieval are correctly gated behind the control-policy tests.

What remains is not a change of direction but a tighter implementation of the same direction: more natural and balanced data, cleaner reward decomposition, and analysis that asks whether behavior changed rather than whether meta text merely appeared.
