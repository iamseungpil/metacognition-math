# Node Allocation Policy (2026-04-06)

This file defines a strict separation between the main `metacognition` experiment track and the
separate `metacognition-behavior-uncertainty` analysis track.

## Core Rule

Do not mix the main training/eval pipeline with the separate behavior-analysis project on the same
node unless the plan is explicitly revised.

## Reserved Roles

1. `metacognition_e8`
   - Role: reserved mainline execution node
   - Historical assignment:
     `probe rollout -> probe retrain -> probe smoke -> E6 -> E7`
   - Current assignment:
     reserved for the next mainline launch after the `SlotC` gate is resolved
   - Do not use for exploratory launches while `SlotC` remains unresolved.

2. `metacognition_eval`
   - Role: active mainline node
   - Historical assignment:
     `qwen3_base_sft -> control_v5_verify_sft -> E3 -> E5 -> E9`
   - Current assignment:
     `SlotC` eval and mainline analysis.

3. `metacognition_train_b`
   - Role: exploratory sidecar node
   - Historical assignment:
     `qwen3_base_sft -> control_v5_redirect_sft -> E8 -> E9b -> E9c`
   - Current assignment:
     `SlotB` exploratory RL.
   - Results from this node must be marked as side evidence unless the launcher base matches the active plan.

4. `metacognition_run_c`
   - Role: analysis-only node
   - Assignment: `metacognition-behavior-uncertainty` project, parser/analysis/reporting, external
     behavior-uncertainty experiments

## Scheduling Rule

The active plan is defined by:

1. `results/plan_metacot_v6.4_active_2026_04_06.md`
2. the long-lived RQ contract in `results/experiment_analysis_plan_2026_04_01.md`

Until the `SlotC` gate resolves, the scheduling rule is:

1. finish `SlotC` eval and analyze:
   - accuracy
   - switch rate
   - verify effectiveness
2. treat `SlotB` as exploratory side evidence, not mainline proof
3. keep `metacognition_e8` free for the next mainline launch
4. if `SlotC` passes the gate:
   - launch mainline RL from the approved checkpoint
5. if `SlotC` fails the gate:
   - launch clean-data restart from `base_sft`

The analysis node remains separate and should not be consumed by SFT or RL unless the user
explicitly changes the policy.

## Safety Rule

1. Never cancel, pause, kill, or delete AMLT jobs unless the user explicitly instructs it.
2. If a node is idle because a prerequisite artifact is missing, leave the holder job intact and
   fix the upstream artifact or launch condition instead.
