# Node Allocation Policy (2026-04-02)

This file defines a strict separation between the main `metacognition` experiment track and the
separate `metacognition-behavior-uncertainty` analysis track.

## Core Rule

Do not mix the main training/eval pipeline with the separate behavior-analysis project on the same
node unless the plan is explicitly revised.

## Reserved Roles

1. `metacognition_e8`
   - Role: main experiment node A
   - First assignment: `qwen3_base_sft -> control_v5_all_sft`
   - Later assignment: primary RL continuation candidate after SFT comparison

2. `metacognition_eval`
   - Role: main experiment node B
   - First assignment: `qwen3_base_sft -> control_v5_verify_sft`
   - Later assignment: evaluation or specialist follow-up, still within the main `metacognition`
     project only

3. `metacognition_train_b`
   - Role: main experiment node C
   - First assignment: `qwen3_base_sft -> control_v5_redirect_sft`
   - Later assignment: main-project follow-up training only

4. `metacognition_run_c`
   - Role: analysis-only node
   - Assignment: `metacognition-behavior-uncertainty` project, parser/analysis/reporting, external
     behavior-uncertainty experiments

## Scheduling Rule

The main three-node training sequence stays:

1. wait for `data/control_v5_10k.parquet`
2. run 3 SFT branches in parallel on the three main-project nodes
3. compare outputs on `gsm8k`, `math500`, `aime2024`
4. continue RL comparisons in the main project

The analysis node remains separate and should not be consumed by SFT or RL unless the user
explicitly changes the policy.

## Safety Rule

1. Never cancel, pause, kill, or delete AMLT jobs unless the user explicitly instructs it.
2. If a node is idle because a prerequisite artifact is missing, leave the holder job intact and
   fix the upstream artifact or launch condition instead.
