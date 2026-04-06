# Problem Description
We need to converge on one active experiment plan for the metacognition project.
The current repository has multiple plan documents and partially divergent running experiments. I need critical feedback on the active plan and the implementation alignment.

## Codebase Context

Locations:
- /home/v-seungplee/metacognition/results/experiment_analysis_plan_2026_04_01.md : stable RQ-level contract (RQ1 Meta-CoT, RQ2 Meta-RL, RQ3 Curriculum)
- /home/v-seungplee/metacognition/results/experiment_plan_v5.md : v5 failure analysis and E9v2/E9bv2/E10v2 plan
- /home/v-seungplee/metacognition/results/plan_metacot_v6.3_final_2026_04_05.md : v6.3 plan after E11 pilot
- /home/v-seungplee/metacognition/results/plan_12gpu_experiments_2026_04_05.md : scenario-based 12 GPU branching plan
- /home/v-seungplee/metacognition/src/training/grpo_v2.py : reward configs and RL modes
- /home/v-seungplee/metacognition/src/training/rewards.py : reward implementations
- /home/v-seungplee/metacognition/scripts/launch_slotB_rl_e9c.sh : current SlotB launcher
- /home/v-seungplee/metacognition/scripts/launch_slotC_sft.sh : current SlotC launcher
- /home/v-seungplee/metacognition/results/autoresearch_e6e7_loop/results.tsv : factual recent run log

Current Situation:
- v5 conclusion: meta behavior exists, but verify repeats same route and redirect lacks execution.
- v6 E11 pilot result: accuracy improved to 64.1% over E9 62.1%, but structural switch/approach_change remained near zero. Multi-meta increased strongly.
- Current running work:
  - SlotC: stronger SFT seed experiment (E9 + 164 seed x 5 epochs), eval running.
  - SlotB: E13 RL running, but it is based on control_v5_E9c/final instead of the E11 checkpoint expected by v6.3 plan.
  - SlotA: 10k clean data generation progress is logged but not directly checked in this prompt.
- Earlier E6/E7 probe experiments completed. Probe smoke passed, but E6 collapsed in earlier analysis. Eval numbers: E6 56.8%, E7 54.0% on 1030 problems.

## Proposed Approaches

### Approach 1: Preserve all plans and just keep running current experiments
Pros:
- Minimal disruption
- No immediate code changes
Cons:
- Hard to interpret results causally
- Current SlotB is misaligned with v6.3 hypothesis
- Active plan is ambiguous

### Approach 2: Declare v6 scenario C the active plan and demote SlotB to exploratory sidecar
Pros:
- Aligns with E11 pilot evidence that E9-based line is too rigid for structural switch
- Keeps SlotC as mainline, which directly tests stronger SFT seeding
- Makes interpretation cleaner
Cons:
- Requires updating plan docs and launcher scripts
- Requires explicitly marking some running results as side evidence only

### Approach 3: Revert to v5 E9v2/E9bv2/E10v2 as the active plan
Pros:
- Matches the recent v5 failure analysis document
- Code already contains those modes
Cons:
- Seems inconsistent with the newer E11 pilot evidence and scenario C pivot
- Risks spending more compute on lines already shown to be rigid

## Questions for Codex
Please provide critical feedback on:
1. Which approach is strategically soundest and why?
2. What are the biggest plan-implementation mismatches right now?
3. Which exact experiments should count as mainline evidence vs exploratory side evidence?
4. What minimal plan document structure would best express intent / hypothesis / verification clearly?
5. What code or launcher changes are required immediately before more experiments should be launched?
