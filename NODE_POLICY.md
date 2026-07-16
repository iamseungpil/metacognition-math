> ⚠️ **DEPRECATED** (pre-rq3 V8/H200 세대 노드 정책): 현행 실험은 **rq3 매치드 래더** — `README.md` 및 `docs/redesign/` 참조.

# Node Allocation Policy (2026-04-15)

This file defines the current fixed node ownership for all active projects.
The purpose is to keep each execution lane separate from other ongoing projects.

## Core Rule

Do not mix projects across these long-lived AMLT holders unless the user explicitly revises the
policy. A job being idle does not make the node available for another project by default.

For claim-bearing paired runs, this file is a contract rather than a suggestion.
If a launcher diverges from the frozen contract, the resulting run is not `mainline`.

## Infrastructure Change (2026-04-15)

NCv4 (PCIe A100) capacity retired by GCR. All previous A100 nodes killed.
New nodes allocated on **msrresrchbasicvc** with **H200 141GB × 4** (Basic tier).
AMLT experiment: `node-recovery-h200-0415`

## Active Ownership Map

1. `metacognition_eval`
   - AMLT experiment: `node-recovery-h200-0415`
   - Hardware: 4× H200 141GB (msrresrchbasicvc, Basic)
   - Project owner: main `metacognition`
   - Allowed work:
     - mainline eval
     - mainline Meta-CoT RL / GDPO runs
     - reward debugging and checkpoint validation for the active plan
   - Disallowed work:
     - behavior-uncertainty project jobs
     - boltzmann-attention jobs
     - softprompt GRPO jobs

2. `metacognition_train_b`
   - AMLT experiment: `node-recovery-h200-0415`
   - Hardware: 4× H200 141GB (msrresrchbasicvc, Basic)
   - Project owner: main `metacognition`
   - Allowed work:
     - base-matched SFT / RL baselines
     - mainline comparison runs paired with `metacognition_eval`
     - HF/eval follow-up directly tied to the active Meta-CoT plan
   - Disallowed work:
     - behavior-uncertainty project jobs
     - boltzmann-attention jobs
     - softprompt GRPO jobs

3. `metacognition_run_c`
   - AMLT experiment: `node-recovery-h200-0415`
   - Hardware: 4× H200 141GB (msrresrchbasicvc, Basic)
   - Project owner: `metacognition-behavior-uncertainty`
   - Allowed work:
     - behavior/uncertainty analysis experiments
     - Four Habits style reproduction or audit work
     - analysis-specific PPO / reporting pipelines for that repository
   - Disallowed work:
     - mainline Meta-CoT SFT / RL
     - boltzmann-attention jobs
     - softprompt GRPO jobs

4. `metacognition_e8`
   - AMLT experiment: `node-recovery-h200-0415`
   - Hardware: 4× H200 141GB (msrresrchbasicvc, Basic)
   - Project owner: `boltzmann-attention`
   - Allowed work:
     - boltzmann-attention experiments
     - boltzmann follow-up evaluation and reporting
   - Disallowed work:
     - mainline Meta-CoT SFT / RL
     - behavior-uncertainty jobs
     - softprompt GRPO jobs

5. `rsp_grpo_node_1`
   - AMLT experiment: `node-recovery-h200-0415`
   - Hardware: 4× H200 141GB (msrresrchbasicvc, Basic)
   - Project owner: softprompt-GRPO (RandomSoftPrompt)
   - Allowed work:
     - RSP GRPO experiments (0.5B, 1.5B, 7B)
     - GRPO / DAPO / GPPO algorithm comparison
     - RSP on/off A/B comparison
   - Disallowed work:
     - mainline Meta-CoT work
     - behavior-uncertainty work
     - boltzmann-attention work

6. `rsp_grpo_node_2`
   - AMLT experiment: `node-recovery-h200-0415`
   - Hardware: 4× H200 141GB (msrresrchbasicvc, Basic)
   - Project owner: softprompt-GRPO (RandomSoftPrompt)
   - Allowed work:
     - RSP GRPO experiments (0.5B, 1.5B, 7B)
     - GRPO / DAPO / GPPO algorithm comparison
     - RSP on/off A/B comparison
   - Disallowed work:
     - mainline Meta-CoT work
     - behavior-uncertainty work
     - boltzmann-attention work

## Scheduling Rule For Main `metacognition`

The active execution plan is:

1. `results/plan_metacot_v8_active_2026_04_09.md`

Under this policy, the main `metacognition` pipeline may use only:

1. `metacognition_eval`
2. `metacognition_train_b`

`metacognition_run_c`, `metacognition_e8`, and `rsp_grpo_exp` are treated as unavailable capacity
for the main Meta-CoT scheduler even if they appear idle.

## Mainline SFT Rule

The current claim-bearing paired SFT comparison is:

1. `metacognition_eval` -> `configs/sft_v8_meta_inside_strict.yaml`
2. `metacognition_train_b` -> `configs/sft_v8_base_matched_strict.yaml`

Required initializer:

1. both lanes must start from raw `Qwen/Qwen3-8B`

Disallowed for the current mainline strict SFT claim:

1. starting from `checkpoints/qwen3_base_sft`
2. starting from any earlier SFT checkpoint
3. silently changing dataset parity or max length on one side only

## Paired Comparison Integrity

When `metacognition_eval` and `metacognition_train_b` are used as a paired comparison for the active
Meta-CoT plan, they must share the frozen hyperparameters defined in:

1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `configs/mainline_contract.yaml`

Required shared keys:

1. `prompt_length=2048`
2. `response_length=4096`
3. `train_batch_size=64`
4. `actor.ppo_mini_batch_size=16`
5. `actor.ppo_micro_batch_size_per_gpu=1`
6. `actor.ppo_max_token_len_per_gpu=16384`
7. `critic.ppo_mini_batch_size=16`
8. `critic.ppo_micro_batch_size_per_gpu=1`
9. `critic.ppo_max_token_len_per_gpu=32768`
10. `rollout.n=4`
11. `learning_rate=1e-6`
12. `kl_coef=0.001`
13. `temperature=0.7`
14. `top_p=0.95`
15. `rollout.tensor_model_parallel_size=2`
16. `rollout.gpu_memory_utilization=0.4`
17. `rollout.log_prob_micro_batch_size_per_gpu=16`
18. `ref.log_prob_micro_batch_size_per_gpu=16`
19. `total_training_steps=300`
20. `save_freq=10`
21. `test_freq=10`
22. `remove_previous_ckpt=False`

Canonical launcher for this contract:

1. `scripts/launch_e21_vs_base_matched_0410.sh`

Allowed paired-run differences are restricted to:

1. checkpoint / model
2. parquet / data definition
3. algorithm, only if the active plan explicitly treats it as an experimental variable
4. reward function

Preflight rule:

1. before any claim-bearing run, execute `python scripts/verify_mainline_alignment.py`
2. if the verifier fails on blocking checks, do not launch
3. use `configs/mainline_contract.yaml` as the machine-readable source of truth
4. if a config or launcher diverges from that contract, it may still be used only as `side_evidence`

Operational rule:

1. Do not claim a `mainline paired comparison` from runs launched with ad hoc overrides that change
   the frozen shared keys.
2. If a launcher on either node diverges from the frozen shared keys, record the run as
   `side_evidence` until the active plan is revised.
3. Do not silently keep one node on an old budget and the other on a new budget.

## Runtime Interpretation

This file defines ownership and allowed usage. It does not itself guarantee that the runtime on a
node is healthy. Runtime state must still be checked separately with:

1. AMLT job status
2. `nvidia-smi`
3. active process inspection
4. checkpoint / log update checks

## Safety Rule

1. Never cancel, pause, kill, or delete AMLT jobs unless the user explicitly instructs it.
2. Do not repurpose a long-lived holder job just because one project is temporarily blocked.
3. If a node is idle because an upstream artifact is missing, fix the artifact or launcher first.
4. If a duplicate in-node training process is started by mistake, stop only the duplicate process and record it.
