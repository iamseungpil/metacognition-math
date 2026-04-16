#!/usr/bin/env bash
# Unified launch script for E21 historical (meta GDPO) vs Base (GRPO) comparison.
# Both use Four-Habits-aligned hyperparameters with matched data.
#
# Usage:
#   MODE=meta  bash scripts/launch_e21r_vs_base_0410.sh   # EVAL node
#   MODE=base  bash scripts/launch_e21r_vs_base_0410.sh   # TRAIN_B node
#
# Hyperparameters aligned with Four Habits PPO (Gandhi et al., 2025):
#   lr=1e-6, kl=0.001, n=4, save/test_freq=10, ~6 epochs
set -euo pipefail

# ── Mode ──
MODE="${MODE:?Set MODE=meta or MODE=base}"
if [[ "$MODE" != "meta" && "$MODE" != "base" ]]; then
  echo "ERROR: MODE must be 'meta' or 'base', got '$MODE'" >&2
  exit 1
fi

# ── Paths ──
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition:${PYTHONPATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:-2f4e627868f1f9dad10bcb1a14fbf96817e6baa9}"
export WANDB_PROJECT="metacot-math"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CUMEM_ENABLE=0

source /scratch/simplerl_venv/bin/activate

# ── Shared hyperparameters (Four Habits aligned) ──
LR=1e-6
KL_COEF=0.001
BATCH_SIZE=64
ROLLOUT_N=4
RESPONSE_LENGTH=4096
PROMPT_LENGTH=2048
TEMPERATURE=0.7
TOP_P=0.95
TOTAL_STEPS=300
SAVE_FREQ=10
TEST_FREQ=10
TP_SIZE=2
GPU_MEM_UTIL=0.4
PPO_MINI_BATCH=16
PPO_MICRO_PER_GPU=1
PPO_MAX_TOKEN_PER_GPU=16384
LOG_PROB_MICRO_PER_GPU=16
MAX_BATCHED_TOKENS=8192
MAX_NUM_SEQS=256

# ── Mode-specific settings ──
if [[ "$MODE" == "meta" ]]; then
  MODEL_PATH="checkpoints/v8_meta_inside_strict_sft"
  TRAIN_DATA="data/verl_train_redirect.parquet"
  VAL_DATA="data/verl_val_redirect.parquet"
  ADV_ESTIMATOR="gdpo"
  REWARD_NAME="compute_score_e21r_v2"
  GDPO_KEYS="++algorithm.gdpo_reward_keys=[correctness,outcome_calibration]"
  REWARD_MANAGER="reward.reward_manager.name=gdpo"
  EXPERIMENT="verl_e21r_v2_0413"
  WANDB_NAME="verl_e21r_v2_0413"
  WANDB_GROUP="e21r-v2"
  LOG_FILE="/scratch/verl_e21r_v2_0413.log"
else
  MODEL_PATH="checkpoints/v8_base_matched_strict_sft"
  TRAIN_DATA="data/verl_train_redirect_base.parquet"
  VAL_DATA="data/verl_val_redirect_base.parquet"
  ADV_ESTIMATOR="grpo"
  REWARD_NAME="compute_score_base"
  GDPO_KEYS=""
  REWARD_MANAGER=""
  EXPERIMENT="verl_base_matched_0410"
  WANDB_NAME="verl_base_matched_0410"
  WANDB_GROUP="base-matched"
  LOG_FILE="/scratch/verl_base_matched_0410.log"
fi

export WANDB_NAME
export WANDB_RUN_GROUP="$WANDB_GROUP"

# ── Preflight checks ──
for f in "$MODEL_PATH/config.json" "$TRAIN_DATA" "$VAL_DATA" \
         "src/training/verl_reward.py" "src/training/rewards.py"; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting $MODE ($EXPERIMENT)"
echo "  Model: $MODEL_PATH"
echo "  Data: $TRAIN_DATA"
echo "  Algorithm: $ADV_ESTIMATOR"
echo "  Reward: $REWARD_NAME"
echo "  Steps: $TOTAL_STEPS | Batch: $BATCH_SIZE | N: $ROLLOUT_N"
echo "  LR: $LR | KL: $KL_COEF | Resp: $RESPONSE_LENGTH"
echo "  Log: $LOG_FILE"

# ── Kill previous runs ──
pkill -f "verl.trainer.main_ppo.*${EXPERIMENT}" 2>/dev/null || true
/scratch/simplerl_venv/bin/ray stop --force >/dev/null 2>&1 || true
sleep 2

# ── Build command ──
CMD=(
  python -m verl.trainer.main_ppo
  --config-path /scratch/simplerl_venv/lib/python3.10/site-packages/verl/trainer/config
  --config-name ppo_trainer
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF"
  actor_rollout_ref.actor.optim.lr="$LR"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH"
  actor_rollout_ref.actor.ppo_micro_batch_size=null
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_PER_GPU"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_PER_GPU"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.temperature="$TEMPERATURE"
  actor_rollout_ref.rollout.top_p="$TOP_P"
  actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH"
  actor_rollout_ref.rollout.prompt_length="$PROMPT_LENGTH"
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS"
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS"
  actor_rollout_ref.rollout.do_sample=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size=null
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_PER_GPU"
  actor_rollout_ref.ref.log_prob_micro_batch_size=null
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_PER_GPU"
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa
  data.train_files="$TRAIN_DATA"
  data.val_files="$VAL_DATA"
  data.max_prompt_length="$PROMPT_LENGTH"
  data.max_response_length="$RESPONSE_LENGTH"
  data.train_batch_size="$BATCH_SIZE"
  algorithm.adv_estimator="$ADV_ESTIMATOR"
  algorithm.use_kl_in_reward=False
  ++reward.custom_reward_function.path=src/training/verl_reward.py
  ++reward.custom_reward_function.name="$REWARD_NAME"
  critic.ppo_mini_batch_size="$PPO_MINI_BATCH"
  critic.ppo_micro_batch_size=null
  critic.ppo_micro_batch_size_per_gpu=1
  critic.forward_micro_batch_size=null
  critic.forward_micro_batch_size_per_gpu=1
  critic.ppo_max_token_len_per_gpu=32768
  trainer.project_name=metacot-math
  trainer.experiment_name="$EXPERIMENT"
  "trainer.logger=[console,wandb]"
  trainer.nnodes=1
  trainer.n_gpus_per_node=4
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.default_local_dir="checkpoints/metacot-math/$EXPERIMENT"
  ++trainer.remove_previous_ckpt=False
)

# Add GDPO-specific keys for meta mode
if [[ -n "$GDPO_KEYS" ]]; then
  CMD+=("$GDPO_KEYS")
fi
if [[ -n "$REWARD_MANAGER" ]]; then
  CMD+=("$REWARD_MANAGER")
fi

# ── Launch ──
nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
echo $! > "/scratch/${EXPERIMENT}.pid"
echo "STARTED $MODE PID=$(cat /scratch/${EXPERIMENT}.pid)"

# ── Verify start ──
sleep 10
if kill -0 "$(cat /scratch/${EXPERIMENT}.pid)" 2>/dev/null; then
  echo "Process alive. Checking first log output..."
  head -5 "$LOG_FILE" 2>/dev/null || true
else
  echo "ERROR: Process died within 10s. Last log lines:" >&2
  tail -20 "$LOG_FILE" 2>/dev/null >&2
  exit 1
fi
