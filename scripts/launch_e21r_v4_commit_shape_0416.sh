#!/usr/bin/env bash
# Launch Meta-GDPO smoke run with analysis-driven commit-shape reward.
#
# Purpose:
#   Prepare a directly comparable RL lane against later self-distill runs:
#   - same strict meta-SFT initializer
#   - same redirect parquet
#   - same 4-GPU veRL stack
#   - reward additionally penalizes no-boxed / repeated-meta / decoherence-like tails
#
# Usage:
#   bash scripts/launch_e21r_v4_commit_shape_0416.sh
# Optional env overrides:
#   EXPERIMENT=verl_e21r_v4_commit_shape_0416
#   WANDB_NAME=...
#   TOTAL_STEPS=300 RESPONSE_LENGTH=4096 PROMPT_LENGTH=2048
set -euo pipefail

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition:${PYTHONPATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_API_KEY}}"
export WANDB_PROJECT="${WANDB_PROJECT:-metacot-math}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CUMEM_ENABLE=0

source /scratch/simplerl_venv/bin/activate

MODEL_PATH="${MODEL_PATH:-checkpoints/v8_meta_inside_strict_sft}"
TRAIN_DATA="${TRAIN_DATA:-data/verl_train_redirect.parquet}"
VAL_DATA="${VAL_DATA:-data/verl_val_redirect.parquet}"
EXPERIMENT="${EXPERIMENT:-verl_e21r_v4_commit_shape_0416}"
WANDB_NAME="${WANDB_NAME:-$EXPERIMENT}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-rq2-smoke-v4}"
LOG_FILE="${LOG_FILE:-/scratch/${EXPERIMENT}.log}"

LR="${LR:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
PROMPT_LENGTH="${PROMPT_LENGTH:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-16}"
PPO_MICRO_PER_GPU="${PPO_MICRO_PER_GPU:-1}"
PPO_MAX_TOKEN_PER_GPU="${PPO_MAX_TOKEN_PER_GPU:-16384}"
LOG_PROB_MICRO_PER_GPU="${LOG_PROB_MICRO_PER_GPU:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

export WANDB_NAME
export WANDB_RUN_GROUP

for f in "$MODEL_PATH/config.json" "$TRAIN_DATA" "$VAL_DATA" \
         "src/training/verl_reward.py" "src/training/rewards.py"; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting meta GDPO smoke ($EXPERIMENT)"
echo "  Model: $MODEL_PATH"
echo "  Data: $TRAIN_DATA"
echo "  Reward: compute_score_e21r_v4_smoke"
echo "  Steps: $TOTAL_STEPS | Batch: $BATCH_SIZE | N: $ROLLOUT_N"
echo "  LR: $LR | KL: $KL_COEF | Resp: $RESPONSE_LENGTH"
echo "  Log: $LOG_FILE"

pkill -f "verl.trainer.main_ppo.*${EXPERIMENT}" 2>/dev/null || true
/scratch/simplerl_venv/bin/ray stop --force >/dev/null 2>&1 || true
sleep 2

nohup python -m verl.trainer.main_ppo \
  --config-path /scratch/simplerl_venv/lib/python3.10/site-packages/verl/trainer/config \
  --config-name ppo_trainer \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size=null \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_PER_GPU" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_PER_GPU" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$TOP_P" \
  actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.prompt_length="$PROMPT_LENGTH" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_PER_GPU" \
  actor_rollout_ref.ref.log_prob_micro_batch_size=null \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_PER_GPU" \
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.max_prompt_length="$PROMPT_LENGTH" \
  data.max_response_length="$RESPONSE_LENGTH" \
  data.train_batch_size="$BATCH_SIZE" \
  algorithm.adv_estimator=gdpo \
  "++algorithm.gdpo_reward_keys=[correctness,outcome_calibration,confidence_revision,redirect_execution,verify_execution,meta_floor,meta_structure,meta_commit_shape]" \
  algorithm.use_kl_in_reward=False \
  reward.reward_manager.name=gdpo \
  ++reward.custom_reward_function.path=src/training/verl_reward.py \
  ++reward.custom_reward_function.name=compute_score_e21r_v4_smoke \
  critic.ppo_mini_batch_size="$PPO_MINI_BATCH" \
  critic.ppo_micro_batch_size=null \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.forward_micro_batch_size=null \
  critic.forward_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=32768 \
  trainer.project_name=metacot-math \
  trainer.experiment_name="$EXPERIMENT" \
  "trainer.logger=[console,wandb]" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="checkpoints/metacot-math/$EXPERIMENT" \
  ++trainer.remove_previous_ckpt=False \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "/scratch/${EXPERIMENT}.pid"
echo "STARTED META_V4 PID=$(cat /scratch/${EXPERIMENT}.pid)"

sleep 10
if kill -0 "$(cat /scratch/${EXPERIMENT}.pid)" 2>/dev/null; then
  echo "Process alive. First log lines:"
  head -5 "$LOG_FILE" 2>/dev/null || true
else
  echo "ERROR: Process died within 10s. Last log lines:" >&2
  tail -20 "$LOG_FILE" 2>/dev/null >&2
  exit 1
fi
