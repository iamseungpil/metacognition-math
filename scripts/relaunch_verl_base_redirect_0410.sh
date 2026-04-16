#!/usr/bin/env bash
set -euo pipefail

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition:${PYTHONPATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:-2f4e627868f1f9dad10bcb1a14fbf96817e6baa9}"
export WANDB_PROJECT="${WANDB_PROJECT:-metacot-math}"
export WANDB_NAME="${WANDB_NAME:-verl_base_redirect_recovery_0410}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-redirect-pilot}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CUMEM_ENABLE=0

source /scratch/simplerl_venv/bin/activate

pkill -f 'verl.trainer.main_ppo.*verl_base_redirect_recovery_0410' 2>/dev/null || true
/scratch/simplerl_venv/bin/ray stop --force >/dev/null 2>&1 || true
sleep 2

nohup python -m verl.trainer.main_ppo \
  --config-path /scratch/simplerl_venv/lib/python3.10/site-packages/verl/trainer/config \
  --config-name ppo_trainer \
  actor_rollout_ref.model.path=checkpoints/v8_base_matched_clean_sft \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.002 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size=null \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.response_length=1536 \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  ++actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=null \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  data.train_files=data/verl_train_redirect_base.parquet \
  data.val_files=data/verl_val_redirect_base.parquet \
  data.max_prompt_length=512 \
  data.max_response_length=1536 \
  data.train_batch_size=32 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  ++reward.custom_reward_function.path=src/training/verl_reward.py \
  ++reward.custom_reward_function.name=compute_score_base \
  critic.ppo_mini_batch_size=8 \
  critic.ppo_micro_batch_size=null \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.forward_micro_batch_size=null \
  critic.forward_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=32768 \
  trainer.project_name=metacot-math \
  trainer.experiment_name=verl_base_redirect_recovery_0410 \
  trainer.logger='[console,wandb]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  > /scratch/verl_base_redirect_recovery_0410.log 2>&1 < /dev/null &

echo $! > /scratch/verl_base_redirect_recovery_0410.pid
echo STARTED_BASE $(cat /scratch/verl_base_redirect_recovery_0410.pid)
