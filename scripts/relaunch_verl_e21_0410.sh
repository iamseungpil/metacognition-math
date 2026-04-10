#!/usr/bin/env bash
set -euo pipefail

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition:${PYTHONPATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:-2f4e627868f1f9dad10bcb1a14fbf96817e6baa9}"
export WANDB_PROJECT="${WANDB_PROJECT:-metacot-math}"
export WANDB_NAME="${WANDB_NAME:-verl_e21_historical_0410}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e21-historical}"
export HF_TOKEN="${HF_TOKEN:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CUMEM_ENABLE=0

source /scratch/simplerl_venv/bin/activate
mkdir -p data tmp

if [ ! -f data/verl_train.parquet ] || [ ! -f data/verl_val.parquet ]; then
  python src/training/verl_gdpo_data.py --mode mixed --split train --output data/verl_train.parquet
  python src/training/verl_gdpo_data.py --mode mixed --split val --output data/verl_val.parquet
fi

pkill -f 'verl.trainer.main_ppo.*verl_e21_historical_0410' 2>/dev/null || true
/scratch/simplerl_venv/bin/ray stop --force >/dev/null 2>&1 || true
sleep 2

nohup python -m verl.trainer.main_ppo \
  --config-path /scratch/simplerl_venv/lib/python3.10/site-packages/verl/trainer/config \
  --config-name ppo_trainer \
  actor_rollout_ref.model.path=checkpoints/v8_meta_inside_E20a \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.002 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
  data.train_files=data/verl_train.parquet \
  data.val_files=data/verl_val.parquet \
  data.max_prompt_length=512 \
  data.max_response_length=4096 \
  data.train_batch_size=128 \
  data.val_batch_size=64 \
  algorithm.adv_estimator=gdpo \
  ++algorithm.gdpo_reward_keys='[correctness,switch_v2,verify_v2,conf_traj,meta_floor]' \
  algorithm.use_kl_in_reward=False \
  ++reward.custom_reward_function.path=src/training/verl_reward.py \
  ++reward.custom_reward_function.name=compute_score \
  trainer.experiment_name=verl_e21_historical_0410 \
  trainer.project_name=metacot-math \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.default_local_dir=checkpoints/verl_e21_historical_0410 \
  trainer.logger='[console,wandb]' \
  > /scratch/verl_e21_historical_0410.log 2>&1 < /dev/null &

echo $! > /scratch/verl_e21_historical_0410.pid
echo STARTED_E21 $(cat /scratch/verl_e21_historical_0410.pid)

if [ -n "${HF_TOKEN}" ]; then
  nohup bash -lc '
    while [ ! -d /scratch/metacognition/checkpoints/verl_e21_historical_0410 ]; do
      sleep 60
    done
    cd /scratch/metacognition
    HF_TOKEN='"'"${HF_TOKEN}"'"' python scripts/sync_checkpoint_to_hf.py \
      --local-dir /scratch/metacognition/checkpoints/verl_e21_historical_0410 \
      --repo-id iamseungpil/metacot-verl-e21-historical \
      --repo-type model \
      --interval-sec 1800
  ' \
    > /scratch/hf_sync_e21_historical_0410.log 2>&1 < /dev/null &
  echo $! > /scratch/hf_sync_e21_historical_0410.pid
  echo STARTED_E21_HF_SYNC $(cat /scratch/hf_sync_e21_historical_0410.pid)
fi
