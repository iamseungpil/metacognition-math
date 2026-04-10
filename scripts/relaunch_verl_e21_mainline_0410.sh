#!/usr/bin/env bash
set -euo pipefail

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition:${PYTHONPATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:-2f4e627868f1f9dad10bcb1a14fbf96817e6baa9}"
export WANDB_PROJECT="${WANDB_PROJECT:-metacot-math}"
export WANDB_NAME="${WANDB_NAME:-verl_e21_mainline_0410}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-historical-e21}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CUMEM_ENABLE=0
export HF_TOKEN="${HF_TOKEN:-}"

source /scratch/simplerl_venv/bin/activate

if [[ ! -f data/verl_train.parquet ]]; then
  python src/training/verl_gdpo_data.py --mode mixed --split train --output data/verl_train.parquet
fi
if [[ ! -f data/verl_val.parquet ]]; then
  python src/training/verl_gdpo_data.py --mode mixed --split val --output data/verl_val.parquet
fi

pkill -f 'verl.trainer.main_ppo.*verl_e21_mainline_0410' 2>/dev/null || true
pkill -f 'verl.trainer.main_ppo.*verl_e21_gdpo_v2' 2>/dev/null || true
pkill -f '/scratch/run_e21.sh' 2>/dev/null || true
pkill -f 'src/training/verl_gdpo.py.*verl_gdpo_e21' 2>/dev/null || true
pkill -f 'hf_sync_latest.py --source-dir checkpoints/verl07_e21_mainline' 2>/dev/null || true
/scratch/simplerl_venv/bin/ray stop --force >/dev/null 2>&1 || true
sleep 2

nohup python src/training/verl_gdpo.py \
  --config-name verl_gdpo_e21 \
  mode=E21 \
  actor_rollout_ref.model.path=checkpoints/v8_meta_inside_E20a \
  data.train_files=data/verl_train.parquet \
  data.val_files=data/verl_val.parquet \
  actor_rollout_ref.actor.ppo_micro_batch_size=null \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=null \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  critic.ppo_micro_batch_size=null \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.forward_micro_batch_size=null \
  critic.forward_micro_batch_size_per_gpu=1 \
  algorithm.adv_estimator=gdpo \
  trainer.project_name=metacot-math \
  trainer.experiment_name=verl_e21_mainline_0410 \
  trainer.logger='[console,wandb]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.default_local_dir=checkpoints/verl07_e21_mainline \
  > /scratch/verl_e21_mainline_0410.log 2>&1 < /dev/null &

echo $! > /scratch/verl_e21_mainline_0410.pid
echo STARTED_E21 $(cat /scratch/verl_e21_mainline_0410.pid)

if [[ -n "${HF_TOKEN}" ]]; then
  nohup bash -lc '
    cd /scratch/metacognition
    source /scratch/simplerl_venv/bin/activate
    while true; do
      python scripts/hf_sync_latest.py \
        --source-dir checkpoints/verl07_e21_mainline \
        --repo-id iamseungpil/metacot \
        --repo-type dataset \
        --path-in-repo models/verl07_e21_mainline_latest || true
      sleep 900
    done
  ' > /scratch/hf_sync_e21_mainline.log 2>&1 < /dev/null &
  echo $! > /scratch/hf_sync_e21_mainline.pid
  echo STARTED_HF_SYNC $(cat /scratch/hf_sync_e21_mainline.pid)
fi
