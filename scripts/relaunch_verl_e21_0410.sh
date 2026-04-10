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

if [ ! -f /scratch/metacognition/configs/verl07_e21.yaml ]; then
  echo "Missing config: /scratch/metacognition/configs/verl07_e21.yaml" >&2
  exit 1
fi

if [ ! -f data/verl_train.parquet ] || [ ! -f data/verl_val.parquet ]; then
  python src/training/verl_gdpo_data.py --mode mixed --split train --output data/verl_train.parquet
  python src/training/verl_gdpo_data.py --mode mixed --split val --output data/verl_val.parquet
fi

pkill -f 'verl.trainer.main_ppo.*verl_e21_historical_0410' 2>/dev/null || true
/scratch/simplerl_venv/bin/ray stop --force >/dev/null 2>&1 || true
sleep 2

nohup python -m verl.trainer.main_ppo \
  --config-path /scratch/metacognition/configs \
  --config-name verl07_e21 \
  trainer.experiment_name=verl_e21_historical_0410 \
  trainer.project_name=metacot-math \
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
