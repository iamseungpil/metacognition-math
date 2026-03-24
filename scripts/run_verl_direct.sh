#!/bin/bash
# Run GRPO directly with verl (no Agent Lightning)
# 4 GPU: vLLM rollout + FSDP training
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate verl
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

# Prepare data if not exists
if [ ! -f verl_train.parquet ]; then
    echo "=== Preparing data ==="
    python -c "
import pandas as pd
df = pd.read_parquet('rollouts/rollouts_final.parquet')
problems = df.drop_duplicates('problem_id')
verl_data = [{'data_source': 'metacot_math', 'prompt': [{'role': 'user', 'content': row['question']}], 'reward_model': {'ground_truth': row['gold_answer']}} for _, row in problems.iterrows()]
pd.DataFrame(verl_data[:len(verl_data)-100]).to_parquet('verl_train.parquet', index=False)
pd.DataFrame(verl_data[len(verl_data)-100:]).to_parquet('verl_val.parquet', index=False)
print(f'Train: {len(verl_data)-100}, Val: 100')
"
fi

echo "=== Starting verl GRPO (4 GPU, direct) ==="
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=verl_train.parquet \
    data.val_files=verl_val.parquet \
    actor_rollout_ref.model.path=checkpoints/meta_sft \
    trainer.n_gpus_per_node=4 \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=metacot-math \
    trainer.experiment_name=metacot-verl-grpo-direct \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=100 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=1000

echo "VERL_DIRECT_DONE"
