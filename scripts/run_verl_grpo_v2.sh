#!/bin/bash
# veRL GRPO v2: 4 GPU, vLLM rollout, FSDP training
# SimpleCorrectnessProbe integration via text-based R_calib (Phase 1)
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate verl
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

# Kill any existing training
pkill -f "grpo\|verl\|vllm" 2>/dev/null || true
sleep 2

echo "=== Preparing verl training data ==="
python << 'PYEOF'
import pandas as pd
import json

df = pd.read_parquet("rollouts/rollouts_final.parquet")
problems = df.drop_duplicates("problem_id").reset_index(drop=True)

verl_data = []
for _, row in problems.iterrows():
    verl_data.append({
        "data_source": "metacot_math",
        "prompt": [{"role": "user", "content": row["question"]}],
        "reward_model": {"ground_truth": row["gold_answer"]},
    })

n = len(verl_data)
train_data = verl_data[:n-100]
val_data = verl_data[n-100:]

pd.DataFrame(train_data).to_parquet("verl_train.parquet", index=False)
pd.DataFrame(val_data).to_parquet("verl_val.parquet", index=False)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
PYEOF

echo "=== Ensuring <|meta|> tokens in tokenizer ==="
python << 'PYEOF'
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/meta_sft", trust_remote_code=True)
meta_tokens = ["<|meta|>", "<|/meta|>"]
existing = set(tokenizer.additional_special_tokens or [])
to_add = [t for t in meta_tokens if t not in existing]
if to_add:
    tokenizer.add_special_tokens({"additional_special_tokens": list(existing) + to_add})
    tokenizer.save_pretrained("checkpoints/meta_sft")
    print(f"Added {to_add} to tokenizer")
else:
    print("Meta tokens already in tokenizer")
PYEOF

echo "=== Starting veRL GRPO v2 (4 GPU, vLLM + FSDP) ==="
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=verl_train.parquet \
    data.val_files=verl_val.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=checkpoints/meta_sft \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=src/training/verl_reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=metacot-math \
    trainer.experiment_name=metacot-verl-grpo-v2 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=200 \
    trainer.test_freq=100 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=1000

echo "VERL_GRPO_V2_DONE"
