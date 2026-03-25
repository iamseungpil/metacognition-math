#!/bin/bash
# Phase 3: GRPO + Full Gnosis on Qwen3-8B Meta SFT
# Uses gnosis_repo's TRL GRPOTrainer with vLLM colocate
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate verl
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Use gnosis_repo's transformers and TRL (with Gnosis integration)
export PYTHONPATH="/scratch/metacognition/gnosis_repo/transformers/src:/scratch/metacognition/gnosis_repo/trl:$PYTHONPATH"
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

echo "=== Phase 3: GRPO + Full Gnosis ==="
echo "Model: checkpoints/qwen3_meta_sft (Qwen3-8B + Meta-CoT SFT)"
echo "Gnosis: Full (attention + hidden + confidence feature extractors)"
echo "Reward: R_correct + R_calib(p̂) + R_penalty"
echo "Generation: vLLM colocate"

python src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_gnosis \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
