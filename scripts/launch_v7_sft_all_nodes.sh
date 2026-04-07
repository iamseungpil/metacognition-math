#!/bin/bash
# Launch V7 Think-Meta SFT on 3 nodes
# Plan: Phase 1 of V7 experiments
#
# Node allocation:
#   EVAL    → E19v2  (V7 full, 3ep, lr=2e-6, mainline)
#   TRAIN_B → E19v2b (V7 full, 5ep, lr=1e-6, epoch ablation)
#   E8      → E19v2c (V7 real-only, 5ep, lr=2e-6, template vs real)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="$HOME/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="$HOME/.ssh/id_rsa"

URL_EVAL="wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms"
URL_TRAIN_B="wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms"
URL_E8="wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms"

run_remote() {
    local url="$1" cmd="$2"
    ssh -T -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
        -i "$SSH_KEY" azureuser@placeholder "$cmd" 2>/dev/null
}

copy_to_remote() {
    local url="$1" src="$2" dst="$3"
    scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
        -i "$SSH_KEY" "$src" "azureuser@placeholder:$dst" 2>/dev/null
}

declare -A NODE_URLS=(
    ["EVAL"]="$URL_EVAL"
    ["TRAIN_B"]="$URL_TRAIN_B"
    ["E8"]="$URL_E8"
)
declare -A NODE_CONFIGS=(
    ["EVAL"]="configs/sft_v7_think_meta.yaml"
    ["TRAIN_B"]="configs/sft_v7_think_meta_5ep.yaml"
    ["E8"]="configs/sft_v7_real_only.yaml"
)
declare -A NODE_DATA=(
    ["EVAL"]="data/v7_think_meta_merged.parquet"
    ["TRAIN_B"]="data/v7_think_meta_merged.parquet"
    ["E8"]="data/v7_real_only.parquet"
)
declare -A NODE_LABELS=(
    ["EVAL"]="E19v2-mainline"
    ["TRAIN_B"]="E19v2b-5ep"
    ["E8"]="E19v2c-real-only"
)

echo "Uploading to nodes..."
for node in EVAL TRAIN_B E8; do
    url="${NODE_URLS[$node]}"
    config="${NODE_CONFIGS[$node]}"
    data="${NODE_DATA[$node]}"
    label="${NODE_LABELS[$node]}"

    echo "  [$node] Creating dirs..."
    run_remote "$url" "mkdir -p /scratch/metacognition/data /scratch/metacognition/configs"

    echo "  [$node] Uploading data..."
    copy_to_remote "$url" "$data" "/scratch/metacognition/$data"

    echo "  [$node] Uploading config..."
    copy_to_remote "$url" "$config" "/scratch/metacognition/$config"

    echo "  [$node] Uploading accelerate config..."
    copy_to_remote "$url" "configs/accelerate_sft.yaml" "/scratch/metacognition/configs/accelerate_sft.yaml"

    echo "  [$node] ✓ Upload complete for $label"
done

echo ""
echo "Launching SFT training..."
for node in EVAL TRAIN_B E8; do
    url="${NODE_URLS[$node]}"
    config="${NODE_CONFIGS[$node]}"
    label="${NODE_LABELS[$node]}"

    echo "  [$node] Launching $label..."
    run_remote "$url" "$(cat <<EOF
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=\$(cat ~/.wandb_key 2>/dev/null || echo 2f4e627868f1f9dad10bcb1a14fbf96817e6baa9)

echo "plan_id: v7-Phase1"
echo "run_label: $label"
echo "mode: SFT"
echo "base_checkpoint: checkpoints/qwen3_base_sft"
echo "config: $config"

nohup accelerate launch --config_file configs/accelerate_sft.yaml \
    src/training/sft.py --config $config \
    > /scratch/${label}_sft.log 2>&1 &
echo "PID=\$!"
EOF
)"
    echo "  [$node] ✓ Launched"
done

echo ""
echo "═══════════════════════════════════════════"
echo "  ALL 3 NODES LAUNCHED"
echo "  EVAL:    E19v2  (V7 full, 3ep)"
echo "  TRAIN_B: E19v2b (V7 full, 5ep)"
echo "  E8:      E19v2c (V7 real-only, 5ep)"
echo "═══════════════════════════════════════════"
