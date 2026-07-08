#!/bin/bash
# Launch V6 clean 10K SFT on 3 nodes after data merge
# Plan: Case B mainline (clean-data restart from base_sft)
#
# Node allocation:
#   EVAL    → E19  (3ep, lr=2e-6, mainline)
#   TRAIN_B → E19b (5ep, lr=1e-6, ablation)
#   E8      → E19c (3ep, lr=5e-6, ablation)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="$HOME/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="$HOME/.ssh/id_rsa"

URL_EVAL="wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms"
URL_TRAIN_B="wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms"
URL_E8="wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms"

MERGED_DATA="data/v6_clean_10k_merged.parquet"
HF_TOKEN="${HF_TOKEN}"
HF_REPO="iamseungpil/metacot"

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

# ── Step 1: Check that merged data exists ──
if [ ! -f "$MERGED_DATA" ]; then
    echo "ERROR: $MERGED_DATA not found. Run merge_v6_clean_10k.py first."
    exit 1
fi
echo "✓ Merged data exists: $(python3 -c "import pandas as pd; print(f'{len(pd.read_parquet(\"$MERGED_DATA\"))} rows')")"

# ── Step 2: Push merged data to HuggingFace ──
echo "Pushing merged data to HuggingFace..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_file(
    path_or_fileobj='$MERGED_DATA',
    path_in_repo='v6_clean_10k_merged.parquet',
    repo_id='$HF_REPO',
    repo_type='dataset'
)
print('✓ HF push complete')
" || echo "WARNING: HF push failed, continuing..."

# ── Step 3: Upload data + configs to all nodes ──
echo "Uploading to nodes..."

declare -A NODE_URLS=(
    ["EVAL"]="$URL_EVAL"
    ["TRAIN_B"]="$URL_TRAIN_B"
    ["E8"]="$URL_E8"
)
declare -A NODE_CONFIGS=(
    ["EVAL"]="configs/sft_v6_clean_10k.yaml"
    ["TRAIN_B"]="configs/sft_v6_clean_10k_5ep.yaml"
    ["E8"]="configs/sft_v6_clean_10k_highlr.yaml"
)
declare -A NODE_LABELS=(
    ["EVAL"]="E19-mainline"
    ["TRAIN_B"]="E19b-5ep-ablation"
    ["E8"]="E19c-highlr-ablation"
)

for node in EVAL TRAIN_B E8; do
    url="${NODE_URLS[$node]}"
    config="${NODE_CONFIGS[$node]}"
    label="${NODE_LABELS[$node]}"

    echo "  [$node] Creating dirs..."
    run_remote "$url" "mkdir -p /scratch/metacognition/data /scratch/metacognition/configs"

    echo "  [$node] Uploading merged data..."
    copy_to_remote "$url" "$MERGED_DATA" "/scratch/metacognition/$MERGED_DATA"

    echo "  [$node] Uploading config ($config)..."
    copy_to_remote "$url" "$config" "/scratch/metacognition/$config"

    echo "  [$node] Uploading accelerate config..."
    copy_to_remote "$url" "configs/accelerate_sft.yaml" "/scratch/metacognition/configs/accelerate_sft.yaml"

    echo "  [$node] ✓ Upload complete for $label"
done

# ── Step 4: Launch SFT on all nodes ──
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
conda activate grpo
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=\$(cat ~/.wandb_key 2>/dev/null || echo ${WANDB_API_KEY})

# Print launcher contract
echo "plan_id: v6.4-CaseB"
echo "run_label: $label"
echo "mode: SFT"
echo "base_checkpoint: checkpoints/qwen3_base_sft"
echo "config: $config"
echo "output_dir: \$(grep output_dir $config | awk '{print \$2}')"

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
echo "  EVAL:    E19  (3ep, lr=2e-6, mainline)"
echo "  TRAIN_B: E19b (5ep, lr=1e-6, ablation)"
echo "  E8:      E19c (3ep, lr=5e-6, ablation)"
echo "═══════════════════════════════════════════"
echo ""
echo "Monitor logs:"
echo "  ssh EVAL 'tail -f /scratch/E19-mainline_sft.log'"
echo "  ssh TRAIN_B 'tail -f /scratch/E19b-5ep-ablation_sft.log'"
echo "  ssh E8 'tail -f /scratch/E19c-highlr-ablation_sft.log'"
