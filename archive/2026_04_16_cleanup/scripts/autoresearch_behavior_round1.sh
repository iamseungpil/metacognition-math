#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN_PID="${1:-}"
LOG_DIR="$PROJECT_ROOT/results/autoresearch_round1"
LOG_FILE="$LOG_DIR/pipeline.log"
mkdir -p "$LOG_DIR"

if [[ -z "$GEN_PID" ]]; then
    echo "usage: $0 <generator_pid>" >&2
    exit 1
fi

DATA_FILE="$PROJECT_ROOT/data/metacot_behavior_trapi_round1.parquet"
AZ_PY="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
TOPS_URL="wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms"
E8_URL="wss://ssh-2etszrmvdrq4cwqdql4al50f32aqiwdcl036benvkg6kmzk8bpc.westus2.nodes.azureml.ms"
SSH_KEY="$HOME/.ssh/id_rsa"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

proxy_ssh() {
    local url="$1"
    shift
    ssh -o ConnectTimeout=20 \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o "ProxyCommand=$AZ_PY $CONNECTOR $url" \
        -i "$SSH_KEY" \
        azureuser@placeholder "$@"
}

copy_file() {
    local url="$1"
    local src="$2"
    local dst="$3"
    local dst_dir
    dst_dir="$(dirname "$dst")"
    proxy_ssh "$url" "mkdir -p '$dst_dir'"
    scp -o ConnectTimeout=20 \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o "ProxyCommand=$AZ_PY $CONNECTOR $url" \
        -i "$SSH_KEY" \
        "$src" \
        "azureuser@placeholder:$dst"
}

log "waiting for TRAPI generation pid=$GEN_PID"
deadline=$(( $(date +%s) + 43200 ))
while [[ ! -f "$DATA_FILE" ]]; do
    now=$(date +%s)
    if (( now >= deadline )); then
        log "timed out waiting for data file: $DATA_FILE"
        exit 1
    fi
    if kill -0 "$GEN_PID" 2>/dev/null; then
        sleep 30
    else
        log "generator pid not visible yet; still waiting for output file"
        sleep 60
    fi
done

log "building behavior SFT variants"
cd "$PROJECT_ROOT"
python scripts/build_behavior_sft_variants.py \
    --input "$DATA_FILE" \
    --output-dir "$PROJECT_ROOT/data" | tee -a "$LOG_FILE"

FILES_TO_COPY=(
    "src/training/grpo_v2.py"
    "src/training/rewards.py"
    "src/training/sft.py"
    "src/metacot/prompt.py"
    "data/behavior_all_sft.parquet"
    "data/behavior_redirect_sft.parquet"
    "configs/sft_behavior_all.yaml"
    "configs/sft_behavior_redirect.yaml"
)

log "copying code and data to tops-caiman"
proxy_ssh "$TOPS_URL" "mkdir -p /scratch/metacognition/src/training /scratch/metacognition/src/metacot /scratch/metacognition/configs /scratch/metacognition/data"
for rel in "${FILES_TO_COPY[@]}"; do
    copy_file "$TOPS_URL" "$PROJECT_ROOT/$rel" "/scratch/metacognition/$rel"
done

log "copying code and data to metacognition_e8"
proxy_ssh "$E8_URL" "mkdir -p /scratch/metacognition/src/training /scratch/metacognition/src/metacot /scratch/metacognition/configs /scratch/metacognition/data"
for rel in "${FILES_TO_COPY[@]}"; do
    copy_file "$E8_URL" "$PROJECT_ROOT/$rel" "/scratch/metacognition/$rel"
done

log "launching combined behavior SFT on tops-caiman"
proxy_ssh "$TOPS_URL" "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && cd /scratch/metacognition && export PYTHONPATH=/scratch/metacognition && nohup accelerate launch --config_file configs/accelerate_sft.yaml src/training/sft.py --config configs/sft_behavior_all.yaml > /scratch/behavior_all_sft.log 2>&1 & echo \$! > /scratch/behavior_all_sft.pid'"

log "launching redirect-focused behavior SFT on metacognition_e8"
proxy_ssh "$E8_URL" "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && cd /scratch/metacognition && export PYTHONPATH=/scratch/metacognition && nohup accelerate launch --config_file configs/accelerate_sft.yaml src/training/sft.py --config configs/sft_behavior_redirect.yaml > /scratch/behavior_redirect_sft.log 2>&1 & echo \$! > /scratch/behavior_redirect_sft.pid'"

log "waiting for combined behavior SFT to finish before E9 GDPO"
proxy_ssh "$TOPS_URL" "bash -lc 'while [ ! -f /scratch/metacognition/checkpoints/qwen3_metacot_behavior_all_sft/tokenizer_config.json ]; do sleep 60; done; source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && cd /scratch/metacognition && export PYTHONPATH=/scratch/metacognition && nohup accelerate launch --config_file configs/accelerate_grpo.yaml src/training/grpo_v2.py --mode E9 --max_steps 300 --model_path checkpoints/qwen3_metacot_behavior_all_sft --data mixed_train --output_dir checkpoints/grpo_v2_behavior_all_E9 > /scratch/grpo_behavior_all_e9.log 2>&1 & echo \$! > /scratch/grpo_behavior_all_e9.pid'" &

log "round1 pipeline launched"
