#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"

URL_ALL="${URL_ALL:-wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms}"
URL_VERIFY="${URL_VERIFY:-wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms}"
URL_REDIRECT="${URL_REDIRECT:-wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms}"
URL_EVAL="${URL_EVAL:-wss://ssh-2etszrmvdrq4cwqdql4al50f36wf6zisw5m1s0rcb6go29r732c.westus2.nodes.azureml.ms}"

LOG_DIR="$ROOT/results/autoresearch_control_v5"
mkdir -p "$LOG_DIR"

FULL_OUT="${FULL_OUT:-$ROOT/data/control_v5_10k.parquet}"
REJECTIONS_OUT="${REJECTIONS_OUT:-$ROOT/data/control_v5_10k.rejections.jsonl}"

proxy_ssh() {
  local url="$1"
  shift
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" azureuser@placeholder "$@"
}

copy_to_remote() {
  local url="$1"
  local src="$2"
  local dst="$3"
  scp -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" "$src" "azureuser@placeholder:$dst"
}

wait_for_file() {
  local path="$1"
  echo "[wait] waiting for $path"
  while [[ ! -f "$path" ]]; do
    sleep 30
  done
}

bootstrap_remote() {
  local url="$1"
  proxy_ssh "$url" "mkdir -p /scratch/metacognition/{data,configs,scripts,src/training,checkpoints,results}"
}

prepare_remote_base_sft() {
  local url="$1"
  proxy_ssh "$url" "bash -lc '
set -euo pipefail
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
if [[ ! -f checkpoints/qwen3_base_sft/tokenizer_config.json ]]; then
  python - <<\"PY\"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=\"iamseungpil/metacot\",
    repo_type=\"dataset\",
    allow_patterns=[\"models/qwen3_base_sft/*\"],
    local_dir=\"/scratch/metacognition/hf_cache\",
    local_dir_use_symlinks=False,
)
PY
  rm -rf checkpoints/qwen3_base_sft
  mkdir -p checkpoints
  cp -r /scratch/metacognition/hf_cache/models/qwen3_base_sft checkpoints/qwen3_base_sft
fi
'"
}

launch_sft() {
  local url="$1"
  local launcher="$2"
  local log_name="$3"
  proxy_ssh "$url" "bash -lc 'chmod +x /scratch/metacognition/scripts/$launcher && nohup bash /scratch/metacognition/scripts/$launcher > /scratch/$log_name 2>&1 < /dev/null & echo \$!'"
}

wait_for_file "$FULL_OUT"

echo "[qc] running control-v5 QC"
python3 scripts/qc_control_v5_samples.py --input "$FULL_OUT" \
  > "$LOG_DIR/control_v5_qc.txt"

echo "[build] building control-v5 SFT variants"
python3 scripts/build_control_v5_sft_variants.py \
  --input "$FULL_OUT" \
  --output-dir "$ROOT/data" | tee "$LOG_DIR/build_variants.log"

echo "[hf] uploading v4/v5 data artifacts"
python3 scripts/upload_dataset_artifacts.py --files \
  "$ROOT/data/control_v4_trapi_round1.parquet" \
  "$ROOT/data/control_v4_all_sft.parquet" \
  "$ROOT/data/control_v4_verify_sft.parquet" \
  "$ROOT/data/control_v4_redirect_sft.parquet" \
  "$FULL_OUT" \
  "$REJECTIONS_OUT" \
  "$ROOT/data/control_v5_all_sft.parquet" \
  "$ROOT/data/control_v5_verify_sft.parquet" \
  "$ROOT/data/control_v5_redirect_sft.parquet" \
  | tee "$LOG_DIR/hf_upload.log"

for url in "$URL_ALL" "$URL_VERIFY" "$URL_REDIRECT" "$URL_EVAL"; do
  bootstrap_remote "$url"
  prepare_remote_base_sft "$url"
done

for url in "$URL_ALL" "$URL_VERIFY" "$URL_REDIRECT"; do
  copy_to_remote "$url" "$ROOT/configs/accelerate_sft.yaml" "/scratch/metacognition/configs/accelerate_sft.yaml"
  copy_to_remote "$url" "$ROOT/src/training/sft.py" "/scratch/metacognition/src/training/sft.py"
  copy_to_remote "$url" "$ROOT/scripts/launch_sft_remote.sh" "/scratch/metacognition/scripts/launch_sft_remote.sh"
done

copy_to_remote "$URL_ALL" "$ROOT/data/control_v5_all_sft.parquet" "/scratch/metacognition/data/control_v5_all_sft.parquet"
copy_to_remote "$URL_VERIFY" "$ROOT/data/control_v5_verify_sft.parquet" "/scratch/metacognition/data/control_v5_verify_sft.parquet"
copy_to_remote "$URL_REDIRECT" "$ROOT/data/control_v5_redirect_sft.parquet" "/scratch/metacognition/data/control_v5_redirect_sft.parquet"

copy_to_remote "$URL_ALL" "$ROOT/configs/sft_control_v5_all.yaml" "/scratch/metacognition/configs/sft_control_v5_all.yaml"
copy_to_remote "$URL_VERIFY" "$ROOT/configs/sft_control_v5_verify.yaml" "/scratch/metacognition/configs/sft_control_v5_verify.yaml"
copy_to_remote "$URL_REDIRECT" "$ROOT/configs/sft_control_v5_redirect.yaml" "/scratch/metacognition/configs/sft_control_v5_redirect.yaml"

copy_to_remote "$URL_ALL" "$ROOT/scripts/launch_control_v5_all_sft_remote.sh" "/scratch/metacognition/scripts/launch_control_v5_all_sft_remote.sh"
copy_to_remote "$URL_VERIFY" "$ROOT/scripts/launch_control_v5_verify_sft_remote.sh" "/scratch/metacognition/scripts/launch_control_v5_verify_sft_remote.sh"
copy_to_remote "$URL_REDIRECT" "$ROOT/scripts/launch_control_v5_redirect_sft_remote.sh" "/scratch/metacognition/scripts/launch_control_v5_redirect_sft_remote.sh"

echo "[launch] control-v5 all"
launch_sft "$URL_ALL" "launch_control_v5_all_sft_remote.sh" "control_v5_all_sft.log" \
  | tee "$LOG_DIR/launch_all.pid"

echo "[launch] control-v5 verify"
launch_sft "$URL_VERIFY" "launch_control_v5_verify_sft_remote.sh" "control_v5_verify_sft.log" \
  | tee "$LOG_DIR/launch_verify.pid"

echo "[launch] control-v5 redirect"
launch_sft "$URL_REDIRECT" "launch_control_v5_redirect_sft_remote.sh" "control_v5_redirect_sft.log" \
  | tee "$LOG_DIR/launch_redirect.pid"

echo "[done] launched control-v5 SFT runs"
