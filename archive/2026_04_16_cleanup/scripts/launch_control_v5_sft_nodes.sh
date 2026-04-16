#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"

URL_ALL="${URL_ALL:-wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms}"
URL_VERIFY="${URL_VERIFY:-wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms}"
URL_REDIRECT="${URL_REDIRECT:-wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms}"

LOG_DIR="$ROOT/results/autoresearch_control_v5"
mkdir -p "$LOG_DIR"

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

bootstrap_remote() {
  local url="$1"
  proxy_ssh "$url" "mkdir -p /scratch/metacognition/{data,configs,scripts,src/training,src/metacot,checkpoints,results}"
}

prepare_remote_base_sft() {
  local url="$1"
  proxy_ssh "$url" "REMOTE_CONDA_ENV=$REMOTE_CONDA_ENV bash -s" <<'EOF'
set -euo pipefail
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate "$REMOTE_CONDA_ENV"
if [[ ! -f checkpoints/qwen3_base_sft/tokenizer_config.json ]]; then
  python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="iamseungpil/metacot",
    repo_type="dataset",
    allow_patterns=["models/qwen3_base_sft/*"],
    local_dir="/scratch/metacognition/hf_cache",
    local_dir_use_symlinks=False,
)
PY
  rm -rf checkpoints/qwen3_base_sft
  mkdir -p checkpoints
  cp -r /scratch/metacognition/hf_cache/models/qwen3_base_sft checkpoints/qwen3_base_sft
fi
EOF
}

copy_common_files() {
  local url="$1"
  copy_to_remote "$url" "$ROOT/configs/accelerate_sft.yaml" "/scratch/metacognition/configs/accelerate_sft.yaml"
  copy_to_remote "$url" "$ROOT/src/training/sft.py" "/scratch/metacognition/src/training/sft.py"
  copy_to_remote "$url" "$ROOT/src/metacot/prompt.py" "/scratch/metacognition/src/metacot/prompt.py"
  copy_to_remote "$url" "$ROOT/scripts/launch_sft_remote.sh" "/scratch/metacognition/scripts/launch_sft_remote.sh"
}

launch_remote() {
  local url="$1"
  local config_src="$2"
  local data_src="$3"
  local launcher_src="$4"
  local remote_log="$5"

  local config_name
  local data_name
  local launcher_name
  config_name="$(basename "$config_src")"
  data_name="$(basename "$data_src")"
  launcher_name="$(basename "$launcher_src")"

  copy_common_files "$url"
  copy_to_remote "$url" "$config_src" "/scratch/metacognition/configs/$config_name"
  copy_to_remote "$url" "$data_src" "/scratch/metacognition/data/$data_name"
  copy_to_remote "$url" "$launcher_src" "/scratch/metacognition/scripts/$launcher_name"

  proxy_ssh "$url" "bash -lc 'export REMOTE_CONDA_ENV=$REMOTE_CONDA_ENV && chmod +x /scratch/metacognition/scripts/launch_sft_remote.sh /scratch/metacognition/scripts/$launcher_name && cd /scratch/metacognition && nohup bash /scratch/metacognition/scripts/$launcher_name > /scratch/$remote_log 2>&1 < /dev/null & echo \$!'"
}

echo "[bootstrap] all"
bootstrap_remote "$URL_ALL"
prepare_remote_base_sft "$URL_ALL"
echo "[bootstrap] verify"
bootstrap_remote "$URL_VERIFY"
prepare_remote_base_sft "$URL_VERIFY"
echo "[bootstrap] redirect"
bootstrap_remote "$URL_REDIRECT"
prepare_remote_base_sft "$URL_REDIRECT"

echo "[launch] all"
launch_remote \
  "$URL_ALL" \
  "$ROOT/configs/sft_control_v5_all.yaml" \
  "$ROOT/data/control_v5_all_sft.parquet" \
  "$ROOT/scripts/launch_control_v5_all_sft_remote.sh" \
  "control_v5_all_sft.log" | tee "$LOG_DIR/launch_all.pid"

echo "[launch] verify"
launch_remote \
  "$URL_VERIFY" \
  "$ROOT/configs/sft_control_v5_verify.yaml" \
  "$ROOT/data/control_v5_verify_sft.parquet" \
  "$ROOT/scripts/launch_control_v5_verify_sft_remote.sh" \
  "control_v5_verify_sft.log" | tee "$LOG_DIR/launch_verify.pid"

echo "[launch] redirect"
launch_remote \
  "$URL_REDIRECT" \
  "$ROOT/configs/sft_control_v5_redirect.yaml" \
  "$ROOT/data/control_v5_redirect_sft.parquet" \
  "$ROOT/scripts/launch_control_v5_redirect_sft_remote.sh" \
  "control_v5_redirect_sft.log" | tee "$LOG_DIR/launch_redirect.pid"

echo "[done] launched control-v5 SFT runs on 3 nodes"
