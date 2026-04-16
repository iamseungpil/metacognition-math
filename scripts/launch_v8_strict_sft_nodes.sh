#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"

URL_EVAL="${URL_EVAL:-wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms}"
URL_TRAIN_B="${URL_TRAIN_B:-wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms}"

LOG_DIR="$ROOT/results/strict_data"
mkdir -p "$LOG_DIR"

proxy_ssh() {
  local url="$1"
  shift
  ssh -T -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" azureuser@placeholder "$@"
}

copy_to_remote() {
  local url="$1"
  local src="$2"
  local dst="$3"
  scp -q -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" "$src" "azureuser@placeholder:$dst"
}

bootstrap_remote() {
  local url="$1"
  proxy_ssh "$url" "mkdir -p /scratch/metacognition/{data,configs,scripts,src/training,src/metacot,checkpoints,results}"
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

  local config_name data_name launcher_name
  config_name="$(basename "$config_src")"
  data_name="$(basename "$data_src")"
  launcher_name="$(basename "$launcher_src")"

  copy_common_files "$url"
  copy_to_remote "$url" "$config_src" "/scratch/metacognition/configs/$config_name"
  copy_to_remote "$url" "$data_src" "/scratch/metacognition/data/$data_name"
  copy_to_remote "$url" "$launcher_src" "/scratch/metacognition/scripts/$launcher_name"

  proxy_ssh "$url" "export REMOTE_CONDA_ENV=$REMOTE_CONDA_ENV; chmod +x /scratch/metacognition/scripts/launch_sft_remote.sh /scratch/metacognition/scripts/$launcher_name; cd /scratch/metacognition; nohup bash /scratch/metacognition/scripts/$launcher_name > /scratch/$remote_log 2>&1 < /dev/null & echo \$!"
}

echo "[bootstrap] eval"
bootstrap_remote "$URL_EVAL"

echo "[bootstrap] train_b"
bootstrap_remote "$URL_TRAIN_B"

echo "[launch] eval -> strict meta"
launch_remote \
  "$URL_EVAL" \
  "$ROOT/configs/sft_v8_meta_inside_strict.yaml" \
  "$ROOT/data/v8_meta_inside_strict.parquet" \
  "$ROOT/scripts/launch_v8_meta_inside_strict_remote.sh" \
  "v8_meta_inside_strict_sft.log" | tee "$LOG_DIR/launch_v8_meta_inside_strict.pid"

echo "[launch] train_b -> strict base"
launch_remote \
  "$URL_TRAIN_B" \
  "$ROOT/configs/sft_v8_base_matched_strict.yaml" \
  "$ROOT/data/v8_base_matched_strict.parquet" \
  "$ROOT/scripts/launch_v8_base_matched_strict_remote.sh" \
  "v8_base_matched_strict_sft.log" | tee "$LOG_DIR/launch_v8_base_matched_strict.pid"

echo "[done] launched strict paired SFT runs on eval/train_b"
