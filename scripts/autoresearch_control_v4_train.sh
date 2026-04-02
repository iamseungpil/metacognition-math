#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"

URL_TOPS="wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms"
URL_E8="wss://ssh-2etszrmvdrq4cwqdql4al50f32aqiwdcl036benvkg6kmzk8bpc.westus2.nodes.azureml.ms"
URL_EVAL_E8="wss://ssh-2etszrmvdrq4cwqdql4al50f365fggn0cs41y3ld90c6m331nlc.westus2.nodes.azureml.ms"

LOG_DIR="$ROOT/results/autoresearch_control_v4"
mkdir -p "$LOG_DIR"

FULL_OUT="$ROOT/data/control_v4_trapi_round1.parquet"

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

echo "[wait] waiting for main control-v4 parquet: $FULL_OUT"
while [[ ! -f "$FULL_OUT" ]]; do
  sleep 30
done

echo "[build] building SFT variants"
python3 scripts/build_control_v4_sft_variants.py \
  --input "$FULL_OUT" \
  --output-dir "$ROOT/data" | tee "$LOG_DIR/build_variants_after_main.log"

for url in "$URL_TOPS" "$URL_E8" "$URL_EVAL_E8"; do
  proxy_ssh "$url" "mkdir -p /scratch/metacognition/data /scratch/metacognition/configs /scratch/metacognition/scripts"
done

for url in "$URL_TOPS" "$URL_E8" "$URL_EVAL_E8"; do
  copy_to_remote "$url" "$ROOT/data/control_v4_all_sft.parquet" "/scratch/metacognition/data/control_v4_all_sft.parquet"
  copy_to_remote "$url" "$ROOT/data/control_v4_redirect_sft.parquet" "/scratch/metacognition/data/control_v4_redirect_sft.parquet"
  copy_to_remote "$url" "$ROOT/data/control_v4_verify_sft.parquet" "/scratch/metacognition/data/control_v4_verify_sft.parquet"
  copy_to_remote "$url" "$ROOT/configs/sft_control_v4_all.yaml" "/scratch/metacognition/configs/sft_control_v4_all.yaml"
  copy_to_remote "$url" "$ROOT/configs/sft_control_v4_redirect.yaml" "/scratch/metacognition/configs/sft_control_v4_redirect.yaml"
  copy_to_remote "$url" "$ROOT/configs/sft_control_v4_verify.yaml" "/scratch/metacognition/configs/sft_control_v4_verify.yaml"
  copy_to_remote "$url" "$ROOT/configs/accelerate_sft.yaml" "/scratch/metacognition/configs/accelerate_sft.yaml"
  copy_to_remote "$url" "$ROOT/src/training/sft.py" "/scratch/metacognition/src/training/sft.py"
  copy_to_remote "$url" "$ROOT/scripts/launch_sft_remote.sh" "/scratch/metacognition/scripts/launch_sft_remote.sh"
done

copy_to_remote "$URL_TOPS" "$ROOT/scripts/launch_control_v4_all_sft_remote.sh" "/scratch/metacognition/scripts/launch_control_v4_all_sft_remote.sh"
copy_to_remote "$URL_E8" "$ROOT/scripts/launch_control_v4_redirect_sft_remote.sh" "/scratch/metacognition/scripts/launch_control_v4_redirect_sft_remote.sh"
copy_to_remote "$URL_EVAL_E8" "$ROOT/scripts/launch_control_v4_verify_sft_remote.sh" "/scratch/metacognition/scripts/launch_control_v4_verify_sft_remote.sh"

echo "[launch] v4 all on tops"
proxy_ssh "$URL_TOPS" "bash -lc 'chmod +x /scratch/metacognition/scripts/launch_control_v4_all_sft_remote.sh && nohup bash /scratch/metacognition/scripts/launch_control_v4_all_sft_remote.sh > /scratch/control_v4_all_sft.log 2>&1 < /dev/null & echo \$!'" | tee "$LOG_DIR/launch_all.pid"

echo "[launch] v4 redirect on e8"
proxy_ssh "$URL_E8" "bash -lc 'chmod +x /scratch/metacognition/scripts/launch_control_v4_redirect_sft_remote.sh && nohup bash /scratch/metacognition/scripts/launch_control_v4_redirect_sft_remote.sh > /scratch/control_v4_redirect_sft.log 2>&1 < /dev/null & echo \$!'" | tee "$LOG_DIR/launch_redirect.pid"

echo "[launch] v4 verify on eval-e8"
proxy_ssh "$URL_EVAL_E8" "bash -lc 'chmod +x /scratch/metacognition/scripts/launch_control_v4_verify_sft_remote.sh && nohup bash /scratch/metacognition/scripts/launch_control_v4_verify_sft_remote.sh > /scratch/control_v4_verify_sft.log 2>&1 < /dev/null & echo \$!'" | tee "$LOG_DIR/launch_verify.pid"

echo "[done] launched v4 SFT runs"
