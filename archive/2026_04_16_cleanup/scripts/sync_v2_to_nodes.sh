#!/usr/bin/env bash
# Sync V2 reward code + launch scripts to all 3 remote nodes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"

URL_E8="${URL_E8:-wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms}"
URL_EVAL="${URL_EVAL:-wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms}"
URL_TRAIN_B="${URL_TRAIN_B:-wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms}"

copy_to_remote() {
  local url="$1" src="$2" dst="$3"
  scp -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" "$src" "azureuser@placeholder:$dst"
}

proxy_ssh() {
  local url="$1"; shift
  ssh -T -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" azureuser@placeholder "$@"
}

sync_node() {
  local url="$1" name="$2"
  echo "=== Syncing to $name ==="
  proxy_ssh "$url" "mkdir -p /scratch/metacognition/{src/training,scripts,tests,results/control_v5_v2_lane}"
  copy_to_remote "$url" "$ROOT/src/training/rewards.py" "/scratch/metacognition/src/training/rewards.py"
  copy_to_remote "$url" "$ROOT/src/training/grpo_v2.py" "/scratch/metacognition/src/training/grpo_v2.py"
  copy_to_remote "$url" "$ROOT/scripts/launch_v2_experiments.sh" "/scratch/metacognition/scripts/launch_v2_experiments.sh"
  proxy_ssh "$url" "chmod +x /scratch/metacognition/scripts/launch_v2_experiments.sh"
  echo "$name: sync complete"
}

sync_node "$URL_EVAL" "EVAL"
sync_node "$URL_TRAIN_B" "TRAIN_B"
sync_node "$URL_E8" "E8"

echo "=== All 3 nodes synced ==="
