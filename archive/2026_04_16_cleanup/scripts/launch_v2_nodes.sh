#!/usr/bin/env bash
# Launch V2 experiments on 3 remote nodes:
#   EVAL:    E9v2 (verify quality)
#   TRAIN_B: E9bv2 (redirect execution)
#   E8:      E10v2 (full controller)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
MAX_STEPS="${MAX_STEPS:-300}"

URL_E8="${URL_E8:-wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms}"
URL_EVAL="${URL_EVAL:-wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms}"
URL_TRAIN_B="${URL_TRAIN_B:-wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms}"

LOG_DIR="$ROOT/results/autoresearch_v2_experiments"
mkdir -p "$LOG_DIR"

proxy_ssh() {
  local url="$1"; shift
  ssh -T -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" azureuser@placeholder "$@"
}

launch_v2() {
  local url="$1" role="$2" log_name="$3"
  echo "[launch] $role on $log_name"
  proxy_ssh "$url" "bash -lc 'cd /scratch/metacognition && chmod +x scripts/*.sh && export REMOTE_CONDA_ENV=$REMOTE_CONDA_ENV MAX_STEPS=$MAX_STEPS NODE_ROLE=$role && nohup bash scripts/launch_v2_experiments.sh > results/control_v5_v2_lane/${role}.out 2>&1 < /dev/null & echo \$!'"
}

echo "=== Launching V2 experiments on 3 nodes ==="

# EVAL node: E9v2 (verify quality)
launch_v2 "$URL_EVAL" "E9v2" "EVAL" | tee "$LOG_DIR/eval_v2.pid"

# TRAIN_B node: E9bv2 (redirect execution)
launch_v2 "$URL_TRAIN_B" "E9bv2" "TRAIN_B" | tee "$LOG_DIR/trainb_v2.pid"

# E8 node: E10v2 (full controller)
launch_v2 "$URL_E8" "E10v2" "E8" | tee "$LOG_DIR/e8_v2.pid"

echo "=== All 3 V2 experiments launched ==="
echo "Training: ~300 steps × 2 min/step ≈ 10 hours"
echo "Then auto-eval: ~26 hours"
echo "Monitor with:"
echo "  proxy_ssh URL 'tail -f /scratch/metacognition/results/control_v5_v2_lane/E9v2_train.log'"
