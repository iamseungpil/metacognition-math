#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
MAX_STEPS="${MAX_STEPS:-200}"

URL_E8="${URL_E8:-wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms}"
URL_EVAL="${URL_EVAL:-wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms}"
URL_TRAIN_B="${URL_TRAIN_B:-wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms}"

LOG_DIR="$ROOT/results/autoresearch_control_v5_rl"
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
  scp -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" "$src" "azureuser@placeholder:$dst"
}

bootstrap_remote() {
  local url="$1"
  proxy_ssh "$url" "mkdir -p /scratch/metacognition/{configs,scripts,src/training,src/probes,src/metacot,src/curriculum,src/eval,results,checkpoints,data}"
}

copy_runtime_bundle() {
  local url="$1"
  copy_to_remote "$url" "$ROOT/configs/accelerate_grpo.yaml" "/scratch/metacognition/configs/accelerate_grpo.yaml"
  copy_to_remote "$url" "$ROOT/scripts/run_grpo_v2.sh" "/scratch/metacognition/scripts/run_grpo_v2.sh"
  copy_to_remote "$url" "$ROOT/scripts/push_models_hf.py" "/scratch/metacognition/scripts/push_models_hf.py"
  copy_to_remote "$url" "$ROOT/scripts/check_runtime_env.py" "/scratch/metacognition/scripts/check_runtime_env.py"
  copy_to_remote "$url" "$ROOT/scripts/ensure_hf_model.py" "/scratch/metacognition/scripts/ensure_hf_model.py"
  copy_to_remote "$url" "$ROOT/scripts/run_control_v5_eval_matrix.sh" "/scratch/metacognition/scripts/run_control_v5_eval_matrix.sh"
  copy_to_remote "$url" "$ROOT/scripts/analyze_control_v5_eval.py" "/scratch/metacognition/scripts/analyze_control_v5_eval.py"
  copy_to_remote "$url" "$ROOT/scripts/build_probe_rollouts_hf.py" "/scratch/metacognition/scripts/build_probe_rollouts_hf.py"
  copy_to_remote "$url" "$ROOT/scripts/smoke_probe_pipeline.py" "/scratch/metacognition/scripts/smoke_probe_pipeline.py"
  copy_to_remote "$url" "$ROOT/src/eval/eval_hf.py" "/scratch/metacognition/src/eval/eval_hf.py"
  copy_to_remote "$url" "$ROOT/src/training/grpo_v2.py" "/scratch/metacognition/src/training/grpo_v2.py"
  copy_to_remote "$url" "$ROOT/src/training/tokenizer_utils.py" "/scratch/metacognition/src/training/tokenizer_utils.py"
  copy_to_remote "$url" "$ROOT/src/training/rewards.py" "/scratch/metacognition/src/training/rewards.py"
  copy_to_remote "$url" "$ROOT/src/probes/simple_probe.py" "/scratch/metacognition/src/probes/simple_probe.py"
  copy_to_remote "$url" "$ROOT/src/probes/retrain.py" "/scratch/metacognition/src/probes/retrain.py"
  copy_to_remote "$url" "$ROOT/src/metacot/prompt.py" "/scratch/metacognition/src/metacot/prompt.py"
  copy_to_remote "$url" "$ROOT/src/curriculum/control_rag.py" "/scratch/metacognition/src/curriculum/control_rag.py"
}

launch_remote_nohup() {
  local url="$1"
  local launcher="$2"
  local log_name="$3"
  proxy_ssh "$url" "bash -lc 'cd /scratch/metacognition && chmod +x scripts/*.sh && export REMOTE_CONDA_ENV=$REMOTE_CONDA_ENV MAX_STEPS=$MAX_STEPS && nohup bash scripts/$launcher > results/$log_name 2>&1 < /dev/null & echo \$!'"
}

for url in "$URL_E8" "$URL_EVAL" "$URL_TRAIN_B"; do
  bootstrap_remote "$url"
  copy_runtime_bundle "$url"
done

copy_to_remote "$URL_E8" "$ROOT/scripts/launch_control_v5_rl_probe_lane_remote.sh" "/scratch/metacognition/scripts/launch_control_v5_rl_probe_lane_remote.sh"
copy_to_remote "$URL_EVAL" "$ROOT/scripts/launch_control_v5_rl_eval_lane_remote.sh" "/scratch/metacognition/scripts/launch_control_v5_rl_eval_lane_remote.sh"
copy_to_remote "$URL_TRAIN_B" "$ROOT/scripts/launch_control_v5_rl_train_b_lane_remote.sh" "/scratch/metacognition/scripts/launch_control_v5_rl_train_b_lane_remote.sh"

echo "[hf] uploading unified control-v5 all SFT from e8 node"
proxy_ssh "$URL_E8" "bash -lc 'cd /scratch/metacognition && source /opt/conda/etc/profile.d/conda.sh && conda activate $REMOTE_CONDA_ENV && export PYTHONPATH=/scratch/metacognition && export HF_TOKEN=${HF_TOKEN:-${HF_TOKEN}} && nohup python scripts/push_models_hf.py --model_path checkpoints/qwen3_metacot_control_v5_all_sft --model_name qwen3_metacot_control_v5_all_sft > results/control_v5_all_sft_hf_upload.log 2>&1 < /dev/null & echo \$!'" | tee "$LOG_DIR/hf_upload.pid"

echo "[launch] probe lane on metacognition_e8"
launch_remote_nohup "$URL_E8" "launch_control_v5_rl_probe_lane_remote.sh" "control_v5_probe_lane.out" | tee "$LOG_DIR/probe_lane.pid"

echo "[launch] calibration lane on metacognition_eval"
launch_remote_nohup "$URL_EVAL" "launch_control_v5_rl_eval_lane_remote.sh" "control_v5_eval_lane.out" | tee "$LOG_DIR/eval_lane.pid"

echo "[launch] behavior lane on metacognition_train_b"
launch_remote_nohup "$URL_TRAIN_B" "launch_control_v5_rl_train_b_lane_remote.sh" "control_v5_train_b_lane.out" | tee "$LOG_DIR/train_b_lane.pid"

echo "[done] launched control-v5 RL lanes on 3 nodes"
