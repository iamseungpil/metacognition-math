#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/v-seungplee/metacognition"
RESULTS_DIR="$ROOT/results"
LOG_FILE="$RESULTS_DIR/autoresearch_followup_2026_04_01.log"
STATE_FILE="$RESULTS_DIR/autoresearch_followup_2026_04_01.state"
PID_FILE="$RESULTS_DIR/autoresearch_followup_2026_04_01.pid"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"

URL_TOPS="wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms"
URL_TRAIN_B="wss://ssh-2etszrmvdrq4cwqdql4al50f34l7xlwuwchd1gpv63o2790grqc.westus2.nodes.azureml.ms"
URL_EVAL_E8="wss://ssh-2etszrmvdrq4cwqdql4al50f365fggn0cs41y3ld90c6m331nlc.westus2.nodes.azureml.ms"

SLEEP_SEC="${SLEEP_SEC:-180}"

run_remote() {
  local url="$1"
  local cmd="$2"
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" \
      azureuser@placeholder "$cmd"
}

log() {
  local ts
  ts="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

init_state() {
  mkdir -p "$RESULTS_DIR"
  echo "$$" > "$PID_FILE"
  if [[ ! -f "$STATE_FILE" ]]; then
    cat > "$STATE_FILE" <<'EOF'
E8_UPLOAD_STARTED=0
E8_EVAL_STARTED=0
EOF
  fi
}

load_state() {
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

save_state() {
  cat > "$STATE_FILE" <<EOF
E8_UPLOAD_STARTED=$E8_UPLOAD_STARTED
E8_EVAL_STARTED=$E8_EVAL_STARTED
EOF
}

collect_tops() {
  run_remote "$URL_TOPS" "$(cat <<'EOF'
set -euo pipefail
e8_active=0
ps -eo cmd | grep -F "src/training/grpo_v2.py --mode E8" | grep -v grep >/dev/null && e8_active=1 || true
e8_final=0
[[ -d /scratch/metacognition/checkpoints/grpo_v2_E8/final ]] && e8_final=1
e8_step="$(tail -n 200 /scratch/metacognition/grpo_e8_bg.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/200' | tail -n 1 | cut -d/ -f1)"
if [[ -z "$e8_step" ]]; then
  e8_step="$(tail -n 200 /scratch/metacognition/grpo_e8.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/200' | tail -n 1 | cut -d/ -f1)"
fi
if [[ -z "$e8_step" ]]; then e8_step=0; fi
e8_uploaded=0
grep -F "UPLOADED_E8" /scratch/metacognition/grpo_v2_E8_hf_upload.log >/dev/null 2>&1 && e8_uploaded=1 || true
printf 'e8_active=%s\ne8_final=%s\ne8_step=%s\ne8_uploaded=%s\n' "$e8_active" "$e8_final" "$e8_step" "$e8_uploaded"
EOF
)"
}

collect_eval_host() {
  local url="$1"
  run_remote "$url" "$(cat <<'EOF'
set -euo pipefail
gpus="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 == 0) print $1}' | paste -sd, -)"
if [[ -z "$gpus" ]]; then gpus=none; fi
e8_eval_active=0
ps -eo cmd | grep -F "1030_grpo_v2_E8" | grep -v grep >/dev/null && e8_eval_active=1 || true
e8_eval_done=0
find /scratch -maxdepth 2 -type f -name 'eval_1030_grpo_v2_E8.json' | grep -q . && e8_eval_done=1 || true
printf 'free_gpus=%s\ne8_eval_active=%s\ne8_eval_done=%s\n' "$gpus" "$e8_eval_active" "$e8_eval_done"
EOF
)"
}

start_e8_upload() {
  run_remote "$URL_TOPS" "$(cat <<'EOF'
set -euo pipefail
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
nohup bash -lc "cd /scratch/metacognition && source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && python -c 'from huggingface_hub import HfApi; api = HfApi(token=\"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\"); api.upload_folder(repo_id=\"iamseungpil/metacot\", repo_type=\"dataset\", folder_path=\"/scratch/metacognition/checkpoints/grpo_v2_E8/final\", path_in_repo=\"models/grpo_v2_E8\", commit_message=\"Upload grpo_v2_E8\", ignore_patterns=[\"checkpoint-*\", \"optimizer*\", \"scheduler*\", \"trainer_state*\", \"training_args*\", \"wandb/*\", \"runs/*\"]); print(\"UPLOADED_E8\")' >/scratch/metacognition/grpo_v2_E8_hf_upload.log 2>&1" >/dev/null 2>&1 &
EOF
)"
}

start_e8_eval() {
  local url="$1"
  local gpu="$2"
  local prefix="$3"
  run_remote "$url" "$(cat <<EOF
set -euo pipefail
cd /scratch/metacognition
mkdir -p /scratch/e8_eval_hf_models /scratch/e8_eval_results
nohup env PYTHONPATH=/scratch/metacognition CUDA_VISIBLE_DEVICES=$gpu /opt/conda/envs/grpo/bin/python -u -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"iamseungpil/metacot\", repo_type=\"dataset\", allow_patterns=[\"models/grpo_v2_E8/*\"], local_dir=\"/scratch/e8_eval_hf_models\", local_dir_use_symlinks=False); import os, runpy, sys; os.chdir(\"/scratch/metacognition\"); sys.argv=[\"src/eval/eval_hf.py\", \"--model_path\", \"/scratch/e8_eval_hf_models/models/grpo_v2_E8\", \"--model_name\", \"1030_grpo_v2_E8\", \"--benchmarks\", \"gsm8k\", \"math500\", \"aime2024\", \"--max_problems\", \"500\", \"--output_dir\", \"/scratch/e8_eval_results\"]; runpy.run_path(\"src/eval/eval_hf.py\", run_name=\"__main__\")" > /scratch/e8_eval_results/${prefix}_eval_e8.log 2>&1 &
EOF
)"
}

pick_gpu() {
  local free_csv="$1"
  if [[ "$free_csv" == "none" ]]; then
    return 1
  fi
  IFS=',' read -r first _ <<<"$free_csv"
  if [[ -n "$first" ]]; then
    echo "$first"
    return 0
  fi
  return 1
}

main() {
  init_state
  load_state

  while :; do
    local tops eval_e8 train_b
    tops="$(collect_tops)" || { log "tops-caiman collection failed"; sleep "$SLEEP_SEC"; continue; }
    eval_e8="$(collect_eval_host "$URL_EVAL_E8")" || { log "eval-e8 collection failed"; sleep "$SLEEP_SEC"; continue; }
    train_b="$(collect_eval_host "$URL_TRAIN_B")" || { log "train_b collection failed"; sleep "$SLEEP_SEC"; continue; }

    eval "$tops"
    eval "$eval_e8"
    local eval_e8_free_gpus="$free_gpus"
    local eval_e8_eval_active="$e8_eval_active"
    local eval_e8_eval_done="$e8_eval_done"
    eval "$train_b"
    local train_b_free_gpus="$free_gpus"
    local train_b_eval_active="$e8_eval_active"
    local train_b_eval_done="$e8_eval_done"

    log "status e8_step=$e8_step/200 e8_active=$e8_active e8_final=$e8_final e8_uploaded=$e8_uploaded eval_e8_free=$eval_e8_free_gpus train_b_free=$train_b_free_gpus"

    if (( e8_final == 1 && e8_uploaded == 0 && E8_UPLOAD_STARTED == 0 )); then
      start_e8_upload
      E8_UPLOAD_STARTED=1
      save_state
      log "started E8 HF upload"
    fi

    if (( e8_uploaded == 1 && E8_EVAL_STARTED == 0 && eval_e8_eval_done == 0 && train_b_eval_done == 0 )); then
      local gpu
      if gpu="$(pick_gpu "$eval_e8_free_gpus")"; then
        start_e8_eval "$URL_EVAL_E8" "$gpu" "eval_e8"
        E8_EVAL_STARTED=1
        save_state
        log "started E8 eval on eval-e8 gpu=$gpu"
      elif gpu="$(pick_gpu "$train_b_free_gpus")"; then
        start_e8_eval "$URL_TRAIN_B" "$gpu" "train_b"
        E8_EVAL_STARTED=1
        save_state
        log "started E8 eval on train_b gpu=$gpu"
      fi
    fi

    sleep "$SLEEP_SEC"
  done
}

main "$@"
