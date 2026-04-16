#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/v-seungplee/metacognition"
RESULTS_DIR="$ROOT/results"
LOG_FILE="$RESULTS_DIR/autoresearch_behavior_phase2_2026_04_01.log"
STATE_FILE="$RESULTS_DIR/autoresearch_behavior_phase2_2026_04_01.state"
PID_FILE="$RESULTS_DIR/autoresearch_behavior_phase2_2026_04_01.pid"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"
HF_TOKEN="${HF_TOKEN:-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}"
WANDB_KEY_FALLBACK="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"

URL_TOPS="wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms"
URL_E8="wss://ssh-2etszrmvdrq4cwqdql4al50f32aqiwdcl036benvkg6kmzk8bpc.westus2.nodes.azureml.ms"
URL_TRAIN_B="wss://ssh-2etszrmvdrq4cwqdql4al50f34l7xlwuwchd1gpv63o2790grqc.westus2.nodes.azureml.ms"
URL_EVAL_E8="wss://ssh-2etszrmvdrq4cwqdql4al50f365fggn0cs41y3ld90c6m331nlc.westus2.nodes.azureml.ms"

SLEEP_SEC="${SLEEP_SEC:-180}"

run_remote() {
  local url="$1"
  local cmd="$2"
  ssh -T -o LogLevel=ERROR \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" \
      azureuser@placeholder "$cmd"
}

copy_to_remote() {
  local url="$1"
  local src="$2"
  local dst="$3"
  scp -q -o LogLevel=ERROR \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" \
      "$src" "azureuser@placeholder:$dst"
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
E9_STARTED=0
VERIFY_FILES_COPIED=0
VERIFY_SFT_STARTED=0
ALL_UPLOAD_STARTED=0
REDIRECT_UPLOAD_STARTED=0
VERIFY_UPLOAD_STARTED=0
ALL_EVAL_STARTED=0
REDIRECT_EVAL_STARTED=0
VERIFY_EVAL_STARTED=0
EOF
  fi
}

load_state() {
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

save_state() {
  cat > "$STATE_FILE" <<EOF
E9_STARTED=$E9_STARTED
VERIFY_FILES_COPIED=$VERIFY_FILES_COPIED
VERIFY_SFT_STARTED=$VERIFY_SFT_STARTED
ALL_UPLOAD_STARTED=$ALL_UPLOAD_STARTED
REDIRECT_UPLOAD_STARTED=$REDIRECT_UPLOAD_STARTED
VERIFY_UPLOAD_STARTED=$VERIFY_UPLOAD_STARTED
ALL_EVAL_STARTED=$ALL_EVAL_STARTED
REDIRECT_EVAL_STARTED=$REDIRECT_EVAL_STARTED
VERIFY_EVAL_STARTED=$VERIFY_EVAL_STARTED
EOF
}

collect_tops() {
  run_remote "$URL_TOPS" "$(cat <<'EOF'
env -i bash --noprofile --norc -c '
e9_active=0
ps -eo cmd | grep -F "src/training/grpo_v2.py --mode E9" | grep -v grep >/dev/null && e9_active=1 || true
e9_final=0
[[ -d /scratch/metacognition/checkpoints/grpo_v2_behavior_all_E9/final ]] && e9_final=1
all_ckpt=0
[[ -f /scratch/metacognition/checkpoints/qwen3_metacot_behavior_all_sft/tokenizer_config.json ]] && all_ckpt=1
all_uploaded=0
grep -F "UPLOADED_qwen3_metacot_behavior_all_sft" /scratch/metacognition/qwen3_metacot_behavior_all_sft_hf_upload.log >/dev/null 2>&1 && all_uploaded=1 || true
printf "e9_active=%s\ne9_final=%s\nall_ckpt=%s\nall_uploaded=%s\n" "$e9_active" "$e9_final" "$all_ckpt" "$all_uploaded"
'
EOF
)"
}

collect_e8_train() {
  run_remote "$URL_E8" "$(cat <<'EOF'
env -i bash --noprofile --norc -c '
verify_data=0
[[ -f /scratch/metacognition/data/behavior_verify_sft.parquet ]] && verify_data=1
verify_cfg=0
[[ -f /scratch/metacognition/configs/sft_behavior_verify.yaml ]] && verify_cfg=1
verify_active=0
ps -eo cmd | grep -F "configs/sft_behavior_verify.yaml" | grep -v grep >/dev/null && verify_active=1 || true
verify_ckpt=0
[[ -f /scratch/metacognition/checkpoints/qwen3_metacot_behavior_verify_sft/tokenizer_config.json ]] && verify_ckpt=1
redirect_uploaded=0
grep -F "UPLOADED_qwen3_metacot_behavior_redirect_sft" /scratch/metacognition/qwen3_metacot_behavior_redirect_sft_hf_upload.log >/dev/null 2>&1 && redirect_uploaded=1 || true
verify_uploaded=0
grep -F "UPLOADED_qwen3_metacot_behavior_verify_sft" /scratch/metacognition/qwen3_metacot_behavior_verify_sft_hf_upload.log >/dev/null 2>&1 && verify_uploaded=1 || true
printf "verify_data=%s\nverify_cfg=%s\nverify_active=%s\nverify_ckpt=%s\nredirect_uploaded=%s\nverify_uploaded=%s\n" "$verify_data" "$verify_cfg" "$verify_active" "$verify_ckpt" "$redirect_uploaded" "$verify_uploaded"
'
EOF
)"
}

collect_eval_host() {
  run_remote "$URL_EVAL_E8" "$(cat <<'EOF'
env -i bash --noprofile --norc -c '
free_gpus="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F"," '\''{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 == 0) print $1}'\'' | paste -sd, -)"
if [[ -z "$free_gpus" ]]; then free_gpus=none; fi
all_done=0
[[ -f /scratch/behavior_all_eval_results/eval_1030_behavior_all_sft.json ]] && all_done=1
redirect_done=0
[[ -f /scratch/behavior_redirect_eval_results/eval_1030_behavior_redirect_sft.json ]] && redirect_done=1
verify_done=0
[[ -f /scratch/behavior_verify_eval_results/eval_1030_behavior_verify_sft.json ]] && verify_done=1
printf "free_gpus=%s\nall_done=%s\nredirect_done=%s\nverify_done=%s\n" "$free_gpus" "$all_done" "$redirect_done" "$verify_done"
'
EOF
)"
}

collect_train_b() {
  run_remote "$URL_TRAIN_B" "$(cat <<'EOF'
env -i bash --noprofile --norc -c '
e3_done=0
[[ -f /scratch/e3_eval_results/eval_1030_grpo_v2_E3_500.json ]] && e3_done=1
e3_step="$(tail -n 80 /scratch/e3_eval_results/eval_e3_500.log 2>/dev/null | tr "\r" "\n" | grep -oE "[0-9]+/1030" | tail -n 1)"
if [[ -z "$e3_step" ]]; then e3_step=0/1030; fi
printf "e3_done=%s\ne3_step=%s\n" "$e3_done" "$e3_step"
'
EOF
)"
}

copy_verify_files() {
  run_remote "$URL_E8" "mkdir -p /scratch/metacognition/data /scratch/metacognition/configs"
  copy_to_remote "$URL_E8" "$ROOT/data/behavior_verify_sft.parquet" "/scratch/metacognition/data/behavior_verify_sft.parquet"
  copy_to_remote "$URL_E8" "$ROOT/configs/sft_behavior_verify.yaml" "/scratch/metacognition/configs/sft_behavior_verify.yaml"
}

start_e9() {
  run_remote "$URL_TOPS" "$(cat <<EOF
env -i bash --noprofile --norc -c '
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="\$(cat ~/.wandb_key 2>/dev/null || echo "$WANDB_KEY_FALLBACK")"
nohup accelerate launch --config_file configs/accelerate_grpo.yaml src/training/grpo_v2.py --mode E9 --max_steps 300 --model_path checkpoints/qwen3_metacot_behavior_all_sft --data mixed_train --output_dir checkpoints/grpo_v2_behavior_all_E9 > /scratch/grpo_behavior_all_e9.log 2>&1 &
'
EOF
)"
}

start_verify_sft() {
  run_remote "$URL_E8" "$(cat <<EOF
env -i bash --noprofile --norc -c '
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="\$(cat ~/.wandb_key 2>/dev/null || echo "$WANDB_KEY_FALLBACK")"
nohup accelerate launch --config_file configs/accelerate_sft.yaml src/training/sft.py --config configs/sft_behavior_verify.yaml > /scratch/behavior_verify_sft.log 2>&1 &
'
EOF
)"
}

start_upload() {
  local url="$1"
  local model_dir="$2"
  local repo_subdir="$3"
  run_remote "$url" "$(cat <<EOF
env -i bash --noprofile --norc -c '
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
nohup python -c "from huggingface_hub import HfApi; api = HfApi(token=\"$HF_TOKEN\"); api.upload_folder(repo_id=\"iamseungpil/metacot\", repo_type=\"dataset\", folder_path=\"$model_dir\", path_in_repo=\"models/$repo_subdir\", commit_message=\"Upload $repo_subdir\", ignore_patterns=[\"checkpoint-*\", \"optimizer*\", \"scheduler*\", \"trainer_state*\", \"training_args*\", \"wandb/*\", \"runs/*\"]); print(\"UPLOADED_$repo_subdir\")" > /scratch/metacognition/${repo_subdir}_hf_upload.log 2>&1 &
'
EOF
)"
}

pick_gpu() {
  local free_csv="$1"
  if [[ "$free_csv" == "none" ]]; then
    return 1
  fi
  IFS=',' read -r first _ <<<"$free_csv"
  [[ -n "$first" ]] || return 1
  echo "$first"
}

start_eval() {
  local gpu="$1"
  local repo_subdir="$2"
  local model_name="$3"
  local output_dir="$4"
  local log_name="$5"
  run_remote "$URL_EVAL_E8" "$(cat <<EOF
env -i bash --noprofile --norc -c '
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
mkdir -p /scratch/${output_dir%/*} /scratch/$output_dir /scratch/${repo_subdir}_hf_models
nohup env PYTHONPATH=/scratch/metacognition CUDA_VISIBLE_DEVICES=$gpu /opt/conda/envs/grpo/bin/python -u -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"iamseungpil/metacot\", repo_type=\"dataset\", allow_patterns=[\"models/$repo_subdir/*\"], local_dir=\"/scratch/${repo_subdir}_hf_models\", local_dir_use_symlinks=False); import os, runpy, sys; os.chdir(\"/scratch/metacognition\"); sys.argv=[\"src/eval/eval_hf.py\", \"--model_path\", \"/scratch/${repo_subdir}_hf_models/models/$repo_subdir\", \"--model_name\", \"$model_name\", \"--benchmarks\", \"gsm8k\", \"math500\", \"aime2024\", \"--max_problems\", \"500\", \"--output_dir\", \"/scratch/$output_dir\"]; runpy.run_path(\"src/eval/eval_hf.py\", run_name=\"__main__\")" > /scratch/$output_dir/$log_name 2>&1 &
'
EOF
)"
}

main() {
  init_state
  load_state

  while :; do
    local tops e8 eval_host train_b

    tops="$(collect_tops)" || { log "tops collection failed"; sleep "$SLEEP_SEC"; continue; }
    e8="$(collect_e8_train)" || { log "metacognition_e8 collection failed"; sleep "$SLEEP_SEC"; continue; }
    eval_host="$(collect_eval_host)" || { log "eval host collection failed"; sleep "$SLEEP_SEC"; continue; }
    train_b="$(collect_train_b)" || { log "train_b collection failed"; sleep "$SLEEP_SEC"; continue; }

    eval "$tops"
    eval "$e8"
    eval "$eval_host"
    local eval_free_gpus="$free_gpus"
    local all_done="$all_done"
    local redirect_done="$redirect_done"
    local verify_done="$verify_done"
    eval "$train_b"

    log "status e9_active=$e9_active e9_final=$e9_final verify_active=$verify_active verify_ckpt=$verify_ckpt all_uploaded=$all_uploaded redirect_uploaded=$redirect_uploaded verify_uploaded=$verify_uploaded eval_free=$eval_free_gpus e3_step=$e3_step"

    if (( E9_STARTED == 0 && all_ckpt == 1 && e9_active == 0 && e9_final == 0 )); then
      start_e9
      E9_STARTED=1
      save_state
      log "started E9 from behavior_all_sft on tops-caiman"
    fi

    if (( VERIFY_FILES_COPIED == 0 )) && (( verify_data == 0 || verify_cfg == 0 )); then
      copy_verify_files
      VERIFY_FILES_COPIED=1
      save_state
      log "copied behavior_verify data/config to metacognition_e8"
    fi

    if (( VERIFY_SFT_STARTED == 0 && verify_data == 1 && verify_cfg == 1 && verify_active == 0 && verify_ckpt == 0 )); then
      start_verify_sft
      VERIFY_SFT_STARTED=1
      save_state
      log "started behavior_verify_sft on metacognition_e8"
    fi

    if (( ALL_UPLOAD_STARTED == 0 && all_ckpt == 1 && all_uploaded == 0 )); then
      start_upload "$URL_TOPS" "/scratch/metacognition/checkpoints/qwen3_metacot_behavior_all_sft" "qwen3_metacot_behavior_all_sft"
      ALL_UPLOAD_STARTED=1
      save_state
      log "started HF upload for behavior_all_sft"
    fi

    if (( REDIRECT_UPLOAD_STARTED == 0 && redirect_uploaded == 0 )); then
      start_upload "$URL_E8" "/scratch/metacognition/checkpoints/qwen3_metacot_behavior_redirect_sft" "qwen3_metacot_behavior_redirect_sft"
      REDIRECT_UPLOAD_STARTED=1
      save_state
      log "started HF upload for behavior_redirect_sft"
    fi

    if (( VERIFY_UPLOAD_STARTED == 0 && verify_ckpt == 1 && verify_uploaded == 0 )); then
      start_upload "$URL_E8" "/scratch/metacognition/checkpoints/qwen3_metacot_behavior_verify_sft" "qwen3_metacot_behavior_verify_sft"
      VERIFY_UPLOAD_STARTED=1
      save_state
      log "started HF upload for behavior_verify_sft"
    fi

    local gpu
    if (( ALL_EVAL_STARTED == 0 && all_uploaded == 1 && all_done == 0 )); then
      if gpu="$(pick_gpu "$eval_free_gpus")"; then
        start_eval "$gpu" "qwen3_metacot_behavior_all_sft" "1030_behavior_all_sft" "behavior_all_eval_results" "eval_behavior_all_sft.log"
        ALL_EVAL_STARTED=1
        save_state
        log "started behavior_all_sft eval on eval-e8 gpu=$gpu"
        eval_host="$(collect_eval_host)" || true
        if [[ -n "${eval_host:-}" ]]; then
          eval "$eval_host"
          eval_free_gpus="$free_gpus"
          all_done="$all_done"
          redirect_done="$redirect_done"
          verify_done="$verify_done"
        fi
      fi
    fi

    if (( REDIRECT_EVAL_STARTED == 0 && redirect_uploaded == 1 && redirect_done == 0 )); then
      if gpu="$(pick_gpu "$eval_free_gpus")"; then
        start_eval "$gpu" "qwen3_metacot_behavior_redirect_sft" "1030_behavior_redirect_sft" "behavior_redirect_eval_results" "eval_behavior_redirect_sft.log"
        REDIRECT_EVAL_STARTED=1
        save_state
        log "started behavior_redirect_sft eval on eval-e8 gpu=$gpu"
        eval_host="$(collect_eval_host)" || true
        if [[ -n "${eval_host:-}" ]]; then
          eval "$eval_host"
          eval_free_gpus="$free_gpus"
          all_done="$all_done"
          redirect_done="$redirect_done"
          verify_done="$verify_done"
        fi
      fi
    fi

    if (( VERIFY_EVAL_STARTED == 0 && verify_uploaded == 1 && verify_done == 0 )); then
      if gpu="$(pick_gpu "$eval_free_gpus")"; then
        start_eval "$gpu" "qwen3_metacot_behavior_verify_sft" "1030_behavior_verify_sft" "behavior_verify_eval_results" "eval_behavior_verify_sft.log"
        VERIFY_EVAL_STARTED=1
        save_state
        log "started behavior_verify_sft eval on eval-e8 gpu=$gpu"
      fi
    fi

    sleep "$SLEEP_SEC"
  done
}

main "$@"
