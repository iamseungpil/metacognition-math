#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/v-seungplee/metacognition"
RESULTS_DIR="$ROOT/results"
TSV_LOG="$RESULTS_DIR/autoresearch_monitor_2026_03_31.tsv"
TEXT_LOG="$RESULTS_DIR/autoresearch_monitor_2026_03_31.log"
STATE_FILE="$RESULTS_DIR/autoresearch_monitor_2026_03_31.state"
PID_FILE="$RESULTS_DIR/autoresearch_monitor_2026_03_31.pid"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"

URL_METACOGNITION="wss://ssh-2etszrmvdrq4cwqdql4al50f32aqiwdcl036benvkg6kmzk8bpc.westus2.nodes.azureml.ms"
URL_TOPS="wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms"
# Use metacognition_train_b as the 4-GPU eval slot for the 12-GPU policy.
URL_EVAL="wss://ssh-2etszrmvdrq4cwqdql4al50f34l7xlwuwchd1gpv63o2790grqc.westus2.nodes.azureml.ms"

SLEEP_SEC="${SLEEP_SEC:-120}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"

run_remote() {
  local url="$1"
  local cmd="$2"
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" \
      azureuser@placeholder "$cmd"
}

log_text() {
  local ts
  ts="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "[$ts] $*" | tee -a "$TEXT_LOG"
}

init_logs() {
  mkdir -p "$RESULTS_DIR"
  echo "$$" > "$PID_FILE"
  if [[ ! -e "$TSV_LOG" ]]; then
    {
      echo "# metric_direction: higher_is_better"
      echo -e "iteration\tcommit\tmetric\tdelta\tguard\tstatus\tdescription"
      echo -e "0\t-\t0\t0\tpass\tbaseline\tmonitor initialized for E3->Base SFT->E5 and E7->E8 pipeline"
    } > "$TSV_LOG"
  fi
  if [[ ! -e "$STATE_FILE" ]]; then
    cat > "$STATE_FILE" <<'EOF'
LAST_METRIC=0
BASE_EVAL_STARTED=0
V3_EVAL_STARTED=0
E8_STARTED=0
E5_UPLOAD_STARTED=0
E8_UPLOAD_STARTED=0
EOF
  fi
}

source_state() {
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

write_state() {
  cat > "$STATE_FILE" <<EOF
LAST_METRIC=$LAST_METRIC
BASE_EVAL_STARTED=$BASE_EVAL_STARTED
V3_EVAL_STARTED=$V3_EVAL_STARTED
E8_STARTED=$E8_STARTED
E5_UPLOAD_STARTED=$E5_UPLOAD_STARTED
E8_UPLOAD_STARTED=$E8_UPLOAD_STARTED
EOF
}

append_tsv() {
  local iteration="$1"
  local metric="$2"
  local delta="$3"
  local guard="$4"
  local status="$5"
  local description="$6"
  echo -e "${iteration}\t-\t${metric}\t${delta}\t${guard}\t${status}\t${description}" >> "$TSV_LOG"
}

collect_metacognition_state() {
  run_remote "$URL_METACOGNITION" "$(cat <<'EOF'
set -euo pipefail
e3_done=0
[[ -d /scratch/metacognition/checkpoints/grpo_v2_E3_500/final ]] && e3_done=1
e3_active=0
ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /src\/training\/grpo_v2.py/ && $0 ~ /--mode E3/ && $0 ~ /--max_steps 300/ {found=1} END{exit(found?0:1)}' && e3_active=1 || true
after_e3_active=0
ps -eo cmd | grep -F "/scratch/metacognition/after_e3.log" | grep -v grep >/dev/null && after_e3_active=1 || true
base_sft_active=0
ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /src\/training\/sft.py/ && $0 ~ /configs\/sft_base.yaml/ {found=1} END{exit(found?0:1)}' && base_sft_active=1 || true
base_sft_done=0
grep -F "Base SFT done" /scratch/metacognition/after_e3.log >/dev/null 2>&1 && base_sft_done=1 || true
e5_active=0
ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /src\/training\/grpo_v2.py/ && $0 ~ /--mode E5/ {found=1} END{exit(found?0:1)}' && e5_active=1 || true
e5_done=0
grep -F "E5 done!" /scratch/metacognition/after_e3.log >/dev/null 2>&1 && e5_done=1 || true
e3_step="$(tail -n 80 /scratch/metacognition/e3_cont.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/300' | tail -n 1 | cut -d/ -f1)"
if [[ -z "$e3_step" ]]; then e3_step=0; fi
printf 'e3_done=%s\ne3_active=%s\nafter_e3_active=%s\nbase_sft_active=%s\nbase_sft_done=%s\ne5_active=%s\ne5_done=%s\ne3_step=%s\n' \
  "$e3_done" "$e3_active" "$after_e3_active" "$base_sft_active" "$base_sft_done" "$e5_active" "$e5_done" "$e3_step"
EOF
)"
}

collect_tops_state() {
  run_remote "$URL_TOPS" "$(cat <<'EOF'
set -euo pipefail
e7_done=0
[[ -d /scratch/metacognition/checkpoints/grpo_v2_E7/final ]] && e7_done=1
e7_active=0
ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /src\/training\/grpo_v2.py/ && $0 ~ /--mode E7/ {found=1} END{exit(found?0:1)}' && e7_active=1 || true
e8_active=0
ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /src\/training\/grpo_v2.py/ && $0 ~ /--mode E8/ {found=1} END{exit(found?0:1)}' && e8_active=1 || true
e8_done=0
[[ -d /scratch/metacognition/checkpoints/grpo_v2_E8/final ]] && e8_done=1
e7_step="$(tail -n 80 /scratch/metacognition/grpo_e7.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/500' | tail -n 1 | cut -d/ -f1)"
if [[ -z "$e7_step" ]]; then e7_step=0; fi
printf 'e7_done=%s\ne7_active=%s\ne8_active=%s\ne8_done=%s\ne7_step=%s\n' \
  "$e7_done" "$e7_active" "$e8_active" "$e8_done" "$e7_step"
EOF
)"
}

collect_eval_state() {
  run_remote "$URL_EVAL" "$(cat <<'EOF'
set -euo pipefail
base_eval_active=0
ps -eo cmd | grep -F "src/eval/eval_hf.py" | grep -F "1030_base_sft" | grep -v grep >/dev/null && base_eval_active=1 || true
base_eval_done=0
[[ -f /scratch/base_eval_results/eval_1030_base_sft.json ]] && base_eval_done=1
v3_eval_active=0
ps -eo cmd | grep -F "src/eval/eval_hf.py" | grep -F "1030_v3_sft" | grep -v grep >/dev/null && v3_eval_active=1 || true
v3_eval_done=0
[[ -f /scratch/v3_eval_results/eval_1030_v3_sft.json ]] && v3_eval_done=1
printf 'base_eval_active=%s\nbase_eval_done=%s\nv3_eval_active=%s\nv3_eval_done=%s\n' \
  "$base_eval_active" "$base_eval_done" "$v3_eval_active" "$v3_eval_done"
EOF
)"
}

start_base_eval() {
  run_remote "$URL_EVAL" "$(cat <<'EOF'
set -euo pipefail
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
nohup bash -lc "cd /scratch/metacognition && source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && export PYTHONPATH=/scratch/metacognition && mkdir -p /scratch/base_eval_hf_models /scratch/base_eval_results && python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"iamseungpil/metacot\", repo_type=\"dataset\", allow_patterns=[\"models/qwen3_base_sft/*\"], local_dir=\"/scratch/base_eval_hf_models\", local_dir_use_symlinks=False)' && CUDA_VISIBLE_DEVICES=0 python -u src/eval/eval_hf.py --model_path /scratch/base_eval_hf_models/models/qwen3_base_sft --model_name 1030_base_sft --benchmarks gsm8k math500 aime2024 --max_problems 500 --output_dir /scratch/base_eval_results > /scratch/base_eval_results/eval_base_sft.log 2>&1" >/dev/null 2>&1 &
EOF
)"
}

start_v3_eval() {
  run_remote "$URL_EVAL" "$(cat <<'EOF'
set -euo pipefail
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
nohup bash -lc "cd /scratch/metacognition && source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && export PYTHONPATH=/scratch/metacognition && mkdir -p /scratch/v3_eval_hf_models /scratch/v3_eval_results && python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"iamseungpil/metacot\", repo_type=\"dataset\", allow_patterns=[\"models/qwen3_metacot_v3_sft/*\"], ignore_patterns=[\"models/qwen3_metacot_v3_sft/checkpoint-*/*\"], local_dir=\"/scratch/v3_eval_hf_models\", local_dir_use_symlinks=False)' && CUDA_VISIBLE_DEVICES=1 python -u src/eval/eval_hf.py --model_path /scratch/v3_eval_hf_models/models/qwen3_metacot_v3_sft --model_name 1030_v3_sft --benchmarks gsm8k math500 aime2024 --max_problems 500 --output_dir /scratch/v3_eval_results > /scratch/v3_eval_results/eval_v3_sft.log 2>&1" >/dev/null 2>&1 &
EOF
)"
}

start_after_e3() {
  run_remote "$URL_METACOGNITION" "$(cat <<'EOF'
set -euo pipefail
cd /scratch/metacognition
nohup bash -c '
set -e
LOG=/scratch/metacognition/after_e3.log
echo "$(date -u "+%a %b %d %H:%M:%S %Y UTC"): after_e3 pipeline RESTARTED by monitor" >> "$LOG"
echo "$(date -u "+%a %b %d %H:%M:%S %Y UTC"): Waiting for grpo_v2_E3_500/final..." >> "$LOG"
while [ ! -d /scratch/metacognition/checkpoints/grpo_v2_E3_500/final ]; do
  sleep 60
done
echo "$(date -u "+%a %b %d %H:%M:%S %Y UTC"): E3_500 final checkpoint found!" >> "$LOG"
echo "$(date -u "+%a %b %d %H:%M:%S %Y UTC"): Starting Base SFT..." >> "$LOG"
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9
accelerate launch --config_file configs/accelerate_sft.yaml src/training/sft.py --config configs/sft_base.yaml > /scratch/metacognition/base_sft.log 2>&1
echo "$(date -u "+%a %b %d %H:%M:%S %Y UTC"): Base SFT done" >> "$LOG"
echo "$(date -u "+%a %b %d %H:%M:%S %Y UTC"): Starting E5 GRPO (500 steps, from qwen3_metacot_v2_sft)..." >> "$LOG"
accelerate launch --config_file configs/accelerate_grpo.yaml src/training/grpo_v2.py --mode E5 --max_steps 500 --model_path checkpoints/qwen3_metacot_v2_sft --data mixed --output_dir checkpoints/grpo_v2_E5 > /scratch/metacognition/grpo_e5.log 2>&1
echo "$(date -u "+%a %b %d %H:%M:%S %Y UTC"): E5 done!" >> "$LOG"
' >/dev/null 2>&1 &
EOF
)"
}

start_e8() {
  run_remote "$URL_TOPS" "$(cat <<'EOF'
set -euo pipefail
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
nohup bash -lc 'cd /scratch/metacognition && source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && export PYTHONPATH=/scratch/metacognition && accelerate launch --config_file configs/accelerate_grpo.yaml src/training/grpo_v2.py --mode E8 --max_steps 200 --model_path checkpoints/grpo_v2_E7/final --data mixed_train --output_dir checkpoints/grpo_v2_E8 > /scratch/metacognition/grpo_e8.log 2>&1' >/dev/null 2>&1 &
EOF
)"
}

start_hf_upload() {
  local url="$1"
  local model_dir="$2"
  local repo_subdir="$3"
  run_remote "$url" "$(cat <<EOF
set -euo pipefail
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
nohup bash -lc "cd /scratch/metacognition && source /opt/conda/etc/profile.d/conda.sh && conda activate grpo && python -c 'from huggingface_hub import HfApi; api = HfApi(token=\"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\"); api.upload_folder(repo_id=\"iamseungpil/metacot\", repo_type=\"dataset\", folder_path=\"$model_dir\", path_in_repo=\"models/$repo_subdir\", commit_message=\"Upload $repo_subdir\", ignore_patterns=[\"checkpoint-*\", \"optimizer*\", \"scheduler*\", \"trainer_state*\", \"training_args*\", \"wandb/*\", \"runs/*\"]); print(\"UPLOADED_$repo_subdir\")'" >/scratch/metacognition/${repo_subdir}_hf_upload.log 2>&1" >/dev/null 2>&1 &
EOF
)"
}

main() {
  init_logs
  source_state

  local iteration=1
  if [[ -s "$TSV_LOG" ]]; then
    local last_iteration
    last_iteration="$(tail -n 1 "$TSV_LOG" | awk -F'\t' '{print $1}')"
    if [[ "$last_iteration" =~ ^[0-9]+$ ]]; then
      iteration=$((last_iteration + 1))
    fi
  fi

  while :; do
    local status="no-op"
    local guard="pass"
    local desc_parts=()

    local meta_state tops_state eval_state
    meta_state="$(collect_metacognition_state)" || { guard="fail"; status="crash"; desc_parts+=("metacognition state collection failed"); }
    tops_state="$(collect_tops_state)" || { guard="fail"; status="crash"; desc_parts+=("tops state collection failed"); }
    eval_state="$(collect_eval_state)" || { guard="fail"; status="crash"; desc_parts+=("eval state collection failed"); }

    if [[ "$status" != "crash" ]]; then
      eval "$meta_state"
      eval "$tops_state"
      eval "$eval_state"

      local metric=0
      (( e3_active == 1 || e3_done == 1 )) && metric=$((metric + 1))
      (( after_e3_active == 1 )) && metric=$((metric + 1))
      (( base_sft_active == 1 || base_sft_done == 1 )) && metric=$((metric + 1))
      (( e5_active == 1 || e5_done == 1 )) && metric=$((metric + 1))
      (( e7_active == 1 || e7_done == 1 )) && metric=$((metric + 1))
      (( e8_active == 1 || e8_done == 1 )) && metric=$((metric + 1))
      (( base_eval_active == 1 || base_eval_done == 1 )) && metric=$((metric + 1))
      (( v3_eval_active == 1 || v3_eval_done == 1 )) && metric=$((metric + 1))

      if (( after_e3_active == 0 )) && (( base_sft_active == 0 )) && (( base_sft_done == 0 )) && (( e5_active == 0 )) && (( e5_done == 0 )); then
        if start_after_e3; then
          status="keep"
          desc_parts+=("restarted after_e3 pipeline")
          after_e3_active=1
          metric=$((metric + 1))
        else
          guard="fail"
          status="crash"
          desc_parts+=("failed to restart after_e3 pipeline")
        fi
      fi

      if (( base_sft_done == 1 )) && (( base_eval_active == 0 )) && (( base_eval_done == 0 )) && (( BASE_EVAL_STARTED == 0 )); then
        if start_base_eval; then
          BASE_EVAL_STARTED=1
          status="keep"
          desc_parts+=("started base_sft eval on train_b")
        else
          guard="fail"
          status="crash"
          desc_parts+=("failed to start base_sft eval")
        fi
      fi

      if (( v3_eval_active == 0 )) && (( v3_eval_done == 0 )) && (( V3_EVAL_STARTED == 0 )); then
        if start_v3_eval; then
          V3_EVAL_STARTED=1
          status="keep"
          desc_parts+=("started v3_sft eval on train_b")
        else
          guard="fail"
          status="crash"
          desc_parts+=("failed to start v3_sft eval")
        fi
      fi

      if (( e7_done == 1 )) && (( e8_active == 0 )) && (( e8_done == 0 )) && (( E8_STARTED == 0 )); then
        if start_e8; then
          E8_STARTED=1
          status="keep"
          desc_parts+=("started E8 on tops-caiman")
        else
          guard="fail"
          status="crash"
          desc_parts+=("failed to start E8")
        fi
      fi

      if (( e5_done == 1 )) && (( E5_UPLOAD_STARTED == 0 )); then
        if start_hf_upload "$URL_METACOGNITION" "/scratch/metacognition/checkpoints/grpo_v2_E5/final" "grpo_v2_E5"; then
          E5_UPLOAD_STARTED=1
          status="keep"
          desc_parts+=("started HF upload for E5")
        else
          guard="fail"
          status="crash"
          desc_parts+=("failed to start HF upload for E5")
        fi
      fi

      if (( e8_done == 1 )) && (( E8_UPLOAD_STARTED == 0 )); then
        if start_hf_upload "$URL_TOPS" "/scratch/metacognition/checkpoints/grpo_v2_E8/final" "grpo_v2_E8"; then
          E8_UPLOAD_STARTED=1
          status="keep"
          desc_parts+=("started HF upload for E8")
        else
          guard="fail"
          status="crash"
          desc_parts+=("failed to start HF upload for E8")
        fi
      fi

      desc_parts+=("e3=$( (( e3_done == 1 )) && echo done || ( (( e3_active == 1 )) && echo active || echo waiting ) )")
      desc_parts+=("e3_step=${e3_step}/300")
      desc_parts+=("after_e3=$( (( after_e3_active == 1 )) && echo active || echo missing )")
      desc_parts+=("base_sft=$( (( base_sft_active == 1 )) && echo active || ( (( base_sft_done == 1 )) && echo done || echo waiting ) )")
      desc_parts+=("e5=$( (( e5_active == 1 )) && echo active || ( (( e5_done == 1 )) && echo done || echo waiting ) )")
      desc_parts+=("e7_step=${e7_step}/500")
      desc_parts+=("e8=$( (( e8_active == 1 )) && echo active || ( (( e8_done == 1 )) && echo done || echo waiting ) )")
      desc_parts+=("base_eval=$( (( base_eval_active == 1 )) && echo active || ( (( base_eval_done == 1 )) && echo done || echo waiting ) )")
      desc_parts+=("v3_eval=$( (( v3_eval_active == 1 )) && echo active || ( (( v3_eval_done == 1 )) && echo done || echo waiting ) )")

      local delta=$((metric - LAST_METRIC))
      local desc
      desc="$(IFS='; '; echo "${desc_parts[*]}")"
      append_tsv "$iteration" "$metric" "$delta" "$guard" "$status" "$desc"
      log_text "iteration=$iteration metric=$metric delta=$delta status=$status guard=$guard :: $desc"
      LAST_METRIC="$metric"
      write_state
    else
      append_tsv "$iteration" "0" "0" "$guard" "$status" "$(IFS='; '; echo "${desc_parts[*]}")"
      log_text "iteration=$iteration metric=0 delta=0 status=$status guard=$guard :: ${desc_parts[*]}"
    fi

    if [[ "$MAX_ITERATIONS" != "0" ]] && (( iteration >= MAX_ITERATIONS )); then
      log_text "stopping after bounded iteration budget: $MAX_ITERATIONS"
      break
    fi

    iteration=$((iteration + 1))
    sleep "$SLEEP_SEC"
  done
}

main "$@"
