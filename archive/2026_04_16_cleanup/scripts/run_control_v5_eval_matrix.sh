#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
conda activate "$REMOTE_CONDA_ENV"
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

OUTPUT_DIR="${OUTPUT_DIR:-results/control_v5_eval}"
BENCHMARKS="${BENCHMARKS:-gsm8k math500 aime2024}"
MAX_PROBLEMS="${MAX_PROBLEMS:-30}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
RAG_BANKS="${RAG_BANKS:-}"

mkdir -p "$OUTPUT_DIR"

if [[ "$#" -eq 0 ]]; then
  echo "usage: bash scripts/run_control_v5_eval_matrix.sh name=path [name=path ...]"
  exit 1
fi

for spec in "$@"; do
  name="${spec%%=*}"
  path="${spec#*=}"
  if [[ -z "$name" || -z "$path" || "$name" == "$path" ]]; then
    echo "invalid spec: $spec"
    exit 1
  fi

  if [[ ! -f "$path/config.json" ]]; then
    echo "=== Materializing $name to $path before eval ==="
    python scripts/ensure_hf_model.py \
      --model-name "$name" \
      --output-dir "$path" \
      --wait \
      --poll-seconds 60 \
      --timeout-seconds 7200
  fi

  cmd=(
    python -u src/eval/eval_hf.py
    --model_path "$path"
    --model_name "$name"
    --benchmarks $BENCHMARKS
    --max_problems "$MAX_PROBLEMS"
    --num_samples "$NUM_SAMPLES"
    --output_dir "$OUTPUT_DIR"
  )

  if [[ -n "$RAG_BANKS" ]]; then
    cmd+=(--rag_example_bank $RAG_BANKS)
  fi

  echo "=== Evaluating $name ==="
  "${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/${name}.log"
done

python scripts/analyze_control_v5_eval.py \
  --results_dir "$OUTPUT_DIR" \
  --output_prefix "$OUTPUT_DIR/control_v5_summary"
