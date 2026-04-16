#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <mode:{question_only_best_of_n|fixed_k_repair}> <model_path> <input_path> <output_dir> <dataset_mode:{naive|epistemic}> <claim_bearing:{0|1}> [example_bank ...]" >&2
  exit 1
fi

MODE="$1"
MODEL_PATH="$2"
INPUT_PATH="$3"
OUTPUT_DIR="$4"
DATASET_MODE="$5"
CLAIM_BEARING_FLAG="$6"
shift 6

EXAMPLE_BANK_ARGS=()
for item in "$@"; do
  EXAMPLE_BANK_ARGS+=(--example_bank "$item")
done

RAG_TOP_K=0
RETRIEVAL_QUERY_MODE=none
SELECTOR_MODE=correctness_only
if [[ "$MODE" == "fixed_k_repair" ]]; then
  SELECTOR_MODE=reward_weighted
  if [[ ${#EXAMPLE_BANK_ARGS[@]} -gt 0 ]]; then
    RAG_TOP_K=1
    RETRIEVAL_QUERY_MODE=question_only
  fi
fi

mkdir -p "$OUTPUT_DIR"

CLAIM_BEARING_ARGS=()
if [[ "$CLAIM_BEARING_FLAG" == "1" ]]; then
  CLAIM_BEARING_ARGS+=(--claim-bearing)
fi

PYTHONPATH=. python scripts/run_online_sdpo_regen.py \
  --model_path "$MODEL_PATH" \
  --input_path "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --dataset_mode "$DATASET_MODE" \
  --repair_candidates 4 \
  --selector_mode "$SELECTOR_MODE" \
  --rag_top_k "$RAG_TOP_K" \
  --retrieval_query_mode "$RETRIEVAL_QUERY_MODE" \
  "${CLAIM_BEARING_ARGS[@]}" \
  "${EXAMPLE_BANK_ARGS[@]}"

echo "[next] Generated dataset: $OUTPUT_DIR/online_sdpo_regen.parquet"
echo "[next] Summary: $OUTPUT_DIR/summary.json"
echo "[info] mode=$MODE selector=$SELECTOR_MODE retrieval_mode=$RETRIEVAL_QUERY_MODE top_k=$RAG_TOP_K"
