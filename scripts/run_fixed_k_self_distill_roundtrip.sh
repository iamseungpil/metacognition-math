#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <model_path> <input_path> <output_dir> <dataset_mode:{naive|epistemic}> <claim_bearing:{0|1}> [example_bank ...]" >&2
  exit 1
fi

MODEL_PATH="$1"
INPUT_PATH="$2"
OUTPUT_DIR="$3"
DATASET_MODE="$4"
CLAIM_BEARING_FLAG="$5"
shift 5

EXAMPLE_BANK_ARGS=()
for item in "$@"; do
  EXAMPLE_BANK_ARGS+=(--example_bank "$item")
done

RAG_TOP_K=0
RETRIEVAL_QUERY_MODE=none
if [[ ${#EXAMPLE_BANK_ARGS[@]} -gt 0 ]]; then
  RAG_TOP_K=1
  RETRIEVAL_QUERY_MODE=question_only
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
  --mode fixed_k_repair \
  --dataset_mode "$DATASET_MODE" \
  --repair_candidates 4 \
  --rag_top_k "$RAG_TOP_K" \
  --retrieval_query_mode "$RETRIEVAL_QUERY_MODE" \
  "${CLAIM_BEARING_ARGS[@]}" \
  "${EXAMPLE_BANK_ARGS[@]}"

echo "[next] Generated dataset: $OUTPUT_DIR/online_sdpo_regen.parquet"
echo "[next] Summary: $OUTPUT_DIR/summary.json"
if [[ ${#EXAMPLE_BANK_ARGS[@]} -gt 0 ]]; then
  echo "[info] retrieval active with ${#EXAMPLE_BANK_ARGS[@]} example-bank argument(s); mode=$RETRIEVAL_QUERY_MODE top_k=$RAG_TOP_K"
else
  echo "[info] retrieval disabled for this roundtrip because no example bank was supplied"
fi
echo "[next] Train with one of:"
echo "  PYTHONPATH=. python src/training/sft.py --config configs/sft_self_distill_base_fixedk_naive.yaml"
echo "  PYTHONPATH=. python src/training/sft.py --config configs/sft_self_distill_meta_fixedk_epistemic.yaml"
