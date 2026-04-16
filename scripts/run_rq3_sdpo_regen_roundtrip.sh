#!/usr/bin/env bash
set -euo pipefail

# Minimal round-trip:
# 1. generate on-policy sdpo_regen traces
# 2. build SFT-ready parquet
# 3. run SFT with configs/sft_rq3_sdpo_regen.yaml
#
# Example:
#   bash scripts/run_rq3_sdpo_regen_roundtrip.sh \
#     checkpoints/v8_meta_inside_strict_sft \
#     results/control_rag_real_audit_with_seed.json \
#     results/rq3_online_sdpo_regen

MODEL_PATH="${1:-}"
INPUT_PATH="${2:-}"
OUTPUT_DIR="${3:-results/rq3_online_sdpo_regen}"

if [[ -z "$MODEL_PATH" || -z "$INPUT_PATH" ]]; then
  echo "Usage: bash scripts/run_rq3_sdpo_regen_roundtrip.sh <model_path> <input_path> [output_dir]" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHONPATH=. python scripts/run_online_sdpo_regen.py \
  --model_path "$MODEL_PATH" \
  --input_path "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR"

echo
echo "Generated artifacts:"
echo "  $OUTPUT_DIR/online_sdpo_traces.jsonl"
echo "  $OUTPUT_DIR/online_sdpo_regen.parquet"
echo
echo "Next:"
echo "  1. Edit configs/sft_rq3_sdpo_regen.yaml dataset_path/output_dir if needed"
echo "  2. Run: PYTHONPATH=. python src/training/sft.py --config configs/sft_rq3_sdpo_regen.yaml"
