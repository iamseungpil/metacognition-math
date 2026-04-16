#!/usr/bin/env bash
set -euo pipefail

# Build OPD-style teacher top-k targets from an sdpo_regen parquet.
#
# Example:
#   bash scripts/run_rq3_teacher_query_roundtrip.sh \
#     results/rq3_online_sdpo_regen/online_sdpo_regen.parquet \
#     checkpoints/v8_meta_inside_strict_sft \
#     results/rq3_teacher_topk

INPUT_PARQUET="${1:-}"
TEACHER_MODEL="${2:-}"
OUTPUT_DIR="${3:-results/rq3_teacher_topk}"

if [[ -z "$INPUT_PARQUET" || -z "$TEACHER_MODEL" ]]; then
  echo "Usage: bash scripts/run_rq3_teacher_query_roundtrip.sh <sdpo_regen_parquet> <teacher_model> [output_dir]" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHONPATH=. python scripts/build_teacher_topk_targets.py \
  --input "$INPUT_PARQUET" \
  --teacher_model_path "$TEACHER_MODEL" \
  --output "$OUTPUT_DIR/teacher_topk_targets.parquet" \
  --summary_json "$OUTPUT_DIR/teacher_topk_summary.json"

echo
echo "Generated:"
echo "  $OUTPUT_DIR/teacher_topk_targets.parquet"
echo "  $OUTPUT_DIR/teacher_topk_summary.json"
