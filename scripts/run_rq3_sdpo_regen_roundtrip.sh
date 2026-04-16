#!/usr/bin/env bash
set -euo pipefail

# Full round-trip:
# 1. generate on-policy sdpo_regen traces
# 2. optionally build teacher top-k targets
# 3. generate an SFT config with exact dataset/output paths
# 4. optionally launch SFT on H200 x4
#
# Example:
#   bash scripts/run_rq3_sdpo_regen_roundtrip.sh \
#     checkpoints/v8_meta_inside_strict_sft \
#     results/control_rag_real_audit_with_seed.json \
#     results/rq3_online_sdpo_regen

MODEL_PATH="${1:-}"
INPUT_PATH="${2:-}"
OUTPUT_DIR="${3:-results/rq3_online_sdpo_regen}"
TEACHER_MODEL="${4:-}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/sft_rq3_sdpo_regen.yaml}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-checkpoints/rq3_sdpo_regen_sft}"
RUN_NAME="${RUN_NAME:-rq3-sdpo-regen-sft}"
ENABLE_TEACHER_KL="${ENABLE_TEACHER_KL:-0}"
LAUNCH_SFT="${LAUNCH_SFT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAG_TOP_K="${RAG_TOP_K:-0}"
RETRIEVAL_QUERY_MODE="${RETRIEVAL_QUERY_MODE:-none}"
SELECTOR_MODE="${SELECTOR_MODE:-reward_weighted}"
REPAIR_CANDIDATES="${REPAIR_CANDIDATES:-4}"
TP_SIZE="${TP_SIZE:-4}"
shift_count=0

if [[ -z "$MODEL_PATH" || -z "$INPUT_PATH" ]]; then
  echo "Usage: bash scripts/run_rq3_sdpo_regen_roundtrip.sh <model_path> <input_path> [output_dir] [teacher_model]" >&2
  exit 1
fi

if [[ $# -ge 4 ]]; then
  shift_count=4
elif [[ $# -eq 3 ]]; then
  shift_count=3
elif [[ $# -eq 2 ]]; then
  shift_count=2
fi
shift "$shift_count"

EXAMPLE_BANK_ARGS=()
for item in "$@"; do
  EXAMPLE_BANK_ARGS+=(--example_bank "$item")
done

if [[ "$ENABLE_TEACHER_KL" == "1" && -z "$TEACHER_MODEL" ]]; then
  echo "FATAL: ENABLE_TEACHER_KL=1 requires <teacher_model> as the 4th argument." >&2
  exit 1
fi

if [[ "$RAG_TOP_K" -gt 0 && "$RETRIEVAL_QUERY_MODE" != "none" && ${#EXAMPLE_BANK_ARGS[@]} -eq 0 ]]; then
  echo "FATAL: RAG was requested for sdpo_regen but no example_bank paths were provided." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

AVAILABLE_GPUS="$("$PYTHON_BIN" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if [[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]] && [[ "$AVAILABLE_GPUS" -gt 0 ]] && [[ "$TP_SIZE" -gt "$AVAILABLE_GPUS" ]]; then
  echo "[warn] TP_SIZE=$TP_SIZE but only $AVAILABLE_GPUS GPU(s) visible; lowering TP_SIZE to $AVAILABLE_GPUS" >&2
  TP_SIZE="$AVAILABLE_GPUS"
fi

PYTHONPATH=. "$PYTHON_BIN" scripts/run_online_sdpo_regen.py \
  --model_path "$MODEL_PATH" \
  --input_path "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --mode sdpo_regen \
  --selector_mode "$SELECTOR_MODE" \
  --repair_candidates "$REPAIR_CANDIDATES" \
  --rag_top_k "$RAG_TOP_K" \
  --retrieval_query_mode "$RETRIEVAL_QUERY_MODE" \
  --tp_size "$TP_SIZE" \
  --dataset_mode sdpo_regen \
  "${EXAMPLE_BANK_ARGS[@]}"

DATASET_PATH="$OUTPUT_DIR/online_sdpo_regen.parquet"

if [[ -n "$TEACHER_MODEL" ]]; then
  PYTHONPATH=. "$PYTHON_BIN" scripts/build_teacher_topk_targets.py \
    --input "$DATASET_PATH" \
    --teacher_model_path "$TEACHER_MODEL" \
    --output "$OUTPUT_DIR/teacher_topk_targets.parquet" \
    --summary_json "$OUTPUT_DIR/teacher_topk_summary.json"
fi

CONFIG_OUT="$OUTPUT_DIR/generated_sft_config.yaml"
TRAIN_DATASET="$DATASET_PATH"
PREPARE_ARGS=()
if [[ "$ENABLE_TEACHER_KL" == "1" ]]; then
  TRAIN_DATASET="$OUTPUT_DIR/teacher_topk_targets.parquet"
  [[ -f "$TRAIN_DATASET" ]] || { echo "FATAL: missing teacher top-k dataset at $TRAIN_DATASET" >&2; exit 1; }
  PREPARE_ARGS+=(--enable_teacher_kl)
fi

PYTHONPATH=. "$PYTHON_BIN" scripts/prepare_self_distill_sft_config.py \
  --template_config "$TEMPLATE_CONFIG" \
  --output_config "$CONFIG_OUT" \
  --model_path "$MODEL_PATH" \
  --dataset_path "$TRAIN_DATASET" \
  --train_output_dir "$TRAIN_OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  "${PREPARE_ARGS[@]}"

echo
echo "Generated artifacts:"
echo "  $OUTPUT_DIR/online_sdpo_traces.jsonl"
echo "  $OUTPUT_DIR/online_sdpo_regen.parquet"
if [[ -n "$TEACHER_MODEL" ]]; then
  echo "  $OUTPUT_DIR/teacher_topk_targets.parquet"
  echo "  $OUTPUT_DIR/teacher_topk_summary.json"
fi
echo "  $CONFIG_OUT"
echo
echo "Next:"
echo "  1. Review generated config: $CONFIG_OUT"
echo "  2. Run: bash scripts/run_self_distill_sft_h200.sh $CONFIG_OUT"

if [[ "$LAUNCH_SFT" == "1" ]]; then
  bash scripts/run_self_distill_sft_h200.sh "$CONFIG_OUT"
fi
