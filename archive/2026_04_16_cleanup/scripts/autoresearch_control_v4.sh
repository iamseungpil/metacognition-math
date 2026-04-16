#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/results/autoresearch_control_v4"
mkdir -p "$LOG_DIR"

SMOKE_OUT="$ROOT/data/control_v4_trapi_smoke.parquet"
FULL_OUT="$ROOT/data/control_v4_trapi_round1.parquet"

echo "[1/4] Smoke generation"
python3 scripts/gen_control_v4_trapi.py \
  --straight-easy 1 \
  --verify-easy 1 \
  --straight-medium 1 \
  --verify-medium 1 \
  --redirect-medium 1 \
  --verify-hard 1 \
  --redirect-hard 1 \
  --concurrent 4 \
  --output "$SMOKE_OUT" | tee "$LOG_DIR/smoke_generation.log"

echo "[2/4] Smoke QC"
python3 scripts/qc_control_v4_samples.py \
  --input "$SMOKE_OUT" \
  --samples-per-bucket 1 | tee "$LOG_DIR/smoke_qc.log"

echo "[3/4] Main generation"
python3 scripts/gen_control_v4_trapi.py \
  --straight-easy 600 \
  --verify-easy 600 \
  --straight-medium 900 \
  --verify-medium 900 \
  --redirect-medium 900 \
  --verify-hard 1050 \
  --redirect-hard 1050 \
  --oversample-factor 2 \
  --concurrent 12 \
  --output "$FULL_OUT" | tee "$LOG_DIR/main_generation.log"

echo "[4/4] Build variants"
python3 scripts/build_control_v4_sft_variants.py \
  --input "$FULL_OUT" \
  --output-dir "$ROOT/data" | tee "$LOG_DIR/build_variants.log"

echo "Artifacts:"
echo "  $SMOKE_OUT"
echo "  $FULL_OUT"
echo "  $ROOT/data/control_v4_all_sft.parquet"
echo "  $ROOT/data/control_v4_verify_sft.parquet"
echo "  $ROOT/data/control_v4_redirect_sft.parquet"
