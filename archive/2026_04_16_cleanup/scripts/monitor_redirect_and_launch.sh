#!/bin/bash
# Monitor redirect data generation, then auto-merge and launch SFT
# Run: nohup bash scripts/monitor_redirect_and_launch.sh > logs/monitor_redirect.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

REDIRECT_PID=1560955
REDIRECT_PARQUET="data/v6_10k_redirect_full.parquet"
VERIFY_STRAIGHT="data/v6_10k_verify_straight.parquet"
MERGED_OUTPUT="data/v6_clean_10k_merged.parquet"
LOG="data/v6_redirect_full.log"

echo "[$(date -u)] Monitor started. Watching PID $REDIRECT_PID"

# ── Wait for redirect process to finish ──
while kill -0 $REDIRECT_PID 2>/dev/null; do
    # Print progress every 5 minutes
    tail -1 "$LOG" 2>/dev/null | head -1
    sleep 300
done

echo "[$(date -u)] Redirect generation process finished."

# ── Check if parquet was created ──
if [ ! -f "$REDIRECT_PARQUET" ]; then
    echo "ERROR: $REDIRECT_PARQUET not found after process finished!"
    echo "Check $LOG for errors."
    exit 1
fi

ROW_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$REDIRECT_PARQUET')))")
echo "[$(date -u)] Redirect data: $ROW_COUNT rows"

if [ "$ROW_COUNT" -lt 1000 ]; then
    echo "ERROR: Too few rows ($ROW_COUNT < 1000). Something went wrong."
    exit 1
fi

# ── Step 1: Merge + Audit ──
echo "[$(date -u)] Running merge + quality audit..."
python3 scripts/merge_v6_clean_10k.py \
    --redirect "$REDIRECT_PARQUET" \
    --verify-straight "$VERIFY_STRAIGHT" \
    --output "$MERGED_OUTPUT" \
    --audit \
    --filter-theatrical

if [ ! -f "$MERGED_OUTPUT" ]; then
    echo "ERROR: Merge failed, $MERGED_OUTPUT not created."
    exit 1
fi

MERGED_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$MERGED_OUTPUT')))")
echo "[$(date -u)] Merged data: $MERGED_COUNT rows"

# ── Step 2: Launch SFT on all nodes ──
echo "[$(date -u)] Launching SFT on all 3 nodes..."
bash scripts/launch_v6_clean_sft_all_nodes.sh

echo "[$(date -u)] All done. SFT training launched on 3 nodes."
echo "[$(date -u)] Check WandB for training progress."
