#!/bin/bash
# ================================================================
# Compute Node Cleanup Script
# Run on the compute node to free ~434 GB of disk space
#
# WHAT THIS DOES:
#   1. Deletes old Qwen2.5 checkpoints that are no longer needed
#   2. Cleans up __pycache__, wandb runs, and temp files
#   3. Removes old verl data prep artifacts
#
# WHAT THIS KEEPS:
#   - base_sft (Qwen2.5 control group for paper comparison)
#   - qwen3_meta_sft (currently training)
#   - rollouts/ (needed for GRPO training data)
#   - sft_data/ (needed for SFT re-training)
#   - metacot_chains/ (GPT-5.4 generated chains, expensive to regenerate)
#   - profiles/ (lightweight, useful reference)
#
# Usage:
#   bash scripts/cleanup_compute.sh           # dry run (default)
#   bash scripts/cleanup_compute.sh --execute # actually delete
# ================================================================
set -e

SCRATCH="/scratch/metacognition"
DRY_RUN=true

if [ "$1" = "--execute" ]; then
    DRY_RUN=false
    echo "*** EXECUTE MODE: Files will be permanently deleted ***"
    echo "Press Ctrl+C within 5 seconds to abort..."
    sleep 5
else
    echo "*** DRY RUN MODE: No files will be deleted ***"
    echo "Run with --execute to actually delete files."
    echo ""
fi

# Helper function
delete_path() {
    local path="$1"
    local desc="$2"
    if [ -e "$path" ]; then
        local size
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would delete: $path ($size) -- $desc"
        else
            echo "Deleting: $path ($size) -- $desc"
            rm -rf "$path"
        fi
    else
        echo "[SKIP] Not found: $path"
    fi
}

echo ""
echo "========================================="
echo "1. Old Qwen2.5 Checkpoints (~434 GB)"
echo "========================================="

# meta_sft: Old Qwen2.5 meta SFT (replaced by qwen3_meta_sft)
delete_path "$SCRATCH/checkpoints/meta_sft" \
    "Old Qwen2.5 Meta-SFT (replaced by qwen3_meta_sft)"

# meta_grpo: Old v1 GRPO experiment
delete_path "$SCRATCH/checkpoints/meta_grpo" \
    "Old v1 GRPO experiment (43G)"

# meta_grpo_v2: ZeRO-3 failed run
delete_path "$SCRATCH/checkpoints/meta_grpo_v2" \
    "Failed ZeRO-3 GRPO run (15G)"

# probe_meta_sft: Qwen2.5 probe data (can retrain on Qwen3)
delete_path "$SCRATCH/checkpoints/probe_meta_sft" \
    "Qwen2.5 probe hidden states (21G, can retrain on Qwen3)"

# phase1_sft: Old intermediate SFT checkpoint
delete_path "$SCRATCH/checkpoints/phase1_sft" \
    "Old Phase 1 SFT checkpoint (Qwen2.5)"

# phase2_grpo: Old Phase 2 GRPO checkpoint
delete_path "$SCRATCH/checkpoints/phase2_grpo" \
    "Old Phase 2 GRPO checkpoint (Qwen2.5)"

# simple_probe: Old probe checkpoint
delete_path "$SCRATCH/checkpoints/simple_probe" \
    "Old simple probe checkpoint (Qwen2.5)"

echo ""
echo "========================================="
echo "2. Cache and Temp Files"
echo "========================================="

# __pycache__ directories
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would delete all __pycache__ directories under $SCRATCH"
    find "$SCRATCH" -type d -name "__pycache__" 2>/dev/null | while read -r d; do
        echo "  $d"
    done
else
    echo "Deleting __pycache__ directories..."
    find "$SCRATCH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
fi

# wandb runs
delete_path "$SCRATCH/wandb" \
    "Old wandb run logs"

# verl temp data
delete_path "$SCRATCH/verl_train.parquet" \
    "verl training data prep artifact"
delete_path "$SCRATCH/verl_val.parquet" \
    "verl validation data prep artifact"

# Old gnosis_data (hidden states for Qwen2.5 probe)
delete_path "$SCRATCH/gnosis_data" \
    "Qwen2.5 cached hidden states for probe training"

# Eval results (lightweight but outdated)
delete_path "$SCRATCH/eval_results.json" \
    "Old eval results (Qwen2.5)"
delete_path "$SCRATCH/eval_results_hf.json" \
    "Old HF eval results (Qwen2.5)"

# Curriculum outputs (from skeleton code)
delete_path "$SCRATCH/curriculum" \
    "Old curriculum learning outputs"

echo ""
echo "========================================="
echo "3. Verification"
echo "========================================="

echo ""
echo "Remaining checkpoints:"
if [ -d "$SCRATCH/checkpoints" ]; then
    du -sh "$SCRATCH/checkpoints"/*/ 2>/dev/null || echo "  (none)"
else
    echo "  (checkpoints dir not found)"
fi

echo ""
echo "Remaining data:"
for d in rollouts sft_data metacot_chains profiles; do
    if [ -d "$SCRATCH/$d" ]; then
        size=$(du -sh "$SCRATCH/$d" 2>/dev/null | cut -f1)
        echo "  $d: $size"
    fi
done

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete. Run with --execute to actually delete files."
else
    echo "Cleanup complete!"
    echo ""
    echo "Total disk usage after cleanup:"
    du -sh "$SCRATCH" 2>/dev/null || echo "  (could not measure)"
fi
