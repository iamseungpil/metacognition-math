#!/bin/bash
# =============================================================================
# autoresearch.sh — Run a single hypothesis experiment for Meta-CoT autoresearch
#
# Takes a hypothesis ID (H1-H6) and runs the corresponding experiment:
#   H1: Eval with max_tokens=4096 (verification that truncation is fixed)
#   H3: Verify-only meta (strip pre/mid meta, keep post-solution verification)
#   H5: Continue GRPO training from checkpoint for more steps
#
# Each hypothesis: prepare -> train (if needed) -> eval -> compare vs Base SFT
#
# Usage:
#   bash scripts/autoresearch.sh H1
#   bash scripts/autoresearch.sh H3
#   bash scripts/autoresearch.sh H5 [--grpo-steps 500] [--grpo-mode E3]
#   bash scripts/autoresearch.sh H1 --dry-run
#
# Environment:
#   Expects conda env 'grpo' with torch, trl, transformers
#   PYTHONPATH must include project root
#   4x A100 80GB GPUs available
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults and argument parsing
# ---------------------------------------------------------------------------
HYPOTHESIS="${1:-}"
DRY_RUN=false
GRPO_STEPS=500
GRPO_MODE="E3"
BASE_SFT_MODEL="checkpoints/qwen3_base_sft"
META_SFT_MODEL="checkpoints/qwen3_metacot_v2_sft"
BENCHMARKS="gsm8k math500 aime2024"
MAX_PROBLEMS=500
RESULTS_DIR="results/autoresearch"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --grpo-steps) GRPO_STEPS="$2"; shift 2 ;;
        --grpo-mode) GRPO_MODE="$2"; shift 2 ;;
        --max-problems) MAX_PROBLEMS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$HYPOTHESIS" ]]; then
    echo "Usage: bash scripts/autoresearch.sh <H1|H3|H5> [options]"
    echo ""
    echo "Hypotheses:"
    echo "  H1  Eval with max_tokens=4096 (verify truncation fix)"
    echo "  H3  Verify-only meta (keep only post-solution meta block)"
    echo "  H5  Continue GRPO training (more steps from checkpoint)"
    echo ""
    echo "Options:"
    echo "  --dry-run          Show what would happen without executing"
    echo "  --grpo-steps N     GRPO training steps for H5 (default: 500)"
    echo "  --grpo-mode MODE   GRPO experiment mode for H5 (default: E3)"
    echo "  --max-problems N   Max problems per benchmark for eval (default: 500)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Conda activation (works on both local and AMLT)
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    eval "$(conda shell.bash hook 2>/dev/null || true)"
fi
conda activate grpo 2>/dev/null || echo "Warning: grpo env not found, using current env"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
export WANDB_PROJECT="metacot-math"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_eval() {
    # $1 = model_path, $2 = model_name, $3 = gpu_id (optional, default 0)
    local model_path="$1"
    local model_name="$2"
    local gpu_id="${3:-0}"

    log "Eval: $model_name ($model_path) on GPU $gpu_id"

    if $DRY_RUN; then
        log "[DRY-RUN] Would run eval: model=$model_path, name=$model_name"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id python -u src/eval/eval_hf.py \
        --model_path "$model_path" \
        --model_name "${model_name}" \
        --benchmarks $BENCHMARKS \
        --max_problems $MAX_PROBLEMS \
        --output_dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/eval_${model_name}.log"
}

extract_accuracy() {
    # Extract overall accuracy from eval JSON
    local eval_file="$1"
    if [[ ! -f "$eval_file" ]]; then
        echo "0.0"
        return
    fi
    python3 -c "
import json
with open('$eval_file') as f:
    data = json.load(f)
results = data['results']
correct = sum(1 for r in results if r['is_correct'])
total = len(results)
print(f'{correct / total:.4f}' if total > 0 else '0.0')
"
}

compare_results() {
    # Compare Meta-CoT accuracy vs Base SFT accuracy
    local meta_name="$1"
    local base_name="$2"

    local meta_file="$RESULTS_DIR/eval_${meta_name}.json"
    local base_file="$RESULTS_DIR/eval_${base_name}.json"

    # If base eval doesn't exist yet, run it
    if [[ ! -f "$base_file" ]]; then
        log "Base SFT eval not found, running baseline eval..."
        run_eval "$BASE_SFT_MODEL" "$base_name" 1
    fi

    local meta_acc=$(extract_accuracy "$meta_file")
    local base_acc=$(extract_accuracy "$base_file")

    log "=============================================="
    log "  COMPARISON: $meta_name vs $base_name"
    log "=============================================="
    log "  Meta-CoT accuracy: ${meta_acc}"
    log "  Base SFT accuracy: ${base_acc}"

    # Compare
    python3 -c "
meta = float('$meta_acc')
base = float('$base_acc')
diff = meta - base
if meta >= base:
    print(f'  PASS: Meta-CoT ({meta:.1%}) >= Base SFT ({base:.1%}), diff = +{diff:.1%}')
else:
    print(f'  FAIL: Meta-CoT ({meta:.1%}) < Base SFT ({base:.1%}), diff = {diff:.1%}')
"
    # Return exit code: 0 if pass, 1 if fail
    python3 -c "exit(0 if float('$meta_acc') >= float('$base_acc') else 1)"
}

# ---------------------------------------------------------------------------
# Hypothesis implementations
# ---------------------------------------------------------------------------

run_h1() {
    # H1: Eval with max_tokens=4096 (already set in eval_hf.py)
    # This is a verification that truncation was causing 31% of errors.
    log "=== H1: Verify max_tokens=4096 eval ==="
    log "eval_hf.py already uses max_tokens=4096 by default."
    log "Running eval on existing Meta SFT model to verify."

    local model_name="h1_meta_sft_4096_${TIMESTAMP}"
    local base_name="h1_base_sft_${TIMESTAMP}"

    # Run Meta-CoT eval
    run_eval "$META_SFT_MODEL" "$model_name" 0

    # Run Base SFT eval for fair comparison
    run_eval "$BASE_SFT_MODEL" "$base_name" 1

    # Also check if there's a GRPO checkpoint to eval
    local grpo_path=""
    if [[ -d "checkpoints/grpo_v2_E3/final" ]]; then
        grpo_path="checkpoints/grpo_v2_E3/final"
    else
        grpo_path=$(ls -d checkpoints/grpo_v2_E3/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    fi

    if [[ -n "$grpo_path" && -d "$grpo_path" ]]; then
        local grpo_name="h1_grpo_e3_4096_${TIMESTAMP}"
        run_eval "$grpo_path" "$grpo_name" 2
    fi

    # Compare
    compare_results "$model_name" "$base_name"
}


run_h3() {
    # H3: Verification-only meta (strip pre/mid, keep post)
    log "=== H3: Verification-Only Meta ==="

    # Step 1: Create verify-only SFT dataset
    log "Step 1: Creating verify-only SFT dataset..."
    if $DRY_RUN; then
        log "[DRY-RUN] Would create verify-only SFT data at data/verifyonly_sft.parquet"
    else
        python scripts/create_verifyonly_sft.py --output data/verifyonly_sft.parquet
    fi

    # Step 2: Create SFT config for verify-only training
    local sft_config="configs/verifyonly_sft.yaml"
    local sft_output="checkpoints/qwen3_verifyonly_sft"
    log "Step 2: SFT training on verify-only data..."

    if $DRY_RUN; then
        log "[DRY-RUN] Would create config at $sft_config and train SFT"
    else
        cat > "$sft_config" <<YAML
# H3: Verify-Only Meta SFT
model_name_or_path: Qwen/Qwen3-8B
dataset_path: ${PROJECT_ROOT}/data/verifyonly_sft.parquet
output_dir: ${PROJECT_ROOT}/${sft_output}

num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
max_length: 4096
save_steps: 200

wandb_project: metacot-math
run_name: qwen3-verifyonly-sft
YAML

        # Run SFT training
        accelerate launch --config_file configs/accelerate_grpo.yaml \
            src/training/sft.py --config "$sft_config" \
            2>&1 | tee "$RESULTS_DIR/sft_verifyonly.log"
    fi

    # Step 3: GRPO on verify-only SFT model
    local grpo_output="checkpoints/grpo_verifyonly_${GRPO_MODE}"
    log "Step 3: GRPO training on verify-only SFT model..."

    if $DRY_RUN; then
        log "[DRY-RUN] Would run GRPO: mode=$GRPO_MODE, steps=$GRPO_STEPS"
    else
        accelerate launch --config_file configs/accelerate_grpo.yaml \
            src/training/grpo_v2.py \
            --mode "$GRPO_MODE" \
            --max_steps "$GRPO_STEPS" \
            --model_path "$sft_output" \
            --data mixed \
            --output_dir "$grpo_output" \
            2>&1 | tee "$RESULTS_DIR/grpo_verifyonly.log"
    fi

    # Step 4: Eval
    local grpo_final="$grpo_output/final"
    if [[ ! -d "$grpo_final" ]]; then
        grpo_final=$(ls -d ${grpo_output}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || echo "$sft_output")
    fi

    local meta_name="h3_verifyonly_${TIMESTAMP}"
    local base_name="h3_base_sft_${TIMESTAMP}"

    run_eval "$grpo_final" "$meta_name" 0
    run_eval "$BASE_SFT_MODEL" "$base_name" 1

    compare_results "$meta_name" "$base_name"
}


run_h5() {
    # H5: Continue GRPO training from checkpoint for more steps
    log "=== H5: Extended GRPO Training ($GRPO_STEPS steps) ==="

    # Find latest checkpoint
    local resume_path=""
    if [[ -d "checkpoints/grpo_v2_${GRPO_MODE}/checkpoint-200" ]]; then
        resume_path="checkpoints/grpo_v2_${GRPO_MODE}/checkpoint-200"
    else
        resume_path=$(ls -d checkpoints/grpo_v2_${GRPO_MODE}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    fi

    if [[ -z "$resume_path" || ! -d "$resume_path" ]]; then
        log "ERROR: No GRPO checkpoint found at checkpoints/grpo_v2_${GRPO_MODE}/"
        log "Run GRPO training first: bash scripts/run_grpo_v2.sh $GRPO_MODE 200"
        exit 1
    fi
    log "Resuming from: $resume_path"

    local grpo_output="checkpoints/grpo_v2_${GRPO_MODE}_extended"

    if $DRY_RUN; then
        log "[DRY-RUN] Would continue GRPO from $resume_path for $GRPO_STEPS steps"
    else
        # Continue training with more steps
        accelerate launch --config_file configs/accelerate_grpo.yaml \
            src/training/grpo_v2.py \
            --mode "$GRPO_MODE" \
            --max_steps "$GRPO_STEPS" \
            --model_path "$resume_path" \
            --data mixed \
            --output_dir "$grpo_output" \
            2>&1 | tee "$RESULTS_DIR/grpo_extended.log"
    fi

    # Eval
    local grpo_final="$grpo_output/final"
    if [[ ! -d "$grpo_final" ]]; then
        grpo_final=$(ls -d ${grpo_output}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || echo "$resume_path")
    fi

    local meta_name="h5_extended_${GRPO_MODE}_${GRPO_STEPS}s_${TIMESTAMP}"
    local base_name="h5_base_sft_${TIMESTAMP}"

    run_eval "$grpo_final" "$meta_name" 0
    run_eval "$BASE_SFT_MODEL" "$base_name" 1

    compare_results "$meta_name" "$base_name"
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

log "=============================================="
log "  AUTORESEARCH: Hypothesis $HYPOTHESIS"
log "  Dry run: $DRY_RUN"
log "  Timestamp: $TIMESTAMP"
log "  Results: $RESULTS_DIR"
log "=============================================="

case "$HYPOTHESIS" in
    H1|h1) run_h1 ;;
    H3|h3) run_h3 ;;
    H5|h5) run_h5 ;;
    *)
        log "ERROR: Unsupported hypothesis: $HYPOTHESIS"
        log "Supported: H1, H3, H5"
        log "H2 (difficulty-adaptive), H4 (GPT-5.4 data), H6 (stepwise reward)"
        log "are handled by autoresearch_loop.py via GRPO mode selection."
        exit 1
        ;;
esac

RESULT=$?
if [[ $RESULT -eq 0 ]]; then
    log "HYPOTHESIS $HYPOTHESIS: PASS"
else
    log "HYPOTHESIS $HYPOTHESIS: FAIL (Meta-CoT < Base SFT)"
fi

exit $RESULT
