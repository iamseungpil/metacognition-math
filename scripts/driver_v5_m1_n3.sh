#!/bin/bash
# driver_v5_m1_n3.sh — Unified M1/N3 driver with HF resume + monitor.
#
# Runs on mainline nodes (eval or train_b per NODE_POLICY).
# Composes: stage_n3_e8.sh + smoke_n3.sh + n3_monitor_push.sh
# with preempt-safe HF push (5-min) and auto-resume from HF checkpoint.
#
# Usage (on node):
#   HF_TOKEN=... VARIANT=m1|n3|sdc-split|sdc-shared bash scripts/driver_v5_m1_n3.sh
#
# Env overrides:
#   VARIANT   : m1 | n3 | n3-random | n3-fullmask | sdc-split | sdc-uniform | sdc-noise | sdc-shared (default: n3)
#   SEED      : random seed (default: 42)
#   TOTAL     : total training steps (default: 300 — full run; 50 = smoke)
#   SAVE_EVERY: checkpoint save interval (default: 20)
#   HF_POLL   : HF push poll interval (seconds, default: 300 = 5 min)
#   RUN_ID    : override run id (default: auto-generated)
#
# Writes:
#   /scratch/meta/run/<run_id>/ (stdout.log, stderr.log, metrics.jsonl, checkpoints/)
#   HF: iamseungpil/metacot / n3_runs/<run_id>/ (mirrored every HF_POLL sec)
set -uo pipefail

# ── Args / env ────────────────────────────────────────────────────────────
HF_TOKEN="${HF_TOKEN:?HF_TOKEN required}"
VARIANT="${VARIANT:-n3}"
SEED="${SEED:-42}"
TOTAL="${TOTAL:-300}"
SAVE_EVERY="${SAVE_EVERY:-20}"
HF_POLL="${HF_POLL:-300}"

STAGE_ROOT=/scratch/meta
CODE_DIR="$STAGE_ROOT/code"

# ── RUN_ID persistence (review iter1 Fix 1) ───────────────────────────────
# Persist per (variant, seed) so preempt restarts resume the same RUN_ID
# (and therefore the same checkpoints dir).
RUN_ID_FILE="$STAGE_ROOT/run/LAST_RUN_ID.${VARIANT}.s${SEED}"
mkdir -p "$(dirname "$RUN_ID_FILE")"
if [ -z "${RUN_ID:-}" ]; then
    if [ -f "$RUN_ID_FILE" ]; then
        RUN_ID="$(cat "$RUN_ID_FILE")"
        echo "[driver] resume: found existing RUN_ID=$RUN_ID in $RUN_ID_FILE"
    else
        RUN_ID="${VARIANT}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
        echo "$RUN_ID" > "$RUN_ID_FILE"
        echo "[driver] fresh: RUN_ID=$RUN_ID written to $RUN_ID_FILE"
    fi
fi

RUN_DIR="$STAGE_ROOT/run/$RUN_ID"
LOG="$RUN_DIR/driver.log"

mkdir -p "$RUN_DIR"
# All logs from this driver go to driver.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) DRIVER v5 START variant=$VARIANT seed=$SEED total=$TOTAL run=$RUN_ID ==="

# ── Stage: code + D2 checkpoint + data (idempotent, with tokenizer patch) ──
echo "=== stage begin ==="
bash "$CODE_DIR/scripts/stage_n3_e8.sh" 2>&1 | tee -a "$RUN_DIR/stage.log"
STAGE_RC=${PIPESTATUS[0]}
if [ "$STAGE_RC" != "0" ] && [ ! -d "$CODE_DIR/checkpoints/self_distill_rebuilt_d2_epistemic_h200" ]; then
    echo "FAIL: stage script failed and checkpoint missing (rc=$STAGE_RC)"
    echo "FAILED" > "$RUN_DIR/FAILED"
    exit 1
fi
echo "=== stage done ==="

# ── HF resume: check if this run has prior checkpoint on HF ───────────────
echo "=== resume probe: hf://n3_runs/$RUN_ID/checkpoints/ ==="
python3 - "$RUN_ID" "$RUN_DIR" <<'PY'
import sys, os
from huggingface_hub import HfApi, snapshot_download
run_id, run_dir = sys.argv[1:3]
token = os.environ['HF_TOKEN']
api = HfApi(token=token)
try:
    files = list(api.list_repo_tree('iamseungpil/metacot', repo_type='dataset', path_in_repo=f'n3_runs/{run_id}/checkpoints'))
    if files:
        print(f'[resume] pulling {len(files)} checkpoint entries for {run_id}')
        os.makedirs(f'{run_dir}/checkpoints', exist_ok=True)
        snapshot_download('iamseungpil/metacot', repo_type='dataset', token=token,
                          allow_patterns=[f'n3_runs/{run_id}/checkpoints/**'],
                          local_dir='/scratch/meta/hf')
        # Link to expected location
        src = f'/scratch/meta/hf/n3_runs/{run_id}/checkpoints'
        if os.path.exists(src):
            import shutil
            for item in os.listdir(src):
                tgt = f'{run_dir}/checkpoints/{item}'
                if not os.path.exists(tgt):
                    os.symlink(f'{src}/{item}', tgt)
            print('[resume] checkpoint linked')
    else:
        print(f'[resume] no prior checkpoint for {run_id} — fresh start')
except Exception as e:
    print(f'[resume] probe failed (likely fresh run): {e}')
PY

# ── Start monitor daemon ──────────────────────────────────────────────────
echo "=== starting HF monitor daemon (poll=${HF_POLL}s) ==="
RUN_DIR="$RUN_DIR" POLL_SEC="$HF_POLL" HF_REPO="iamseungpil/metacot" HF_TOKEN="$HF_TOKEN" \
    bash "$CODE_DIR/scripts/n3_monitor_push.sh"

# ── Launch training ───────────────────────────────────────────────────────
echo "=== launching training (variant=$VARIANT seed=$SEED total=$TOTAL save_every=$SAVE_EVERY) ==="
cd "$CODE_DIR"

# Make sure tokenizer_config patch applied (re-run in case stage skipped)
TC=/scratch/meta/hf/models/self_distill_rebuilt_d2_epistemic_h200/tokenizer_config.json
if [ -f "$TC" ]; then
    python3 - "$TC" <<'PY'
import json, sys
p = sys.argv[1]
tc = json.load(open(p))
est = tc.get('extra_special_tokens')
if isinstance(est, list):
    tc['extra_special_tokens'] = {}
    json.dump(tc, open(p, 'w'), ensure_ascii=False, indent=2)
    print(f'PATCHED {p}')
else:
    print(f'OK extra_special_tokens={type(est).__name__}')
PY
fi

# Build config from full (not smoke) template
export HF_TOKEN
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-$(cat ~/.wandb_key 2>/dev/null || true)}}"
export WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
export WANDB_PROJECT="${WANDB_PROJECT:-metacot-meta-rlsd}"
export WANDB_RUN_NAME="$RUN_ID"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"
# Review iter1 Fix 2: trainer writes metrics.jsonl here (next to driver.log).
export META_RLSD_RUN_DIR="$RUN_DIR"

RUN_CFG="$RUN_DIR/run_config.yaml"
cp configs/contrastive_meta_rlsd.yaml "$RUN_CFG"
python3 - "$RUN_CFG" "$TOTAL" "$RUN_DIR" "$SEED" "$SAVE_EVERY" <<'PY'
import sys, yaml, pathlib
cfg_p, steps, run_dir, seed, save_every = sys.argv[1:]
cfg = yaml.safe_load(pathlib.Path(cfg_p).read_text())
cfg['total_steps'] = int(steps)
cfg['output_dir'] = f'{run_dir}/checkpoints'
cfg['seed'] = int(seed)
cfg['save_interval'] = int(save_every)
cfg['eval_interval'] = max(int(save_every), 10)
cfg['run_name'] = pathlib.Path(run_dir).name
pathlib.Path(cfg_p).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f'[cfg] total_steps={cfg["total_steps"]} save={cfg["save_interval"]} out={cfg["output_dir"]}')
PY

# Launch with accelerate + DeepSpeed ZeRO-3 (4-GPU)
accelerate launch \
    --config_file configs/accelerate_ds3.yaml \
    src/training/contrastive_meta_rlsd_trainer.py \
    --config "$RUN_CFG" \
    --variant "$VARIANT" \
    --seed "$SEED" \
    > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log"
TRAIN_RC=$?

echo "=== training exit rc=$TRAIN_RC ==="
if [ "$TRAIN_RC" = "0" ]; then
    echo "DONE" > "$RUN_DIR/DONE"
else
    echo "FAILED" > "$RUN_DIR/FAILED"
fi

# Final HF push — tail the training logs one more time
python3 - "$RUN_DIR" "$RUN_ID" <<'PY'
import sys, os
from huggingface_hub import HfApi
run_dir, run_id = sys.argv[1:3]
api = HfApi(token=os.environ['HF_TOKEN'])
for f in ['stdout.log', 'stderr.log', 'driver.log', 'metrics.jsonl', 'DONE', 'FAILED', 'run_config.yaml']:
    p = f'{run_dir}/{f}'
    if os.path.exists(p):
        try:
            api.upload_file(path_or_fileobj=p, path_in_repo=f'n3_runs/{run_id}/{f}',
                            repo_id='iamseungpil/metacot', repo_type='dataset')
            print(f'[final-push] {f}')
        except Exception as e:
            print(f'[final-push] {f}: {e}')
PY

echo "=== $(date) DRIVER v5 END rc=$TRAIN_RC ==="
exit $TRAIN_RC
