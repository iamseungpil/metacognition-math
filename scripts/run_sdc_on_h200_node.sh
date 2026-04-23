#!/bin/bash
# Runs on the remote H200 node. Orchestrates:
#   A. Immediately start GPU keeper (prevents BSC idle-suspend) in tmux 'gpukeeper'
#   B. Run bootstrap (installs verl 0.7.1 + vllm 0.6.3 + ray 2.10 + torch 2.4) in tmux 'boot'
#   C. When bootstrap completes, stop gpukeeper + launch SDC in tmux 'sdc' with retry loop
#   D. Keep-alive heartbeat daemon (every 5min)
#   E. HF ckpt push daemon (handled inside launch_sdc_verl.sh)

set -uo pipefail

# Environment
[ -e /tmp/amlt-env ] && source /tmp/amlt-env || true
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
export WANDB_KEY="$WANDB_API_KEY"
export WANDB_PROJECT="${WANDB_PROJECT:-skilldiscovery2}"
export WANDB_NAME="${WANDB_NAME:-sdc_h200_$(hostname)_$(date +%m%d_%H%M)}"
export FORCE_SYNC="${FORCE_SYNC:-0}"
export CONFIG="${CONFIG:-verl_sdc_e21r_shared_h200_4x16k}"

SCRATCH="/scratch"
LOG_DIR="$SCRATCH/logs"
mkdir -p "$LOG_DIR"

echo "[$(date)] === outer launcher start ==="

# === A. GPU KEEPER === (pre-installed ptca conda env has torch)
echo "[$(date)] [A] starting gpu_keeper tmux session"
rm -f "$SCRATCH/gpu_keeper.stop"
# Prefer /scratch/metacognition/scripts/gpu_keeper.py; if missing, pull from HF
KEEPER="$SCRATCH/metacognition/scripts/gpu_keeper.py"
if [ ! -f "$KEEPER" ]; then
    KEEPER="$SCRATCH/amlt_code/scripts/gpu_keeper.py"
fi
if [ ! -f "$KEEPER" ]; then
    # Last resort: inline
    cat > "$SCRATCH/gpu_keeper.py" <<'PYEOF'
import os, time, torch
STOP = "/scratch/gpu_keeper.stop"
try: os.remove(STOP)
except FileNotFoundError: pass
n=torch.cuda.device_count()
print(f"[gpu_keeper] n_gpus={n}", flush=True)
tensors=[torch.randn(4096,4096,device=f"cuda:{g}",dtype=torch.float16) for g in range(n)]
i=0
while not os.path.exists(STOP):
    for g,t in enumerate(tensors): tensors[g]=(t@t)*0.5+torch.randn_like(t)*0.01
    if i%30==0: print(f"[gpu_keeper] iter={i}", flush=True)
    i+=1; time.sleep(1)
print("[gpu_keeper] stop seen", flush=True)
PYEOF
    KEEPER="$SCRATCH/gpu_keeper.py"
fi
tmux kill-session -t gpukeeper 2>/dev/null || true
# Use default python (ptca has torch preinstalled)
tmux new-session -d -s gpukeeper "python -u $KEEPER 2>&1 | tee $LOG_DIR/gpu_keeper.log"
sleep 3
echo "[$(date)] [A] gpukeeper tmux started — log: $LOG_DIR/gpu_keeper.log"
tmux list-sessions | tee -a "$LOG_DIR/sessions.log"

# === B. BOOTSTRAP === (pull code snapshot if needed, then install)
echo "[$(date)] [B] starting bootstrap tmux session"
# Make sure /scratch/metacognition has the latest scripts
if [ ! -f "$SCRATCH/metacognition/scripts/bootstrap_sdc_node.sh" ]; then
    echo "[$(date)] [B] pulling code snapshot from HF"
    pip install -q huggingface_hub 2>/dev/null || true
    python - <<PY
from huggingface_hub import hf_hub_download
import os, tarfile
p = hf_hub_download(
    repo_id="iamseungpil/metacot",
    filename="code_snapshots/metacognition.tar.gz",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
)
with tarfile.open(p) as t: t.extractall("/scratch")
print("extracted to /scratch/metacognition")
PY
fi

tmux kill-session -t boot 2>/dev/null || true
tmux new-session -d -s boot "bash $SCRATCH/metacognition/scripts/bootstrap_sdc_node.sh 2>&1 | tee -a $LOG_DIR/bootstrap.log; touch $SCRATCH/bootstrap.done"
echo "[$(date)] [B] bootstrap tmux started — waiting for $SCRATCH/bootstrap.done"

# === D. Keep-alive heartbeat (outside tmux, nohup'd) ===
if ! pgrep -f sdc_keepalive_beacon >/dev/null 2>&1; then
    nohup bash -c '
        while true; do
            echo "[heartbeat $(date)]" >> /scratch/logs/keepalive.log
            nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -4 >> /scratch/logs/keepalive.log
            : sdc_keepalive_beacon
            sleep 180
        done
    ' >/dev/null 2>&1 &
    echo "[$(date)] [D] keep-alive beacon started PID=$!"
fi

# === C. WAIT for bootstrap, then launch SDC with retry loop ===
# Wait up to 60 min for bootstrap; if done_marker already present (idempotent), skip
echo "[$(date)] [C] waiting for bootstrap.done"
for i in $(seq 1 360); do  # 360 × 10s = 60 min
    if [ -f "$SCRATCH/bootstrap.done" ] || [ -f "$SCRATCH/simplerl.done" ]; then
        break
    fi
    sleep 10
done

if [ ! -f "$SCRATCH/simplerl.done" ] && [ ! -f "$SCRATCH/bootstrap.done" ]; then
    echo "[$(date)] [C] bootstrap did NOT finish in 60min — aborting SDC launch"
    exit 1
fi
echo "[$(date)] [C] bootstrap complete"

# NOTE: gpu_keeper auto-detects bootstrap.done and switches to LOW mode (60s sleep,
# ~3% util). Leave it running as idle-suspend watchdog through SDC initialization.
# Only stop it if training actually reaches rollout phase (not needed for now).
echo "[$(date)] [C] gpu_keeper stays in LOW mode (auto-downshifted via bootstrap.done marker)"

# Source env set by bootstrap
[ -f "$SCRATCH/env.sh" ] && source "$SCRATCH/env.sh"

cd "$SCRATCH/metacognition"
echo "[$(date)] [C] === SDC launch loop (CONFIG=$CONFIG) ==="

MAX_RETRIES="${MAX_RETRIES:-10}"
for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo "[$(date)] attempt $attempt/$MAX_RETRIES"
    if DETACH=1 bash scripts/launch_sdc_verl.sh 2>&1 | tee -a "$LOG_DIR/launch.log"; then
        PID=$(cat "$SCRATCH/sdc_direct.pid" 2>/dev/null || echo "")
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "[$(date)] training PID=$PID — waiting..."
            while kill -0 "$PID" 2>/dev/null; do sleep 60; done
            echo "[$(date)] training PID=$PID exited"
        fi
    else
        echo "[$(date)] attempt $attempt: launch script failed"
    fi
    echo "[$(date)] backoff 60s..."
    sleep 60
done

echo "[$(date)] === all retries exhausted ==="
