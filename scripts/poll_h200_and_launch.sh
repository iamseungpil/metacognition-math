#!/usr/bin/env bash
# Poll the 6 watched H200 jobs every ~5 min.
# When any flips to Running, SSH in, run bootstrap + SDC launch (detached).
#
# Uses amlt status to detect RUNNING. Records state transitions to a log.
# Idempotent: will not re-launch on a job it already launched (marker file).

set -uo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set for remote code staging}"

# amlt reads .amltconfig from CWD and parents; ~/.amltconfig is the root one.
# Ensure we're in $HOME so amlt can resolve the 'skilldiscovery2' project.
cd "$HOME" || exit 1

LOG_DIR="/home/v-seungplee/metacognition/logs"
mkdir -p "$LOG_DIR"
POLL_LOG="$LOG_DIR/poll_h200.log"
LAUNCHED_DIR="$LOG_DIR/launched_markers"
mkdir -p "$LAUNCHED_DIR"

# Watched jobs: EXP:JOB pairs — 2026-04-22 6-node auto-yaml
WATCH=(
  "metacot-h200-6node-auto-0422:sdc_n1"
  "metacot-h200-6node-auto-0422:sdc_n2"
  "metacot-h200-6node-auto-0422:sdc_n3"
  "metacot-h200-6node-auto-0422:sdc_n4"
  "metacot-h200-6node-auto-0422:sdc_n5"
  "metacot-h200-6node-auto-0422:sdc_n6"
)
# Self-launching jobs (auto-launch in yaml): skip SSH, just log status
SELF_LAUNCH_PATTERN="^metacot-h200-6node-auto-"

ts() { date +'%Y-%m-%d %H:%M:%S'; }

check_job() {
    # $1 = exp:job
    local pair="$1"
    local exp="${pair%%:*}"
    local job="${pair##*:}"
    local out
    out=$(amlt status "$exp" 2>/dev/null | awk -v j=":$job" '$0 ~ j {print}' | head -1)
    local status
    status=$(echo "$out" | awk '{for(i=1;i<=NF;i++) if($i ~ /^(running|queued|killed|failed|preparing|pass|paused)$/) {print $i; exit}}')
    echo "$status"
}

launch_if_running() {
    local pair="$1"
    local status="$2"
    local marker="$LAUNCHED_DIR/${pair//:/__}"
    if [ "$status" = "running" ] && [ ! -f "$marker" ]; then
        # Self-launching pairs: skip SSH, just record marker for dedup
        if [[ "$pair" =~ $SELF_LAUNCH_PATTERN ]]; then
            echo "[$(ts)] DETECTED RUNNING (self-launch yaml): $pair" | tee -a "$POLL_LOG"
            touch "$marker"
            return 0
        fi
        echo "[$(ts)] DETECTED RUNNING: $pair — launching bootstrap+SDC via SSH..." | tee -a "$POLL_LOG"
        touch "$marker"
        local exp="${pair%%:*}"
        local job="${pair##*:}"
        # Compose the bootstrap+launch command for the remote node.
        # Use a tmux session so SSH disconnect doesn't kill it.
        # Bootstrap-and-launch — base64-encoded to avoid shell quoting hell.
        local cmdfile="$LOG_DIR/remote_cmd_${pair//:/__}.sh"
        cat > "$cmdfile" <<REMOTE_EOF
set -e
export HF_TOKEN="${HF_TOKEN}"
mkdir -p /scratch/logs
if [ ! -f /scratch/metacognition/scripts/run_sdc_on_h200_node.sh ]; then
  [ -e /tmp/amlt-env ] && source /tmp/amlt-env || true
  pip install -q huggingface_hub 2>/dev/null || true
  python -c "import os,tarfile; from huggingface_hub import hf_hub_download; p=hf_hub_download(repo_id='iamseungpil/metacot',filename='code_snapshots/metacognition.tar.gz',repo_type='dataset',token=os.environ['HF_TOKEN']); t=tarfile.open(p); t.extractall('/scratch'); t.close()"
fi
# Run orchestrator detached; it spawns its own tmux sessions ('gpukeeper','boot','sdc_train')
nohup bash /scratch/metacognition/scripts/run_sdc_on_h200_node.sh >/scratch/logs/sdc_outer.log 2>&1 </dev/null &
disown
sleep 2
echo "orchestrator launched, pid_pattern:"
pgrep -f run_sdc_on_h200_node.sh | head -3
tmux list-sessions 2>/dev/null | head -5
echo LAUNCHED
REMOTE_EOF
        local b64; b64=$(base64 -w0 "$cmdfile")
        local remote_invoke="echo $b64 | base64 -d | bash"
        ( amlt ssh "$exp" ":$job" -c "$remote_invoke" 2>&1 | tee -a "$LOG_DIR/ssh_${pair//:/__}.log" ) &
        echo "[$(ts)] spawned SSH+launch for $pair (PID $!)" | tee -a "$POLL_LOG"
    fi
}

echo "[$(ts)] poll loop started; watching ${#WATCH[@]} jobs" | tee -a "$POLL_LOG"

while true; do
    line="[$(ts)]"
    any_running=false
    for pair in "${WATCH[@]}"; do
        s=$(check_job "$pair")
        s="${s:-unknown}"
        line="$line $pair=$s"
        if [ "$s" = "running" ]; then any_running=true; fi
        launch_if_running "$pair" "$s"
    done
    echo "$line" >> "$POLL_LOG"
    # 5 minutes
    sleep 300
done
