#!/bin/bash
# Relentless retry daemon for the s3b format-SFT job under basicvc H100
# capacity starvation + workspace TooManyRequests (queue >15000).
# Strategy: keep submitting fresh-named jobs; on TooManyRequests back off,
# on instant-fail resubmit fast, stop as soon as one job reaches `running`.
cd /home/v-seungplee/metacognition-math
set -a; source .env 2>/dev/null; set +a
AMLT=/home/v-seungplee/miniconda3/envs/amlt/bin/amlt
YAML=${S3B_DAEMON_YAML:-h100std_s3b_chain.yaml}
NAME_PREFIX=${S3B_DAEMON_PREFIX:-s3b-chain-a}
LOG=/tmp/s3b_retry_daemon.log
: > "$LOG"
echo "=== s3b retry daemon start $(date -u +%H:%M:%S) ===" >> "$LOG"

poll_status () {  # $1=name -> echoes running|completed|failed|preparing|queued|unknown
  $AMLT status "$1" 2>/dev/null | grep -oiE "running|completed|failed|preparing|queued" | head -1 | tr 'A-Z' 'a-z'
}

watch_for_run () {  # $1=name, poll up to ~8min; return 0 if running/completed, 1 if failed/timeout
  local name=$1 st
  for p in 1 2 3 4; do
    sleep 120
    st=$(poll_status "$name")
    echo "[$(date -u +%H:%M:%S)] poll $name -> ${st:-none}" >> "$LOG"
    if [ "$st" = "running" ] || [ "$st" = "completed" ]; then return 0; fi
    if [ "$st" = "failed" ]; then return 1; fi
  done
  return 1
}

# Relentless submit loop (v7 preempted + weights lost; the fixed yaml now bakes a
# durable dataset-models upload, so any job that reaches running is self-sufficient).
i=0
MAX=80
while [ $i -lt $MAX ]; do
  i=$((i+1))
  name="${NAME_PREFIX}$i"
  out=$($AMLT run "$YAML" "$name" -y -d "auto-retry $i (capacity starvation)" 2>&1)
  if echo "$out" | grep -q "TooManyRequests"; then
    echo "[$(date -u +%H:%M:%S)] $name REJECTED TooManyRequests -> backoff 240s" >> "$LOG"
    sleep 240
    continue
  fi
  if ! echo "$out" | grep -qiE "preparing|Created new experiment"; then
    echo "[$(date -u +%H:%M:%S)] $name submit-error:" >> "$LOG"
    echo "$out" | tail -4 >> "$LOG"
    sleep 120
    continue
  fi
  echo "[$(date -u +%H:%M:%S)] $name submitted (preparing), watching for node" >> "$LOG"
  if watch_for_run "$name"; then
    echo "=== SUCCESS $name reached running $(date -u +%H:%M:%S) ===" >> "$LOG"
    exit 0
  fi
  echo "[$(date -u +%H:%M:%S)] $name did not get a node (instant-fail/timeout), resubmitting" >> "$LOG"
  sleep 30
done
echo "=== daemon exhausted $MAX attempts without a node $(date -u +%H:%M:%S) ===" >> "$LOG"
exit 2
