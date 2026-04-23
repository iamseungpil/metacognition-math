#!/bin/bash
# N3 monitor + incremental HF push daemon (plan §9.8).
#
# Runs in background, polls every POLL_SEC (default 1200s = 20 min),
# pushes new/changed artifacts from RUN_DIR to HF.
#
# Exits when:
#   - DONE marker file appears in RUN_DIR
#   - FAILED marker file appears in RUN_DIR
#   - MAX_RUNTIME exceeded (default 10h)
#
# Usage:
#   RUN_DIR=/scratch/meta/run/<run_id> bash scripts/n3_monitor_push.sh

set -u
RUN_DIR="${RUN_DIR:?RUN_DIR required}"
POLL_SEC="${POLL_SEC:-1200}"
MAX_RUNTIME="${MAX_RUNTIME:-36000}"   # 10h
HF_REPO="${HF_REPO:-iamseungpil/metacot}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required}"
RUN_ID="$(basename "$RUN_DIR")"
HF_SUBDIR="n3_runs/${RUN_ID}"
MON_LOG="${RUN_DIR}/monitor.log"

if [ "${MON_DAEMONIZED:-0}" != "1" ]; then
  mkdir -p "$RUN_DIR"
  MON_DAEMONIZED=1 RUN_DIR="$RUN_DIR" POLL_SEC="$POLL_SEC" \
      MAX_RUNTIME="$MAX_RUNTIME" HF_REPO="$HF_REPO" HF_TOKEN="$HF_TOKEN" \
      nohup setsid bash "$0" > "$MON_LOG" 2>&1 </dev/null &
  disown
  echo "N3_MONITOR_STARTED pid=$! run_id=$RUN_ID log=$MON_LOG"
  exit 0
fi

echo "=== $(date) N3 monitor START run=$RUN_ID poll=${POLL_SEC}s ==="
START_TS=$(date +%s)

push_artifacts() {
  echo "--- $(date) pushing to HF $HF_REPO:$HF_SUBDIR ---"
  HF_TOKEN="$HF_TOKEN" python3 - "$RUN_DIR" "$HF_REPO" "$HF_SUBDIR" <<'PY'
import os, sys, pathlib
from huggingface_hub import upload_file, HfApi

run_dir_s, repo, subdir = sys.argv[1:4]
run_dir = pathlib.Path(run_dir_s)
tok = os.environ["HF_TOKEN"]

# Tracked files: anything small enough to upload individually.
patterns = [
    "launch.log", "stdout.log", "stderr.log", "monitor.log",
    "smoke_config.yaml", "smoke_acceptance.json", "final_metrics.json",
    "metrics.jsonl", "DONE", "FAILED",
]
pushed = 0
for pat in patterns:
    for f in run_dir.glob(pat):
        if not f.is_file():
            continue
        # Skip if empty (not yet written by child).
        if f.stat().st_size == 0:
            continue
        try:
            upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"{subdir}/{f.name}",
                repo_id=repo,
                repo_type="dataset",
                token=tok,
                commit_message=f"N3 monitor: {f.name} @ {f.stat().st_mtime:.0f}",
            )
            print(f"  pushed {f.name} ({f.stat().st_size} B)")
            pushed += 1
        except Exception as e:
            print(f"  FAIL {f.name}: {e!r}")

# Also push the most recent checkpoint dir if present (lightweight: only
# config.json + adapter_model.* — big files are blob, limit to <500MB total).
ckpt_root = run_dir / "checkpoints"
if ckpt_root.is_dir():
    ckpts = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        latest = ckpts[-1]
        for f in latest.rglob("*"):
            if f.is_file() and f.stat().st_size < 50 * 1024 * 1024:  # ≤50MB
                try:
                    rel = f.relative_to(run_dir)
                    upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=f"{subdir}/{rel}",
                        repo_id=repo,
                        repo_type="dataset",
                        token=tok,
                        commit_message=f"N3 monitor: ckpt {f.name}",
                    )
                    pushed += 1
                except Exception as e:
                    print(f"  FAIL {f.name}: {e!r}")

print(f"total pushed: {pushed}")
PY
}

gpu_status() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
      | tr '\n' ' '
  echo
}

while true; do
  ELAPSED=$(( $(date +%s) - START_TS ))
  if [ "$ELAPSED" -gt "$MAX_RUNTIME" ]; then
    echo "=== $(date) monitor MAX_RUNTIME exceeded ($ELAPSED s) ==="
    push_artifacts || true
    break
  fi

  if [ -f "$RUN_DIR/DONE" ]; then
    echo "=== $(date) DONE marker detected ==="
    push_artifacts || true
    break
  fi

  if [ -f "$RUN_DIR/FAILED" ]; then
    echo "=== $(date) FAILED marker detected ==="
    push_artifacts || true
    break
  fi

  echo "--- $(date) tick (elapsed=${ELAPSED}s) ---"
  gpu_status
  push_artifacts || echo "push FAILED (continuing)"

  sleep "$POLL_SEC"
done

echo "=== $(date) monitor EXIT ==="
