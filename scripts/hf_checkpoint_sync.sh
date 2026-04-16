#!/bin/bash
# Periodic HF checkpoint sync — run in background on training nodes.
# Polls frequently and uploads each NEW `global_step_*` checkpoint to HF
# under the path `checkpoints/{experiment}/latest`, REPLACING the previous
# upload (not accumulating step directories). Also overwrites
# `checkpoints/{experiment}/latest_checkpointed_iteration.txt` so a fresh
# rebuild can resume from the right step.
#
# veRL saves every `trainer.save_freq` steps (we use save_freq=10), so with a
# 60-second poll interval each 10-step checkpoint lands on HF within a minute
# of being written to disk.
#
# Usage: nohup bash scripts/hf_checkpoint_sync.sh &
set -euo pipefail

INTERVAL_SEC="${INTERVAL_SEC:-60}"  # poll every minute
HF_TOKEN="${HF_TOKEN:-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}"
REPO_ID="${REPO_ID:-iamseungpil/metacot}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/metacognition/checkpoints/metacot-math}"
PYTHON="${PYTHON:-/scratch/simplerl_venv/bin/python3}"
STATE_FILE="${STATE_FILE:-/scratch/hf_checkpoint_sync.state}"

echo "$(date) HF checkpoint sync started (replace-mode, $INTERVAL_SEC s poll)"
echo "  checkpoint_root: $CHECKPOINT_ROOT"
echo "  repo: $REPO_ID"
echo "  state:    $STATE_FILE"

touch "$STATE_FILE"

while true; do
    sleep "$INTERVAL_SEC"

    # For each experiment dir, find the newest global_step_N
    for EXP_DIR in "$CHECKPOINT_ROOT"/*/; do
        [ -d "$EXP_DIR" ] || continue
        EXPERIMENT=$(basename "$EXP_DIR")

        LATEST_LOCAL=$(find "$EXP_DIR" -maxdepth 1 -name "global_step_*" -type d 2>/dev/null \
            | sort -V | tail -1)
        [ -n "$LATEST_LOCAL" ] || continue

        STEP=$(basename "$LATEST_LOCAL")

        # Skip if we already uploaded this exact step for this experiment
        MARKER="${EXPERIMENT}:${STEP}"
        if grep -Fqx "$MARKER" "$STATE_FILE"; then
            continue
        fi

        REMOTE_PATH="checkpoints/${EXPERIMENT}/latest"
        ITER_FILE="${EXP_DIR}latest_checkpointed_iteration.txt"

        echo "$(date) Uploading $LATEST_LOCAL → $REPO_ID/$REMOTE_PATH (replaces previous)"

        $PYTHON - <<PYEOF 2>&1 || { echo "$(date) Upload failed"; continue; }
from huggingface_hub import HfApi
import os, sys

api = HfApi(token="$HF_TOKEN")
repo = "$REPO_ID"
folder = "$LATEST_LOCAL"
remote = "$REMOTE_PATH"
iter_file = "$ITER_FILE"
step = "$STEP"
experiment = "$EXPERIMENT"

# Delete existing "latest" folder on HF so we get a clean replace
try:
    api.delete_folder(
        path_in_repo=remote,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Delete stale {experiment}/latest before replace",
    )
    print(f"  deleted stale {remote}")
except Exception as e:
    # Folder may not exist on first upload — ignore
    print(f"  (no stale folder to delete: {type(e).__name__})")

# Upload new contents
api.upload_folder(
    folder_path=folder,
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=remote,
    commit_message=f"Sync {experiment} {step}",
)

# Also push the iteration marker
if os.path.exists(iter_file):
    api.upload_file(
        path_or_fileobj=iter_file,
        path_in_repo=f"checkpoints/{experiment}/latest_checkpointed_iteration.txt",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Update {experiment} iteration marker to {step}",
    )

print("UPLOAD_DONE")
PYEOF

        # Record that we uploaded this step
        echo "$MARKER" >> "$STATE_FILE"
    done
done
