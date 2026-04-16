#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="/home/v-seungplee/.ssh/id_rsa"

resolve_proxy() {
  local display_name="$1"
  PYTHONPATH=. python scripts/get_active_proxy_endpoint.py --display-name "$display_name"
}

URL_EVAL="${URL_EVAL:-$(resolve_proxy metacognition_eval)}"
URL_TRAIN_B="${URL_TRAIN_B:-$(resolve_proxy metacognition_train_b)}"

LOCAL_META_DIR="$ROOT/checkpoints/v8_meta_inside_strict_sft"
LOCAL_BASE_DIR="$ROOT/checkpoints/v8_base_matched_strict_sft"
REMOTE_META_DIR="/scratch/metacognition/checkpoints/v8_meta_inside_strict_sft"
REMOTE_BASE_DIR="/scratch/metacognition/checkpoints/v8_base_matched_strict_sft"

proxy_ssh() {
  local url="$1"
  shift
  ssh -T -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" azureuser@placeholder "$@"
}

copy_from_remote() {
  local url="$1"
  local src="$2"
  local dst="$3"
  scp -r -q -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
      -i "$SSH_KEY" "azureuser@placeholder:$src" "$dst"
}

copy_model_root_from_remote() {
  local url="$1"
  local src="$2"
  local dst="$3"
  rm -rf "$dst"
  mkdir -p "$dst"
  local rel
  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    local tries=0
    while true; do
      tries=$((tries + 1))
      if scp -q -o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null \
          -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $url" \
          -i "$SSH_KEY" "azureuser@placeholder:$src/$rel" "$dst/$rel"; then
        break
      fi
      if [[ "$tries" -ge 3 ]]; then
        echo "[error] failed to copy $src/$rel after $tries attempts" >&2
        return 1
      fi
      echo "[retry] $rel ($tries)" >&2
      sleep 2
    done
  done < <(proxy_ssh "$url" "cd '$src' && find . -maxdepth 1 -type f | sed 's#^./##' | sort")
}

ensure_remote_ready() {
  local url="$1"
  local path="$2"
  local label="$3"
  proxy_ssh "$url" "test -f '$path/tokenizer_config.json' && echo '[ready] $label'" >/dev/null
}

echo "[check] remote strict checkpoints"
ensure_remote_ready "$URL_EVAL" "$REMOTE_META_DIR" "strict meta sft"
ensure_remote_ready "$URL_TRAIN_B" "$REMOTE_BASE_DIR" "strict base sft"

echo "[sync] meta checkpoint"
copy_model_root_from_remote "$URL_EVAL" "$REMOTE_META_DIR" "$LOCAL_META_DIR"

echo "[sync] base checkpoint"
copy_model_root_from_remote "$URL_TRAIN_B" "$REMOTE_BASE_DIR" "$LOCAL_BASE_DIR"

test -f "$LOCAL_META_DIR/tokenizer_config.json"
test -f "$LOCAL_BASE_DIR/tokenizer_config.json"

echo "[run] post-SFT bundle"
python scripts/run_post_sft_bundle.py "$@"
