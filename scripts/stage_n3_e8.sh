#!/bin/bash
# Stage code + D2 student checkpoint + train data on E8 /scratch (plan §9.5).
#
# Idempotent — re-runs safely. Pulls only missing assets.
#
# Usage (on E8):
#   HF_TOKEN=... bash scripts/stage_n3_e8.sh

set -euo pipefail

HF_TOKEN="${HF_TOKEN:?HF_TOKEN required}"
STAGE_ROOT="${STAGE_ROOT:-/scratch/meta}"
HF_LOCAL="${STAGE_ROOT}/hf"
CODE_DIR="${STAGE_ROOT}/code"
CKPT_NAME="v8_meta_inside_strict_sft"

mkdir -p "$HF_LOCAL" "$CODE_DIR" "${STAGE_ROOT}/run"

echo "=== $(date) stage: code snapshot ==="
if [ ! -d "$CODE_DIR/src" ]; then
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="iamseungpil/metacot", repo_type="dataset",
                  local_dir="${HF_LOCAL}", token="${HF_TOKEN}",
                  allow_patterns=["code_snapshot/metacot_code.tar.gz"])
PY
  tar xzf "${HF_LOCAL}/code_snapshot/metacot_code.tar.gz" -C "$CODE_DIR"
  echo "extracted to $CODE_DIR"
else
  echo "code already present at $CODE_DIR"
fi

echo "=== $(date) stage: D2 checkpoint ==="
CKPT_HF_PATH="${HF_LOCAL}/models/${CKPT_NAME}"
CKPT_CODE_PATH="${CODE_DIR}/checkpoints/${CKPT_NAME}"

if [ ! -d "$CKPT_HF_PATH" ]; then
  python3 - <<PY
from huggingface_hub import snapshot_download
# Skip checkpoint-* subdirs (mid-training intermediate ckpts, we only need final shard set).
snapshot_download(repo_id="iamseungpil/metacot", repo_type="dataset",
                  local_dir="${HF_LOCAL}", token="${HF_TOKEN}",
                  allow_patterns=["models/${CKPT_NAME}/*"],
                  ignore_patterns=["models/${CKPT_NAME}/checkpoint-*/**",
                                   "models/${CKPT_NAME}/**/training_args.bin",
                                   "models/${CKPT_NAME}/**/optimizer.pt"])
PY
fi

mkdir -p "${CODE_DIR}/checkpoints"
if [ ! -e "$CKPT_CODE_PATH" ]; then
  ln -s "$CKPT_HF_PATH" "$CKPT_CODE_PATH"
  echo "linked: $CKPT_CODE_PATH → $CKPT_HF_PATH"
fi

# Patch tokenizer_config.json — transformers 4.5x/5.x compatibility
# (extra_special_tokens must be dict not list; easiest: remove entirely — other fields carry it).
TC_PATH="${CKPT_HF_PATH}/tokenizer_config.json"
if [ -f "$TC_PATH" ]; then
  python3 - "$TC_PATH" <<'PY'
import json, sys
p = sys.argv[1]
tc = json.load(open(p))
changed = False
est = tc.get('extra_special_tokens')
if isinstance(est, list):
    tc['extra_special_tokens'] = {}
    changed = True
if changed:
    json.dump(tc, open(p, 'w'), ensure_ascii=False, indent=2)
    print(f'PATCHED {p}: extra_special_tokens list -> dict')
else:
    print(f'OK {p}: extra_special_tokens type={type(est).__name__}')
PY
fi

echo "=== $(date) stage: train data ==="
DATA_HF="${HF_LOCAL}/data"
DATA_CODE="${CODE_DIR}/data"
mkdir -p "$DATA_CODE"

# Data parquets are now embedded in the code snapshot tarball (data/verl_*.parquet).
# They are extracted alongside src/ and scripts/ into $CODE_DIR by the code-snapshot step.
# Verify they exist at the expected paths.
for name in verl_train_redirect.parquet verl_val_redirect.parquet verl_train_redirect_base.parquet verl_val_redirect_base.parquet; do
  if [ -f "${DATA_CODE}/$name" ]; then
    echo "data ok: ${DATA_CODE}/$name"
  else
    echo "WARN: missing ${DATA_CODE}/$name (not in code snapshot)"
  fi
done

echo "=== $(date) stage: summary ==="
du -sh "$HF_LOCAL" 2>/dev/null || true
echo "code: $CODE_DIR"
echo "ckpt: $CKPT_CODE_PATH"
echo "data: ${DATA_CODE}/verl_train_redirect.parquet"
echo "=== $(date) stage DONE ==="
