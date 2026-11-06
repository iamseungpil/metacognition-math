#!/bin/bash
# get_data.sh [--no-models] [--no-prefetch]   원격 서버 준비: 데이터(비공개 HF iamseungpil/metacot-cd9-data → $ROOT/scratch/{data,eval}) ·
#   기본 모델(Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-1.7B → $ROOT/scratch/models/) · 시험 데이터셋 HF 캐시 선받기(a79 는 오프라인으로 읽는다).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/hf_env.sh"
exec "$PY" -I "$R_DIR/hf_sync.py" get-data "$@"
