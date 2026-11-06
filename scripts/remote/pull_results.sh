#!/bin/bash
# pull_results.sh [--list] [--force] [RUN_NAME ...]   (이 서버에서) HF 의 runs/*/ 중 DONE 이 있고 아직 안 받은 것을 받는다.
#   평가 폴더 → $ROOT/scratch/eval/<같은 이름>/ (summary.json·meta.json·texts.jsonl — 압축 풀어서; 이미 있는 이름은 덮지 않고 경고)
#   나머지 → $ROOT/scratch/remote_runs/<RUN_NAME>/ (.pulled 표시로 중복 받기 방지)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/hf_env.sh"
exec "$PY" -I "$R_DIR/hf_sync.py" pull "$@"
