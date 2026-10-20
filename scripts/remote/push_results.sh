#!/bin/bash
# push_results.sh RUN_NAME [--weights] [--failed] [--dry-run]
#   끝난 런 하나를 비공개 HF 데이터셋 iamseungpil/metacot-cd9-remote 의 runs/<RUN_NAME>/ 로 올린다(없으면 private 으로 만든다).
#   올리는 것: 평가 폴더마다 summary.json·meta.json·texts.jsonl.gz · 체크포인트의 RUN_ENV.txt·TRAIN_DONE·pfx_rollouts.jsonl.gz(있으면) ·
#   학습 로그(gz)·큐 로그(gz) · DONE (--failed 면 FAILED + 로그만). --weights = 병합된 최종 모델도(수 GB).
#   어느 폴더·계보를 올릴지는 $ROOT/scratch/remote_meta/<RUN_NAME>.env (a78c 복사본 · r1_eval.sh · q17_prep.sh 가 쓴다)가 정한다.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/hf_env.sh"
exec "$PY" -I "$R_DIR/hf_sync.py" push "$@"
