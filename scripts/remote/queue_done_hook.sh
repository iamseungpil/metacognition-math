#!/bin/bash
# 큐 완료 훅: QUEUE_DONE_HOOK=$PWD/scripts/remote/queue_done_hook.sh python3 scripts/local/gpu_queue.py start-workers 0 1 2 3
#   gpu_queue 가 잡이 끝날 때 `훅 <잡이름> <done|failed|aborted> <rc>` 를 부른다. 잡 이름 = RUN_NAME.
NAME=$1; STATE=$2
case "$STATE" in
  done)   bash "$(dirname "${BASH_SOURCE[0]}")/push_results.sh" "$NAME" ${PUSH_WEIGHTS:+--weights} ;;
  failed) bash "$(dirname "${BASH_SOURCE[0]}")/push_results.sh" "$NAME" --failed ;;
  *)      echo "[hook] $NAME $STATE — 올리지 않음" ;;
esac
