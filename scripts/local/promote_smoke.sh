#!/bin/bash
# ★0911: 스모크(짧은 스텝)가 무사히 끝나면 사람이 다시 확인할 때까지 기다리지 않고
#   그 자리에서 본실험(100스텝)으로 자동 승격한다. 사용자 지시(0911): "트리거를 걸어서
#   끝나면 바로 실행". run_arm.sh 는 resume_mode=auto 라 같은 계보(같은 ARM/SEED/VARIANT)
#   로 다시 제출하면 스모크가 남긴 체크포인트에서 그대로 이어간다 — 처음부터 다시 안 돈다.
#
# 사용법: promote_smoke.sh <스모크잡이름> <ARM> <SEED> <VARIANT> <본실험잡이름> <envvar=val...>
# 예: promote_smoke.sh v10_FIXEDA_CHK_smoke20_s1 FIXEDA_CHK 1 chk w11_FIXEDA_CHK_full_s1 \
#       CHK_REGION=1 PAGED=1 REF_OFFLOAD=1 ACTOR_OFFLOAD=1 SLIM=1 VLLM_UTIL=0.15 CHK_AMP=1.5
set -u
SMOKE_NAME="$1"; ARM="$2"; SEED="$3"; VARIANT="$4"; FULL_NAME="$5"; shift 5
ENV_PREFIX="$*"
cd "$(dirname "$0")/../.." || exit 1
LOG=/hdd_data/seungpil/scratch/logs/cd8_decisions.log
echo "[promote_smoke] watching '${SMOKE_NAME}' -> on clean finish, submit '${FULL_NAME}' ($(date -Is))"
while true; do
  # 큐의 running/pending 어디에도 이 이름이 없으면(=done 또는 failed 로 옮겨감) 끝난 것이다.
  if ! grep -rl "\"name\": \"${SMOKE_NAME}\"" /hdd_data/seungpil/queue/running /hdd_data/seungpil/queue/pending >/dev/null 2>&1; then
    break
  fi
  sleep 60
done
DONE_FILE=$(grep -rl "\"name\": \"${SMOKE_NAME}\"" /hdd_data/seungpil/queue/done 2>/dev/null | head -1)
FAIL_FILE=$(grep -rl "\"name\": \"${SMOKE_NAME}\"" /hdd_data/seungpil/queue/failed 2>/dev/null | head -1)
if [ -n "$FAIL_FILE" ] || [ -z "$DONE_FILE" ]; then
  echo "[$(date '+%m-%d %H:%M')] promote_smoke: '${SMOKE_NAME}' 실패로 끝남(또는 못 찾음) — '${FULL_NAME}' 자동 승격 안 함, 사람 확인 필요" >> "$LOG"
  exit 1
fi
echo "[$(date '+%m-%d %H:%M')] promote_smoke: '${SMOKE_NAME}' 정상 종료 확인 — '${FULL_NAME}' 100스텝 본실험 제출(같은 계보라 체크포인트에서 이어감)" >> "$LOG"
source scripts/local/env.sh >/dev/null 2>&1
# ENV_PREFIX 는 제출자(이 스크립트)가 아니라 --cmd 문자열 **안에** 그대로 박혀야 한다 —
# 나중에 워커가 그 문자열을 새 셸에서 실행할 때 비로소 적용돼야 학습에 실제로 걸린다.
/hdd_data/seungpil/envs/simplerl/bin/python scripts/local/gpu_queue.py submit \
  --name "${FULL_NAME}" --priority 90 --need-mb 46000 \
  --cmd "${ENV_PREFIX} bash scripts/local/run_arm_retry.sh 60 300 -- ${ARM} ${SEED} 100 ${VARIANT}" \
  >> "$LOG" 2>&1
