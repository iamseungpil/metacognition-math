#!/bin/bash
# chain_submit — 파일 하나가 나타나면 큐 잡 하나를 제출한다(범용 의존성 연결).
#
# promote_smoke.sh 는 «잡 이름»을 지켜보는데, 산출물 파일을 기다려야 하는 경우가
# 따로 있다(예: 자리 캐기가 기증 메타 롤아웃의 texts.jsonl 을 필요로 한다).
# GPU 슬롯을 잡고 기다리면 카드가 논다 — 그래서 큐 **밖에서** 기다렸다가 제출한다.
#
# 사용법: chain_submit.sh <기다릴파일> <타임아웃초> <잡이름> <우선순위> <need_mb> <커맨드>
set -u
WAIT_FILE="$1"; TIMEOUT="$2"; NAME="$3"; PRIO="$4"; NEED="$5"; shift 5
CMD="$*"
cd "$(dirname "$0")/../.." || exit 1
LOG=/hdd_data/seungpil/scratch/logs/cd8_decisions.log
START=$(date +%s)
echo "[chain_submit] '${NAME}' 대기 시작: ${WAIT_FILE} ($(date -Is))"
while [ ! -s "$WAIT_FILE" ]; do
  if [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
    echo "[$(date '+%m-%d %H:%M')] chain_submit: '${NAME}' 타임아웃 — ${WAIT_FILE} 가 안 나타남, 제출 안 함" >> "$LOG"
    exit 1
  fi
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] chain_submit: ${WAIT_FILE} 확인 — '${NAME}' 제출" >> "$LOG"
/hdd_data/seungpil/envs/simplerl/bin/python scripts/local/gpu_queue.py submit \
  --name "$NAME" --priority "$PRIO" --need-mb "$NEED" --cmd "$CMD" >> "$LOG" 2>&1
