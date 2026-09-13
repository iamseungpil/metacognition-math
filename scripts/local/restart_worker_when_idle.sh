#!/bin/bash
# ★0914: 워커 프로세스는 시작 시점의 gpu_queue.py 코드를 물고 돈다(pid 기록·aborted 처리 같은
#   수리가 반영되지 않음). 도는 잡을 죽이지 않으려면 «그 GPU 에 running 잡이 없을 때» 옛 워커를
#   내리고 새 워커를 띄워야 한다. 사용법: restart_worker_when_idle.sh <gpu>
set -u
G="$1"; Q=/hdd_data/seungpil/queue; PY=/hdd_data/seungpil/envs/simplerl/bin/python
cd "$(dirname "$0")/../.."
LOG=/hdd_data/seungpil/scratch/logs/cd8_decisions.log
busy() { for f in "$Q"/running/*.json; do [ -f "$f" ] || continue; $PY - "$f" "$G" <<'PY' && return 0
import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if str(j.get("assigned_gpu"))==sys.argv[2] else 1)
PY
done; return 1; }
while busy; do sleep 60; done
OLD=$(pgrep -f "gpu_queue.py worker --gpu $G ")
touch "$Q/STOP"                       # 새 잡을 집지 못하게 잠깐 멈춤
sleep 5
[ -n "$OLD" ] && kill -KILL $OLD
rm -f "$Q/gpu_${G}.lock"
nohup $PY scripts/local/gpu_queue.py worker --gpu "$G" --jobs-per-gpu 1 \
  >> /hdd_data/seungpil/scratch/logs/worker_gpu${G}_slot0.log 2>&1 &
NEW=$!
rm -f "$Q/STOP"
echo "[$(date '+%m-%d %H:%M')] restart_worker: gpu$G 옛 워커($OLD) → 새 워커($NEW), 새 코드(pid 기록·aborted)" >> "$LOG"
