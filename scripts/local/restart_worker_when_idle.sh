#!/bin/bash
# ★0914: 워커 프로세스는 시작 시점의 gpu_queue.py 코드를 물고 돈다(pid 기록·aborted 처리 같은
#   수리가 반영되지 않음). 도는 잡을 죽이지 않으려면 «그 GPU 에 running 잡이 없을 때» 옛 워커를
#   내리고 새 워커를 띄워야 한다. 사용법: restart_worker_when_idle.sh <gpu>
#
# ★감사 8(0914) 경쟁 수리 — 순서가 중요하다:
#   1. «비었다» 확인과 kill 사이에 옛 워커가 새 잡을 집는 창을 STOP 으로 닫는다. 단 STOP 은 네 워커
#      모두에게 전역이라 몇 시간짜리 대기 내내 들고 있으면 GPU 0/1 도 새 잡을 못 집는다. 그래서:
#      STOP 없이 비기를 기다림 → touch STOP → 워커 poll(5s) 보다 긴 정착 대기 → busy 재확인
#      (그 사이 잡이 붙었으면 STOP 을 풀고 처음부터) → 확정. STOP 이 있는 동안은 어떤 워커도
#      새 잡을 못 집으므로 재확인 통과 뒤에는 kill 까지 창이 없다.
#   2. (위 루프가 그 GPU 에 running 잡이 없음을 STOP 아래에서 확정한다.)
#   3. 옛 워커에 TERM → 종료 대기(최대 RESTART_TERM_WAIT_S, 기본 60s) → 안 죽으면 KILL.
#      (KILL 부터 하면 flock 파일 기술자·자식 정리가 안 된다.)
#   4. 잠금 파일 gpu_<G>.lock 은 **절대 지우지 않는다** — flock 은 inode 에 걸리므로, 지우면 새 워커가
#      다른 inode 에 잠그고 남아 있던 옛 워커/잡과 배타가 깨진다.
#   5. 새 워커 pid 를 queue/pids/gpu_<G>_slot_0.pid 에 쓴다(gpu_queue.py start-workers/stop-workers 규약).
#   6. STOP 을 지운다.
# env(테스트용 재지정): QUEUE_ROOT PY GPU_QUEUE_PY RESTART_LOG WORKER_LOG_DIR RESTART_POLL_S RESTART_TERM_WAIT_S RESTART_STOP_SETTLE_S
set -u
G="${1:?gpu required}"
Q="${QUEUE_ROOT:-/hdd_data/seungpil/queue}"
PY="${PY:-/hdd_data/seungpil/envs/simplerl/bin/python}"
cd "$(dirname "$0")/../.."
GQ="${GPU_QUEUE_PY:-$(pwd)/scripts/local/gpu_queue.py}"
LOG="${RESTART_LOG:-/hdd_data/seungpil/scratch/logs/cd8_decisions.log}"
WLOG_DIR="${WORKER_LOG_DIR:-/hdd_data/seungpil/scratch/logs}"
POLL="${RESTART_POLL_S:-60}"
TERM_WAIT="${RESTART_TERM_WAIT_S:-60}"
mkdir -p "$Q/pids" "$WLOG_DIR" "$(dirname "$LOG")"

log() { echo "[$(date '+%m-%d %H:%M')] restart_worker: gpu$G $*" | tee -a "$LOG"; }

busy() { for f in "$Q"/running/*.json; do [ -f "$f" ] || continue; $PY - "$f" "$G" <<'PY' && return 0
import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if str(j.get("assigned_gpu"))==sys.argv[2] else 1)
PY
done; return 1; }

alive() { kill -0 "$1" 2>/dev/null && [ "$(awk '/^State:/{print $2}' "/proc/$1/status" 2>/dev/null)" != "Z" ]; }

# 1–2. 비기를 기다린 뒤 STOP 아래에서 재확인(전역 STOP 은 짧게만 든다).
SETTLE="${RESTART_STOP_SETTLE_S:-12}"   # gpu_queue 워커 poll 5s + claim 여유
log "running 잡이 비기를 기다린다(STOP 없이)"
while :; do
  while busy; do sleep "$POLL"; done
  touch "$Q/STOP"                       # 이 시점부터 어떤 워커도 새 잡을 집지 않는다
  sleep "$SETTLE"
  if busy; then                         # 창 사이에 잡이 붙었다 — 풀고 다시 기다린다
    rm -f "$Q/STOP"; log "STOP 직전에 잡이 붙음 — 풀고 다시 기다린다"; continue
  fi
  break
done
log "STOP 아래에서 비었음 확정"
# 3. 옛 워커 TERM → 대기 → KILL.
OLD=$(pgrep -f "gpu_queue.py worker --gpu $G " || true)
how="none"
if [ -n "$OLD" ]; then
  kill -TERM $OLD 2>/dev/null || true
  how="TERM"
  deadline=$(( $(date +%s) + TERM_WAIT ))
  while :; do
    left=""
    for p in $OLD; do alive "$p" && left="$left $p"; done
    [ -z "$left" ] && break
    if [ "$(date +%s)" -ge "$deadline" ]; then
      kill -KILL $left 2>/dev/null || true
      how="TERM→KILL(${TERM_WAIT}s 안에 안 내려감:${left})"
      sleep 1
      break
    fi
    sleep 1
  done
fi
# 4. 잠금 파일은 건드리지 않는다. 5. 새 워커 + pid 파일.
nohup "$PY" "$GQ" worker --gpu "$G" --jobs-per-gpu 1 \
  >> "$WLOG_DIR/worker_gpu${G}_slot0.log" 2>&1 &
NEW=$!
echo "$NEW" > "$Q/pids/gpu_${G}_slot_0.pid"
# 6. STOP 해제.
rm -f "$Q/STOP"
log "옛 워커(${OLD:-없음}, $how) → 새 워커($NEW, pid 파일 기록), 새 코드(pid 기록·aborted·finalize)"
