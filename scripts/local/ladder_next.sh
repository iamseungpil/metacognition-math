#!/usr/bin/env bash
# scripts/local/ladder_next.sh <lineage> <next_submit_cmd>
#
# ★감사 10(0914) — 사전등록 §4 «중단되면 다음 팔 자동 제출»의 자동화. 다음 둘 중 하나가 되면
# <next_submit_cmd> 를 **한 번만** 실행하고 끝난다:
#   (a) ${WORK}/checkpoints/<lineage>/ABORTED.txt 가 생김 (트레이너의 사전등록 중단, rc 75)
#   (b) 그 잡이 queue/running/ 을 떠남 (done/failed/aborted 어느 쪽이든 — 사다리는 다음 칸으로)
# 잡은 running|pending 의 json 에서 name 또는 cmd 에 <lineage>(또는 LADDER_MATCH)가 들어있는 것.
# 아직 pending 이면 계속 기다린다. running 에도 pending 에도 한 번도 안 보이면
# LADDER_APPEAR_TIMEOUT_S(기본 600) 뒤 «never seen» 으로 기록하고 제출한다(잡이 이미 끝난 경우).
# 한 번만: <ckpt_dir>/.ladder_next_submitted 마커 — 다시 불러도 두 번 제출하지 않는다.
#
# env: WORK(기본 /hdd_data/seungpil/scratch) QUEUE_ROOT(기본 /hdd_data/seungpil/queue)
#      LADDER_POLL_S(기본 60) LADDER_APPEAR_TIMEOUT_S(기본 600) LADDER_MATCH(기본 <lineage>)
#      LADDER_LOG(기본 ${WORK}/logs/cd8_decisions.log)
set -u
LINEAGE="${1:?lineage required}"
SUBMIT_CMD="${2:?next_submit_cmd required}"
WORK="${WORK:-/hdd_data/seungpil/scratch}"
Q="${QUEUE_ROOT:-/hdd_data/seungpil/queue}"
POLL="${LADDER_POLL_S:-60}"
APPEAR_TIMEOUT="${LADDER_APPEAR_TIMEOUT_S:-600}"
MATCH="${LADDER_MATCH:-$LINEAGE}"
LOG="${LADDER_LOG:-${WORK}/logs/cd8_decisions.log}"
CKPT_DIR="${WORK}/checkpoints/${LINEAGE}"
MARKER="${CKPT_DIR}/.ladder_next_submitted"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%m-%d %H:%M')] ladder_next(${LINEAGE}): $*" | tee -a "$LOG"; }

# 큐 상태 디렉토리에 MATCH 를 담은 잡이 있나 (name 또는 cmd 부분 문자열). grep 만으로 충분히
# 보수적이다(json 문자열 이스케이프는 우리 이름/커맨드에 없다).
has_job() { grep -lF -- "$MATCH" "$Q/$1"/*.json >/dev/null 2>&1; }

if [ -f "$MARKER" ]; then
  log "이미 제출됨($MARKER) — 두 번 제출하지 않는다"
  exit 0
fi

seen=0
t0=$(date +%s)
reason=""
while :; do
  if [ -f "$CKPT_DIR/ABORTED.txt" ]; then
    reason="ABORTED.txt: $(head -c 200 "$CKPT_DIR/ABORTED.txt" | tr '\n' ' ')"; break
  fi
  if has_job running; then
    seen=1
  elif [ "$seen" = "1" ]; then
    reason="job left running/"; break
  elif has_job pending; then
    :   # 아직 순서를 기다린다
  elif [ $(( $(date +%s) - t0 )) -ge "$APPEAR_TIMEOUT" ]; then
    reason="job never seen in running/pending for ${APPEAR_TIMEOUT}s (already finished?)"; break
  fi
  sleep "$POLL"
done

log "트리거: ${reason} → 다음 팔 제출: ${SUBMIT_CMD}"
mkdir -p "$CKPT_DIR"
if out=$(bash -lc "$SUBMIT_CMD" 2>&1); then
  date '+%Y-%m-%dT%H:%M:%S' > "$MARKER"
  log "제출 성공: $(printf '%s' "$out" | tail -c 300)"
  exit 0
else
  rc=$?
  log "제출 실패 rc=${rc}: $(printf '%s' "$out" | tail -c 300)"
  exit "$rc"
fi
