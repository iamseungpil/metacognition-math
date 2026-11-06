#!/bin/bash
# push_loop.sh — 큐 완료 훅 대신 쓰는 반복문: 10분마다 $ROOT/queue/{done,failed} 의 새 잡을 push_results.sh 로 올린다(한 번 올린 잡은 표시).
#   사용: nohup bash scripts/remote/push_loop.sh > $ROOT/scratch/logs/push_loop.log 2>&1 &
ROOT=${ROOT:-/hdd_data/seungpil}; H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); mkdir -p "$ROOT/queue/pushed"
while true; do
  for st in done failed; do
    for f in "$ROOT"/queue/$st/*.json; do
      [ -e "$f" ] || continue
      n=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$f") || continue
      [ -e "$ROOT/queue/pushed/$n.$st" ] && continue
      bash "$H/queue_done_hook.sh" "$n" "$st" 0 && touch "$ROOT/queue/pushed/$n.$st"
    done
  done
  sleep 600
done
