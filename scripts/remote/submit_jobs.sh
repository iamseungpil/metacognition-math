#!/bin/bash
# submit_jobs.sh GROUP [--dry-run] [--smoke-ok]   remote_jobs.tsv 의 한 그룹을 gpu_queue 에 제출한다.
#   GROUP = r2prep (q17_prep_g*) | r1 (r1_*) | r2rl (q17_<팔>_s<씨앗>; --smoke-ok 필요 + 앞부분 데이터 필요)
#   잡 이름 = RUN_NAME (push_results.sh 키). 명령은 `cd <REPO> && export ROOT=… RUN_NAME=<이름> && <tsv 명령>` 으로 제출한다.
set -euo pipefail
ROOT=${ROOT:-/hdd_data/seungpil}; R_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=${REPO:-$(cd "$R_DIR/../.." && pwd)}
G=${1:?GROUP: r2prep|r1|r2rl}; shift || true
DRY=0; SMOKE=0; for a in "$@"; do case $a in --dry-run) DRY=1;; --smoke-ok) SMOKE=1;; esac; done
case $G in r2prep) PAT='^q17_prep_g';; r1) PAT='^r1_';; r2rl) PAT='^q17_(grpokl|pfx|reset)_s';; *) echo "GROUP?" >&2; exit 2;; esac
if [ $G = r2rl ]; then
  [ $SMOKE = 1 ] || { echo "[submit] r2rl 은 원격 스모크(SMOKE=1, 2스텝) 통과 후 --smoke-ok 로만 제출 (RUNBOOK 4절)" >&2; exit 3; }
  [ -s $ROOT/scratch/data/mc_pfx_commit_dapo17_screen_plain_le6000.parquet ] || { echo "[submit] 앞부분 데이터 없음 — r2prep 먼저" >&2; exit 3; }
  echo "[submit] 주의: r2rl 잡은 GPU 4장을 다 쓴다 — 워커는 GPU 하나(예: start-workers 0)만 띄워 직렬로 돌릴 것" >&2
fi
n=0
while IFS=$'\t' read -r name prio cmd hours check; do
  [[ $name =~ $PAT ]] || continue
  full="cd $REPO && export ROOT=$ROOT RUN_NAME=$name && $cmd"
  if [ $DRY = 1 ]; then echo "[dry] $name prio=$prio ~${hours}h :: $full"
  else python3 "$REPO/scripts/local/gpu_queue.py" submit --name "$name" --cmd "$full" --priority "$prio"; fi
  n=$((n+1))
done < <(grep -v '^#' "$R_DIR/remote_jobs.tsv" | tail -n +2)
echo "[submit] $G: $n개 (dry-run=$DRY)"
