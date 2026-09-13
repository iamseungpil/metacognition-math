#!/bin/bash
# ★0911: 단발(non-resume) 잡용 범용 재시도 래퍼. run_arm_retry.sh 는 trainer.resume_mode=auto
#   에 기대어 "같은 계보로 다시 띄우면 마지막 ckpt 에서 이어간다"는 전제가 있어 run_arm.sh
#   전용이다. eval/이어쓰기 같은 단발 잡은 상태가 없어 그 전제가 필요 없다 — 실패하면
#   그냥 통째로 다시 돌리면 된다. 두 번의 실측 사고(09-10 새벽 VTR/VTRW, 09-10 저녁
#   FIXED_CHK·PERSIST_CHK 평가 3건)가 전부 "단발 제출이라 재시도가 없어 조용히 죽어
#   방치됨"이었다 — ckpt_keeper 가 큐 제출 성공 시점에 완료 마커를 찍어, 잡 자체가 나중에
#   죽어도 다시 제출하지 않기 때문이다. 이 래퍼는 잡 **안에서** 재시도해 그 간극을 메운다.
# 사용법: retry_cmd.sh <최대시도> <대기초> -- <실행할 명령...>
MAX="${1:-5}"; WAIT="${2:-90}"; shift 2; [ "$1" = "--" ] && shift
for i in $(seq 1 "$MAX"); do
  echo "[retry_cmd] attempt $i/$MAX $(date -Is)"
  "$@" && { echo "[retry_cmd] done at attempt $i"; exit 0; }
  rc=$?
  # ★0913: rc 75 = 사전등록 중단(의도된 정지). 재시도 대상이 아니다.
  if [ "$rc" = "75" ]; then echo "[retry_cmd] preregistered ABORT (rc 75) — not retrying"; exit 75; fi
  echo "[retry_cmd] attempt $i failed (rc=$rc); sleeping ${WAIT}s"
  sleep "$WAIT"
done
echo "[retry_cmd] exhausted $MAX attempts"; exit 1
