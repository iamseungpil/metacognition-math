#!/bin/bash
# ★0909: 공유 카드에서 외부 잡이 커졌다 작아졌다 하면 우리 학습이 액터 업데이트 순간 OOM 으로
#   죽는다(0909 새벽 6회). run_arm.sh 는 trainer.resume_mode=auto 라 같은 계보로 다시 띄우면
#   마지막 ckpt(5스텝 간격)에서 이어간다 — 그래서 한 큐 슬롯이 카드를 잡은 채 재시도한다.
#   사용법: run_arm_retry.sh <최대시도> <대기초> -- <run_arm.sh 인자들>
MAX="${1:-20}"; WAIT="${2:-240}"; shift 2; [ "$1" = "--" ] && shift
for i in $(seq 1 "$MAX"); do
  echo "[retry] attempt $i/$MAX $(date -Is)"
  bash scripts/local/run_arm.sh "$@" && { echo "[retry] done at attempt $i"; exit 0; }
  rc=$?
  # ★0913: 사전등록 중단(verl_sdc 가 rc 75 로 나온다)은 크래시가 아니다 — 재시도하면
  #   ckpt 에서 이어져 같은 스텝에서 또 중단된다(PL_NG 41회 재발사, GPU3 하루 낭비).
  if [ "$rc" = "75" ]; then echo "[retry] preregistered ABORT (rc 75) — not retrying"; exit 75; fi
  echo "[retry] attempt $i failed (rc=$rc); sleeping ${WAIT}s"
  sleep "$WAIT"
done
echo "[retry] exhausted $MAX attempts"; exit 1
