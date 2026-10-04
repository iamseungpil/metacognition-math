#!/usr/bin/env python
"""gate_judgment.py — 판정 스텝(기본 30) 지표를 보고 중단/계속을 자동 결정한다.

사용자 지시(0911): "중간중간 모니터링은 아니더라도 확인하고 의도대로 중간 지표들이
안 나오면 바로 중단하고 다음 실험 돌려줘." 이 스크립트는 promote_smoke.sh 와 짝을
이루는 워처다 — promote_smoke.sh 는 "스모크가 무사히 끝났다"만 보고 무조건 승격하고,
이 스크립트는 본실험이 첫 판정 스텝(기본 30)에 도달했을 때 held-out 정확도가 문턱을
넘는지 보고, 못 넘으면 그 잡을 **정확한 pid** 로만 종료한다(이름/커맨드 문자열
grep으로 아무 pid나 잡아 죽이지 않는다 — 이 프로젝트에서 과거 오폭 사고가 있었다).

안전장치:
  1. running/ 큐 json에 gpu_queue.py가 기록한 pid만 쓴다(0911 추가 배선).
  2. 죽이기 전 /proc/<pid>/cmdline 을 읽어 이 잡의 arm 토큰이 실제로 들어있는지
     재확인한다 — pid가 이미 재활용돼 다른 프로세스가 됐으면 죽이지 않고 그냥
     "이미 사라짐"으로 기록하고 종료한다.
  3. killpg(SIGTERM) 후 유예 뒤 SIGKILL — gpu_queue.py 워커 자신의 종료 절차와 동일.
  4. 죽인 잡은 running/ 에서 queue/aborted/ 로 옮긴다(done/failed 를 오염시키지 않음).
  5. 판단 결과는 성공/실패 어느 쪽이든 cd8_decisions.log 에 근거 숫자와 함께 남긴다.

사용법:
  gate_judgment.py --lineage cd7_NOSURR_CHK_chk_s1 --job-name w13_NOSURR_CHK_full_s1 \
    --step 30 --min-acc 0.60 --baseline "EVCM s30 acc" \
    [--on-abort-cmd "python scripts/local/gpu_queue.py submit --name ... --cmd ..."]

★B3(0914): 재시도 팔(M_RETRY/M_RETRY_RAND)은 math_retry_eval.py 가 eval/<lineage>/step_<N>/
math500_retry/telemetry.json 에 final_acc 키로 채점을 낸다(기본 math500/acc 와 다른 레이아웃) —
--eval-subdir math500_retry --acc-key final_acc 로 넘겨라. 기본값(math500/acc)은 기존 팔과
바이트 동일 동작.

--min-acc 미달이면 종료 + (있으면) --on-abort-cmd 실행. 충분하면 아무 것도 안 하고
(잡은 이미 스스로 s50/100까지 이어간다) 로그만 남기고 끝난다.

★감사 10(0914): --abort-file <ckpt_dir>/ABORTED.txt 를 주면 telemetry 를 기다리는 동안 그 파일도
같이 본다 — 트레이너가 사전등록 중단(rc 75)으로 스스로 죽으면 판정 스텝 eval 은 영영 안 나오므로,
파일이 나타나면 바로 --on-abort-cmd(다음 팔 제출)를 실행하고 0 으로 끝난다(잡은 이미 죽었다).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/hdd_data/seungpil/scratch"))
QUEUE = Path(os.environ.get("QUEUE_ROOT", "/hdd_data/seungpil/queue"))
REPO = Path(os.environ.get("REPO_ROOT", "/home/ubuntu/seungpil/metacognition-math"))
LOG = WORK / "logs" / "cd8_decisions.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M')}] gate_judgment: {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def find_running_job(job_name: str) -> tuple[Path, dict] | tuple[None, None]:
    for f in (QUEUE / "running").glob("*.json"):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        if j.get("name") == job_name:
            return f, j
    return None, None


def pid_cmdline_contains(pid: int, token: str) -> bool:
    """/proc/<pid>/cmdline 에 token 이 실제로 들어있는지 — pid 재활용 오폭 방지."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return False
    text = raw.replace(b"\x00", b" ").decode("utf-8", "ignore")
    return token in text


def kill_job_exact_pid(job_path: Path, job: dict, arm_token: str) -> str:
    """정확한 pid만 죽인다. 반환값은 사람이 읽을 상태 문자열."""
    pid = job.get("pid")
    if pid is None:
        return "SKIP(no pid recorded — gpu_queue.py가 이 잡을 pid 배선 이전에 시작했을 수 있다, 수동 확인 필요)"
    # ★0913 감사 수리: 기본 토큰(--job-name)은 프로세스 cmdline 에 **없다** — gpu_queue 는
    #   `bash -lc "<cmd>"` 로 띄우므로 cmdline 엔 잡 이름이 아니라 명령 문자열이 들어간다.
    #   그래서 기본 호출은 항상 SKIP 이었고 트리거가 한 번도 실제로 죽인 적이 없다.
    #   잡 json 의 cmd 문자열(그 잡만의 고유 문자열)도 토큰으로 인정한다.
    _tokens = [t for t in (arm_token, job.get("cmd")) if t]
    if not any(pid_cmdline_contains(int(pid), t) for t in _tokens):
        return f"SKIP(pid {pid}의 cmdline에 '{arm_token}'/잡 cmd 없음 — 이미 끝났거나 pid 재활용, 죽이지 않음)"
    try:
        os.killpg(int(pid), signal.SIGTERM)
        time.sleep(5)
        os.killpg(int(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:  # noqa: BLE001
        return f"ERROR(killpg 실패: {e!r})"
    aborted_dir = QUEUE / "aborted"
    aborted_dir.mkdir(parents=True, exist_ok=True)
    job["aborted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    job["aborted_reason"] = "gate_judgment: 판정 스텝 지표 미달"
    try:
        job_path.write_text(json.dumps(job, indent=2))
        job_path.rename(aborted_dir / job_path.name)
    except Exception:
        pass
    return f"KILLED(pid {pid} killpg, queue/aborted/{job_path.name}로 이동)"


def read_acc(lineage: str, step: int, *, eval_subdir: str = "math500_8k", acc_key: str = "acc") -> float | None:
    """★B3(0914): eval_subdir/acc_key 를 인자화 — 재시도 팔은 eval 출력이
    `<lineage>/step_<N>/math500_retry_8k/telemetry.json` 이고 채점 키가 `final_acc` 다
    (math_retry_eval.py). ★EVAL_MAX_TOKENS(0914b): run_math_arm.sh 가 사후 eval 예산을
    RESP_LEN 과 분리해 EVAL_MAX_TOKENS(기본 8192)로 고정하면서 출력 폴더 이름에 예산 태그가
    붙었다(math500 -> math500_8k) — 기본값도 같이 옮긴다."""
    tel = WORK / "eval" / lineage / f"step_{step}" / eval_subdir / "telemetry.json"
    if not tel.exists():
        return None
    try:
        return float(json.loads(tel.read_text())[acc_key])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", required=True, help="cd7_<ARM>_<variant>_s<seed> — eval 출력 폴더 이름")
    ap.add_argument("--job-name", required=True, help="큐 잡 이름(gpu_queue.py submit --name) — running/ 검색용")
    ap.add_argument("--arm-token", default=None, help="pid 재확인용 고유 토큰(기본: --job-name 그대로)")
    ap.add_argument("--step", type=int, default=30)
    # ★EVAL_MAX_TOKENS(0914b): run_math_arm.sh 가 사후 eval 폴더 이름에 예산 태그를 새기도록
    #   바뀌었다(math500 -> math500_8k, 재시도 팔은 math500_retry_8k) — 기본 EVAL_MAX_TOKENS=8192
    #   기준으로 기본값도 같이 옮긴다. run_math_arm.sh 와 바이트 동일하게 맞출 것(둘 다 손으로
    #   고치면 어긋난다).
    ap.add_argument("--eval-subdir", default="math500_8k",
                    help="eval/<lineage>/step_<N>/<subdir>/telemetry.json (재시도 팔: math500_retry_8k)")
    ap.add_argument("--acc-key", default="acc",
                    help="telemetry.json 안 정확도 키(재시도 팔의 math_retry_eval.py 출력: final_acc)")
    ap.add_argument("--min-acc", type=float, required=True)
    ap.add_argument("--baseline", default="", help="사람이 읽을 근거 설명(로그용)")
    ap.add_argument("--on-abort-cmd", default=None, help="문턱 미달 시 실행할 셸 커맨드(다음 실험 제출용)")
    ap.add_argument("--abort-file", default=None,
                    help="<ckpt_dir>/ABORTED.txt — 사전등록 중단 마커. 나타나면 --on-abort-cmd 를 실행한다")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--timeout-s", type=int, default=6 * 3600)
    a = ap.parse_args()
    arm_token = a.arm_token or a.job_name

    log(f"watching lineage={a.lineage} job={a.job_name} step={a.step} min_acc={a.min_acc} baseline='{a.baseline}'")
    deadline = time.time() + a.timeout_s
    while time.time() < deadline:
        if a.abort_file and Path(a.abort_file).exists():
            log(f"ABORTED.txt 감지({a.abort_file}) — 트레이너가 사전등록 중단으로 스스로 멈췄다: "
                f"{Path(a.abort_file).read_text()[:200]!r}")
            _run_on_abort(a)
            return 0
        acc = read_acc(a.lineage, a.step, eval_subdir=a.eval_subdir, acc_key=a.acc_key)
        if acc is not None:
            break
        # 잡 자체가 실패로 끝나 다시는 이 스텝 eval이 안 나올 상황도 감지한다.
        still_alive = find_running_job(a.job_name)[0] is not None
        still_pending = any(
            json.loads(f.read_text()).get("name") == a.job_name for f in (QUEUE / "pending").glob("*.json")
        )
        if not still_alive and not still_pending:
            failed = any(
                json.loads(f.read_text()).get("name") == a.job_name for f in (QUEUE / "failed").glob("*.json")
            )
            if failed:
                log(f"'{a.job_name}' 이 step {a.step} eval 전에 failed로 끝남 — 게이트 판단 불가, 사람 확인 필요")
                return 1
        time.sleep(a.poll_s)
    else:
        log(f"'{a.lineage}' step {a.step} eval telemetry.json이 {a.timeout_s}s 안에 안 나타남 — 타임아웃, 사람 확인 필요")
        return 1

    if acc >= a.min_acc:
        log(f"PASS: {a.lineage} step{a.step} acc={acc:.4f} >= min_acc={a.min_acc:.4f} (기준: {a.baseline}) — 계속 진행, 조치 없음")
        return 0

    log(f"FAIL: {a.lineage} step{a.step} acc={acc:.4f} < min_acc={a.min_acc:.4f} (기준: {a.baseline}) — 잡 중단 시도")
    jp, job = find_running_job(a.job_name)
    if jp is None:
        log(f"'{a.job_name}' 이 이미 running/ 에 없음(스스로 끝났거나 다른 이유로 옮겨짐) — 죽일 필요 없음")
    else:
        status = kill_job_exact_pid(jp, job, arm_token)
        log(f"abort action: {status}")

    _run_on_abort(a)
    return 0


def _run_on_abort(a) -> None:
    if not a.on_abort_cmd:
        return
    log(f"다음 실험 제출 시도: {a.on_abort_cmd}")
    r = subprocess.run(["bash", "-lc", a.on_abort_cmd], cwd=str(REPO), capture_output=True, text=True)
    if r.returncode == 0:
        log(f"다음 실험 제출 성공: {r.stdout.strip()[-300:]}")
    else:
        log(f"다음 실험 제출 실패 rc={r.returncode}: {r.stderr.strip()[-300:]}")


if __name__ == "__main__":
    sys.exit(main())
