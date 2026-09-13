#!/usr/bin/env python3
"""scripts/local/gpu_queue.py — a tiny file-based job queue + one worker per GPU.

Layout (fixed): /hdd_data/seungpil/queue/{pending,running,done,failed}/
Job file: JSON, e.g. {"cmd": "...", "name": "...", "gpus": 1, "priority": 10}

Design notes:
  * Atomicity: a job moves pending -> running via os.rename (same filesystem,
    atomic), so two workers racing on the same file never both "win" it — one
    rename succeeds, the other raises FileNotFoundError/OSError and moves on.
  * Per-GPU exclusivity: `worker --gpu N` takes an flock() on
    <queue_dir>/gpu_N.lock for the duration of running ONE job, so two worker
    processes pinned to the same GPU index never run jobs concurrently on it
    (this is what --jobs-per-gpu 1, the default, relies on). Pass
    --jobs-per-gpu >1 to run multiple worker processes against the same GPU
    with NO lock (explicitly requested in the task spec for future
    higher-density eval jobs) — the caller is responsible for those jobs
    actually fitting in one GPU's memory concurrently.
  * Disk guard: before starting a job, check shutil.disk_usage on /hdd_data and
    on / ; refuse to start (log "[DISK] waiting", requeue via not consuming the
    job — i.e. leave it in pending, actually since we already renamed it into
    running we move it BACK to pending) if /hdd_data usage > 85% or / free < 1GB (0905: 루트 200GB 를 타 프로젝트가 채워 15GB 기준은 상시 차단이 됨; 우리 잡은 /hdd_data 에만 씀).
  * A file named <queue_dir>/STOP pauses ALL workers (they poll and sleep)
    without killing them, so `start-workers` doesn't need to be re-run after
    unpausing.

This is intentionally dependency-free (stdlib only) so it runs inside the
simplerl conda env or the system python equally.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

QUEUE_ROOT = Path("/hdd_data/seungpil/queue")
STATES = ("pending", "running", "done", "failed")
WORK = Path(os.environ.get("WORK", "/hdd_data/seungpil/scratch"))
LOG_DIR = WORK / "logs"
PIDS_DIR = QUEUE_ROOT / "pids"

DISK_USAGE_PCT_MAX = 85.0
ROOT_FREE_GB_MIN = 1.0   # 0905: 루트 200GB 를 타 프로젝트(/tmp/ttso 55GB, envs 79GB)가 채움. 우리 잡은 /hdd_data 에만 쓴다


def _ensure_dirs() -> None:
    for s in STATES:
        (QUEUE_ROOT / s).mkdir(parents=True, exist_ok=True)
    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ── submit ────────────────────────────────────────────────────────────────
def cmd_submit(args: argparse.Namespace) -> int:
    _ensure_dirs()
    job_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    job = {
        "id": job_id,
        "cmd": args.cmd,
        "name": args.name,
        "gpus": args.gpus,
        "priority": args.priority,
        "need_mb": args.need_mb,
        "submitted_at": _now(),
    }
    path = QUEUE_ROOT / "pending" / f"{job_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job, indent=2))
    tmp.rename(path)  # atomic within pending/
    print(f"[submit] {job_id} name={args.name!r} priority={args.priority} -> {path}")
    return 0


# ── status ────────────────────────────────────────────────────────────────
def _gpu_memory_table(gpu_indices) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception as exc:  # nvidia-smi missing, no GPU visible, etc.
        return [{"index": i, "memory": f"<nvidia-smi failed: {exc}>"} for i in gpu_indices]
    rows = []
    for line in out.strip().splitlines():
        idx, used, total = (p.strip() for p in line.split(","))
        if int(idx) in gpu_indices:
            rows.append({"index": int(idx), "used_mb": int(used), "total_mb": int(total)})
    return rows


def cmd_status(args: argparse.Namespace) -> int:
    _ensure_dirs()
    counts = {}
    for s in STATES:
        files = sorted((QUEUE_ROOT / s).glob("*.json"))
        counts[s] = files

    print(f"queue root: {QUEUE_ROOT}")
    stop_flag = QUEUE_ROOT / "STOP"
    print(f"STOP flag present: {stop_flag.exists()}")
    print()
    for s in STATES:
        files = counts[s]
        print(f"== {s} ({len(files)}) ==")
        for f in files[:20]:
            try:
                job = json.loads(f.read_text())
            except Exception as exc:
                print(f"  {f.name}: <unreadable: {exc}>")
                continue
            extra = ""
            if s == "done":
                extra = f" rc={job.get('exit_code')} finished_at={job.get('finished_at')}"
            elif s == "failed":
                extra = f" rc={job.get('exit_code')} finished_at={job.get('finished_at')}"
            elif s == "running":
                extra = f" gpu={job.get('assigned_gpu')} started_at={job.get('started_at')}"
            print(f"  {f.stem} name={job.get('name')!r} priority={job.get('priority')}{extra}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        print()

    print("== GPUs 0-3 (nvidia-smi memory.used/memory.total MiB) ==")
    for row in _gpu_memory_table({0, 1, 2, 3}):
        print(f"  {row}")

    du_hdd = shutil.disk_usage("/hdd_data") if Path("/hdd_data").exists() else None
    du_root = shutil.disk_usage("/")
    if du_hdd:
        pct = 100.0 * du_hdd.used / du_hdd.total
        print(f"\n/hdd_data usage: {pct:.1f}% ({du_hdd.used/1e9:.0f}GB / {du_hdd.total/1e9:.0f}GB)")
    print(f"/ free: {du_root.free/1e9:.1f}GB")
    return 0


# ── disk guard ────────────────────────────────────────────────────────────
def _disk_ok() -> tuple[bool, str]:
    try:
        du_hdd = shutil.disk_usage("/hdd_data")
        pct = 100.0 * du_hdd.used / du_hdd.total
        if pct > DISK_USAGE_PCT_MAX:
            return False, f"/hdd_data at {pct:.1f}% > {DISK_USAGE_PCT_MAX}%"
    except FileNotFoundError:
        pass
    du_root = shutil.disk_usage("/")
    free_gb = du_root.free / 1e9
    if free_gb < ROOT_FREE_GB_MIN:
        return False, f"/ free {free_gb:.1f}GB < {ROOT_FREE_GB_MIN}GB"
    return True, "ok"


# ── worker ────────────────────────────────────────────────────────────────
def _gpu_free_mb(gpu: int) -> int | None:
    """nvidia-smi 로 이 카드의 남은 메모리(MiB). 못 읽으면 None(필터 안 함)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
            text=True, timeout=20)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None



def _gpu_cap_mb(gpu):
    """QUEUE_ROOT/caps.json = {"0": 45000, "3": 45000} — 그 카드에 올릴 수 있는 잡의 need_mb 상한. 없으면 None."""
    if gpu is None:
        return None
    try:
        caps = json.loads((QUEUE_ROOT / "caps.json").read_text())
    except Exception:
        return None
    v = caps.get(str(gpu))
    return int(v) if v is not None else None

PICK_HOLD_S = 90


def _pick_job(gpu: int | None = None) -> Path | None:
    """Return a pending job file sorted by (-priority, submitted order), or None.

    ★0904: 타인의 프로세스가 같은 카드에 상주할 수 있다(GPU 2 에 14GB). 잡의 `need_mb` 보다
    카드의 남은 메모리가 작으면 그 잡은 이 워커가 집지 않는다(다른 카드의 워커가 집는다).
    need_mb 가 없는 옛 잡은 필터하지 않는다.
    """
    pending = sorted((QUEUE_ROOT / "pending").glob("*.json"))
    if not pending:
        return None
    free = _gpu_free_mb(gpu) if gpu is not None else None

    def load(p: Path):
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    # ★0908: «큰 잡 기아» 수정. 카드가 막 비었을 때 nvidia-smi 의 free 는 직전 잡의 메모리가
    #   아직 회수되기 전 값이라, 우선순위 최상의 큰 잡(need 60GB)이 free 필터에 걸리고
    #   need 가 작은 평가 잡이 카드를 채 간다 — 이것이 11:06·11:26 두 번 반복돼 p99 학습 잡이
    #   p98 평가 뒤로 밀렸다. 우선순위 1등 후보가 cap 은 통과하지만 free 만 모자라면 최대
    #   PICK_HOLD_S 초 동안 5초 간격으로 free 를 다시 재고, 그동안 맞으면 그 잡을 집는다.
    #   끝내 안 맞으면 기존 규칙대로 맞는 잡 중 최상위를 집는다.
    _top = None
    for p in pending:
        job = load(p)
        need = int(job.get("need_mb", 0) or 0)
        cap = _gpu_cap_mb(gpu)
        if cap is not None and need and need > cap:
            continue
        key = (-int(job.get("priority", 0)), p.name)
        if _top is None or key < _top[0]:
            _top = (key, p, need)
    if _top is not None and free is not None and _top[2] and _top[2] > free:
        deadline = time.time() + PICK_HOLD_S
        while time.time() < deadline:
            time.sleep(5)
            free = _gpu_free_mb(gpu)
            if free is None or _top[2] <= free:
                print(f"[pick gpu={gpu}] hold ok: free={free} need={_top[2]} -> {_top[1].name}")
                return _top[1]
        print(f"[pick gpu={gpu}] hold expired: free={free} need={_top[2]} — falling back to fitting jobs")

    cands = []
    for p in pending:
        job = load(p)
        need = int(job.get("need_mb", 0) or 0)
        # ★0907: 카드별 상한(caps.json, 핫리로드). 타인이 «간헐적으로」 쓰는 카드(GPU 0·3)에는 큰 학습 잡을
        #   올리지 않는다 — 빈 틈에 올렸다가 상대가 돌아오면 vLLM/옵티마이저가 OOM 으로 죽는다(0907 3회).
        cap = _gpu_cap_mb(gpu)
        if cap is not None and need and need > cap:
            continue
        if free is not None and need and need > free:
            continue
        cands.append((-int(job.get("priority", 0)), p.name, p))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def _claim(job_path: Path) -> Path | None:
    """Atomically move a pending job file to running/. Returns the new path, or
    None if another worker already claimed it (race lost)."""
    dest = QUEUE_ROOT / "running" / job_path.name
    try:
        job_path.rename(dest)
        return dest
    except OSError:
        return None


def _run_one_job(gpu: int, job_path: Path, lock_fd) -> None:
    job = json.loads(job_path.read_text())
    job["assigned_gpu"] = gpu
    job["started_at"] = _now()
    job_path.write_text(json.dumps(job, indent=2))

    name = job.get("name", job_path.stem)
    log_path = LOG_DIR / f"queue_{name}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[worker gpu={gpu}] running {job_path.name} name={name!r} -> {log_path}")
    with open(log_path, "a") as logf:
        logf.write(f"\n=== [{_now()}] gpu={gpu} job={job_path.name} cmd={job['cmd']!r} ===\n")
        logf.flush()
        # 잡을 자기 세션(프로세스 그룹)으로 띄운다: 잡이 죽거나 우리가 죽이면 vLLM EngineCore
        # 같은 spawn 자식까지 함께 정리해야 GPU 메모리가 고아로 남지 않는다(0904 실측: 부모만
        # 죽자 EngineCore 49GB 가 카드에 남아 다음 잡이 OOM).
        proc = subprocess.Popen(["bash", "-lc", job["cmd"]], stdout=logf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        # ★0911 지표 게이트: proc.pid(=새 세션의 프로세스그룹 리더)를 running/ json에 즉시
        #   기록한다 — 외부 감시자가 "이름으로 grep해서 아무 pid나 죽이기"가 아니라 정확히
        #   이 pid의 killpg만 하도록. job_path는 이미 running/으로 옮겨져 있다.
        try:
            job["pid"] = proc.pid
            job_path.write_text(json.dumps(job, indent=2))
        except Exception:
            pass
        try:
            rc = proc.wait()
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(3)
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                pass

    job["exit_code"] = rc
    job["finished_at"] = _now()
    final_state = "done" if rc == 0 else "failed"
    # ★0913 감사 수리(경쟁): gate_judgment 가 잡을 죽이고 running/→aborted/ 로 옮긴 뒤 여기서
    #   write_text 하면 running/ 에 파일이 **되살아나** failed/ 로도 기록됐다(이중 기록).
    #   running/ 에 파일이 없으면 aborted/ 쪽 기록에 종료 정보만 덧붙이고 끝낸다.
    #   rc 75 는 사전등록 중단(verl_sdc.ABORT_EXIT_CODE) — failed 가 아니라 aborted 로 둔다.
    if not job_path.exists():
        alt = QUEUE_ROOT / "aborted" / job_path.name
        if alt.exists():
            try:
                j2 = json.loads(alt.read_text()); j2.update(exit_code=rc, finished_at=job["finished_at"])
                alt.write_text(json.dumps(j2, indent=2))
            except Exception:
                pass
        print(f"[worker gpu={gpu}] {job_path.name} already moved out of running/ (rc={rc})")
        return
    if rc == 75:
        final_state = "aborted"
        job["aborted_reason"] = job.get("aborted_reason") or "preregistered abort (rc 75)"
        (QUEUE_ROOT / "aborted").mkdir(parents=True, exist_ok=True)
    dest = QUEUE_ROOT / final_state / job_path.name
    job_path.write_text(json.dumps(job, indent=2))
    job_path.rename(dest)
    print(f"[worker gpu={gpu}] {job_path.name} -> {final_state} (rc={rc})")


def _worker_loop(gpu: int, use_lock: bool, poll_s: float) -> None:
    _ensure_dirs()
    lock_path = QUEUE_ROOT / f"gpu_{gpu}.lock"
    print(f"[worker gpu={gpu}] starting, pid={os.getpid()}, lock={'on' if use_lock else 'off'}")

    running = True

    def _sigterm(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    while running:
        if (QUEUE_ROOT / "STOP").exists():
            print(f"[worker gpu={gpu}] STOP flag present, sleeping")
            time.sleep(poll_s)
            continue

        ok, reason = _disk_ok()
        if not ok:
            print(f"[DISK] waiting: {reason}")
            time.sleep(poll_s)
            continue

        job_path = _pick_job(gpu)
        if job_path is None:
            time.sleep(poll_s)
            continue

        claimed = _claim(job_path)
        if claimed is None:
            continue  # lost the race, try again immediately

        if use_lock:
            lock_fd = open(lock_path, "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                _run_one_job(gpu, claimed, lock_fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        else:
            _run_one_job(gpu, claimed, None)


def cmd_worker(args: argparse.Namespace) -> int:
    _worker_loop(args.gpu, use_lock=(args.jobs_per_gpu <= 1), poll_s=args.poll_interval)
    return 0


# ── start-workers / stop-workers ─────────────────────────────────────────
def cmd_start_workers(args: argparse.Namespace) -> int:
    _ensure_dirs()
    this_file = os.path.abspath(__file__)
    for gpu in args.gpus:
        for slot in range(args.jobs_per_gpu):
            pid_file = PIDS_DIR / f"gpu_{gpu}_slot_{slot}.pid"
            if pid_file.exists():
                try:
                    old_pid = int(pid_file.read_text().strip())
                    os.kill(old_pid, 0)
                    print(f"[start-workers] gpu={gpu} slot={slot} already running (pid={old_pid}), skipping")
                    continue
                except (OSError, ValueError):
                    pass  # stale pid file
            log_path = LOG_DIR / f"worker_gpu{gpu}_slot{slot}.log"
            with open(log_path, "a") as logf:
                proc = subprocess.Popen(
                    [sys.executable, this_file, "worker", "--gpu", str(gpu),
                     "--jobs-per-gpu", str(args.jobs_per_gpu)],
                    stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid_file.write_text(str(proc.pid))
            print(f"[start-workers] gpu={gpu} slot={slot} pid={proc.pid} log={log_path}")
    return 0


def cmd_stop_workers(args: argparse.Namespace) -> int:
    _ensure_dirs()
    any_found = False
    for pid_file in sorted(PIDS_DIR.glob("*.pid")):
        any_found = True
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"[stop-workers] sent SIGTERM to {pid_file.name} pid={pid}")
        except (OSError, ValueError) as exc:
            print(f"[stop-workers] {pid_file.name}: {exc}")
        pid_file.unlink(missing_ok=True)
    if not any_found:
        print("[stop-workers] no pid files found (nothing to stop)")
    return 0


# ── argparse plumbing ────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("submit", help="add one job to pending/")
    sp.add_argument("--name", required=True)
    sp.add_argument("--cmd", required=True)
    sp.add_argument("--gpus", type=int, default=1)
    sp.add_argument("--priority", type=int, default=0)
    sp.add_argument("--need-mb", dest="need_mb", type=int, default=0,
                    help="이 잡이 필요로 하는 GPU 여유 메모리(MiB). 학습 70000, 생성/채점 40000 권장")
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("status", help="table of pending/running/done/failed + GPU memory")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("worker", help="run one worker loop pinned to --gpu (internal; used by start-workers)")
    sp.add_argument("--gpu", type=int, required=True)
    sp.add_argument("--jobs-per-gpu", type=int, default=1,
                     help="if >1, no flock is taken (multiple workers share the GPU with no exclusivity)")
    sp.add_argument("--poll-interval", type=float, default=5.0)
    sp.set_defaults(func=cmd_worker)

    sp = sub.add_parser("start-workers", help="nohup one worker process per GPU index given")
    sp.add_argument("gpus", type=int, nargs="+", help="GPU indices, e.g. 0 1 2 3")
    sp.add_argument("--jobs-per-gpu", type=int, default=1)
    sp.set_defaults(func=cmd_start_workers)

    sp = sub.add_parser("stop-workers", help="SIGTERM every worker started via start-workers")
    sp.set_defaults(func=cmd_stop_workers)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
